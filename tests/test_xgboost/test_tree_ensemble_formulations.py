"""Tests for the ensemble-level tree formulations with XGBoost.

Mirrors the cross-formulation agreement, lifecycle and model-size tests of
``tests/test_sklearn/test_tree_ensemble_formulations.py`` (where the checks
are documented) for the XGBoost wiring.

XGBoost branches left iff ``x < t`` (strict) and its thresholds sit at
(float32) data values; extraction normalizes to the ``x <= t`` convention by
shifting thresholds down by epsilon. At the default ``epsilon = 0`` the
comparisons below share the closure semantics of the other frameworks.
"""

import json
import unittest
import warnings

import gurobipy as gp
import numpy as np
import xgboost as xgb
from gurobipy import GRB, GurobiError
from sklearn import datasets

from gurobi_ml import add_predictor_constr

from ..tree_ensemble import (
    representable_cells,
    thresholds_by_feature,
    ties_duplicated_threshold,
)

EPSILON = 0.0


def _tree_pairs(regressor):
    """The (feature, threshold) pairs of each tree of the booster.

    Thresholds are read exactly as the extraction does (float32, no shift at
    epsilon=0) so that cell probing and tie detection see the MIP's values.
    """
    raw = json.loads(regressor.get_booster().save_raw(raw_format="json"))
    pairs = []
    for tree in raw["learner"]["gradient_booster"]["model"]["trees"]:
        split_nodes = np.array(tree["left_children"]) >= 0
        features = np.array(tree["split_indices"])[split_nodes]
        thresholds = np.array(tree["split_conditions"], dtype=np.float32)[split_nodes]
        pairs.append(set(zip(features.tolist(), thresholds.tolist())))
    return pairs


class TestCrossFormulationAgreement(unittest.TestCase):
    """Cross-formulation agreement for XGBoost regressors (reg:squarederror)."""

    def _optimize(self, predictor, X, formulation, sense):
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, X.shape[1]), lb=X.min(axis=0), ub=X.max(axis=0))
            pred_constr = add_predictor_constr(
                gpm, predictor, x, epsilon=EPSILON, formulation=formulation
            )
            gpm.setObjective(pred_constr.output.sum(), sense)
            try:
                gpm.optimize()
            except GurobiError as error:
                if error.errno == 10010:
                    warnings.warn(UserWarning("Limited license"))
                    self.skipTest("Model too large for limited license")
                raise
            self.assertEqual(gpm.Status, GRB.OPTIMAL)
            self.assertLessEqual(gpm.MaxVio, gpm.Params.FeasibilityTol)
            return gpm.ObjVal, np.array(x.X[0, :]), gpm.Params.FeasibilityTol

    def _assert_predict_reproduces(self, predictor, objective, x_star, feas_tol):
        cells = representable_cells(
            x_star, thresholds_by_feature(_tree_pairs(predictor)), feas_tol
        )
        predictions = predictor.predict(cells)
        tolerance = 1e-4 * (1.0 + abs(objective))
        self.assertLessEqual(
            np.min(np.abs(predictions - objective)),
            tolerance,
            msg="MIP optimum not reproduced by predict on any representable cell",
        )

    def test_diabetes_agreement(self):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]

        for random_state in (17, 42):
            predictor = xgb.XGBRegressor(
                n_estimators=5, max_depth=3, random_state=random_state
            ).fit(X, y)
            for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
                with self.subTest(random_state=random_state, sense=sense):
                    obj_leaf, x_leaf, feas_tol = self._optimize(
                        predictor, X, "leaf", sense
                    )
                    obj_misic, x_misic, _ = self._optimize(predictor, X, "misic", sense)

                    # The shared z couples the trees, so the misic optimum
                    # always corresponds to a real input.
                    self._assert_predict_reproduces(
                        predictor, obj_misic, x_misic, feas_tol
                    )

                    # Each solve is proven optimal within the default
                    # relative MIPGap of 1e-4.
                    tolerance = 3e-4 * max(1.0, abs(obj_misic))

                    # Every misic-feasible point is leaf-feasible, so the
                    # leaf optimum dominates.
                    if sense == GRB.MAXIMIZE:
                        self.assertGreaterEqual(obj_leaf, obj_misic - tolerance)
                    else:
                        self.assertLessEqual(obj_leaf, obj_misic + tolerance)

                    # Without a duplicated-threshold tie the leaf formulation
                    # cannot mix branches across trees.
                    if not ties_duplicated_threshold(
                        _tree_pairs(predictor), x_leaf, feas_tol
                    ):
                        self.assertLessEqual(abs(obj_leaf - obj_misic), tolerance)
                        self._assert_predict_reproduces(
                            predictor, obj_leaf, x_leaf, feas_tol
                        )


class TestLifecycle(unittest.TestCase):
    """``remove()`` restores the model; unknown formulations raise; a
    positive epsilon warns."""

    def setUp(self):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        self.n_features = X.shape[1]
        self.predictor = xgb.XGBRegressor(
            n_estimators=4, max_depth=2, random_state=0
        ).fit(X, y)

    def test_add_remove(self):
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((2, self.n_features), lb=-GRB.INFINITY)
            y = gpm.addMVar((2, 1), lb=-GRB.INFINITY)
            gpm.update()
            numvars = gpm.NumVars

            pred_constr = add_predictor_constr(
                gpm, self.predictor, x, y, formulation="misic"
            )
            self.assertEqual(gpm.NumVars, numvars + len(pred_constr.vars))
            self.assertEqual(gpm.NumConstrs, len(pred_constr.constrs))
            self.assertEqual(gpm.NumGenConstrs, len(pred_constr.genconstrs))

            pred_constr.remove()
            gpm.update()
            self.assertEqual(gpm.NumVars, numvars)
            self.assertEqual(gpm.NumConstrs, 0)
            self.assertEqual(gpm.NumGenConstrs, 0)

    def test_unknown_formulation(self):
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((1, self.n_features), lb=-GRB.INFINITY)
            with self.assertRaisesRegex(ValueError, "Unknown formulation"):
                add_predictor_constr(
                    gpm, self.predictor, x, formulation="no_such_formulation"
                )

    def test_positive_epsilon_warns_for_ensemble_formulation(self):
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((1, self.n_features), lb=-GRB.INFINITY)
            with self.assertWarnsRegex(UserWarning, "applies globally"):
                add_predictor_constr(
                    gpm, self.predictor, x, epsilon=1e-5, formulation="misic"
                )


class TestModelSize(unittest.TestCase):
    """With unbounded inputs the binary count equals the number of distinct
    (feature, threshold) pairs of the ensemble — not one copy per tree."""

    def test_misic_binaries_are_shared_split_variables(self):
        rng = np.random.RandomState(0)
        # Integer-valued features so split thresholds coincide across trees
        # and threshold sharing is actually exercised.
        X = rng.randint(0, 5, size=(200, 4)).astype(float)
        y = rng.uniform(size=200)
        nex = 2

        for n_estimators in (2, 6):
            predictor = xgb.XGBRegressor(
                n_estimators=n_estimators, max_depth=3, random_state=0
            ).fit(X, y)

            tree_pairs = _tree_pairs(predictor)
            n_shared = sum(
                len(values) for values in thresholds_by_feature(tree_pairs).values()
            )
            n_splits = sum(len(pairs) for pairs in tree_pairs)
            raw = json.loads(predictor.get_booster().save_raw(raw_format="json"))
            n_leaves = sum(
                int((np.array(tree["left_children"]) < 0).sum())
                for tree in raw["learner"]["gradient_booster"]["model"]["trees"]
            )

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar(
                        (nex, X.shape[1]), lb=-GRB.INFINITY, ub=GRB.INFINITY
                    )
                    add_predictor_constr(gpm, predictor, x, formulation="misic")
                    gpm.update()

                    self.assertEqual(gpm.NumBinVars, nex * n_shared)
                    self.assertLess(n_shared, n_splits)
                    # Total: input + output + the trees-sum variable + split
                    # binaries + one continuous leaf variable per example
                    # and leaf.
                    self.assertEqual(
                        gpm.NumVars,
                        nex * X.shape[1] + nex + nex + nex * n_shared + nex * n_leaves,
                    )
                    self.assertEqual(gpm.NumGenConstrs, 2 * nex * n_shared)


if __name__ == "__main__":
    unittest.main()
