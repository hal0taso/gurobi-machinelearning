"""Tests for the ensemble-level tree formulations with LightGBM.

Mirrors the cross-formulation agreement, lifecycle and model-size tests of
``tests/test_sklearn/test_tree_ensemble_formulations.py`` (where the checks
are documented) for the LightGBM wiring.
"""

import unittest
import warnings

import gurobipy as gp
import lightgbm as lgb
import numpy as np
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

    Thresholds are cast to float32 exactly as the extraction does, so that
    cell probing and tie detection see the MIP's values.
    """
    pairs = []
    for tree in regressor.booster_.dump_model()["tree_info"]:
        tree_pairs = set()
        heap = [tree["tree_structure"]]
        while heap:
            node = heap.pop()
            if "split_index" in node:
                tree_pairs.add(
                    (node["split_feature"], float(np.float32(node["threshold"])))
                )
                heap.append(node["left_child"])
                heap.append(node["right_child"])
        pairs.append(tree_pairs)
    return pairs


class TestCrossFormulationAgreement(unittest.TestCase):
    """Cross-formulation agreement for LightGBM regressors."""

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

    def _diabetes_predictor(self, random_state):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        return X, lgb.sklearn.LGBMRegressor(
            n_estimators=5,
            max_depth=3,
            random_state=random_state,
            verbose=-1,
        ).fit(X, y)

    def _diabetes_agreement(self, formulation):
        """Compare the given shared-z formulation against the leaf baseline
        (one test method per formulation, so failures name it)."""
        for random_state in (17, 42):
            X, predictor = self._diabetes_predictor(random_state)
            for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
                with self.subTest(random_state=random_state, sense=sense):
                    obj_leaf, x_leaf, feas_tol = self._optimize(
                        predictor, X, "leaf", sense
                    )
                    objective, x_star, _ = self._optimize(
                        predictor, X, formulation, sense
                    )
                    # The shared z couples the trees, so a shared-z optimum
                    # always corresponds to a real input.
                    self._assert_predict_reproduces(
                        predictor, objective, x_star, feas_tol
                    )

                    # Each solve is proven optimal within the default
                    # relative MIPGap of 1e-4.
                    tolerance = 3e-4 * max(1.0, abs(objective))

                    # Every shared-z-feasible point is leaf-feasible, so the
                    # leaf optimum dominates.
                    if sense == GRB.MAXIMIZE:
                        self.assertGreaterEqual(obj_leaf, objective - tolerance)
                    else:
                        self.assertLessEqual(obj_leaf, objective + tolerance)

                    # Without a duplicated-threshold tie the leaf formulation
                    # cannot mix branches across trees.
                    if not ties_duplicated_threshold(
                        _tree_pairs(predictor), x_leaf, feas_tol
                    ):
                        self.assertLessEqual(abs(obj_leaf - objective), tolerance)
                        self._assert_predict_reproduces(
                            predictor, obj_leaf, x_leaf, feas_tol
                        )

    def test_diabetes_agreement_misic(self):
        self._diabetes_agreement("misic")

    def test_diabetes_agreement_parmentier_vidal(self):
        self._diabetes_agreement("parmentier_vidal")

    def test_diabetes_agreement_ocean(self):
        """Ocean sits between the families: misic <= ocean <= leaf
        (maximize). See tests/test_sklearn/test_tree_ensemble_formulations.py."""
        for random_state in (17, 42):
            X, predictor = self._diabetes_predictor(random_state)
            for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
                with self.subTest(random_state=random_state, sense=sense):
                    obj_leaf, _, feas_tol = self._optimize(predictor, X, "leaf", sense)
                    obj_misic, _, _ = self._optimize(predictor, X, "misic", sense)
                    objective, x_star, _ = self._optimize(predictor, X, "ocean", sense)
                    tolerance = 3e-4 * max(1.0, abs(objective))
                    # Exact-arithmetic ordering: misic <= ocean <= leaf
                    # (maximize; reversed when minimizing) — see the sklearn
                    # test module for why equality with leaf is unsound.
                    if sense == GRB.MAXIMIZE:
                        self.assertGreaterEqual(objective, obj_misic - tolerance)
                        self.assertLessEqual(objective, obj_leaf + tolerance)
                    else:
                        self.assertLessEqual(objective, obj_misic + tolerance)
                        self.assertGreaterEqual(objective, obj_leaf - tolerance)
                    if not ties_duplicated_threshold(
                        _tree_pairs(predictor), x_star, feas_tol
                    ):
                        self._assert_predict_reproduces(
                            predictor, objective, x_star, feas_tol
                        )

    def test_diabetes_agreement_biggs_perakis(self):
        """Sandwich misic <= biggs_perakis <= leaf (maximize) — see the
        sklearn test module."""
        for random_state in (17, 42):
            X, predictor = self._diabetes_predictor(random_state)
            for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
                with self.subTest(random_state=random_state, sense=sense):
                    obj_leaf, _, feas_tol = self._optimize(predictor, X, "leaf", sense)
                    obj_misic, _, _ = self._optimize(predictor, X, "misic", sense)
                    objective, x_star, _ = self._optimize(
                        predictor, X, "biggs_perakis", sense
                    )
                    tolerance = 3e-4 * max(1.0, abs(objective))
                    if sense == GRB.MAXIMIZE:
                        self.assertGreaterEqual(objective, obj_misic - tolerance)
                        self.assertLessEqual(objective, obj_leaf + tolerance)
                    else:
                        self.assertLessEqual(objective, obj_misic + tolerance)
                        self.assertGreaterEqual(objective, obj_leaf - tolerance)
                    if not ties_duplicated_threshold(
                        _tree_pairs(predictor), x_star, feas_tol
                    ):
                        self._assert_predict_reproduces(
                            predictor, objective, x_star, feas_tol
                        )

    def test_shared_formulations_agree(self):
        """The shared-z formulations solve the same problem: their optima
        must always agree (no knife-edge exception applies)."""
        for random_state in (17, 42):
            X, predictor = self._diabetes_predictor(random_state)
            for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
                with self.subTest(random_state=random_state, sense=sense):
                    obj_misic, _, _ = self._optimize(predictor, X, "misic", sense)
                    tolerance = 3e-4 * max(1.0, abs(obj_misic))
                    for formulation in ("parmentier_vidal",):
                        objective, _, _ = self._optimize(
                            predictor, X, formulation, sense
                        )
                        self.assertLessEqual(
                            abs(obj_misic - objective), tolerance, msg=formulation
                        )


class TestLifecycle(unittest.TestCase):
    """``remove()`` restores the model; unknown formulations raise; a
    positive epsilon warns."""

    def setUp(self):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        self.n_features = X.shape[1]
        self.predictor = lgb.sklearn.LGBMRegressor(
            n_estimators=4, max_depth=2, random_state=0, verbose=-1
        ).fit(X, y)

    def test_add_remove_misic(self):
        self._check_add_remove("misic")

    def test_add_remove_parmentier_vidal(self):
        self._check_add_remove("parmentier_vidal")

    def test_add_remove_ocean(self):
        self._check_add_remove("ocean")

    def test_add_remove_biggs_perakis(self):
        self._check_add_remove("biggs_perakis")

    def _check_add_remove(self, formulation):
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((2, self.n_features), lb=-1000.0, ub=1000.0)
            y = gpm.addMVar((2, 1), lb=-GRB.INFINITY)
            gpm.update()
            numvars = gpm.NumVars

            pred_constr = add_predictor_constr(
                gpm, self.predictor, x, y, formulation=formulation
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
        X = rng.randint(0, 5, size=(300, 4)).astype(float)
        y = rng.uniform(size=300)
        nex = 2

        for n_estimators in (2, 6):
            predictor = lgb.sklearn.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=3,
                random_state=0,
                verbose=-1,
            ).fit(X, y)

            tree_pairs = _tree_pairs(predictor)
            n_shared = sum(
                len(values) for values in thresholds_by_feature(tree_pairs).values()
            )
            n_splits = sum(len(pairs) for pairs in tree_pairs)
            n_leaves = sum(
                tree["num_leaves"]
                for tree in predictor.booster_.dump_model()["tree_info"]
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
                    # Total: input + output + split binaries + one continuous
                    # leaf variable per example and leaf.
                    self.assertEqual(
                        gpm.NumVars,
                        nex * X.shape[1] + nex + nex * n_shared + nex * n_leaves,
                    )
                    self.assertEqual(gpm.NumGenConstrs, 2 * nex * n_shared)

    def test_parmentier_vidal_binaries_scale_with_depth(self):
        rng = np.random.RandomState(0)
        X = rng.randint(0, 5, size=(300, 4)).astype(float)
        y = rng.uniform(size=300)
        nex = 2

        for n_estimators in (2, 6):
            predictor = lgb.sklearn.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=3,
                random_state=0,
                verbose=-1,
            ).fit(X, y)

            tree_pairs = _tree_pairs(predictor)
            n_shared = sum(
                len(values) for values in thresholds_by_feature(tree_pairs).values()
            )
            n_nodes = 0
            n_levels = 0
            for tree in predictor.booster_.dump_model()["tree_info"]:
                max_split_depth = 0
                count = 0
                heap = [(tree["tree_structure"], 0)]
                while heap:
                    node, depth = heap.pop()
                    count += 1
                    if "split_index" in node:
                        max_split_depth = max(max_split_depth, depth)
                        heap.append((node["left_child"], depth + 1))
                        heap.append((node["right_child"], depth + 1))
                n_nodes += count
                n_levels += max_split_depth + 1

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar(
                        (nex, X.shape[1]), lb=-GRB.INFINITY, ub=GRB.INFINITY
                    )
                    add_predictor_constr(
                        gpm, predictor, x, formulation="parmentier_vidal"
                    )
                    gpm.update()

                    # Shared split binaries plus one branching binary per
                    # example, tree and depth level.
                    self.assertEqual(gpm.NumBinVars, nex * (n_shared + n_levels))
                    # Total: input + output + split and branching binaries +
                    # one continuous flow variable per example and node.
                    self.assertEqual(
                        gpm.NumVars,
                        nex * X.shape[1]
                        + nex
                        + nex * (n_shared + n_levels)
                        + nex * n_nodes,
                    )
                    self.assertEqual(gpm.NumGenConstrs, 2 * nex * n_shared)

    def test_ocean_binaries_are_only_branching_binaries(self):
        rng = np.random.RandomState(0)
        X = rng.randint(0, 5, size=(300, 4)).astype(float)
        y = rng.uniform(size=300)
        nex = 2

        for n_estimators in (2, 6):
            predictor = lgb.sklearn.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=3,
                random_state=0,
                verbose=-1,
            ).fit(X, y)

            thresholds = thresholds_by_feature(_tree_pairs(predictor))
            n_intervals = sum(len(values) + 1 for values in thresholds.values())
            n_nodes = 0
            n_levels = 0
            for tree in predictor.booster_.dump_model()["tree_info"]:
                max_split_depth = 0
                count = 0
                heap = [(tree["tree_structure"], 0)]
                while heap:
                    node, depth = heap.pop()
                    count += 1
                    if "split_index" in node:
                        max_split_depth = max(max_split_depth, depth)
                        heap.append((node["left_child"], depth + 1))
                        heap.append((node["right_child"], depth + 1))
                n_nodes += count
                n_levels += max_split_depth + 1

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar((nex, X.shape[1]), lb=-100.0, ub=100.0)
                    add_predictor_constr(gpm, predictor, x, formulation="ocean")
                    gpm.update()

                    self.assertEqual(gpm.NumBinVars, nex * n_levels)
                    self.assertEqual(gpm.NumGenConstrs, 0)
                    self.assertEqual(
                        gpm.NumVars,
                        nex * X.shape[1]
                        + nex
                        + nex * n_levels
                        + nex * n_nodes
                        + nex * n_intervals,
                    )

    def test_biggs_perakis_binaries_are_leaf_selectors(self):
        rng = np.random.RandomState(0)
        X = rng.randint(0, 5, size=(300, 4)).astype(float)
        y = rng.uniform(size=300)
        nex = 2

        for n_estimators in (2, 6):
            predictor = lgb.sklearn.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=3,
                random_state=0,
                verbose=-1,
            ).fit(X, y)

            n_leaves = sum(
                tree["num_leaves"]
                for tree in predictor.booster_.dump_model()["tree_info"]
            )

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar((nex, X.shape[1]), lb=-100.0, ub=100.0)
                    add_predictor_constr(gpm, predictor, x, formulation="biggs_perakis")
                    gpm.update()

                    self.assertEqual(gpm.NumBinVars, nex * n_leaves)
                    self.assertEqual(gpm.NumGenConstrs, 0)
                    self.assertEqual(
                        gpm.NumVars,
                        nex * X.shape[1] + nex + nex * n_leaves,
                    )


if __name__ == "__main__":
    unittest.main()
