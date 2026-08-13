"""Tests for the ensemble-level tree formulations (currently ``"misic"``).

Beyond the fixed-input equivalence tests in test_sklearn_formulations.py,
this module checks:

- cross-formulation agreement: the shared-z optimum is always reproduced by
  ``predict`` on one of the threshold cells the solution can represent; the
  leaf optimum dominates it and equals it whenever the leaf solution does not
  tie a threshold duplicated across trees (at such a tie the leaf
  formulation may mix branch choices across trees — a combination no real
  input reproduces — while the shared ``z`` forbids the mix by construction);
- existing-coverage parity for ``epsilon`` and fixed features, lifecycle
  (``remove()`` restores the model, unknown formulation raises);
- model-size validation: with unbounded inputs the binary count equals the
  number of distinct split thresholds — independent of the number of trees.
"""

import itertools
import unittest
import warnings

import gurobipy as gp
import numpy as np
from gurobipy import GRB, GurobiError
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from gurobi_ml import add_predictor_constr

# Cross-formulation comparisons must run at the default epsilon=0: a positive
# epsilon has a different scope per formulation — path-wise in "leaf" but
# ensemble-wise (every threshold, all trees) in the shared-z formulations —
# so the feasible sets, and hence the optimal objectives, may legitimately
# differ. Positive epsilons are additionally treacherous below Gurobi's
# IntFeasTol (default 1e-5): the indicator linking is translated into SOS1
# constraints whose feasibility check treats values up to IntFeasTol as zero,
# so with epsilon <= IntFeasTol the solver may return an input sitting on the
# wrong side of a threshold by exactly epsilon (observed with epsilon=1e-5:
# MaxVio == epsilon, vanishing for epsilon=1e-4 or IntFeasTol=1e-9).
EPSILON = 0.0


def _sklearn_trees(predictor):
    """Return the fitted ``Tree`` objects of a sklearn tree predictor."""
    if isinstance(predictor, GradientBoostingRegressor):
        return [est[0].tree_ for est in predictor.estimators_]
    if isinstance(predictor, RandomForestRegressor):
        return [est.tree_ for est in predictor.estimators_]
    return [predictor.tree_]


def _thresholds_by_feature(predictor):
    """Collect each feature's split thresholds over all trees of predictor."""
    thresholds = {}
    for tree in _sklearn_trees(predictor):
        split_nodes = tree.children_left >= 0
        for feature, threshold in zip(
            tree.feature[split_nodes], tree.threshold[split_nodes]
        ):
            thresholds.setdefault(feature, set()).add(threshold)
    return thresholds


def _ties_duplicated_threshold(predictor, x_star, feas_tol):
    """True if x_star ties a threshold that appears in more than one tree.

    Trees fitted on resamples of the same data reuse the exact same split
    thresholds (midpoints of the same value pairs). When ``x_star`` sits
    exactly on such a threshold, the leaf formulation — whose trees branch
    independently — may route different trees to different sides: a branch
    combination no real input reproduces. Optima equality across
    formulations is only asserted without such a tie.
    """
    seen, duplicated = set(), set()
    for tree in _sklearn_trees(predictor):
        split_nodes = tree.children_left >= 0
        pairs = set(zip(tree.feature[split_nodes], tree.threshold[split_nodes]))
        duplicated |= pairs & seen
        seen |= pairs
    for feature, threshold in duplicated:
        tie_tol = max(feas_tol, float(np.spacing(np.float32(threshold))))
        if abs(x_star[feature] - threshold) <= tie_tol:
            return True
    return False


def _representable_cells(x_star, thresholds, feas_tol):
    """Points of every threshold cell that x_star can represent.

    The solution's own cell, plus both adjacent cells for every coordinate
    that ties with a split threshold (the solver may have taken either branch
    at such a knife edge).  sklearn's ``predict`` casts inputs to float32
    before comparing them to the (float64) thresholds, so the tie window and
    the points probing both sides of a threshold must be one float32 ulp
    wide — a float64 ulp would be erased by the cast.
    """
    candidates = []
    for feature, value in enumerate(x_star):
        coordinate_candidates = [value]
        for threshold in thresholds.get(feature, ()):
            tie_tol = max(feas_tol, float(np.spacing(np.float32(threshold))))
            if abs(value - threshold) <= tie_tol:
                below = float(np.nextafter(np.float32(threshold), np.float32(-np.inf)))
                above = float(np.nextafter(np.float32(threshold), np.float32(np.inf)))
                coordinate_candidates += [below, above]
        candidates.append(coordinate_candidates)
    return np.array(list(itertools.product(*candidates)))


class TestCrossFormulationAgreement(unittest.TestCase):
    """All formulations must find the same optimal value on tiny instances."""

    def _optimize(self, predictor, X, formulation, sense):
        """Solve max/min of the prediction over the box of feature ranges.

        Returns ``(objective, x_star, feas_tol)``.
        """
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
            # At epsilon=0 there is no band for a solution to hide in; any
            # violation above FeasibilityTol means a solver tolerance is
            # masking a wrong-side threshold again (see the EPSILON note).
            self.assertLessEqual(gpm.MaxVio, gpm.Params.FeasibilityTol)
            return gpm.ObjVal, np.array(x.X[0, :]), gpm.Params.FeasibilityTol

    def _assert_predict_reproduces(self, predictor, objective, x_star, feas_tol):
        """The MIP optimum must be attained by ``predict`` on one of the
        threshold cells the solution can represent."""
        cells = _representable_cells(
            x_star, _thresholds_by_feature(predictor), feas_tol
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
            predictors = [
                GradientBoostingRegressor(
                    n_estimators=5, max_depth=3, random_state=random_state
                ).fit(X, y),
                RandomForestRegressor(
                    n_estimators=5, max_depth=3, random_state=random_state
                ).fit(X, y),
                DecisionTreeRegressor(max_depth=4, random_state=random_state).fit(X, y),
            ]
            for predictor in predictors:
                for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
                    with self.subTest(
                        predictor=type(predictor).__name__,
                        random_state=random_state,
                        sense=sense,
                    ):
                        obj_leaf, x_leaf, feas_tol = self._optimize(
                            predictor, X, "leaf", sense
                        )
                        obj_misic, x_misic, _ = self._optimize(
                            predictor, X, "misic", sense
                        )

                        # The shared z couples the trees, so the misic
                        # optimum always corresponds to a real input.
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

                        # Without a duplicated-threshold tie the leaf
                        # formulation cannot mix branches across trees: the
                        # optima must agree and the leaf optimum must also
                        # correspond to a real input.
                        if not _ties_duplicated_threshold(predictor, x_leaf, feas_tol):
                            self.assertLessEqual(abs(obj_leaf - obj_misic), tolerance)
                            self._assert_predict_reproduces(
                                predictor, obj_leaf, x_leaf, feas_tol
                            )


def _add_predictor_constr_silently(gpm, predictor, x, **kwargs):
    """``add_predictor_constr`` suppressing the global-epsilon warning.

    Used by tests that intentionally combine a positive epsilon with an
    ensemble formulation; the warning itself is asserted in
    ``test_positive_epsilon_warns_for_ensemble_formulation``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return add_predictor_constr(gpm, predictor, x, **kwargs)


class TestEpsilonAndFixedFeatures(unittest.TestCase):
    """Design decisions 2-3: epsilon semantics and fixed-feature handling."""

    def setUp(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        self.predictor = DecisionTreeRegressor(max_depth=1).fit(X, y)
        self.threshold = float(self.predictor.tree_.threshold[0])

    def test_epsilon_enforced_on_right_branch(self):
        """Selecting the right branch must push x at least epsilon above the
        threshold, for every formulation."""
        epsilon = 1e-2
        for formulation in ("leaf", "misic"):
            with self.subTest(formulation=formulation):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar((1, 1), lb=0.0, ub=1.0)
                    pred_constr = _add_predictor_constr_silently(
                        gpm,
                        self.predictor,
                        x,
                        epsilon=epsilon,
                        formulation=formulation,
                    )
                    # Force the right leaf (value 1.0) and pull x down.
                    gpm.addConstr(pred_constr.output.sum() >= 0.5)
                    gpm.setObjective(x.sum(), GRB.MINIMIZE)
                    gpm.optimize()
                    self.assertEqual(gpm.Status, GRB.OPTIMAL)
                    self.assertGreaterEqual(
                        x.X[0, 0],
                        self.threshold + epsilon - gpm.Params.FeasibilityTol,
                    )

    def test_fixed_feature_in_epsilon_band_is_feasible(self):
        """A feature fixed to a constant inside ``(t, t + epsilon)`` must not
        make the model infeasible: epsilon is dropped for fixed features and
        the constant routes to the right branch."""
        epsilon = 1e-2
        value = self.threshold + epsilon / 2.0
        for formulation in ("leaf", "misic"):
            with self.subTest(formulation=formulation):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar((1, 1), lb=value, ub=value)
                    pred_constr = _add_predictor_constr_silently(
                        gpm,
                        self.predictor,
                        x,
                        epsilon=epsilon,
                        formulation=formulation,
                    )
                    gpm.optimize()
                    self.assertEqual(gpm.Status, GRB.OPTIMAL)
                    self.assertAlmostEqual(pred_constr.output.X[0, 0], 1.0)

    def test_unfixed_box_inside_epsilon_band_has_no_leaf(self):
        """A non-fixed input boxed strictly inside ``(t, t + epsilon)`` can
        reach no leaf; every formulation reports it at build time."""
        epsilon = 1e-2
        lb = self.threshold + epsilon / 4.0
        ub = self.threshold + epsilon / 2.0
        for formulation in ("leaf", "misic"):
            with self.subTest(formulation=formulation):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar((1, 1), lb=lb, ub=ub)
                    with self.assertRaisesRegex(ValueError, "No reachable leaf nodes"):
                        _add_predictor_constr_silently(
                            gpm,
                            self.predictor,
                            x,
                            epsilon=epsilon,
                            formulation=formulation,
                        )

    def test_positive_epsilon_warns_for_ensemble_formulation(self):
        """Ensemble formulations apply epsilon globally (every threshold of
        every tree, not only along the selected paths); passing a positive
        epsilon must warn. The leaf formulation stays silent."""
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar((1, 1), lb=0.0, ub=1.0)
            with self.assertWarnsRegex(UserWarning, "applies\\s+globally"):
                add_predictor_constr(
                    gpm, self.predictor, x, epsilon=1e-2, formulation="misic"
                )
        for formulation, epsilon in (("misic", 0.0), ("leaf", 1e-2)):
            with self.subTest(formulation=formulation, epsilon=epsilon):
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar((1, 1), lb=0.0, ub=1.0)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        add_predictor_constr(
                            gpm,
                            self.predictor,
                            x,
                            epsilon=epsilon,
                            formulation=formulation,
                        )
                    self.assertFalse(
                        [w for w in caught if "globally" in str(w.message)]
                    )


class TestLifecycle(unittest.TestCase):
    """``remove()`` restores the model; unknown formulations raise."""

    def setUp(self):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        self.n_features = X.shape[1]
        self.predictors = [
            GradientBoostingRegressor(n_estimators=4, max_depth=2, random_state=0).fit(
                X, y
            ),
            RandomForestRegressor(n_estimators=4, max_depth=2, random_state=0).fit(
                X, y
            ),
            DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y),
        ]

    def test_add_remove(self):
        for predictor in self.predictors:
            with self.subTest(predictor=type(predictor).__name__):
                with gp.Model() as gpm:
                    gpm.Params.OutputFlag = 0
                    x = gpm.addMVar((2, self.n_features), lb=-GRB.INFINITY)
                    y = gpm.addMVar((2, 1), lb=-GRB.INFINITY)
                    gpm.update()
                    numvars = gpm.NumVars

                    pred_constr = add_predictor_constr(
                        gpm, predictor, x, y, epsilon=EPSILON, formulation="misic"
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
        for predictor in self.predictors:
            with self.subTest(predictor=type(predictor).__name__):
                with gp.Model() as gpm:
                    gpm.Params.OutputFlag = 0
                    x = gpm.addMVar((1, self.n_features), lb=-GRB.INFINITY)
                    with self.assertRaisesRegex(ValueError, "Unknown formulation"):
                        add_predictor_constr(
                            gpm, predictor, x, formulation="no_such_formulation"
                        )


class TestModelSize(unittest.TestCase):
    """Criterion #4: with unbounded inputs (no pruning) the variable counts
    must match the size formulas of the formulation."""

    def test_misic_binaries_are_shared_split_variables(self):
        rng = np.random.RandomState(0)
        # Integer-valued features so split thresholds (midpoints) coincide
        # across trees and threshold sharing is actually exercised.
        X = rng.randint(0, 5, size=(200, 4)).astype(float)
        y = rng.uniform(size=200)
        nex = 2

        for n_estimators in (2, 6):
            predictor = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=3, random_state=0
            ).fit(X, y)

            trees = _sklearn_trees(predictor)
            thresholds = _thresholds_by_feature(predictor)
            n_shared = sum(len(values) for values in thresholds.values())
            n_splits = sum(int((tree.children_left >= 0).sum()) for tree in trees)
            n_leaves = sum(int((tree.children_left < 0).sum()) for tree in trees)

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar(
                        (nex, X.shape[1]), lb=-GRB.INFINITY, ub=GRB.INFINITY
                    )
                    add_predictor_constr(gpm, predictor, x, formulation="misic")
                    gpm.update()

                    # One binary per example and distinct (feature, threshold)
                    # of the whole ensemble — not per tree.
                    self.assertEqual(gpm.NumBinVars, nex * n_shared)
                    self.assertLess(n_shared, n_splits)
                    # Total: input + output + split binaries + one continuous
                    # leaf variable per example and leaf.
                    self.assertEqual(
                        gpm.NumVars,
                        nex * X.shape[1] + nex + nex * n_shared + nex * n_leaves,
                    )
                    # Two indicator constraints link each unfixed binary.
                    self.assertEqual(gpm.NumGenConstrs, 2 * nex * n_shared)


if __name__ == "__main__":
    unittest.main()
