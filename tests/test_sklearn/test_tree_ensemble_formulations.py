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

import io
import unittest
import warnings

import gurobipy as gp
import numpy as np
from gurobipy import GRB, GurobiError
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from gurobi_ml import add_predictor_constr

from ..tree_ensemble import (
    representable_cells,
    thresholds_by_feature,
    ties_duplicated_threshold,
)

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


def _tree_pairs(predictor):
    """The (feature, threshold) pairs of each tree of predictor.

    sklearn's ``predict`` casts inputs to float32 before comparing them to
    the (float64) thresholds, hence the float32-aware helpers in
    ``tests.tree_ensemble``.
    """
    pairs = []
    for tree in _sklearn_trees(predictor):
        split_nodes = tree.children_left >= 0
        pairs.append(set(zip(tree.feature[split_nodes], tree.threshold[split_nodes])))
    return pairs


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

    def _diabetes_predictors(self, random_state):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        return X, [
            GradientBoostingRegressor(
                n_estimators=5, max_depth=3, random_state=random_state
            ).fit(X, y),
            RandomForestRegressor(
                n_estimators=5, max_depth=3, random_state=random_state
            ).fit(X, y),
            DecisionTreeRegressor(max_depth=4, random_state=random_state).fit(X, y),
        ]

    def _diabetes_agreement(self, formulation):
        """Compare the given shared-z formulation against the leaf baseline
        (one test method per formulation, so failures name it)."""
        for random_state in (17, 42):
            X, predictors = self._diabetes_predictors(random_state)
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
                        objective, x_star, _ = self._optimize(
                            predictor, X, formulation, sense
                        )
                        # The shared z couples the trees, so a shared-z
                        # optimum always corresponds to a real input.
                        self._assert_predict_reproduces(
                            predictor, objective, x_star, feas_tol
                        )

                        # Each solve is proven optimal within the default
                        # relative MIPGap of 1e-4.
                        tolerance = 3e-4 * max(1.0, abs(objective))

                        # Every shared-z-feasible point is leaf-feasible, so
                        # the leaf optimum dominates.
                        if sense == GRB.MAXIMIZE:
                            self.assertGreaterEqual(obj_leaf, objective - tolerance)
                        else:
                            self.assertLessEqual(obj_leaf, objective + tolerance)

                        # Without a duplicated-threshold tie the leaf
                        # formulation cannot mix branches across trees: the
                        # optima must agree and the leaf optimum must also
                        # correspond to a real input.
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
        """Ocean sits between the families: the shared mu chain forbids
        per-tree tolerance-slop stacking (so ocean <= leaf when maximizing)
        but still allows branch mixing at exact ties (so misic <= ocean).
        Predict-reproduction holds only without a duplicated-threshold tie.
        The paper's margin (our epsilon) restores tie coupling."""
        for random_state in (17, 42):
            X, predictors = self._diabetes_predictors(random_state)
            for predictor in predictors:
                for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
                    with self.subTest(
                        predictor=type(predictor).__name__,
                        random_state=random_state,
                        sense=sense,
                    ):
                        obj_leaf, _, feas_tol = self._optimize(
                            predictor, X, "leaf", sense
                        )
                        objective, x_star, _ = self._optimize(
                            predictor, X, "ocean", sense
                        )
                        tolerance = 3e-4 * max(1.0, abs(objective))
                        self.assertLessEqual(abs(obj_leaf - objective), tolerance)
                        if not ties_duplicated_threshold(
                            _tree_pairs(predictor), x_star, feas_tol
                        ):
                            self._assert_predict_reproduces(
                                predictor, objective, x_star, feas_tol
                            )

    def test_diabetes_agreement_biggs_perakis(self):
        """Exact-arithmetic: biggs_perakis equals leaf (same per-tree closure
        boxes); under solver tolerances only the sandwich
        misic <= biggs_perakis <= leaf (maximize) is sound, since
        per-tree-row encodings may stack independent sub-tolerance
        violations differently."""
        for random_state in (17, 42):
            X, predictors = self._diabetes_predictors(random_state)
            for predictor in predictors:
                for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
                    with self.subTest(
                        predictor=type(predictor).__name__,
                        random_state=random_state,
                        sense=sense,
                    ):
                        obj_leaf, _, feas_tol = self._optimize(
                            predictor, X, "leaf", sense
                        )
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
            X, predictors = self._diabetes_predictors(random_state)
            for predictor in predictors:
                for sense in (GRB.MAXIMIZE, GRB.MINIMIZE):
                    with self.subTest(
                        predictor=type(predictor).__name__,
                        random_state=random_state,
                        sense=sense,
                    ):
                        obj_misic, _, _ = self._optimize(predictor, X, "misic", sense)
                        tolerance = 3e-4 * max(1.0, abs(obj_misic))
                        for formulation in ("parmentier_vidal",):
                            objective, _, _ = self._optimize(
                                predictor, X, formulation, sense
                            )
                            self.assertLessEqual(
                                abs(obj_misic - objective),
                                tolerance,
                                msg=formulation,
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
        self._check_epsilon_enforced_on_right_branch("leaf")

    def test_epsilon_enforced_on_right_branch_misic(self):
        self._check_epsilon_enforced_on_right_branch("misic")

    def test_epsilon_enforced_on_right_branch_parmentier_vidal(self):
        self._check_epsilon_enforced_on_right_branch("parmentier_vidal")

    def test_epsilon_enforced_on_right_branch_ocean(self):
        self._check_epsilon_enforced_on_right_branch("ocean")

    def test_epsilon_enforced_on_right_branch_biggs_perakis(self):
        self._check_epsilon_enforced_on_right_branch("biggs_perakis")

    def _check_epsilon_enforced_on_right_branch(self, formulation):
        """Selecting the right branch must push x at least epsilon above the
        threshold."""
        epsilon = 1e-2
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
        self._check_fixed_feature_in_epsilon_band("leaf")

    def test_fixed_feature_in_epsilon_band_is_feasible_misic(self):
        self._check_fixed_feature_in_epsilon_band("misic")

    def test_fixed_feature_in_epsilon_band_is_feasible_parmentier_vidal(self):
        self._check_fixed_feature_in_epsilon_band("parmentier_vidal")

    def test_fixed_feature_in_epsilon_band_is_feasible_ocean(self):
        self._check_fixed_feature_in_epsilon_band("ocean")

    def test_fixed_feature_in_epsilon_band_is_feasible_biggs_perakis(self):
        self._check_fixed_feature_in_epsilon_band("biggs_perakis")

    def _check_fixed_feature_in_epsilon_band(self, formulation):
        """A feature fixed to a constant inside ``(t, t + epsilon)`` must not
        make the model infeasible: epsilon is dropped for fixed features and
        the constant routes to the right branch."""
        epsilon = 1e-2
        value = self.threshold + epsilon / 2.0
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
        self._check_unfixed_box_inside_epsilon_band("leaf")

    def test_unfixed_box_inside_epsilon_band_has_no_leaf_misic(self):
        self._check_unfixed_box_inside_epsilon_band("misic")

    def test_unfixed_box_inside_epsilon_band_has_no_leaf_parmentier_vidal(self):
        self._check_unfixed_box_inside_epsilon_band("parmentier_vidal")

    def test_unfixed_box_inside_epsilon_band_has_no_leaf_ocean(self):
        self._check_unfixed_box_inside_epsilon_band("ocean")

    def test_unfixed_box_inside_epsilon_band_has_no_leaf_biggs_perakis(self):
        self._check_unfixed_box_inside_epsilon_band("biggs_perakis")

    def _check_unfixed_box_inside_epsilon_band(self, formulation):
        """A non-fixed input boxed strictly inside ``(t, t + epsilon)`` can
        reach no leaf; every formulation reports it at build time."""
        epsilon = 1e-2
        lb = self.threshold + epsilon / 4.0
        ub = self.threshold + epsilon / 2.0
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
        for formulation, epsilon in (
            ("misic", 0.0),
            ("leaf", 1e-2),
            # biggs_perakis carries epsilon path-wise in its leaf boxes, so
            # the global-epsilon warning must not fire for it.
            ("biggs_perakis", 1e-2),
        ):
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

    def test_add_remove_misic(self):
        self._add_remove("misic")

    def test_add_remove_parmentier_vidal(self):
        self._add_remove("parmentier_vidal")

    def test_add_remove_ocean(self):
        self._add_remove("ocean")

    def test_add_remove_biggs_perakis(self):
        self._add_remove("biggs_perakis")

    def test_ocean_requires_finite_bounds(self):
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((1, self.n_features), lb=-GRB.INFINITY)
            with self.assertRaisesRegex(ValueError, "finite bounds"):
                add_predictor_constr(gpm, self.predictors[0], x, formulation="ocean")

    def test_biggs_perakis_requires_finite_bounds(self):
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((1, self.n_features), lb=-GRB.INFINITY)
            with self.assertRaisesRegex(ValueError, "finite bounds"):
                add_predictor_constr(
                    gpm, self.predictors[0], x, formulation="biggs_perakis"
                )

    def _add_remove(self, formulation):
        for predictor in self.predictors:
            with self.subTest(predictor=type(predictor).__name__):
                self._check_add_remove(predictor, formulation)

    def _check_add_remove(self, predictor, formulation):
        with gp.Model() as gpm:
            gpm.Params.OutputFlag = 0
            x = gpm.addMVar((2, self.n_features), lb=-1000.0, ub=1000.0)
            y = gpm.addMVar((2, 1), lb=-GRB.INFINITY)
            gpm.update()
            numvars = gpm.NumVars

            pred_constr = add_predictor_constr(
                gpm, predictor, x, y, epsilon=EPSILON, formulation=formulation
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
            thresholds = thresholds_by_feature(_tree_pairs(predictor))
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

    def test_parmentier_vidal_binaries_scale_with_depth(self):
        rng = np.random.RandomState(0)
        X = rng.randint(0, 5, size=(200, 4)).astype(float)
        y = rng.uniform(size=200)
        nex = 2

        for n_estimators in (2, 6):
            predictor = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=3, random_state=0
            ).fit(X, y)

            trees = _sklearn_trees(predictor)
            thresholds = thresholds_by_feature(_tree_pairs(predictor))
            n_shared = sum(len(values) for values in thresholds.values())
            # One branching binary per tree and depth level of its splits.
            n_levels = sum(int(tree.max_depth) for tree in trees)
            n_nodes = sum(int(tree.node_count) for tree in trees)

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
                    # Total: input + output + split binaries + branching
                    # binaries + one continuous flow variable per example
                    # and node.
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
        X = rng.randint(0, 5, size=(200, 4)).astype(float)
        y = rng.uniform(size=200)
        nex = 2

        for n_estimators in (2, 6):
            predictor = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=3, random_state=0
            ).fit(X, y)

            trees = _sklearn_trees(predictor)
            thresholds = thresholds_by_feature(_tree_pairs(predictor))
            n_intervals = sum(len(values) + 1 for values in thresholds.values())
            n_levels = sum(int(tree.max_depth) for tree in trees)
            n_nodes = sum(int(tree.node_count) for tree in trees)

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar((nex, X.shape[1]), lb=-100.0, ub=100.0)
                    add_predictor_constr(gpm, predictor, x, formulation="ocean")
                    gpm.update()

                    # The paper's selling point: the only binaries are the
                    # branching variables — none per threshold — and there
                    # are no indicator constraints at all.
                    self.assertEqual(gpm.NumBinVars, nex * n_levels)
                    self.assertEqual(gpm.NumGenConstrs, 0)
                    # Total: input + output + branching binaries + one flow
                    # per example and node + one mu per example and interval.
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
        X = rng.randint(0, 5, size=(200, 4)).astype(float)
        y = rng.uniform(size=200)
        nex = 2

        for n_estimators in (2, 6):
            predictor = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=3, random_state=0
            ).fit(X, y)

            trees = _sklearn_trees(predictor)
            n_leaves = sum(int((tree.children_left < 0).sum()) for tree in trees)

            with self.subTest(n_estimators=n_estimators):
                params = {"OutputFlag": 0}
                with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
                    x = gpm.addMVar((nex, X.shape[1]), lb=-100.0, ub=100.0)
                    add_predictor_constr(gpm, predictor, x, formulation="biggs_perakis")
                    gpm.update()

                    # One binary leaf selector per example and leaf — like
                    # the naive leaf count — no indicator constraints and no
                    # shared variables of any kind.
                    self.assertEqual(gpm.NumBinVars, nex * n_leaves)
                    self.assertEqual(gpm.NumGenConstrs, 0)
                    self.assertEqual(
                        gpm.NumVars,
                        nex * X.shape[1] + nex + nex * n_leaves,
                    )


class TestPrintStats(unittest.TestCase):
    """The ensemble formulations print a block-structured size summary in
    ``print_stats`` — they have no per-tree sub-estimators, and the shared
    variables belong to no tree — while the per-tree leaf path keeps its
    per-estimator table."""

    def setUp(self):
        data = datasets.load_diabetes()
        X, y = data["data"], data["target"]
        self.X = X
        self.predictor = GradientBoostingRegressor(
            n_estimators=3, max_depth=3, random_state=0
        ).fit(X, y)

    def _print_stats(self, formulation):
        params = {"OutputFlag": 0}
        with gp.Env(params=params) as env, gp.Model(env=env) as gpm:
            x = gpm.addMVar(
                (1, self.X.shape[1]),
                lb=self.X.min(axis=0),
                ub=self.X.max(axis=0),
            )
            pred_constr = add_predictor_constr(
                gpm, self.predictor, x, formulation=formulation
            )
            gpm.update()
            output = io.StringIO()
            pred_constr.print_stats(file=output)
            return output.getvalue(), gpm.NumBinVars

    def test_ensemble_formulations_print_block_table(self):
        for formulation in ("misic", "parmentier_vidal", "ocean", "biggs_perakis"):
            with self.subTest(formulation=formulation):
                output, _ = self._print_stats(formulation)
                self.assertIn(f"Ensemble formulation '{formulation}': 3 trees", output)
                # One table row per structural block: each tree plus the
                # output linking rows.
                self.assertIn("Block", output)
                self.assertIn("tree0", output)
                self.assertIn("tree2", output)
                self.assertIn("linking", output)
                # No per-estimator table, and no dangling empty header.
                self.assertNotIn("Estimator", output)

    def test_table_binaries_match_model(self):
        # The Binaries column must add up to the model's binary count, and
        # only the shared-variable formulations print a shared row.
        for formulation, has_shared in (("misic", True), ("biggs_perakis", False)):
            with self.subTest(formulation=formulation):
                output, num_bin_vars = self._print_stats(formulation)
                lines = output.splitlines()
                start = next(i for i, s in enumerate(lines) if s.startswith("="))
                rows = [
                    line.split()
                    for line in lines[start + 1 :]
                    if line and not line.startswith("-")
                ]
                self.assertEqual(sum(int(tokens[-4]) for tokens in rows), num_bin_vars)
                self.assertEqual(
                    any(tokens[0] == "shared" for tokens in rows), has_shared
                )

    def test_leaf_keeps_estimator_table(self):
        output, _ = self._print_stats("leaf")
        self.assertIn("Estimator", output)
        self.assertNotIn("Ensemble formulation", output)


if __name__ == "__main__":
    unittest.main()
