# Copyright © 2023-2026 Gurobi Optimization, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for modeling decision trees"""

from typing import NamedTuple
from warnings import warn

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from ..base_predictor_constr import AbstractPredictorConstr


class TreeLeaves(NamedTuple):
    """Leaf variables of one tree: ``variables[k, j]`` is 1 (or carries the
    unit flow) when input row ``k`` reaches leaf node ``nodes[j]``; ``j``
    runs over the leaves reachable given the input variable bounds."""

    variables: gp.MVar
    nodes: np.ndarray


class TreeLeavesAccessor:
    """Mixin exposing per-tree leaf variables."""

    _tree_leaves = None
    _ensemble_stats = None

    #: What the shared variables of each ensemble formulation are.
    _SHARED_LABELS = {
        "misic": "split binaries z, one per distinct unfixed threshold",
        "parmentier_vidal": "split binaries z, one per distinct unfixed threshold",
        "ocean": "interval fractions mu, continuous",
    }
    #: What the per-tree variables of each ensemble formulation are.
    _TREE_LABELS = {
        "misic": "leaf weights, continuous",
        "parmentier_vidal": "node flows (continuous) + depth branching binaries",
        "ocean": "node flows (continuous) + depth branching binaries",
        "biggs_perakis": "leaf selectors, binary",
    }

    @property
    def tree_leaves(self):
        """Tuple of :py:class:`TreeLeaves`, one per tree.

        Raises
        ------
        AttributeError
            If the formulation does not expose leaf variables (``"paths"``).
        """
        if self._tree_leaves is None:
            raise AttributeError(
                "this predictor constraint's formulation does not expose "
                "per-tree leaf variables"
            )
        return self._tree_leaves

    def _print_ensemble_stats(self, file=None):
        """Print the size structure of an ensemble formulation as a table.

        The ensemble formulations build the whole ensemble at once — there
        are no per-tree sub-estimators, and the shared variables belong to
        no tree — so the per-estimator table is replaced by one row per
        structural block: the shared variables, each tree, and the output
        linking rows. The Binaries column separates the integer part of
        each block — the size driver the formulations differ in.
        """
        stats = self._ensemble_stats
        formulation = stats["formulation"]
        header = (
            f"{'Block':13} {'Output Shape':>14} {'Variables':>12} "
            f"{'Binaries':>12} {'Constraints':^38}"
        )

        def _row(label, block):
            print(
                f"{label:13} {'-':>14} {block['vars']:>12} "
                f"{block['binaries']:>12} {block['linear']:>12} "
                f"{block['quadratic']:>12} {block['general']:>12}",
                file=file,
            )

        print(
            f"Ensemble formulation '{formulation}': {stats['n_trees']} trees",
            file=file,
        )
        shared_label = self._SHARED_LABELS.get(formulation)
        if shared_label is not None:
            print(f"Shared variables: {shared_label}.", file=file)
        print(f"Per-tree variables: {self._TREE_LABELS[formulation]}.", file=file)
        print("-" * len(header), file=file)
        print(header, file=file)
        print(
            f"{' ' * 54} {'Linear':>12} {'Quadratic':>12} {'General':>12}",
            file=file,
        )
        print("=" * len(header), file=file)
        if shared_label is not None:
            _row("shared", stats["shared"])
        for i, block in enumerate(stats["trees"]):
            _row(f"tree{i}", block)
        _row("linking", stats["linking"])
        print("-" * len(header), file=file)


def _compute_leafs_bounds(gp_model, tree, feature_is_fixed, epsilon, safety_floor=0.0):
    """Compute the bounds that define each leaf of the tree

    Parameters
    ----------
    tree : dict
        The decision tree to model.
    feature_is_fixed : ndarray
        Boolean array indicating if a feature is fixed.
    epsilon : float
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.
    """
    capacity = tree["capacity"]
    n_features = tree["n_features"]
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    feature = tree["feature"]
    threshold = tree["threshold"]

    node_lb = -np.ones((n_features, capacity)) * GRB.INFINITY
    node_ub = np.ones((n_features, capacity)) * GRB.INFINITY

    stack = [
        0,
    ]

    feas_tol = gp_model.Params.FeasibilityTol

    while len(stack) > 0:
        node = stack.pop()
        left = children_left[node]
        if left < 0:
            continue
        right = children_right[node]
        assert left not in stack
        assert right not in stack
        node_ub[:, right] = node_ub[:, node]
        node_lb[:, right] = node_lb[:, node]
        node_ub[:, left] = node_ub[:, node]
        node_lb[:, left] = node_lb[:, node]

        node_threshold = threshold[node]
        if 0 < abs(node_threshold) < safety_floor:
            node_threshold = np.sign(node_threshold) * safety_floor

        if 0 < abs(node_threshold) < feas_tol:
            warn(
                f"Split threshold {node_threshold} is smaller than Gurobi's "
                f"feasibility tolerance ({feas_tol}). This may lead to numerical issues. "
                "Consider setting 'safety_floor' to a higher value (e.g., 1e-5).",
                UserWarning,
            )

        node_ub[feature[node], left] = node_threshold
        if feature_is_fixed[feature[node]]:
            node_lb[feature[node], right] = node_threshold
        else:
            node_lb[feature[node], right] = node_threshold + epsilon
        stack.append(right)
        stack.append(left)
    return (node_lb, node_ub)


def _compute_reachability(
    gp_model, tree, _input, feature_is_fixed, epsilon, nodes, safety_floor=0.0
):
    """Which examples can reach which of the given tree nodes.

    A node is reachable for an example when the box of inputs routed to it
    (from :py:func:`_compute_leafs_bounds`, so with the same epsilon and
    ``feature_is_fixed`` semantics as the constraints) intersects the
    example's input variable bounds.

    Returns a boolean array of shape ``(n_examples, len(nodes))``.
    """
    (node_lb, node_ub) = _compute_leafs_bounds(
        gp_model, tree, feature_is_fixed, epsilon, safety_floor
    )
    input_lb = _input.getAttr(GRB.Attr.LB)
    input_ub = _input.getAttr(GRB.Attr.UB)

    selected_lb = node_lb[:, nodes]  # (n_features, len(nodes))
    selected_ub = node_ub[:, nodes]
    reachability = np.ones((_input.shape[0], len(nodes)), dtype=bool)
    for f in range(tree["n_features"]):
        reachability &= (input_ub[:, f, None] >= selected_lb[f, None, :]) & (
            input_lb[:, f, None] <= selected_ub[f, None, :]
        )
    return reachability


def _leafs_formulation(
    gp_model, _input, output, tree, epsilon, _name_var, verbose, timer, safety_floor=0.0
):
    """Formulate decision tree using 'leafs' formulation

    We have one variable per leaf of the tree and a series of indicator to
    define when that leaf is reached.
    """
    nex = _input.shape[0]
    n_features = tree["n_features"]

    # Collect leaf nodes
    leafs = tree["children_left"] < 0
    leaf_nodes = leafs.nonzero()[0]

    # Get fixed features we don't want to apply the epsilon for them
    feature_is_fixed = (_input.lb == _input.ub).all(axis=0)

    (node_lb, node_ub) = _compute_leafs_bounds(
        gp_model, tree, feature_is_fixed, epsilon, safety_floor
    )
    input_ub = _input.getAttr(GRB.Attr.UB)
    input_lb = _input.getAttr(GRB.Attr.LB)

    # Reachability: compute (nex, n_leaves) without materializing a (nex, n_features, n_leaves) array.
    leaf_lb = node_lb[:, leaf_nodes]  # (n_features, n_leaves)
    leaf_ub = node_ub[:, leaf_nodes]  # (n_features, n_leaves)
    reachability_matrix = np.ones((nex, leaf_nodes.size), dtype=bool)
    for f in range(n_features):
        reachability_matrix &= (input_ub[:, f, None] >= leaf_lb[f, None, :]) & (
            input_lb[:, f, None] <= leaf_ub[f, None, :]
        )

    # Drop leaves that no example can reach — they contribute nothing.
    any_reachable = reachability_matrix.any(axis=0)  # (n_leaves,)
    active_leaf_nodes = leaf_nodes[any_reachable]
    active_reachability = reachability_matrix[:, any_reachable]  # (nex, n_active)
    n_active = len(active_leaf_nodes)

    if n_active == 0:
        raise ValueError(
            "No reachable leaf nodes given the current input variable bounds; "
            "the decision tree constraint would be infeasible."
        )

    leafs_vars = gp_model.addMVar(
        (nex, n_active), vtype=GRB.BINARY, name=_name_var("leafs")
    )

    if verbose:
        timer.timing(f"Added {nex * n_active} leafs vars")

    for i, node in enumerate(active_leaf_nodes):
        reachable = active_reachability[:, i]
        # Non reachable nodes
        leafs_vars[~reachable, i].setAttr(GRB.Attr.UB, 0.0)
        # Leaf node:
        rhs = output[reachable, :].tolist()
        lhs = leafs_vars[reachable, i].tolist()
        values = tree["value"][node, :]
        n_indicators = sum(reachable)
        for l_var, r_vars in zip(lhs, rhs):
            for r_var, value in zip(r_vars, values):
                gp_model.addGenConstrIndicator(l_var, 1, r_var, GRB.EQUAL, value)

        for feature in range(n_features):
            feat_lb = node_lb[feature, node]
            feat_ub = node_ub[feature, node]

            if feat_lb > -GRB.INFINITY:
                tight = (input_lb[:, feature] < feat_lb) & reachable
                lhs = leafs_vars[tight, i].tolist()
                rhs = _input[tight, feature].tolist()
                n_indicators += sum(tight)
                for l_var, r_var in zip(lhs, rhs):
                    gp_model.addGenConstrIndicator(
                        l_var, 1, r_var, GRB.GREATER_EQUAL, feat_lb
                    )

            if feat_ub < GRB.INFINITY:
                tight = (input_ub[:, feature] > feat_ub) & reachable
                lhs = leafs_vars[tight, i].tolist()
                rhs = _input[tight, feature].tolist()
                n_indicators += sum(tight)
                for l_var, r_var in zip(lhs, rhs):
                    gp_model.addGenConstrIndicator(
                        l_var, 1, r_var, GRB.LESS_EQUAL, feat_ub
                    )
        if verbose:
            timer.timing(f"Added leaf {node} using {n_indicators} indicators")

    # We should attain 1 leaf
    gp_model.addConstr(leafs_vars.sum(axis=1) == 1)

    # Use only active leaves for bounds — tighter than using all leaves.
    values = tree["value"][active_leaf_nodes, :]
    gp_model.addConstr(output <= np.max(values, axis=0))
    gp_model.addConstr(output >= np.min(values, axis=0))

    if verbose:
        timer.timing(f"Added {nex} linear constraints")

    return TreeLeaves(leafs_vars, active_leaf_nodes)


def _paths_formulation(
    gp_model, _input, output, tree, epsilon, _name_var, safety_floor=0.0
):
    """
       Path formulation for decision tree

    We have one variable for each node of the tree and do a formulation
    that reconsistutes paths through the tree. This is inferior to the
    leaf formulation and is deprecated.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    _input : mvar_array_like
        Decision variables used as input for decision tree in gp_model.
    output : mvar_array_like
        Decision variables used as output for decision tree in gp_model.
    tree : dict
        The decision tree to model.
    epsilon : float
        Small value used to impose strict inequalities for splitting nodes in
        MIP formulations.
    _name_var : function
        Function to name variables.
    """

    warn(
        "Path formulation of decision trees is not tested anymore.", DeprecationWarning
    )
    outdim = output.shape[1]
    nex = _input.shape[0]
    nodes = gp_model.addMVar(
        (nex, tree["capacity"]), vtype=GRB.BINARY, name=_name_var("node")
    )

    children_left = tree["children_left"]
    children_right = tree["children_right"]
    threshold = tree["threshold"]
    feature = tree["feature"]
    value = tree["value"]
    # Collect leafs and non-leafs nodes
    not_leafs = children_left >= 0
    leafs = children_left < 0

    # Connectivity constraint
    gp_model.addConstr(
        nodes[:, not_leafs]
        == nodes[:, children_right[not_leafs]] + nodes[:, children_left[not_leafs]]
    )

    # The value of the root is always 1
    nodes[:, 0].LB = 1.0

    feas_tol = gp_model.Params.FeasibilityTol

    # Node splitting
    for node in not_leafs.nonzero()[0]:
        left = children_left[node]
        right = children_right[node]
        node_threshold = threshold[node]
        if 0 < abs(node_threshold) < safety_floor:
            node_threshold = np.sign(node_threshold) * safety_floor

        if 0 < abs(node_threshold) < feas_tol:
            warn(
                f"Split threshold {node_threshold} is smaller than Gurobi's "
                f"feasibility tolerance ({feas_tol}). This may lead to numerical issues. "
                "Consider setting 'safety_floor' to a higher value (e.g., 1e-5).",
                UserWarning,
            )

        # Intermediate node
        node_feature = feature[node]
        feat_var = _input[:, node_feature]

        fixed_input = (feat_var.UB == feat_var.LB).all()

        if fixed_input:
            # Special case where we have an MVarPlusConst object
            # If that feature is a constant we can directly fix it.
            node_value = _input[:, node_feature].LB
            fixed_left = node_value <= node_threshold
            nodes[fixed_left, right].UB = 0.0
            nodes[~fixed_left, left].UB = 0.0
        else:
            lhs = _input[:, node_feature].tolist()
            rhs = nodes[:, left].tolist()
            gp_model.addConstrs(
                ((rhs[k] == 1) >> (lhs[k] <= node_threshold)) for k in range(nex)
            )
            rhs = nodes[:, right].tolist()
            gp_model.addConstrs(
                ((rhs[k] == 1) >> (lhs[k] >= node_threshold + epsilon))
                for k in range(nex)
            )

    for node in leafs.nonzero()[0]:
        # Leaf node:
        lhs = output.tolist()
        rhs = nodes[:, node].tolist()
        node_value = value[node, :]
        gp_model.addConstrs(
            (rhs[k] == 1) >> (lhs[k][i] == node_value[i])
            for k in range(nex)
            for i in range(outdim)
        )

    gp_model.addConstr(output <= np.max(tree["value"], axis=0))
    gp_model.addConstr(output >= np.min(tree["value"], axis=0))


class AbstractTreeEstimator(TreeLeavesAccessor, AbstractPredictorConstr):
    """Abstract class to model a decision tree

    The decision tree should be stored in a dictionary with a similar representation
    as the one that scikit-learn uses:

        "capacity": number of nodes in the tree (size of the arrays that follow),
        "children_left": index of left children (-1 for a leaf)
        "children_right": index of right children (-1 for a leaf)
        "feature": splitting feature of node
        "threshold": threshold for spliting node
        "value": value of the node for output variable
    """

    def __init__(
        self,
        gp_model,
        tree,
        input_vars,
        output_vars,
        epsilon,
        timer=None,
        safety_floor=0.0,
        **kwargs,
    ):
        self._default_name = "tree"
        self._tree = tree
        self._epsilon = epsilon
        self._safety_floor = safety_floor

        self._formulation = kwargs.get("formulation", "leaf")
        if timer is None:
            self._timer = AbstractPredictorConstr._ModelingTimer()
        else:
            self._timer = timer
        AbstractPredictorConstr.__init__(
            self, gp_model, input_vars, output_vars, **kwargs
        )

    def _mip_model(self, **kwargs):
        # Imported here to avoid a circular import: the ensemble formulations
        # build on helpers defined in this module.
        from .ensemble_model import (
            ENSEMBLE_FORMULATIONS,
            add_tree_ensemble_formulation,
        )

        if self._formulation in ("leafs", "leaf"):
            leaves = _leafs_formulation(
                self.gp_model,
                self.input,
                self.output,
                self._tree,
                self._epsilon,
                self._name_var,
                self.verbose,
                self._timer,
                self._safety_floor,
            )
            self._tree_leaves = (leaves,)
        elif self._formulation == "paths":
            _paths_formulation(
                self.gp_model,
                self.input,
                self.output,
                self._tree,
                self._epsilon,
                self._name_var,
                self._safety_floor,
            )
        elif self._formulation in ENSEMBLE_FORMULATIONS:
            # A lone decision tree is an ensemble of one tree.
            self._tree_leaves, self._ensemble_stats = add_tree_ensemble_formulation(
                self.gp_model,
                [self._tree],
                np.ones(1),
                self.input,
                self.output,
                self._formulation,
                self._epsilon,
                self._name_var,
                self._safety_floor,
            )
        else:
            raise ValueError(f"Unknown formulation: {self._formulation}")

    def get_error(self, eps):
        """Functions returns an error for an abstract class

        Child classes should implement this.
        """
        raise NotImplementedError("Child classes must implement get_error")
