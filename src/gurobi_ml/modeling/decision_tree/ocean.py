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

"""Paper-faithful Parmentier–Vidal flow formulation (OCEAN).

Reference: Parmentier & Vidal, "Optimal Counterfactual Explanations in
Tree Ensembles" (2021), https://arxiv.org/abs/2106.06631, and its
reference implementation https://github.com/vidalt/OCEAN.

Unlike ``formulation="parmentier_vidal"`` (which reuses the ensemble's
binary split variables), this variant follows the paper: the ordinal
feature variables ``mu`` are **continuous** — the only binaries are the
per-tree, per-depth branching variables. For each feature, the thresholds
plus the input variable's bounds partition its domain into intervals;
``mu[j]`` is the fraction of interval ``j`` that lies below the input
value, the ``mu`` chain is nonincreasing, and the input is recovered
exactly by ``x == lb + sum_j mu[j] * width[j]`` — no indicator or big-M
constraints. Flow through a split forces the adjacent ``mu`` values:
going right forces the interval below the threshold to be fully crossed,
going left forbids entering the interval above it.

The interval widths appear as constraint coefficients, so every feature
used in a split must have **finite bounds** (the paper assumes bounded
feature domains); a :py:exc:`ValueError` is raised otherwise. This is a
prototype for the formulation benchmark and is self-contained on purpose.
"""

from warnings import warn

import numpy as np
from gurobipy import GRB

from .decision_tree_model import _compute_reachability


class OrdinalMuVariables:
    """Continuous ordinal interval variables shared by the whole ensemble."""

    def __init__(self, gp_model, trees, _input, epsilon, _name_var, safety_floor=0.0):
        self._safety_floor = safety_floor
        self._epsilon = epsilon
        nex, n_features = _input.shape

        input_lb = _input.getAttr(GRB.Attr.LB)
        input_ub = _input.getAttr(GRB.Attr.UB)
        self.feature_is_fixed = (input_lb == input_ub).all(axis=0)

        feas_tol = gp_model.Params.FeasibilityTol

        split_features = np.concatenate(
            [tree["feature"][tree["children_left"] >= 0] for tree in trees]
        )
        split_thresholds = np.concatenate(
            [tree["threshold"][tree["children_left"] >= 0] for tree in trees]
        )
        clamped = (np.abs(split_thresholds) > 0) & (
            np.abs(split_thresholds) < safety_floor
        )
        split_thresholds = np.where(
            clamped, np.sign(split_thresholds) * safety_floor, split_thresholds
        )

        self._thresholds = {}
        self._mu = {}
        self._widths = {}
        for f in range(n_features):
            values = np.unique(split_thresholds[split_features == f])
            if values.size == 0:
                continue

            for value in values[(np.abs(values) > 0) & (np.abs(values) < feas_tol)]:
                warn(
                    f"Split threshold {value} is smaller than Gurobi's "
                    f"feasibility tolerance ({feas_tol}). This may lead to numerical issues. "
                    "Consider setting 'safety_floor' to a higher value (e.g., 1e-5).",
                    UserWarning,
                )

            if not (
                np.isfinite(input_lb[:, f]).all() and np.isfinite(input_ub[:, f]).all()
            ):
                raise ValueError(
                    "The 'ocean' formulation requires finite bounds on every "
                    f"input variable used in a split; feature {f} is unbounded. "
                    "Bound the input variables (e.g. to the observed feature "
                    "ranges) or use another formulation."
                )

            # Interval boundaries per example: bounds plus the thresholds
            # clipped into them. Thresholds outside an example's bounds give
            # zero-width intervals; the corresponding subtrees are pruned by
            # reachability, so their (vacuous) mu values are never forcing.
            boundaries = np.clip(
                values[None, :], input_lb[:, [f]], input_ub[:, [f]]
            )  # (nex, m)
            levels = np.concatenate(
                [input_lb[:, [f]], boundaries, input_ub[:, [f]]], axis=1
            )
            widths = np.diff(levels, axis=1)  # (nex, m + 1)

            mu = gp_model.addMVar(
                (nex, values.size + 1), ub=1.0, name=_name_var(f"mu[{f}]")
            )
            # x consumes the intervals in order: the chain is nonincreasing.
            gp_model.addConstr(mu[:, :-1] >= mu[:, 1:])
            # Exact recovery of the input from the interval fractions.
            gp_model.addConstr(
                _input[:, f] == input_lb[:, f] + (mu * widths).sum(axis=1)
            )

            self._thresholds[f] = values
            self._mu[f] = mu
            self._widths[f] = widths

    def link_split(self, gp_model, feature, threshold, flow_left, flow_right):
        """Force the mu chain consistent with a split's flow variables.

        Threshold ``v_j`` separates interval ``j`` (ending at ``v_j``) from
        interval ``j + 1``; going right requires interval ``j`` fully
        crossed, going left forbids entering interval ``j + 1``. A positive
        epsilon additionally requires the right branch to cross ``epsilon``
        into interval ``j + 1`` (the paper's margin device, with our
        absolute epsilon: ``mu[j+1] >= (epsilon / width) * flow_right``).
        """
        values = self._thresholds[feature]
        threshold_clamped = threshold
        if 0 < abs(threshold) < self._safety_floor:
            threshold_clamped = np.sign(threshold) * self._safety_floor
        j = np.searchsorted(values, threshold_clamped)
        assert j < values.size and values[j] == threshold_clamped
        mu = self._mu[feature]

        if flow_left is not None:
            gp_model.addConstr(flow_left <= 1 - mu[:, j + 1])
        if flow_right is not None:
            gp_model.addConstr(flow_right <= mu[:, j])
            if self._epsilon > 0.0 and not self.feature_is_fixed[feature]:
                widths = self._widths[feature][:, j + 1]
                scale = np.divide(
                    self._epsilon,
                    widths,
                    out=np.zeros_like(widths),
                    where=widths > 0,
                )
                gp_model.addConstr(mu[:, j + 1] >= scale * flow_right)


def add_ocean_tree(
    gp_model, split_vars, tree, _input, epsilon, name=None, safety_floor=0.0
):
    """Add the OCEAN flow formulation of one tree of an ensemble to gp_model.

    Flow structure as in ``parmentier_vidal`` (per-node flows with
    conservation and one branching binary per depth level), but the split
    linking goes through the ensemble's continuous ordinal ``mu`` variables
    instead of binary split variables.

    Returns ``(expression, values)`` like the other tree builders.
    """
    nex = _input.shape[0]
    children_left = tree["children_left"]
    children_right = tree["children_right"]

    # Breadth-first traversal from the root, recording depths.
    order = [0]
    depth = np.zeros(tree["capacity"], dtype=np.int64)
    for node in order:  # grows while iterating
        left = children_left[node]
        if left >= 0:
            right = children_right[node]
            depth[left] = depth[right] = depth[node] + 1
            order.append(left)
            order.append(right)
    nodes = np.array(order)

    reachability = _compute_reachability(
        gp_model,
        tree,
        _input,
        split_vars.feature_is_fixed,
        epsilon,
        nodes,
        safety_floor,
    )

    active = reachability.any(axis=0)
    active_nodes = nodes[active]
    if not any(children_left[node] < 0 for node in active_nodes):
        raise ValueError(
            "No reachable leaf nodes given the current input variable bounds; "
            "the decision tree constraint would be infeasible."
        )
    column = {node: i for i, node in enumerate(active_nodes)}

    flow = gp_model.addMVar((nex, active_nodes.size), ub=1.0, name=name)
    flow[~reachability[:, active]].setAttr(GRB.Attr.UB, 0.0)

    gp_model.addConstr(flow[:, 0] == 1)

    split_nodes = [node for node in active_nodes if children_left[node] >= 0]
    if split_nodes:
        n_levels = int(max(depth[node] for node in split_nodes)) + 1
        branch = gp_model.addMVar(
            (nex, n_levels),
            vtype=GRB.BINARY,
            name=None if name is None else f"branch_{name}",
        )

    feature = tree["feature"]
    threshold = tree["threshold"]
    for node in split_nodes:
        level = branch[:, int(depth[node])]
        conservation = 0
        left = children_left[node]
        flow_left = None
        if left in column:
            flow_left = flow[:, column[left]]
            conservation = conservation + flow_left
            gp_model.addConstr(flow_left <= level)
        right = children_right[node]
        flow_right = None
        if right in column:
            flow_right = flow[:, column[right]]
            conservation = conservation + flow_right
            gp_model.addConstr(flow_right <= 1 - level)
        split_vars.link_split(
            gp_model, feature[node], threshold[node], flow_left, flow_right
        )
        gp_model.addConstr(flow[:, column[node]] == conservation)

    active_leaves = [node for node in active_nodes if children_left[node] < 0]
    leaf_columns = [column[node] for node in active_leaves]
    values = tree["value"][active_leaves, :]
    return flow[:, leaf_columns] @ values, values
