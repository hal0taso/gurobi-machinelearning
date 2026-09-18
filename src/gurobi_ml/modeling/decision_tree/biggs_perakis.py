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

"""Per-tree constraints of the Biggs–Perakis projected formulation.

Reference: Biggs & Perakis, "Tight Mixed-Integer Optimization Formulations
for Prescriptive Trees" (2023), https://arxiv.org/abs/2302.14744 — the
"projected union of polyhedra" formulation, the best performer in the
paper's experiments.

Per tree: one binary selector per reachable leaf with ``sum(z) == 1`` and,
for every feature the tree splits on, the two aggregated box constraints

    sum_l ub[l, i] * z[l]  >=  x[i]  >=  sum_l lb[l, i] * z[l]

where ``lb``/``ub`` are the leaf boxes (with the same epsilon and
``feature_is_fixed`` semantics as the leaf formulation) clipped to the
input variable bounds. When ``z`` selects one leaf the input lies in that
leaf's box; the LP relaxation is the projected convex hull of the boxes'
union — locally ideal per tree. Trees are coupled through the input
variables only (no shared split variables), so epsilon acts path-wise as
in the leaf baseline and, at exact threshold ties, trees may branch
independently (tie-uncoupled family).

The box bounds appear as constraint coefficients, so every feature used in
a split must have **finite bounds**; a :py:exc:`ValueError` is raised
otherwise (same requirement and rationale as the "ocean" formulation).
"""

import numpy as np
from gurobipy import GRB

from .decision_tree_model import (
    TreeLeaves,
    _compute_leafs_bounds,
    _compute_reachability,
)
from .misic import _leaf_intervals


def add_biggs_perakis_tree(
    gp_model, split_vars, tree, _input, epsilon, name=None, safety_floor=0.0
):
    """Add the projected formulation of one tree of an ensemble to gp_model.

    ``split_vars`` is unused (the formulation shares no ensemble-level
    variables) and accepted only for the common builder signature.

    Returns ``(expression, values)`` like the other tree builders.
    """
    nex = _input.shape[0]

    input_lb = _input.getAttr(GRB.Attr.LB)
    input_ub = _input.getAttr(GRB.Attr.UB)
    feature_is_fixed = (input_lb == input_ub).all(axis=0)

    split_features = np.unique(tree["feature"][tree["children_left"] >= 0])
    for f in split_features:
        if not (
            np.isfinite(input_lb[:, f]).all() and np.isfinite(input_ub[:, f]).all()
        ):
            raise ValueError(
                "The 'biggs_perakis' formulation requires finite bounds on "
                f"every input variable used in a split; feature {f} is "
                "unbounded. Bound the input variables (e.g. to the observed "
                "feature ranges) or use another formulation."
            )

    leaves_order, first, last = _leaf_intervals(tree)

    reachability = _compute_reachability(
        gp_model,
        tree,
        _input,
        feature_is_fixed,
        epsilon,
        leaves_order,
        safety_floor,
    )

    active_positions = reachability.any(axis=0).nonzero()[0]
    if active_positions.size == 0:
        raise ValueError(
            "No reachable leaf nodes given the current input variable bounds; "
            "the decision tree constraint would be infeasible."
        )
    active_leaves = leaves_order[active_positions]

    z = gp_model.addMVar((nex, active_positions.size), vtype=GRB.BINARY, name=name)
    z[~reachability[:, active_positions]].setAttr(GRB.Attr.UB, 0.0)

    gp_model.addConstr(z.sum(axis=1) == 1)

    (node_lb, node_ub) = _compute_leafs_bounds(
        gp_model, tree, feature_is_fixed, epsilon, safety_floor
    )
    for f in split_features:
        # Leaf boxes clipped to each example's input bounds: coefficients
        # stay finite and the selected leaf's box never cuts the example's
        # own domain.
        box_lb = np.clip(
            node_lb[f, active_leaves][None, :],
            input_lb[:, [f]],
            input_ub[:, [f]],
        )
        box_ub = np.clip(
            node_ub[f, active_leaves][None, :],
            input_lb[:, [f]],
            input_ub[:, [f]],
        )
        gp_model.addConstr((z * box_lb).sum(axis=1) <= _input[:, f])
        gp_model.addConstr((z * box_ub).sum(axis=1) >= _input[:, f])

    values = tree["value"][active_leaves, :]
    return z @ values, values, TreeLeaves(z, active_leaves)
