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

"""Per-tree constraints of the Mišić tree-ensemble formulation.

Reference: Mišić, "Optimization of Tree Ensembles" (2020),
https://arxiv.org/abs/1705.10883.
"""

import numpy as np
from gurobipy import GRB

from .decision_tree_model import _compute_reachability


def _leaf_intervals(tree):
    """Index the leaves of each subtree by depth-first leaf order.

    In a binary tree the leaves of any subtree are contiguous in left-to-right
    (depth-first) leaf order.  Returns ``(leaves_order, first, last)`` where
    ``leaves_order`` lists the leaf nodes in that order, and the leaves below
    ``node`` are the positions ``first[node]:last[node]`` of ``leaves_order``.
    """
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    first = np.zeros(tree["capacity"], dtype=np.int64)
    last = np.zeros(tree["capacity"], dtype=np.int64)
    leaves_order = []

    stack = [(0, False)]
    while stack:
        node, children_done = stack.pop()
        if children_left[node] < 0:
            first[node] = len(leaves_order)
            leaves_order.append(node)
            last[node] = len(leaves_order)
        elif children_done:
            first[node] = first[children_left[node]]
            last[node] = last[children_right[node]]
        else:
            stack.append((node, True))
            stack.append((children_right[node], False))
            stack.append((children_left[node], False))
    return np.array(leaves_order), first, last


def add_misic_tree(
    gp_model, split_vars, tree, _input, epsilon, name=None, safety_floor=0.0
):
    """Add the Mišić formulation of one tree of an ensemble to gp_model.

    One continuous variable ``y[k, l] >= 0`` per example and reachable leaf,
    with ``sum_l y[k, l] == 1``; for each splitting node the leaves of its
    left (right) subtree are linked to the ensemble's shared split binary:
    ``sum_{l in left} y <= z`` and ``sum_{l in right} y <= 1 - z``.

    Leaves that no example can reach given the input variable bounds are
    dropped; the pruning uses the same leaf boxes as the leaf formulation.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    split_vars : SplitVariables
        The shared split binaries of the ensemble.
    tree : dict
        The decision tree to model.
    _input : mvar_array_like
        Decision variables used as input for the ensemble.
    epsilon : float
        Small value used to impose strict inequalities for splitting nodes.
    name : str, optional
        Name for the leaf variables.
    safety_floor : float, optional
        |SafetyFloorParam|

    Returns
    -------
    tuple
        ``(expression, values)`` where ``expression`` is the tree's output as
        a linear expression of shape ``(n_examples, n_outputs)`` and
        ``values`` are the output values of the reachable leaves.
    """
    nex = _input.shape[0]

    leaves_order, first, last = _leaf_intervals(tree)

    reachability = _compute_reachability(
        gp_model,
        tree,
        _input,
        split_vars.feature_is_fixed,
        epsilon,
        leaves_order,
        safety_floor,
    )

    # Positions (in depth-first leaf order) of leaves some example can reach.
    active_positions = reachability.any(axis=0).nonzero()[0]
    if active_positions.size == 0:
        raise ValueError(
            "No reachable leaf nodes given the current input variable bounds; "
            "the decision tree constraint would be infeasible."
        )

    y = gp_model.addMVar((nex, active_positions.size), ub=1.0, name=name)
    y[~reachability[:, active_positions]].setAttr(GRB.Attr.UB, 0.0)

    gp_model.addConstr(y.sum(axis=1) == 1)

    children_left = tree["children_left"]
    children_right = tree["children_right"]
    feature = tree["feature"]
    threshold = tree["threshold"]
    for node in (children_left >= 0).nonzero()[0]:
        z_col = split_vars.column(feature[node], threshold[node])
        for child, is_left in (
            (children_left[node], True),
            (children_right[node], False),
        ):
            begin, end = np.searchsorted(active_positions, (first[child], last[child]))
            if begin == end:
                # No reachable leaf below this child: nothing to link.
                continue
            side = y[:, begin:end].sum(axis=1)
            if is_left:
                gp_model.addConstr(side <= z_col)
            else:
                gp_model.addConstr(side <= 1 - z_col)

    values = tree["value"][leaves_order[active_positions], :]
    return y @ values, values
