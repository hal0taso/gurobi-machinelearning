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

"""Per-tree constraints of the Parmentier–Vidal flow formulation.

Reference: Parmentier & Vidal, "Optimal Counterfactual Explanations in
Tree Ensembles" (2021), https://arxiv.org/abs/2106.06631.

One unit of flow is routed from the root to a leaf. The flow variables are
continuous; integrality comes from one binary per tree and depth level
("go left at depth d"), shared across all nodes of that level — only one
node per level carries flow, so the shared direction is well defined.
Feature consistency uses the ensemble's shared ordinal split variables
(the paper's "mu" device); in this implementation they are the binary
``z[f, j]`` of :py:class:`.split_variables.SplitVariables`, so the
formulation differs from the Mišić one only in the per-tree linking.
"""

import numpy as np
from gurobipy import GRB

from .decision_tree_model import TreeLeaves, _compute_reachability


def _breadth_first_nodes(tree):
    """Nodes reachable from the root in breadth-first order, with depths."""
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    order = [0]
    depth = np.zeros(tree["capacity"], dtype=np.int64)
    for node in order:  # grows while iterating
        left = children_left[node]
        if left >= 0:
            right = children_right[node]
            depth[left] = depth[right] = depth[node] + 1
            order.append(left)
            order.append(right)
    return np.array(order), depth


def add_parmentier_vidal_tree(
    gp_model, split_vars, tree, _input, epsilon, name=None, safety_floor=0.0
):
    """Add the Parmentier–Vidal flow formulation of one tree to gp_model.

    Per example: one continuous flow variable per reachable node with
    ``flow[root] == 1`` and conservation ``flow[v] == flow[left] +
    flow[right]``; one binary per depth level steering the flow left or
    right; and per splitting node the linking to the ensemble's shared
    split binaries, ``flow[left] <= z`` and ``flow[right] <= 1 - z``.

    Unreachable subtrees (same leaf-box semantics as the other
    formulations) are pruned: their nodes get no flow variables.

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
        Name for the flow variables (the branching binaries get
        ``branch_<name>``).
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
    children_left = tree["children_left"]
    children_right = tree["children_right"]

    nodes, depth = _breadth_first_nodes(tree)

    reachability = _compute_reachability(
        gp_model,
        tree,
        _input,
        split_vars.feature_is_fixed,
        epsilon,
        nodes,
        safety_floor,
    )

    # Nodes some example can reach; a reachable child implies a reachable
    # parent, so the active set is closed towards the root.
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

    # One unit of flow enters at the root and is steered by one binary per
    # depth level: only one node per level carries flow.
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
        z_col = split_vars.column(feature[node], threshold[node])
        level = branch[:, int(depth[node])]
        conservation = 0
        left = children_left[node]
        if left in column:
            left_flow = flow[:, column[left]]
            conservation = conservation + left_flow
            gp_model.addConstr(left_flow <= level)
            gp_model.addConstr(left_flow <= z_col)
        right = children_right[node]
        if right in column:
            right_flow = flow[:, column[right]]
            conservation = conservation + right_flow
            gp_model.addConstr(right_flow <= 1 - level)
            gp_model.addConstr(right_flow <= 1 - z_col)
        gp_model.addConstr(flow[:, column[node]] == conservation)

    active_leaves = [node for node in active_nodes if children_left[node] < 0]
    leaf_columns = [column[node] for node in active_leaves]
    values = tree["value"][active_leaves, :]
    leaf_flow = flow[:, leaf_columns]
    return leaf_flow @ values, values, TreeLeaves(leaf_flow, np.array(active_leaves))
