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

"""Ensemble-level tree formulations built on shared ordinal split variables.

The formulations dispatched here differ from the per-tree ``"leaf"`` and
``"paths"`` formulations of
:py:class:`gurobi_ml.modeling.decision_tree.AbstractTreeEstimator` in that the
split binaries ``z[f, j]`` ("is ``x_f <= v_j``?") exist once per (feature,
threshold) for the whole ensemble — all trees reference the same variables.
The binary count therefore grows with the number of distinct thresholds, not
with the number of trees, and the trees are coupled combinatorially through
the shared binaries instead of only through the continuous input variables.
"""

from warnings import warn

import numpy as np

from .biggs_perakis import add_biggs_perakis_tree
from .misic import add_misic_tree
from .ocean import OrdinalMuVariables, add_ocean_tree
from .parmentier_vidal import add_parmentier_vidal_tree
from .split_variables import SplitVariables

_TREE_BUILDERS = {
    "misic": add_misic_tree,
    "parmentier_vidal": add_parmentier_vidal_tree,
    "ocean": add_ocean_tree,
    "biggs_perakis": add_biggs_perakis_tree,
}

#: Formulations handled by :py:func:`add_tree_ensemble_formulation`.
ENSEMBLE_FORMULATIONS = tuple(_TREE_BUILDERS)


def add_tree_ensemble_formulation(
    gp_model,
    trees,
    weights,
    constant,
    _input,
    output,
    formulation,
    epsilon,
    _name_var,
    safety_floor=0.0,
):
    """Formulate a tree ensemble in gp_model using formulation.

    The formulation predicts
    ``output == sum_t weights[t] * tree_t(input) + constant``.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    trees : list of dict
        The trees of the ensemble (dict representation of
        :py:class:`gurobi_ml.modeling.decision_tree.AbstractTreeEstimator`).
    weights : ndarray
        Weight of each tree in the ensemble prediction.
    constant : float
        Constant offset of the ensemble prediction.
    _input : mvar_array_like
        Decision variables used as input for the ensemble.
    output : mvar_array_like
        Decision variables used as output for the ensemble.
    formulation : str
        The formulation to use; one of ``ENSEMBLE_FORMULATIONS``.
    epsilon : float
        Small value used to impose strict inequalities for splitting nodes.
    _name_var : function
        Function to name variables.
    safety_floor : float, optional
        |SafetyFloorParam|

    Returns
    -------
    list of (mvar_array_like, ndarray)
        Per tree, the leaf variables and the tree node index of each of
        their columns: column ``j`` of the variables corresponds to leaf
        node ``nodes[j]`` (as in sklearn's ``apply``). The variable is 1
        (or carries the flow) exactly when the input reaches that leaf —
        usable to reconstruct decisions or to constrain leaf co-selection
        across examples.
    """
    try:
        tree_builder = _TREE_BUILDERS[formulation]
    except KeyError:
        raise ValueError(f"Unknown formulation: {formulation}") from None

    # The projected Biggs-Perakis formulation shares no ensemble-level
    # variables: its epsilon lives in the per-tree leaf boxes and acts
    # path-wise like the leaf baseline, so the global-epsilon warning does
    # not apply.
    if epsilon > 0.0 and formulation != "biggs_perakis":
        warn(
            f"epsilon={epsilon} with the '{formulation}' formulation applies "
            "globally: the band (t, t + epsilon) of every threshold of the "
            "ensemble is forbidden, not only along the selected paths as in "
            "the 'leaf' formulation. The model becomes infeasible when the "
            "attainable range of an input (e.g. a derived pipeline feature) "
            "is narrower than epsilon, and an epsilon below Gurobi's "
            "IntFeasTol may return solutions that disagree with the "
            "predictor. Consider leaving epsilon at 0.",
            UserWarning,
        )

    # The "ocean" variant shares continuous ordinal interval variables
    # instead of binary split variables; "biggs_perakis" shares nothing —
    # its trees couple through the input variables only.
    if formulation == "biggs_perakis":
        split_vars = None
    else:
        shared_variables = (
            OrdinalMuVariables if formulation == "ocean" else SplitVariables
        )
        split_vars = shared_variables(
            gp_model, trees, _input, epsilon, _name_var, safety_floor
        )

    outdim = output.shape[1]
    output_lb = np.full(outdim, float(constant))
    output_ub = np.full(outdim, float(constant))
    total = constant
    tree_leafs = []
    for i, (tree, weight) in enumerate(zip(trees, weights)):
        expression, values, leaf_vars, leaf_nodes = tree_builder(
            gp_model,
            split_vars,
            tree,
            _input,
            epsilon,
            name=_name_var(f"y[{i}]"),
            safety_floor=safety_floor,
        )
        tree_leafs.append((leaf_vars, leaf_nodes))
        total = total + weight * expression
        output_lb += np.minimum(
            weight * values.min(axis=0), weight * values.max(axis=0)
        )
        output_ub += np.maximum(
            weight * values.min(axis=0), weight * values.max(axis=0)
        )

    gp_model.addConstr(output == total)
    gp_model.addConstr(output >= output_lb)
    gp_model.addConstr(output <= output_ub)
    return tree_leafs
