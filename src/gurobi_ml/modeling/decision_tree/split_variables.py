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

"""Shared ordinal split variables for tree-ensemble MIP formulations."""

from warnings import warn

import numpy as np
from gurobipy import GRB


def _clamp_threshold(threshold, safety_floor):
    """Clamp a split threshold away from zero (same rule as the leaf formulation)."""
    if 0 < abs(threshold) < safety_floor:
        return np.sign(threshold) * safety_floor
    return threshold


class SplitVariables:
    """Shared ordinal split binaries ``z[f, j]`` of a tree ensemble.

    For every example ``k``, feature ``f`` and distinct split threshold
    ``v_j`` (sorted increasingly, collected over all trees of the ensemble)
    a binary variable models ``x[k, f] <= v_j``.  The binaries of one feature
    are ordered by ``z[k, f, j] <= z[k, f, j + 1]``, and indicator
    constraints link them to the input variables:

        ``z = 1  =>  x <= v_j``    and    ``z = 0  =>  x >= v_j + epsilon``.

    Thresholds falling outside the bounds of an input variable fix the
    corresponding binary instead of adding the linking constraints.  Features
    whose input column is fixed (``lb == ub`` for every example) get no
    linking at all: their binaries are fixed by comparing the constant value
    to the threshold, without applying ``epsilon`` — mirroring the
    ``feature_is_fixed`` handling of the leaf formulation.

    Parameters
    ----------
    gp_model : :external+gurobi:py:class:`Model`
        The gurobipy model where the predictor should be inserted.
    trees : list of dict
        The trees of the ensemble (dict representation of
        :py:class:`gurobi_ml.modeling.decision_tree.AbstractTreeEstimator`).
    _input : mvar_array_like
        Decision variables used as input for the ensemble.
    epsilon : float
        Small value used to impose strict inequalities for splitting nodes.
    _name_var : function
        Function to name variables.
    safety_floor : float, optional
        |SafetyFloorParam|
    """

    def __init__(self, gp_model, trees, _input, epsilon, _name_var, safety_floor=0.0):
        self._safety_floor = safety_floor
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
        self._vars = {}
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

            z = gp_model.addMVar(
                (nex, values.size), vtype=GRB.BINARY, name=_name_var(f"z[{f}]")
            )
            self._thresholds[f] = values
            self._vars[f] = z

            # x <= v_j implies x <= v_{j+1}: the binaries are ordered.
            if values.size > 1:
                gp_model.addConstr(z[:, :-1] <= z[:, 1:])

            if self.feature_is_fixed[f]:
                go_left = input_lb[:, [f]] <= values[None, :]
                z[go_left].setAttr(GRB.Attr.LB, 1.0)
                z[~go_left].setAttr(GRB.Attr.UB, 0.0)
                continue

            # Thresholds outside the input bounds fix the binaries; only the
            # remaining ones need linking constraints to the input variables.
            always_left = input_ub[:, [f]] <= values[None, :]
            never_left = input_lb[:, [f]] > values[None, :]
            z[always_left].setAttr(GRB.Attr.LB, 1.0)
            z[never_left].setAttr(GRB.Attr.UB, 0.0)

            for k, j in zip(*(~(always_left | never_left)).nonzero()):
                z_var = z[k, j].item()
                x_var = _input[k, f].item()
                gp_model.addGenConstrIndicator(
                    z_var, 1, x_var, GRB.LESS_EQUAL, values[j]
                )
                gp_model.addGenConstrIndicator(
                    z_var, 0, x_var, GRB.GREATER_EQUAL, values[j] + epsilon
                )

    def column(self, feature, threshold):
        """Return the binaries (one per example) of feature's split at threshold."""
        values = self._thresholds[feature]
        threshold = _clamp_threshold(threshold, self._safety_floor)
        j = np.searchsorted(values, threshold)
        assert j < values.size and values[j] == threshold
        return self._vars[feature][:, j]
