"""Conditional fixed-graph diagnostics. Caller-verified premises are required.

Reachability is not an optimization or accuracy guarantee.
"""
from dataclasses import asdict, dataclass
import math

UNREACHABLE = 'STRUCTURALLY_UNREACHABLE_UNDER_ASSUMPTIONS'
REACHABLE = 'REACHABLE_NOT_GUARANTEED_REPAIRABLE'
UNKNOWN = 'UNKNOWN'

@dataclass(frozen=True)
class Assumptions:
    receiver_only_scope: bool | None = None
    no_parameter_alias: bool | None = None
    upstream_independent_of_receiver: bool | None = None
    router_independent_of_receiver: bool | None = None
    fixed_choice_and_lifecycle: bool | None = None
    no_downstream_cross_position_mixing: bool | None = None
    fixed_tokens: bool | None = None
    fixed_targets_and_weights: bool | None = None
    fixed_finite_state_and_precision: bool | None = None
    cache_valid_or_unused: bool | None = None

    def failures(self):
        return [name for name, value in asdict(self).items() if value is not True]

def classify_position(executed, assumptions, *, local_gradient_norm=None):
    if not isinstance(assumptions, Assumptions):
        assumptions = Assumptions(**assumptions)
    missing = assumptions.failures()
    if missing or executed is None:
        status = UNKNOWN
    else:
        status = REACHABLE if bool(executed) else UNREACHABLE
    return {'status': status, 'unsatisfied_or_unverified_premises': missing,
        'receiver_executed_at_position': executed,
        'local_gradient_norm_not_a_structural_premise': local_gradient_norm,
        'interpretation': 'b=0 can give zero a-gradient on a reachable path; None, numerical zero, and absence of dependency are distinct.'}

def bounds(draws, loss_mode):
    """Weights reflect actual scheduled draws, including all repetitions.

    Each draw supplies four status entries and actual fixed-target KL values.
    Partial U is valid but need not be the maximal unreachable set.  A finite
    precision estimate is not the exact nonnegative mathematical KL theorem.
    """
    if not draws or loss_mode not in ('ALL4', 'FIRST'):
        raise ValueError('nonempty fixed draws and registered loss required')
    n = len(draws)
    weights = [1 / (4 * n)] * 4 if loss_mode == 'ALL4' else [1 / n, 0., 0., 0.]
    numeric = first = objective = 0.
    numeric64 = first64 = 0.
    has64 = all(d.get('KL_FP64_normalized') is not None for d in draws)
    count = {'unreachable': 0, 'reachable': 0, 'unknown': 0}
    weighted_unknown = 0.
    for draw in draws:
        values, statuses = draw['KL_FP32'], draw['statuses']
        if len(values) != 4 or len(statuses) != 4 or any(not math.isfinite(x) for x in values):
            raise ValueError('missing/nonfinite four-position initial evidence')
        for j, (value, status, weight) in enumerate(zip(values, statuses, weights)):
            objective += weight * value
            if not weight:
                continue
            if status == UNREACHABLE:
                count['unreachable'] += 1
                numeric += weight * value
                if has64:
                    numeric64 += weight * draw['KL_FP64_normalized'][j]
            elif status == REACHABLE:
                count['reachable'] += 1
            else:
                count['unknown'] += 1
                weighted_unknown += weight
        if statuses[0] == UNREACHABLE:
            first += values[0] / n
            if has64:
                first64 += draw['KL_FP64_normalized'][0] / n
    return {'N_draws': n, 'loss_mode': loss_mode, 'position_weights_per_draw': weights,
        'mathematical_lower_bound': 'sum_{certified U} w_i KL(Q_i || P_phi0^i), for normalized distributions and the verified structural premises',
        'LB_actual_objective_FP32_estimate': numeric, 'LB_FIRST_FP32_estimate': first,
        'initial_objective_FP32_estimate': objective,
        'LB_actual_objective_FP64_normalized_estimate': numeric64 if has64 else None,
        'LB_FIRST_FP64_normalized_estimate': first64 if has64 else None,
        'numeric_FP64_status': 'NORMALIZED_DISTRIBUTION_RECOMPUTATION' if has64 else 'UNAVAILABLE_FULL_P0_DISTRIBUTIONS',
        'positions_with_positive_loss_weight': count, 'unknown_position_weight_mass': weighted_unknown,
        'certified_set_complete_for_weighted_positions': count['unknown'] == 0,
        'negative_FP32_values_preserved': sum(v < 0 for d in draws for v in d['KL_FP32']),
        'no_automatic_optimizer_or_admission_action': True}

def diagnostic_check(endpoint_objective, predicted_bound, *, atol, rtol):
    if not all(math.isfinite(v) for v in [endpoint_objective, predicted_bound, atol, rtol]) or atol < 0 or rtol < 0:
        raise ValueError('finite preregistered diagnostic values required')
    margin = endpoint_objective - predicted_bound
    tolerance = atol + rtol * abs(predicted_bound)
    return {'endpoint_objective': endpoint_objective, 'predicted_bound_FP32_estimate': predicted_bound,
        'endpoint_minus_bound': margin, 'fixed_tolerance': tolerance,
        'numeric_relation': 'WITHIN_DECLARED_DIAGNOSTIC_TOLERANCE' if margin >= -tolerance else 'NUMERICAL_OR_ASSUMPTION_VIOLATION_REQUIRES_REVIEW',
        'not_a_success_or_accuracy_gate': True}
