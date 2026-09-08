"""Necessary headroom check for nonnegative-loss ladders, never a success gate."""
from fractions import Fraction


def _exact(value):
    if type(value) not in (str, int, Fraction):
        raise ValueError('provide an exact rational value, not an estimated float')
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        raise ValueError('invalid rational value') from None


def assess_headroom(myopic_risk, optimal_risk_lower_bound, adjacent_gains, *, scope):
    """Caller must establish the bound on the SAME population, loss and budget.

    R_opt >= L implies no policy can reach a target risk below L. Insufficient
    headroom rules out the requested ladder; sufficient headroom does not prove
    the ladder exists or authorize experiments. An achievable policy risk is an
    UPPER bound on R_opt and is invalid as L unless independently proved optimal.
    """
    baseline, lower = _exact(myopic_risk), _exact(optimal_risk_lower_bound)
    if baseline <= 0 or lower < 0 or lower > baseline:
        raise ValueError('require 0 <= lower bound <= positive myopic risk')
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError('explicit population/loss/budget scope required')
    gains = tuple(_exact(g) for g in adjacent_gains)
    if not gains or any(g < 0 or g >= 1 for g in gains):
        raise ValueError('one or more fractional gains in [0,1) required')
    retained = Fraction(1)
    for gain in gains:
        retained *= 1-gain
    return dict(status='ruled_out' if lower > baseline*retained else 'not_ruled_out',
                scope=scope, exact_required_total_gain=str(1-retained),
                exact_maximum_possible_total_gain=str(1-lower/baseline),
                exact_required_terminal_risk_ceiling=str(baseline*retained),
                exact_optimal_risk_lower_bound=str(lower),
                sufficient_for_success=False, paid_calls_authorized=False)
