"""Proper terminal loss against a separately supplied rollout target law.

The reference must come from the simulator's joint world/observation law,
conditioned on the branch. A regenerated forecast cannot grade itself.
This utility does not establish that the supplied reference is calibrated.
"""
import math


def _distribution(row):
    if not isinstance(row, dict) or not row:
        raise ValueError('nonempty categorical distribution required')
    if any(not isinstance(k, str) or not k for k in row):
        raise ValueError('nonempty string categories required')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for v in row.values()):
        raise ValueError('finite nonnegative probabilities required')
    if abs(math.fsum(row.values()) - 1) > 1e-10:
        raise ValueError('probabilities must sum to one')


def expected_brier(reference, forecast):
    """Half-Brier risk and exact Bayes-risk/excess-risk decomposition.

Missing categories have zero mass, including categories unique to either
distribution. None denotes abstention with the project's fixed loss of one.
No smoothing, renormalization, or dropping unsupported outcomes is performed.
"""
    _distribution(reference)
    bayes_risk = .5 * math.fsum(p * (1-p) for p in reference.values())
    if forecast is None:
        return dict(expected_loss=1., reference_bayes_risk=bayes_risk,
                    excess_risk=None, forecast_self_risk=None, abstained=True)
    _distribution(forecast)
    categories = reference.keys() | forecast.keys()
    excess = .5 * math.fsum((reference.get(k, 0)-forecast.get(k, 0))**2
                            for k in categories)
    self_risk = .5 * math.fsum(p * (1-p) for p in forecast.values())
    return dict(expected_loss=bayes_risk+excess,
                reference_bayes_risk=bayes_risk, excess_risk=excess,
                forecast_self_risk=self_risk, abstained=False)
