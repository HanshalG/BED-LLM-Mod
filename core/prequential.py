"""Prequential categorical weighting of fixed, history-adaptive predictors.

Weights describe predictive pipelines, not posterior mass on physical mechanisms.
This enforces local forecast/observation ordering, not external information privacy.
No clipping, model calls, fitting to retrospective targets, or automatic restart.
"""

from dataclasses import dataclass
from fractions import Fraction


def _simplex(values, *, positive=False):
    if type(values) not in (list, tuple):
        raise ValueError("probabilities must be a list or tuple")
    values = tuple(values)
    if not values or any(type(v) not in (int, Fraction) or v < 0 for v in values):
        raise ValueError("nonnegative exact probabilities required")
    if sum(values) != 1 or (positive and any(v == 0 for v in values)):
        raise ValueError("normalized probability simplex required")
    return tuple(Fraction(v) for v in values)


@dataclass(frozen=True)
class Forecast:
    event_id: str
    sequence: int
    expert_ids: tuple
    expert_probabilities: tuple
    prior_weights: tuple
    mixture: tuple


@dataclass(frozen=True)
class ScoreReceipt:
    event_id: str
    sequence: int
    outcome_index: int
    predictive_probability: Fraction
    expert_likelihoods: tuple
    posterior_weights: tuple


class PrequentialMixture:
    """One-use forecasts; score only the next outcome, then permit a refresh.

    An expert is a named prediction algorithm, which may generate different structures
    after each observed outcome. Do not insert those structures as new experts here:
    that would require a separate prospective admission/prior rule. Zero predictive
    likelihood eliminates that expert; all-zero likelihood terminates this learner.
    """

    def __init__(self, expert_ids, prior_weights, *, outcomes=2):
        if type(expert_ids) not in (list, tuple):
            raise ValueError("expert IDs must be a list or tuple")
        ids = tuple(expert_ids)
        if any(type(i) is not str or not i or len(i) > 128 for i in ids):
            raise ValueError("invalid expert ID")
        if not ids or len(ids) > 32 or len(set(ids)) != len(ids):
            raise ValueError("1 to 32 distinct expert IDs required")
        if type(outcomes) is not int or not 2 <= outcomes <= 32:
            raise ValueError("2 to 32 outcomes required")
        weights = _simplex(prior_weights, positive=True)
        if len(ids) != len(weights):
            raise ValueError("expert/prior size mismatch")
        self.expert_ids = ids
        self.outcomes = outcomes
        self._weights = weights
        self._pending = None
        self._seen = set()
        self._sequence = 0
        self._failed = False

    @property
    def weights(self):
        return self._weights

    def forecast(self, event_id, forecasts):
        if self._failed:
            raise RuntimeError("learner terminal after unsupported observation")
        if self._pending is not None:
            raise RuntimeError("score the outstanding forecast before refresh")
        if (
            type(event_id) is not str
            or not event_id
            or len(event_id) > 256
            or event_id in self._seen
        ):
            raise ValueError("unique nonempty event ID required")
        if type(forecasts) is not dict or set(forecasts) != set(self.expert_ids):
            raise ValueError("exact fixed expert set required")
        probabilities = tuple(_simplex(forecasts[e]) for e in self.expert_ids)
        if any(len(p) != self.outcomes for p in probabilities):
            raise ValueError("outcome size mismatch")
        mixture = tuple(
            sum((w * p[k] for w, p in zip(self._weights, probabilities)), Fraction(0))
            for k in range(self.outcomes)
        )
        frozen = Forecast(
            event_id,
            self._sequence,
            self.expert_ids,
            probabilities,
            self._weights,
            mixture,
        )
        self._pending = frozen
        self._seen.add(event_id)
        return frozen

    def observe(self, forecast, outcome_index):
        if self._failed:
            raise RuntimeError("learner terminal after unsupported observation")
        if forecast is not self._pending or forecast is None:
            raise ValueError("only the outstanding original forecast can be scored")
        if type(outcome_index) is not int or not 0 <= outcome_index < self.outcomes:
            raise ValueError("invalid categorical outcome index")
        likelihoods = tuple(p[outcome_index] for p in forecast.expert_probabilities)
        evidence = forecast.mixture[outcome_index]
        if evidence == 0:
            self._failed = True
            self._pending = None
            raise RuntimeError(
                "unsupported observation: every weighted expert assigned zero"
            )
        self._weights = tuple(
            w * p / evidence for w, p in zip(forecast.prior_weights, likelihoods)
        )
        self._pending = None
        self._sequence += 1
        return ScoreReceipt(
            forecast.event_id,
            forecast.sequence,
            outcome_index,
            evidence,
            likelihoods,
            self._weights,
        )
