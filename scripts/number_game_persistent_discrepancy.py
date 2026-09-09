"""Candidate full-support world model; epistemic rule discrepancies persist."""
from dataclasses import dataclass
import math

from scripts.number_game_fixed_target_risk import _support, _indices


@dataclass(frozen=True)
class PersistentDiscrepancyBelief:
    """Uniform rule prior; Beta flip rate and one fixed latent bit per coordinate.

    Hyperparameters are mandatory, not estimated from evaluation outcomes.
    This is an opt-in research model, not a qualified predictive interface.
    """
    rules: tuple
    alpha: float
    beta: float
    history: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'rules', _support(self.rules))
        if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0
               for x in (self.alpha, self.beta)):
            raise ValueError('positive finite Beta hyperparameters required')
        history = tuple(self.history)
        seen = {}
        for q, answer in history:
            _indices((q,), len(self.rules[0]))
            if type(answer) is not bool:
                raise ValueError('boolean observation required')
            if q in seen and seen[q] != answer:
                raise ValueError('contradiction in deterministic observations')
            seen[q] = answer
        object.__setattr__(self, 'history', tuple(sorted(seen.items())))

    def _components(self):
        n = len(self.history)
        parameters, logs = [], []
        base = math.lgamma(self.alpha)+math.lgamma(self.beta)-math.lgamma(self.alpha+self.beta)
        for rule in self.rules:
            errors = sum(rule[q] != y for q, y in self.history)
            a, b = self.alpha+errors, self.beta+n-errors
            parameters.append((a, b))
            logs.append(math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)-base)
        shift = max(logs)
        weights = [math.exp(x-shift) for x in logs]
        total = sum(weights)
        return tuple(w/total for w in weights), tuple(parameters)

    def observe(self, query, answer):
        return type(self)(self.rules, self.alpha, self.beta, self.history+((query, answer),))

    def predict(self, targets):
        targets = _indices(targets, len(self.rules[0]))
        observed = dict(self.history)
        weights, parameters = self._components()
        return tuple(float(observed[q]) if q in observed else sum(
            w*((1-a/(a+b)) if rule[q] else a/(a+b))
            for w, (a, b), rule in zip(weights, parameters, self.rules)) for q in targets)

    def sample_world(self, rng):
        weights, parameters = self._components()
        index = rng.choices(range(len(weights)), weights=weights, k=1)[0]
        rate = rng.betavariate(*parameters[index])
        observed = dict(self.history)
        return tuple(observed[q] if q in observed else value ^ (rng.random() < rate)
                     for q, value in enumerate(self.rules[index]))

    def expected_query_brier(self, query, targets):
        targets = _indices(targets, len(self.rules[0]))
        probability = self.predict((query,))[0]
        risk = 0.0
        for answer, mass in ((False, 1-probability), (True, probability)):
            if mass:
                predictions = self.observe(query, answer).predict(targets)
                risk += mass*sum(p*(1-p) for p in predictions)/len(targets)
        return risk
