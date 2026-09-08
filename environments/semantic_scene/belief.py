"""Exact fixed-support conditioning for deterministic membership queries.

The caller owns the prior and target distribution. This is not a model-discovery
prior or a correction for outcome-dependent proposal selection.
"""

from dataclasses import dataclass
from fractions import Fraction

from .rules import CompiledRule, RuleError, parse_scene


class UnsupportedHistory(RuleError):
    pass


def _weights(values):
    if not values or any(type(w) not in (int, Fraction) or w <= 0 for w in values):
        raise RuleError("positive exact prior/target weights required")
    total = sum(values)
    return tuple(Fraction(w) / total for w in values)


@dataclass(frozen=True)
class Posterior:
    rules: tuple
    probabilities: tuple
    evidence: Fraction

    def predict(self, scene):
        return sum(
            (w for r, w in zip(self.rules, self.probabilities) if r.label(scene)),
            Fraction(0),
        )

    def brier_risk(self, targets, weights):
        if not targets or len(targets) != len(weights):
            raise RuleError("fixed target/weight lengths required")
        target_weights = _weights(weights)
        probabilities = [self.predict(s) for s in targets]
        return sum(
            (w * p * (1 - p) for w, p in zip(target_weights, probabilities)),
            Fraction(0),
        )


class RuleBelief:
    def __init__(self, rules, prior_weights):
        self.rules = tuple(rules)
        if len(self.rules) != len(prior_weights) or not self.rules:
            raise RuleError("rule/prior lengths required")
        if any(not isinstance(r, CompiledRule) for r in self.rules):
            raise RuleError("compiled rules required")
        if len({r.key for r in self.rules}) != len(self.rules):
            raise RuleError("duplicate canonical rules: declare prior mass explicitly")
        self.prior = _weights(prior_weights)

    def condition(self, history):
        # Always replay from the prior, never treat a refreshed pool as new evidence.
        labels = {}
        observations = []
        for scene, label in history:
            if type(label) is not bool:
                raise RuleError("membership label must be Boolean")
            key = parse_scene(scene)
            if key in labels and labels[key] != label:
                raise UnsupportedHistory("contradictory deterministic labels")
            labels[key] = label
            observations.append((scene, label))
        masses = tuple(
            w if all(r.label(s) == y for s, y in observations) else Fraction(0)
            for r, w in zip(self.rules, self.prior)
        )
        evidence = sum(masses, Fraction(0))
        if not evidence:
            raise UnsupportedHistory("no represented rule explains the history")
        return Posterior(self.rules, tuple(w / evidence for w in masses), evidence)
