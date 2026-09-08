from fractions import Fraction
import json

import pytest

from environments.semantic_scene.belief import RuleBelief, UnsupportedHistory
from environments.semantic_scene.rules import RuleError, compile_rule


def scene(color):
    return {"objects": [{"color": color, "shape": "block", "size": "small"}]}


def make_rule(color):
    return compile_rule(
        json.dumps(
            {
                "op": "count",
                "comparison": "ge",
                "n": 1,
                "where": {"op": "is", "attribute": "color", "value": color},
            }
        )
    )


def test_exact_prior_conditioning_and_risk_by_independent_world_sum():
    rules = [make_rule(c) for c in ("red", "blue", "yellow")]
    belief = RuleBelief(rules, [1, 2, 3])
    posterior = belief.condition([(scene("red"), False)])
    assert posterior.evidence == Fraction(5, 6)
    assert posterior.probabilities == (0, Fraction(2, 5), Fraction(3, 5))
    targets = [scene(c) for c in ("red", "blue", "yellow")]
    predictions = [posterior.predict(s) for s in targets]
    # Independent literal truth matrix, not another call to the rule evaluator.
    truth_matrix = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    expected = sum(
        w * sum((p - y) ** 2 for p, y in zip(predictions, ys)) / 3
        for w, ys in zip(posterior.probabilities, truth_matrix)
    )
    assert posterior.brier_risk(targets, [1, 1, 1]) == expected == Fraction(4, 25)
    assert belief.condition([(scene("red"), False)] * 2) == posterior
    assert belief.condition([]).probabilities == (
        Fraction(1, 6),
        Fraction(2, 6),
        Fraction(3, 6),
    )


def test_unknown_truth_and_contradictions_never_reset_to_uniform():
    belief = RuleBelief([make_rule("red"), make_rule("blue")], [1, 1])
    with pytest.raises(UnsupportedHistory, match="no represented"):
        belief.condition([(scene("yellow"), True)])
    with pytest.raises(UnsupportedHistory, match="contradictory"):
        belief.condition([(scene("red"), True), (scene("red"), False)])
    with pytest.raises(RuleError, match="Boolean"):
        belief.condition([(scene("red"), 1)])


@pytest.mark.parametrize("weights", [[0, 1], [-1, 2], [True, 1], [0.5, 0.5], [1]])
def test_invalid_prior_rejected(weights):
    with pytest.raises(RuleError):
        RuleBelief([make_rule("red"), make_rule("blue")], weights)


def test_duplicate_proposals_cannot_silently_multiply_mass():
    with pytest.raises(RuleError, match="duplicate"):
        RuleBelief([make_rule("red"), make_rule("red")], [1, 1])
