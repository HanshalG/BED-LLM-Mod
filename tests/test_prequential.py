from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from core.prequential import PrequentialMixture


def new():
    return PrequentialMixture(("initial", "refresh"), (F(1, 2), F(1, 2)))


def test_two_observations_equal_independent_likelihood_product():
    learner = new()
    first = learner.forecast(
        "first", {"initial": [F(1, 4), F(3, 4)], "refresh": [F(3, 4), F(1, 4)]}
    )
    receipt = learner.observe(first, 1)
    assert first.mixture == (F(1, 2), F(1, 2))
    assert receipt.posterior_weights == (F(3, 4), F(1, 4))
    second = learner.forecast(
        "second", {"initial": [F(1, 2), F(1, 2)], "refresh": [F(1, 5), F(4, 5)]}
    )
    learner.observe(second, 1)
    masses = (F(1, 2) * F(3, 4) * F(1, 2), F(1, 2) * F(1, 4) * F(4, 5))
    assert learner.weights == tuple(m / sum(masses) for m in masses)
    assert first.prior_weights == (F(1, 2), F(1, 2))


def test_no_refresh_replacement_or_retroactive_scoring():
    learner = new()
    inputs = {"initial": [F(1, 2), F(1, 2)], "refresh": [F(1, 4), F(3, 4)]}
    forecast = learner.forecast("q", inputs)
    inputs["refresh"][1] = 0
    assert forecast.expert_probabilities[1][1] == F(3, 4)
    with pytest.raises(RuntimeError, match="outstanding"):
        learner.forecast("new", inputs)
    with pytest.raises(ValueError, match="original"):
        learner.observe(replace(forecast), 1)
    learner.observe(forecast, 1)
    with pytest.raises(ValueError):
        learner.observe(forecast, 0)
    with pytest.raises(ValueError):
        learner.forecast("q", inputs)


def test_adaptive_forecasts_form_normalized_sequence_distribution():
    total = F(0)
    for sequence in product((0, 1), repeat=3):
        learner, probability, history = new(), F(1), []
        for t, y in enumerate(sequence):
            # Two adaptive predictors use only prior outcomes, not the current one.
            p = F(1 + sum(history), 2 + len(history))
            forecast = learner.forecast(
                str(t), {"initial": [1 - p, p], "refresh": [p, 1 - p]}
            )
            probability *= forecast.mixture[y]
            learner.observe(forecast, y)
            history.append(y)
        total += probability
    assert total == 1


def test_model_count_cannot_change_predictor_mass():
    learner = new()
    with pytest.raises(ValueError, match="fixed expert"):
        learner.forecast(
            "q",
            {
                "initial": [F(1, 2)] * 2,
                "refresh": [F(1, 2)] * 2,
                "refresh_copy": [F(1, 2)] * 2,
            },
        )
    assert learner.weights == (F(1, 2), F(1, 2))


def test_real_rule_refresh_can_only_earn_credit_on_next_observation():
    import json
    from environments.semantic_scene.belief import RuleBelief
    from environments.semantic_scene.rules import compile_rule

    def model(color):
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

    def scene(color):
        return {"objects": [{"color": color, "shape": "block", "size": "small"}]}

    initial = [model("red"), model("blue")]
    learner = new()
    first = learner.forecast(
        "first", {"initial": [F(1, 2)] * 2, "refresh": [F(1, 2)] * 2}
    )
    learner.observe(first, 1)
    history = [(scene("red"), True)]
    # Generate an executable broader rule after the answer; both rules fit it.
    red_or_blue = compile_rule(
        json.dumps(
            {
                "op": "or",
                "args": [
                    {
                        "op": "count",
                        "comparison": "ge",
                        "n": 1,
                        "where": {"op": "is", "attribute": "color", "value": c},
                    }
                    for c in ("red", "blue")
                ],
            }
        )
    )
    p0 = RuleBelief(initial, [1, 1]).condition(history).predict(scene("blue"))
    p1 = (
        RuleBelief(initial + [red_or_blue], [1, 1, 1])
        .condition(history)
        .predict(scene("blue"))
    )
    assert learner.weights == (F(1, 2), F(1, 2))
    second = learner.forecast(
        "second", {"initial": [1 - p0, p0], "refresh": [1 - p1, p1]}
    )
    assert second.mixture[1] == F(1, 4)
    learner.observe(second, 1)
    assert learner.weights == (0, 1)
    with pytest.raises(ValueError):
        learner.observe(first, 1)


def test_all_zero_observation_fails_terminal_without_retry_or_reset():
    learner = new()
    forecast = learner.forecast("q", {"initial": [1, 0], "refresh": [1, 0]})
    with pytest.raises(RuntimeError, match="unsupported"):
        learner.observe(forecast, 1)
    with pytest.raises(RuntimeError, match="terminal"):
        learner.observe(forecast, 0)
    with pytest.raises(RuntimeError, match="terminal"):
        learner.forecast("new", {"initial": [1, 0], "refresh": [1, 0]})


@pytest.mark.parametrize("bad", [[True, False], [0.5, 0.5], [F(-1), F(2)], [1, 1]])
def test_strict_probability_contract(bad):
    with pytest.raises(ValueError):
        new().forecast("q", {"initial": bad, "refresh": [0, 1]})
