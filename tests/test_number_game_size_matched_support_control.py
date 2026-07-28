import pytest

from scripts.number_game_generator_aware_bed import (
    RuleHypothesis,
    choose_predictive_bayes_risk_root,
    predictive_bayes_risk_scores,
)
from scripts.number_game_size_matched_support_control import (
    average_endpoint_mixture,
    fast_predictive_bayes_risk_root,
    size_matched_static_branches,
)


def _rule(name: str, positives: set[int]) -> RuleHypothesis:
    return RuleHypothesis(
        name=name,
        expression="n == 0",
        extension=tuple(number in positives for number in range(101)),
    )


def test_size_matched_branches_are_deterministic_and_exact_size():
    rules = [
        _rule("a", {1}),
        _rule("b", {1, 2}),
        _rule("c", {2}),
        _rule("d", {3}),
    ]
    actual = {
        (1, False): [rules[2]],
        (1, True): [rules[0], rules[1]],
    }

    first = size_matched_static_branches(
        global_pool=rules,
        actual_branches=actual,
        roots=[1],
        tree_seed=10,
        sample_index=2,
    )
    second = size_matched_static_branches(
        global_pool=rules,
        actual_branches=actual,
        roots=[1],
        tree_seed=10,
        sample_index=2,
    )

    assert first == second
    assert len(first[(1, False)]) == 1
    assert len(first[(1, True)]) == 2
    assert all(not rule.extension[1] for rule in first[(1, False)])
    assert all(rule.extension[1] for rule in first[(1, True)])


def test_fast_root_matches_reference_selector():
    support = [
        _rule("a", {1, 2}),
        _rule("b", {1, 3}),
        _rule("c", {2, 3}),
        _rule("d", {4}),
    ]
    roots = [1, 2]
    branches = {
        (root, label): [
            rule for rule in support if rule.extension[root] == label
        ]
        for root in roots
        for label in (False, True)
    }
    reference = choose_predictive_bayes_risk_root(
        predictive_bayes_risk_scores(
            support=support,
            roots=roots,
            branches=branches,
        )
    )

    assert fast_predictive_bayes_risk_root(
        support=support,
        roots=roots,
        branches=branches,
    ) == reference


def test_endpoint_mixture_preserves_repeated_root_weight():
    def endpoint(value):
        return {
            "mean_posterior_predictive_brier": value,
            "mean_best_hamming_error": value,
            "truth_extension_coverage_rate": 1.0 - value,
            "targets": [
                {
                    "target": "t",
                    "posterior_predictive_brier": value,
                    "best_hamming_error": value,
                    "truth_extension_covered": 1.0 - value,
                }
            ],
        }

    result = average_endpoint_mixture(
        [endpoint(0.1), endpoint(0.1), endpoint(0.4)],
        policy="mixture",
    )

    assert result["mean_posterior_predictive_brier"] == pytest.approx(0.2)
    assert result["truth_extension_coverage_rate"] == pytest.approx(0.8)
