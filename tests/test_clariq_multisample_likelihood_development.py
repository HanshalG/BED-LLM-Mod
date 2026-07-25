import math

from scripts.clariq_multisample_likelihood_development import (
    build_likelihood,
    information_gain,
    policy_scores,
    spearman,
)


def test_multisample_likelihood_and_information_gain() -> None:
    informative = build_likelihood(["YN", "YN", "YU"], 2)
    uninformative = build_likelihood(["YY", "YY", "YY"], 2)
    prior = [0.5, 0.5]
    assert information_gain(prior, informative)[0] > 0.0
    assert information_gain(prior, informative)[0] > information_gain(
        prior, uninformative
    )[0]


def test_policy_scores_are_finite_and_exclude_repeated_root() -> None:
    likelihoods = {
        "Q1": build_likelihood(["YNN", "YNN", "YUN"], 3),
        "Q2": build_likelihood(["NYN", "NYN", "NYY"], 3),
        "Q3": build_likelihood(["NNY", "NNY", "UNY"], 3),
    }
    result = policy_scores(likelihoods)
    assert result["myopic_question_id"] in likelihoods
    assert result["depth_two_question_id"] in likelihoods
    assert all(
        math.isfinite(value)
        for value in result["depth_two_scores"].values()
    )
    assert all(
        followup != question_id
        for question_id, branches in result["best_followups"].items()
        for followup in branches.values()
    )


def test_spearman_handles_ties() -> None:
    assert abs(spearman([1, 2, 3], [10, 20, 30]) - 1.0) < 1e-12
    assert spearman([1, 1, 1], [10, 20, 30]) == 0.0
