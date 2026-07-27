from __future__ import annotations

import numpy as np

from scripts.worldvalues_persona_nonmyopic_opportunity import (
    TASK_SPEC_HASH,
    _canonical_hash,
    analyze_task,
    expected_target_entropy_after_question,
    frozen_task_specs,
    posterior_branches,
)


def test_posterior_branches_normalize_and_recover_predictive_mass() -> None:
    weights = np.asarray([0.25, 0.75])
    likelihoods = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    outcome_probabilities, posteriors = posterior_branches(
        weights, likelihoods
    )
    assert np.allclose(outcome_probabilities, [0.275, 0.725])
    assert np.allclose(posteriors.sum(axis=1), 1.0)
    assert np.allclose(
        posteriors[0], [0.25 * 0.8 / 0.275, 0.75 * 0.1 / 0.275]
    )


def test_expected_entropy_matches_manual_binary_calculation() -> None:
    weights = np.asarray([0.5, 0.5])
    question = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    target = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]])
    observed = expected_target_entropy_after_question(
        weights, question, target
    )

    outcome_zero = 0.55
    posterior_zero = np.asarray([0.45 / 0.55, 0.10 / 0.55])
    outcome_one = 0.45
    posterior_one = np.asarray([0.05 / 0.45, 0.40 / 0.45])

    def entropy(values: np.ndarray) -> float:
        return float(-np.sum(values * np.log(values)))

    expected = (
        outcome_zero * entropy(posterior_zero)
        + outcome_one * entropy(posterior_one)
    )
    assert np.isclose(observed, expected)


def test_toy_problem_contains_true_nonmyopic_tradeoff() -> None:
    probability_of_zero = np.asarray(
        [
            [0.641, 0.038, 0.051, 0.295, 0.886],
            [0.918, 0.954, 0.035, 0.550, 0.073],
            [0.817, 0.588, 0.515, 0.506, 0.553],
            [0.105, 0.026, 0.995, 0.585, 0.529],
            [0.068, 0.366, 0.554, 0.479, 0.360],
            [0.856, 0.280, 0.088, 0.113, 0.135],
            [0.147, 0.505, 0.839, 0.726, 0.994],
            [0.365, 0.684, 0.856, 0.372, 0.191],
        ]
    )
    candidate_probabilities = np.stack(
        [probability_of_zero, 1.0 - probability_of_zero], axis=-1
    )
    target = np.zeros((8, 1, 2), dtype=np.float64)
    target[:4, 0, 0] = 1.0
    target[4:, 0, 1] = 1.0
    probabilities = np.concatenate(
        [target, candidate_probabilities], axis=1
    )
    question_ids = ["target", "a", "b", "c", "d", "e"]
    record = analyze_task(
        {
            "task_index": 0,
            "target_questions": ["target"],
            "candidate_questions": ["a", "b", "c", "d", "e"],
        },
        question_ids,
        probabilities,
    )
    assert record["myopic_root_question"] == "a"
    assert record["adaptive_d2_root_question"] == "b"
    assert record["immediate_sacrifice"] > 0.01
    assert record["final_advantage"] > 0.01
    assert record["strict_tradeoff"]


def test_frozen_task_spec_hash_is_stable() -> None:
    question_ids = [
        "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q27", "Q28", "Q29",
        "Q30", "Q31", "Q32", "Q46", "Q51", "Q52", "Q53", "Q54",
        "Q55", "Q58", "Q59", "Q60", "Q61", "Q62", "Q63", "Q64",
        "Q65", "Q66", "Q67", "Q68", "Q69", "Q70", "Q71", "Q72",
        "Q73", "Q74", "Q75", "Q76", "Q77", "Q78", "Q79", "Q80",
        "Q81", "Q113", "Q114", "Q115", "Q116", "Q117", "Q118",
        "Q130", "Q131", "Q132", "Q133", "Q134", "Q135", "Q136",
        "Q137", "Q138", "Q142", "Q143", "Q146", "Q147", "Q148",
        "Q169", "Q170", "Q196", "Q197", "Q198", "Q199", "Q224",
        "Q225", "Q226", "Q227", "Q228", "Q229", "Q230", "Q231",
        "Q232", "Q233", "Q234", "Q235", "Q236", "Q237", "Q238",
        "Q239", "Q253", "Q254", "Q255", "Q256", "Q257", "Q258",
        "Q259",
    ]
    specs = frozen_task_specs(question_ids)
    assert _canonical_hash(specs) == TASK_SPEC_HASH
    assert len(specs) == 20
    assert all(len(spec["target_questions"]) == 8 for spec in specs)
    assert all(len(spec["candidate_questions"]) == 24 for spec in specs)
    assert all(
        set(spec["target_questions"]).isdisjoint(
            spec["candidate_questions"]
        )
        for spec in specs
    )
