from __future__ import annotations

import math

from scripts.atd_code_first_link_audit import (
    entropy_from_outcomes,
    score_task,
    spearman,
    summarize,
)


def test_entropy_and_spearman_helpers() -> None:
    assert math.isclose(entropy_from_outcomes(["a", "a", "b", "b"]), math.log(2))
    assert spearman([1.0, 2.0, 3.0], [2.0, 4.0, 8.0]) == 1.0
    assert spearman([1.0, 1.0], [0.0, 1.0]) is None


def test_score_task_retains_particle_multiplicity_and_true_branch() -> None:
    row = score_task(
        7,
        ["q0", "q1", "q2"],
        [
            ["yes", "same", "left"],
            ["yes", "same", "left"],
            ["no", "same", "right"],
            ["no", "same", "other"],
        ],
        ["yes", "same", "left"],
        [True, False, False, False],
    )

    assert row["usable"] is False  # The formal audit requires at least eight particles.
    assert row["queries"][0]["survivor_count"] == 2
    assert row["queries"][0]["posterior_pass_fraction"] == 0.5
    assert row["selected_query_index"] == 2
    assert row["oracle_query_index"] == 2


def test_summary_applies_all_frozen_gates() -> None:
    task = {
        "usable": True,
        "initial_pass_fraction": 0.25,
        "pass_fraction_range": 0.5,
        "selected_gain_over_initial": 0.2,
        "selected_gain_over_candidate_mean": 0.1,
        "top1_regret": 0.05,
        "spearman_eig_vs_pass": 0.4,
    }
    result = summarize([task.copy() for _ in range(35)])
    assert result["all_gates_pass"] is True
    assert result["metrics"]["usable_task_count"] == 35
