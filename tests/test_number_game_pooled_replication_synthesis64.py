from __future__ import annotations

import json

from scripts import number_game_pooled_replication_synthesis64 as analysis


def _row(
    seed: int,
    *,
    candidate: float,
    myopic: float,
    parent: float,
    generated: float,
) -> dict:
    return {
        "tree_seed": seed,
        "policy": {
            "candidate_brier": candidate,
            "baseline_brier": myopic,
            "candidate_hamming": candidate / 2.0,
            "baseline_hamming": myopic / 2.0,
            "candidate_coverage": 0.75,
            "baseline_coverage": 0.5,
        },
        "support": {
            "candidate_brier": candidate,
            "parent_only_brier": parent,
            "generated_only_brier": generated,
            "candidate_root": seed,
            "parent_only_root": seed + 1,
            "generated_only_root": seed + 2,
        },
    }


def test_hash_bound_sources_are_two_disjoint_cohorts() -> None:
    first_policy, first_ablation, second = analysis.load_sources()
    cohorts = analysis.cohort_rows(first_policy, first_ablation, second)

    assert [len(cohort) for cohort in cohorts] == [32, 32]
    assert not (
        {row["tree_seed"] for row in cohorts[0]}
        & {row["tree_seed"] for row in cohorts[1]}
    )
    for cohort, source in zip(
        cohorts,
        (first_ablation, second["second_refresh"]),
        strict=True,
    ):
        parent = source["comparisons"]["parent_only"]
        generated = source["comparisons"]["generated_only"]
        assert analysis.summarize_root_differences(
            [cohort],
            "parent_only",
        )["pooled"] == parent["root_differences"]
        assert analysis.summarize_root_differences(
            [cohort],
            "generated_only",
        )["pooled"] == generated["root_differences"]
        assert abs(
            sum(
                row["support"]["candidate_brier"]
                - row["support"]["parent_only_brier"]
                for row in cohort
            )
            / len(cohort)
            - parent["mean_candidate_minus_baseline_brier"]
        ) < 1e-12


def test_stratified_summary_uses_both_cohorts_and_reports_contrast() -> None:
    cohorts = [
        [
            _row(1, candidate=0.1, myopic=0.2, parent=0.15, generated=0.16),
            _row(2, candidate=0.2, myopic=0.3, parent=0.25, generated=0.26),
        ],
        [
            _row(3, candidate=0.3, myopic=0.35, parent=0.31, generated=0.32),
            _row(4, candidate=0.4, myopic=0.45, parent=0.41, generated=0.42),
        ],
    ]
    bootstrap = analysis.stratified_bootstrap_indices(
        cohort_sizes=[2, 2],
        seed=1,
        samples=100,
    )
    summary = analysis.summarize_difference(
        cohorts,
        candidate_path=("support", "candidate_brier"),
        baseline_path=("support", "parent_only_brier"),
        bootstrap_indices=bootstrap,
    )

    assert summary["candidate_mean"] == 0.25
    assert abs(summary["baseline_mean"] - 0.28) < 1e-12
    assert abs(summary["mean_candidate_minus_baseline"] + 0.03) < 1e-12
    assert all(
        abs(actual - expected) < 1e-12
        for actual, expected in zip(
            summary["source_mean_differences"],
            [-0.05, -0.01],
            strict=True,
        )
    )
    assert abs(summary["cohort_one_minus_two_mean_difference"] + 0.04) < 1e-12
    assert (summary["wins"], summary["ties"], summary["losses"]) == (4, 0, 0)


def test_root_differences_are_reported_by_cohort() -> None:
    cohorts = [
        [_row(1, candidate=0.1, myopic=0.2, parent=0.2, generated=0.2)],
        [_row(2, candidate=0.1, myopic=0.2, parent=0.2, generated=0.2)],
    ]

    assert analysis.summarize_root_differences(cohorts, "parent_only") == {
        "pooled": 2,
        "per_cohort": [1, 1],
    }


def test_zero_call_synthesis_preserves_prospective_null(tmp_path) -> None:
    result = analysis.run_synthesis(tmp_path)

    assert result["status"] == "retrospective_policy_robustness_positive"
    assert result["protocol"]["model_calls"] == 0
    assert result["protocol"]["cost_usd"] == 0.0
    assert result["protocol"]["prospective_mechanism_status_remains_binding"]
    assert all(result["robustness_checks"].values())
    assert (
        result["pooled"]["policy_vs_myopic"]["brier"]["relative_reduction"]
        == 0.11217762371675755
    )
    assert (
        result["pooled"]["second_refresh"]["parent_only"]["root_differences"][
            "per_cohort"
        ]
        == [22, 19]
    )
    assert json.loads((tmp_path / "RESULT.json").read_text())["status"] == (
        "retrospective_policy_robustness_positive"
    )
