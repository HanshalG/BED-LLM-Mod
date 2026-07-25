from __future__ import annotations

from scripts import animals_support_retention_holdout as holdout


def _candidate(
    *,
    retention: float,
    eig: float,
    expansion: float,
    covered: int,
    expected: float,
) -> dict[str, object]:
    return {
        "expected_current_support_retention": retention,
        "immediate_eig": eig,
        "expected_support_size": expansion,
        "expected_truth_coverage": expected,
        "realized_endpoint": {
            "truth_covered": covered,
            "uniform_truth_probability": (
                0.1 if covered else 0.0
            ),
        },
    }


def _record(index: int) -> dict[str, object]:
    return {
        "truth_covered_before_counterfactuals": False,
        "candidate_dynamics": [
            _candidate(
                retention=0.9,
                eig=0.1,
                expansion=10.0,
                covered=1,
                expected=0.8,
            ),
            _candidate(
                retention=0.1,
                eig=0.9,
                expansion=5.0,
                covered=int(index % 10 == 0),
                expected=0.2,
            ),
            _candidate(
                retention=0.2,
                eig=0.2,
                expansion=20.0,
                covered=0,
                expected=0.3,
            ),
        ],
        "selector_indices": {"random": index % 3},
    }


def test_holdout_targets_require_frozen_size():
    targets = [f"animal-{index}" for index in range(60)]
    assert holdout.holdout_targets({"holdout_targets": targets}) == targets


def test_summary_passes_strict_positive_paired_gate(monkeypatch):
    monkeypatch.setattr(holdout, "HOLDOUT_SIZE", 60)
    monkeypatch.setattr(holdout, "MIN_SELECTOR_CHANGES", 20)
    records = [_record(index) for index in range(60)]
    usage = {
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "retry_count": 0,
        "adapter_cost_usd": 0.5,
    }
    summary = holdout.summarize_holdout(records, usage)
    assert summary["support_retention_vs_eig"]["mean_paired_gain"] == 0.9
    assert summary["support_retention_vs_eig"][
        "paired_bootstrap_95"
    ][0] > 0.0
    assert summary["within_state_realized_coverage_pairwise"][
        "support_retention"
    ]["accuracy"] == 0.95
    assert summary["gates"]["all_pass"] is True


def test_summary_fails_when_serving_truncates(monkeypatch):
    monkeypatch.setattr(holdout, "HOLDOUT_SIZE", 60)
    records = [_record(index) for index in range(60)]
    summary = holdout.summarize_holdout(
        records,
        {
            "adapter_reasoning_tokens": 0,
            "forced_exits": 1,
            "retry_count": 0,
            "adapter_cost_usd": 0.5,
        },
    )
    assert summary["gates"]["zero_forced_exits"] is False
    assert summary["gates"]["all_pass"] is False
