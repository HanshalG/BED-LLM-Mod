"""Audit the preregistered RockSample[11,11] Gemma StrategyEIG confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_nonmyopic_rock_branch_result import (
    ARMS,
    ARM_LABELS,
    ARM_STYLES,
    BASELINES,
)


EXPECTED_MAP = "11-11"
EXPECTED_RUN_ID = "nonmyopic-rocksample-11-11-gemma-slot-confirmation-20260721"
EXPECTED_MODEL = "google/gemma-4-26b-a4b-it"
EXPECTED_CONFIG = {
    "map_names": [EXPECTED_MAP],
    "num_trials_per_map": 30,
    "num_rounds": 12,
    "num_strategies": 6,
    "planning_horizon": 2,
    "seed": 24079,
    "bootstrap_replicates": 10_000,
    "temperature": 0.0,
    "validation_retries": 1,
    "trial_concurrency": 32,
    "strategy_schema": "branch_policy_v2",
    "primary_endpoint": "entropy_auc",
}


def _mean_step_field(trace: dict[str, Any], field: str) -> float:
    return statistics.fmean(float(step[field]) for step in trace["steps"])


def _assert_float_lists_match(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    assert all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(actual, expected)
    )


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    assert payload["schema_version"] == 1
    assert payload["stage"] == "L1"
    assert payload["dry_run"] is False
    assert payload["run_id"] == EXPECTED_RUN_ID
    assert payload["config"] == EXPECTED_CONFIG
    assert payload["resume"] is None

    mechanics = payload["mechanics"]
    required_mechanics = (
        "all_selected_actions_legal",
        "initial_strategy_cells_shared_with_d1",
        "width_logical_llm_calls_match_strategy_eig",
        "width_exact_scorer_units_match_strategy_eig",
        "random_strategy_cells_have_k_candidates",
    )
    assert mechanics["terminal_cell_failures"] == 0
    assert mechanics["rollout_scoring_llm_calls"] == 0
    assert all(mechanics[key] for key in required_mechanics)
    assert mechanics["accepted_llm_cells"] == len(payload["candidate_requests"])
    assert mechanics["raw_rejected_responses"] == len(payload["invalid_responses"])
    assert mechanics["physical_llm_requests"] == (
        len(payload["candidate_requests"]) + len(payload["invalid_responses"])
    )

    usage = payload["usage"]
    assert usage["backend"] == "openrouter"
    assert usage["model"] == EXPECTED_MODEL
    assert usage["requests"] == mechanics["physical_llm_requests"]
    assert usage["reasoning_tokens"] == 0
    assert usage["forced_exits"] == 0
    assert set(usage["model_usage"]) == {EXPECTED_MODEL}

    traces = payload["traces"][EXPECTED_MAP]
    reference_pairs = [
        (trace["trial_index"], trace["truth_index"])
        for trace in traces["strategy_eig"]
    ]
    assert [trial for trial, _truth in reference_pairs] == list(range(30))
    for arm in ARMS:
        assert len(traces[arm]) == 30
        assert [
            (trace["trial_index"], trace["truth_index"]) for trace in traces[arm]
        ] == reference_pairs
        assert all(len(trace["steps"]) == 12 for trace in traces[arm])
        assert all(trace["map_name"] == EXPECTED_MAP for trace in traces[arm])
        assert all(trace["arm"] == arm for trace in traces[arm])

    paired = payload["maps"][EXPECTED_MAP]["paired"]
    comparisons: dict[str, Any] = {}
    for baseline in BASELINES:
        stored = paired[f"strategy_eig_minus_{baseline}"]
        entropy_values = [
            _mean_step_field(baseline_trace, "entropy_after")
            - _mean_step_field(strategy_trace, "entropy_after")
            for strategy_trace, baseline_trace in zip(
                traces["strategy_eig"], traces[baseline]
            )
        ]
        truth_values = [
            _mean_step_field(strategy_trace, "truth_log_probability")
            - _mean_step_field(baseline_trace, "truth_log_probability")
            for strategy_trace, baseline_trace in zip(
                traces["strategy_eig"], traces[baseline]
            )
        ]
        _assert_float_lists_match(entropy_values, stored["entropy_auc_paired_values"])
        _assert_float_lists_match(
            truth_values, stored["truth_log_probability_auc_paired_values"]
        )
        assert math.isclose(
            statistics.fmean(entropy_values),
            stored["entropy_auc_gain_mean"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert math.isclose(
            statistics.fmean(truth_values),
            stored["truth_log_probability_auc_gain_mean"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        wins = sum(value > 1e-12 for value in entropy_values)
        losses = sum(value < -1e-12 for value in entropy_values)
        ties = len(entropy_values) - wins - losses
        assert [wins, ties, losses] == stored["entropy_auc_wins_ties_losses"]
        comparisons[baseline] = {
            "entropy_auc_gain": stored["entropy_auc_gain_mean"],
            "entropy_auc_ci95": stored["entropy_auc_gain_ci95"],
            "truth_log_auc_gain": stored["truth_log_probability_auc_gain_mean"],
            "truth_log_auc_ci95": stored["truth_log_probability_auc_gain_ci95"],
            "wins_ties_losses": stored["entropy_auc_wins_ties_losses"],
        }

    primary_passed = all(
        comparison["entropy_auc_ci95"][0] > 0.0
        for comparison in comparisons.values()
    )
    truth_log_passed = all(
        comparison["truth_log_auc_ci95"][0] > 0.0
        for comparison in comparisons.values()
    )
    assert primary_passed == payload["gate_passed"]
    assert primary_passed == payload["maps"][EXPECTED_MAP]["gate_passed"]

    arms: dict[str, Any] = {}
    for arm in ARMS:
        steps = [step for trace in traces[arm] for step in trace["steps"]]
        h2_steps = [step for trace in traces[arm] for step in trace["steps"][:-1]]
        move_count = sum(step["action"].startswith("move-") for step in steps)
        arms[arm] = {
            "move_count": move_count,
            "decision_count": len(steps),
            "move_rate": move_count / len(steps),
            "mean_h2_exhaustive_fraction": statistics.fmean(
                float(step["exhaustive_fraction"]) for step in h2_steps
            ),
            "entropy_auc_mean": statistics.fmean(
                _mean_step_field(trace, "entropy_after") for trace in traces[arm]
            ),
            "round_entropy_mean": payload["maps"][EXPECTED_MAP]["summary"][arm][
                "round_entropy_mean"
            ],
        }

    exact = paired["strategy_eig_minus_exhaustive_d2"]
    assert primary_passed
    assert truth_log_passed
    return {
        "schema_version": 1,
        "claim": "positive_nonmyopic_gain_scales_to_standard_11_rock_geometry",
        "primary_gate_passed": primary_passed,
        "truth_log_corroboration_passed": truth_log_passed,
        "comparisons": comparisons,
        "exact_d2_entropy_auc_gap": exact["entropy_auc_gain_mean"],
        "exact_d2_entropy_auc_gap_ci95": exact["entropy_auc_gain_ci95"],
        "arms": arms,
        "mechanics": mechanics,
        "resume": payload["resume"],
        "usage": usage,
    }


def render_summary(audit: dict[str, Any]) -> str:
    lines = [
        "# RockSample[11,11] Gemma Root-Slot Confirmation",
        "",
        "The preregistered eleven-rock scale confirmation passes its primary and "
        "truth-log corroboration gates. Positive paired gains favor StrategyEIG.",
        "",
        "| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |",
        "| --- | --- | --- | --- |",
    ]
    for baseline in BASELINES:
        comparison = audit["comparisons"][baseline]
        entropy_ci = comparison["entropy_auc_ci95"]
        truth_ci = comparison["truth_log_auc_ci95"]
        wtl = comparison["wins_ties_losses"]
        lines.append(
            f"| {ARM_LABELS[baseline]} | "
            f"{comparison['entropy_auc_gain']:+.4f} "
            f"[{entropy_ci[0]:+.4f}, {entropy_ci[1]:+.4f}] | "
            f"{comparison['truth_log_auc_gain']:+.4f} "
            f"[{truth_ci[0]:+.4f}, {truth_ci[1]:+.4f}] | "
            f"{wtl[0]}/{wtl[1]}/{wtl[2]} |"
        )
    strategy = audit["arms"]["strategy_eig"]
    exact_gap = audit["exact_d2_entropy_auc_gap"]
    exact_ci = audit["exact_d2_entropy_auc_gap_ci95"]
    lines.extend(
        [
            "",
            "All three primary and all three truth-log intervals exclude zero. "
            f"StrategyEIG moves on {strategy['move_count']}/{strategy['decision_count']} "
            f"decisions and captures {strategy['mean_h2_exhaustive_fraction']:.1%} of "
            "exhaustive d2 value over nonterminal horizon-two rounds.",
            "",
            f"The remaining entropy-AUC gap to exhaustive d2 is {exact_gap:+.4f} "
            f"[{exact_ci[0]:+.4f}, {exact_ci[1]:+.4f}]. The run made "
            f"{audit['usage']['requests']} physical requests, retained "
            f"{audit['mechanics']['raw_rejected_responses']} rejected response, and cost "
            f"${audit['usage']['run_cost_usd']:.8f}. It had zero terminal failures, "
            "reasoning tokens, forced exits, resumes, or rollout-scoring LLM calls.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_entropy(audit: dict[str, Any], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    rounds = range(1, 13)
    for arm in ARMS:
        axis.plot(
            rounds,
            audit["arms"][arm]["round_entropy_mean"],
            label=ARM_LABELS[arm],
            markersize=3.5,
            **ARM_STYLES[arm],
        )
    axis.set_title("RockSample[11,11]: Gemma root-slot confirmation")
    axis.set_xlabel("Round")
    axis.set_ylabel("Mean posterior entropy (nats)")
    axis.set_xticks(list(rounds))
    axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="best", frameon=False, fontsize=8.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    audit = analyze(json.loads(args.result.read_text(encoding="utf-8")))
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.summary_output.write_text(render_summary(audit), encoding="utf-8")
    plot_entropy(audit, args.plot_output)
    print(
        json.dumps(
            {
                "primary_gate_passed": audit["primary_gate_passed"],
                "truth_log_corroboration_passed": audit[
                    "truth_log_corroboration_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
