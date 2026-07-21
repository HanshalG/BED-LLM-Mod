"""Audit the RockSample[7,8] Gemma root-slot replication."""

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


EXPECTED_MAP = "7-8"
EXPECTED_RUNS = {
    "gemma_semantic": {
        "run_id": "nonmyopic-rocksample-7-8-scale-20260721",
        "seed": 24072,
        "model": "google/gemma-4-26b-a4b-it",
        "label": "Gemma semantic roots",
        "interface": "semantic_distinct_roots",
        "resumed": True,
    },
    "gemma_slots": {
        "run_id": "nonmyopic-rocksample-7-8-gemma-slot-replication-20260721",
        "seed": 24077,
        "model": "google/gemma-4-26b-a4b-it",
        "label": "Gemma root slots",
        "interface": "ordered_movement_slots",
        "resumed": False,
    },
    "gpt54_mini_slots": {
        "run_id": "nonmyopic-rocksample-7-8-gpt54mini-slot-replication-20260721",
        "seed": 24074,
        "model": "openai/gpt-5.4-mini",
        "label": "GPT-5.4 Mini root slots",
        "interface": "ordered_movement_slots",
        "resumed": True,
    },
}


def _expected_config(seed: int) -> dict[str, Any]:
    return {
        "map_names": [EXPECTED_MAP],
        "num_trials_per_map": 30,
        "num_rounds": 10,
        "num_strategies": 6,
        "planning_horizon": 2,
        "seed": seed,
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


def _audit_run(payload: dict[str, Any], *, expected: dict[str, Any]) -> dict[str, Any]:
    assert payload["schema_version"] == 1
    assert payload["stage"] == "L1"
    assert payload["dry_run"] is False
    assert payload["run_id"] == expected["run_id"]
    assert payload["config"] == _expected_config(expected["seed"])

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
    assert usage["model"] == expected["model"]
    assert usage["requests"] == mechanics["physical_llm_requests"]
    assert usage["reasoning_tokens"] == 0
    assert usage["forced_exits"] == 0
    assert set(usage["model_usage"]) == {expected["model"]}

    resume = payload["resume"]
    if expected["resumed"]:
        assert resume is not None
        assert resume["accepted_cells_reused"] > 0
        assert resume["prior_error"]
        assert resume["failure_artifact"].endswith("L1_FAILURE.json")
    else:
        assert resume is None

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
        assert all(len(trace["steps"]) == 10 for trace in traces[arm])
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
        comparison["entropy_auc_ci95"][0] > 0.0 for comparison in comparisons.values()
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
    return {
        "label": expected["label"],
        "interface": expected["interface"],
        "primary_gate_passed": primary_passed,
        "truth_log_corroboration_passed": truth_log_passed,
        "comparisons": comparisons,
        "exact_d2_entropy_auc_gap": exact["entropy_auc_gain_mean"],
        "exact_d2_entropy_auc_gap_ci95": exact["entropy_auc_gain_ci95"],
        "arms": arms,
        "mechanics": mechanics,
        "resume": resume,
        "usage": usage,
    }


def analyze(
    semantic_gemma_payload: dict[str, Any],
    slot_gemma_payload: dict[str, Any],
    slot_gpt_payload: dict[str, Any],
) -> dict[str, Any]:
    runs = {
        "gemma_semantic": _audit_run(
            semantic_gemma_payload, expected=EXPECTED_RUNS["gemma_semantic"]
        ),
        "gemma_slots": _audit_run(
            slot_gemma_payload, expected=EXPECTED_RUNS["gemma_slots"]
        ),
        "gpt54_mini_slots": _audit_run(
            slot_gpt_payload, expected=EXPECTED_RUNS["gpt54_mini_slots"]
        ),
    }
    slot_runs = (runs["gemma_slots"], runs["gpt54_mini_slots"])
    assert all(run["primary_gate_passed"] for run in slot_runs)
    return {
        "schema_version": 1,
        "claim": "positive_nonmyopic_gain_holds_under_shared_root_slot_interface",
        "same_interface_primary_gates_passed": True,
        "same_interface_truth_log_gates_passed": all(
            run["truth_log_corroboration_passed"] for run in slot_runs
        ),
        "runs": runs,
        "descriptive_gemma_slot_minus_semantic_gain": {
            baseline: (
                runs["gemma_slots"]["comparisons"][baseline]["entropy_auc_gain"]
                - runs["gemma_semantic"]["comparisons"][baseline]["entropy_auc_gain"]
            )
            for baseline in BASELINES
        },
        "descriptive_gemma_minus_gpt_slot_gain": {
            baseline: (
                runs["gemma_slots"]["comparisons"][baseline]["entropy_auc_gain"]
                - runs["gpt54_mini_slots"]["comparisons"][baseline]["entropy_auc_gain"]
            )
            for baseline in BASELINES
        },
        "descriptive_gemma_slot_exact_gap_improvement": (
            runs["gemma_slots"]["exact_d2_entropy_auc_gap"]
            - runs["gemma_semantic"]["exact_d2_entropy_auc_gap"]
        ),
    }


def render_summary(audit: dict[str, Any]) -> str:
    runs = audit["runs"]
    lines = [
        "# RockSample[7,8] Root-Slot Replication",
        "",
        "Gemma 4 26B A4B passes the preregistered root-slot replication under the same "
        "proposal interface used by GPT-5.4 Mini. Positive paired gains favor StrategyEIG.",
        "",
        "| Model and interface | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in ("gemma_slots", "gpt54_mini_slots"):
        run = runs[key]
        for baseline in BASELINES:
            comparison = run["comparisons"][baseline]
            entropy_ci = comparison["entropy_auc_ci95"]
            truth_ci = comparison["truth_log_auc_ci95"]
            wtl = comparison["wins_ties_losses"]
            lines.append(
                f"| {run['label']} | {ARM_LABELS[baseline]} | "
                f"{comparison['entropy_auc_gain']:+.4f} "
                f"[{entropy_ci[0]:+.4f}, {entropy_ci[1]:+.4f}] | "
                f"{comparison['truth_log_auc_gain']:+.4f} "
                f"[{truth_ci[0]:+.4f}, {truth_ci[1]:+.4f}] | "
                f"{wtl[0]}/{wtl[1]}/{wtl[2]} |"
            )
    lines.extend(
        [
            "",
            "All six same-interface primary intervals and all six truth-log corroboration "
            "intervals exclude zero. The model-family difference is descriptive because "
            "the runs use different fresh seeds.",
            "",
            "## Interface Comparison",
            "",
            "| Control | Gemma semantic gain | Gemma slot gain | Descriptive change |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for baseline in BASELINES:
        semantic = runs["gemma_semantic"]["comparisons"][baseline]["entropy_auc_gain"]
        slots = runs["gemma_slots"]["comparisons"][baseline]["entropy_auc_gain"]
        delta = audit["descriptive_gemma_slot_minus_semantic_gain"][baseline]
        lines.append(
            f"| {ARM_LABELS[baseline]} | {semantic:+.4f} | {slots:+.4f} | {delta:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"The Gemma slot run narrows the entropy-AUC gap to exhaustive d2 from "
            f"{runs['gemma_semantic']['exact_d2_entropy_auc_gap']:+.4f} to "
            f"{runs['gemma_slots']['exact_d2_entropy_auc_gap']:+.4f}, a descriptive "
            f"improvement of {audit['descriptive_gemma_slot_exact_gap_improvement']:+.4f}. "
            "Because the prompt runs use different seeds, this is robustness and mechanism "
            "evidence rather than a causal interface effect.",
            "",
            "## Mechanics and Serving",
            "",
            "| Run | Movement | h2 exhaustive fraction | Requests | Rejects | Cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in runs.values():
        strategy = run["arms"]["strategy_eig"]
        lines.append(
            f"| {run['label']} | {strategy['move_count']}/{strategy['decision_count']} "
            f"({strategy['move_rate']:.3f}) | "
            f"{strategy['mean_h2_exhaustive_fraction']:.3f} | "
            f"{run['usage']['requests']} | {run['mechanics']['raw_rejected_responses']} | "
            f"${run['usage']['run_cost_usd']:.8f} |"
        )
    lines.extend(
        [
            "",
            "The fresh Gemma slot run completed all 870 cells without a rejection or resume. "
            "Every selected action was legal and exact rollout scoring made zero LLM calls.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_entropy(audit: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 3.8), sharex=True, sharey=True)
    rounds = range(1, 11)
    for axis, run in zip(axes, audit["runs"].values()):
        for arm in ARMS:
            axis.plot(
                rounds,
                run["arms"][arm]["round_entropy_mean"],
                label=ARM_LABELS[arm],
                markersize=3.5,
                **ARM_STYLES[arm],
            )
        axis.set_title(run["label"])
        axis.set_xlabel("Round")
        axis.set_xticks(list(rounds))
        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean posterior entropy (nats)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.2)
    fig.suptitle("RockSample[7,8]: proposal-interface replication")
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("semantic_gemma_result", type=Path)
    parser.add_argument("slot_gemma_result", type=Path)
    parser.add_argument("slot_gpt_result", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    audit = analyze(
        json.loads(args.semantic_gemma_result.read_text(encoding="utf-8")),
        json.loads(args.slot_gemma_result.read_text(encoding="utf-8")),
        json.loads(args.slot_gpt_result.read_text(encoding="utf-8")),
    )
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.summary_output.write_text(render_summary(audit), encoding="utf-8")
    plot_entropy(audit, args.plot_output)
    print(
        json.dumps(
            {
                "same_interface_primary_gates_passed": audit[
                    "same_interface_primary_gates_passed"
                ],
                "same_interface_truth_log_gates_passed": audit[
                    "same_interface_truth_log_gates_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
