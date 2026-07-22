"""Audit the preregistered RockSample[15,15] Gemma StrategyEIG confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_nonmyopic_rock_branch_result import (
    ARMS,
    ARM_LABELS,
    ARM_STYLES,
    BASELINES,
)
from scripts.nonmyopic_rock_strategy_prior import _bootstrap_mean_ci, _stable_seed


EXPECTED_MAP = "15-15"
EXPECTED_RUNS = {
    "gemma": {
        "run_id": "nonmyopic-rocksample-15-15-gemma-slot-confirmation-20260722",
        "seed": 24100,
        "model": "google/gemma-4-26b-a4b-it",
        "label": "Gemma 4 26B A4B",
        "backend": "openrouter",
        "trial_concurrency": 4,
        "resumed": False,
    },
    "vllm": {
        "run_id": "nonmyopic-rocksample-15-15-vllm-replication-20260722",
        "seed": 24101,
        "model": "google/gemma-4-26B-A4B-it",
        "label": "Gemma 4 26B A4B direct vLLM seed 24101",
        "backend": "vllm",
        "trial_concurrency": 1,
        "resumed": False,
    },
    "vllm_seed_24102": {
        "run_id": "nonmyopic-rocksample-15-15-vllm-seed-24102-20260722",
        "seed": 24102,
        "model": "google/gemma-4-26B-A4B-it",
        "label": "Gemma 4 26B A4B direct vLLM seed 24102",
        "backend": "vllm",
        "trial_concurrency": 1,
        "resumed": False,
    },
    "vllm_seed_24103": {
        "run_id": "nonmyopic-rocksample-15-15-vllm-seed-24103-20260722",
        "seed": 24103,
        "model": "google/gemma-4-26B-A4B-it",
        "label": "Gemma 4 26B A4B direct vLLM seed 24103",
        "backend": "vllm",
        "trial_concurrency": 1,
        "resumed": False,
    },
    "e4b_vllm": {
        "run_id": "nonmyopic-rocksample-15-15-e4b-vllm-20260722",
        "seed": 24105,
        "model": "google/gemma-4-E4B-it",
        "label": "Gemma 4 E4B direct vLLM seed 24105",
        "backend": "vllm",
        "trial_concurrency": 1,
        "resumed": False,
    },
    "12b_vllm": {
        "run_id": "nonmyopic-rocksample-15-15-12b-vllm-20260722",
        "seed": 24106,
        "model": "google/gemma-4-12B-it",
        "label": "Gemma 4 12B direct vLLM seed 24106",
        "backend": "vllm",
        "trial_concurrency": 1,
        "resumed": False,
    },
}


def _expected_config(
    seed: int, num_strategies: int = 4, trial_concurrency: int = 4
) -> dict[str, Any]:
    return {
        "map_names": [EXPECTED_MAP],
        "num_trials_per_map": 30,
        "num_rounds": 15,
        "num_strategies": num_strategies,
        "planning_horizon": 2,
        "seed": seed,
        "bootstrap_replicates": 10_000,
        "temperature": 0.0,
        "validation_retries": 1,
        "trial_concurrency": trial_concurrency,
        "strategy_schema": "branch_policy_v2",
        "primary_endpoint": "entropy_auc",
    }


EXPECTED_RUN_ID = EXPECTED_RUNS["gemma"]["run_id"]
EXPECTED_MODEL = EXPECTED_RUNS["gemma"]["model"]
EXPECTED_CONFIG = _expected_config(EXPECTED_RUNS["gemma"]["seed"])


def _mean_step_field(trace: dict[str, Any], field: str) -> float:
    return statistics.fmean(float(step[field]) for step in trace["steps"])


def _assert_float_lists_match(actual: list[float], expected: list[float]) -> None:
    assert len(actual) == len(expected)
    assert all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
        for left, right in zip(actual, expected)
    )


def _assert_bootstrap_ci(
    values: list[float],
    stored: list[float],
    *,
    seed: int,
    label: str,
    metric: str,
) -> None:
    expected = _bootstrap_mean_ci(
        np.asarray(values, dtype=float),
        seed=_stable_seed(seed, "l1-bootstrap", label, metric),
        replicates=10_000,
    )
    _assert_float_lists_match(list(expected), stored)


def _analyze_expected(
    payload: dict[str, Any], *, expected: dict[str, Any]
) -> dict[str, Any]:
    assert payload["schema_version"] == 1
    assert payload["stage"] == "L1"
    assert payload["dry_run"] is False
    assert payload["run_id"] == expected["run_id"]
    num_strategies = expected.get("num_strategies", 4)
    assert payload["config"] == _expected_config(
        expected["seed"], num_strategies, expected["trial_concurrency"]
    )
    if expected["resumed"]:
        resume = payload["resume"]
        assert resume is not None
        assert resume["accepted_cells_reused"] == expected["accepted_cells_reused"]
        assert resume["prior_error"]
        assert resume["failure_artifact"].endswith("L1_FAILURE.json")
    else:
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
    assert usage["backend"] == expected["backend"]
    assert usage["model"] == expected["model"]
    assert usage["requests"] == mechanics["physical_llm_requests"]
    assert usage["reasoning_tokens"] == 0
    assert usage["forced_exits"] == 0
    assert set(usage["model_usage"]) == {expected["model"]}

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
        assert all(len(trace["steps"]) == 15 for trace in traces[arm])
        assert all(trace["map_name"] == EXPECTED_MAP for trace in traces[arm])
        assert all(trace["arm"] == arm for trace in traces[arm])
    if "num_strategies" in expected:
        for arm in ("strategy_eig", "shared_d1", "random_strategy"):
            assert all(
                len(step["candidate_strategies"]) == num_strategies
                for trace in traces[arm]
                for step in trace["steps"]
            )

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
        comparison_label = f"{EXPECTED_MAP}-{baseline}"
        _assert_bootstrap_ci(
            entropy_values,
            stored["entropy_auc_gain_ci95"],
            seed=expected["seed"],
            label=comparison_label,
            metric="entropy-auc",
        )
        _assert_bootstrap_ci(
            truth_values,
            stored["truth_log_probability_auc_gain_ci95"],
            seed=expected["seed"],
            label=comparison_label,
            metric="truth-log-auc",
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
    exact_values = [
        _mean_step_field(baseline_trace, "entropy_after")
        - _mean_step_field(strategy_trace, "entropy_after")
        for strategy_trace, baseline_trace in zip(
            traces["strategy_eig"], traces["exhaustive_d2"]
        )
    ]
    _assert_float_lists_match(exact_values, exact["entropy_auc_paired_values"])
    _assert_bootstrap_ci(
        exact_values,
        exact["entropy_auc_gain_ci95"],
        seed=expected["seed"],
        label=f"{EXPECTED_MAP}-exhaustive_d2",
        metric="entropy-auc",
    )
    assert math.isclose(
        statistics.fmean(exact_values),
        exact["entropy_auc_gain_mean"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    return {
        "schema_version": 1,
        "claim": "positive_nonmyopic_gain_scales_to_standard_15_rock_geometry",
        "label": expected["label"],
        "model": expected["model"],
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


def analyze_run(payload: dict[str, Any], run_key: str) -> dict[str, Any]:
    return _analyze_expected(payload, expected=EXPECTED_RUNS[run_key])


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    return analyze_run(payload, "gemma")


def render_summary(audit: dict[str, Any]) -> str:
    if audit["primary_gate_passed"] and audit["truth_log_corroboration_passed"]:
        outcome = (
            "passes its primary and truth-log corroboration gates. Positive paired "
            "gains favor StrategyEIG."
        )
    elif audit["primary_gate_passed"]:
        outcome = (
            "passes its primary entropy-AUC gate but fails its truth-log "
            "corroboration gate."
        )
    else:
        outcome = "fails its preregistered primary entropy-AUC gate."
    lines = [
        f"# RockSample[15,15] {audit['label']}",
        "",
        f"The registered fifteen-rock run {outcome}",
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
    if exact_ci[0] > 0.0:
        exact_comparison = (
            "Its entropy-AUC advantage over terminal-objective exhaustive d2 is "
        )
    elif exact_ci[1] < 0.0:
        exact_comparison = "Its remaining entropy-AUC gap to exhaustive d2 is "
    else:
        exact_comparison = "Its entropy-AUC difference from exhaustive d2 is "
    lines.extend(
        [
            "",
            f"Primary gate passed: {audit['primary_gate_passed']}. Truth-log "
            f"corroboration passed: {audit['truth_log_corroboration_passed']}. "
            f"StrategyEIG moves on {strategy['move_count']}/{strategy['decision_count']} "
            f"decisions and captures {strategy['mean_h2_exhaustive_fraction']:.1%} of "
            "exhaustive d2 value over nonterminal horizon-two rounds.",
            "",
            f"{exact_comparison}{exact_gap:+.4f} "
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
    rounds = range(1, 16)
    for arm in ARMS:
        axis.plot(
            rounds,
            audit["arms"][arm]["round_entropy_mean"],
            label=ARM_LABELS[arm],
            markersize=3.5,
            **ARM_STYLES[arm],
        )
    axis.set_title(f"RockSample[15,15]: {audit['label']}")
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
    parser.add_argument("--run-key", choices=tuple(EXPECTED_RUNS), default="gemma")
    args = parser.parse_args()
    audit = analyze_run(
        json.loads(args.result.read_text(encoding="utf-8")), args.run_key
    )
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
