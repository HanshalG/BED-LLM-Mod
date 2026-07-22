"""Audit the three direct-vLLM RockSample[15,15] Gemma seeds."""

from __future__ import annotations

import argparse
import json
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
from scripts.analyze_nonmyopic_rocksample_15_15_llm import analyze_run
from scripts.nonmyopic_rock_strategy_prior import _stable_seed


RUN_KEYS = ("vllm", "vllm_seed_24102", "vllm_seed_24103")
FRESH_RUN_KEYS = RUN_KEYS[1:]
BOOTSTRAP_SEED = 24104
BOOTSTRAP_REPLICATES = 10_000


def _trace_auc(trace: dict[str, Any], field: str) -> float:
    return statistics.fmean(float(step[field]) for step in trace["steps"])


def _paired_values(
    payload: dict[str, Any], baseline: str
) -> tuple[list[float], list[float]]:
    traces = payload["traces"]["15-15"]
    entropy = [
        _trace_auc(baseline_trace, "entropy_after")
        - _trace_auc(strategy_trace, "entropy_after")
        for strategy_trace, baseline_trace in zip(
            traces["strategy_eig"], traces[baseline]
        )
    ]
    truth = [
        _trace_auc(strategy_trace, "truth_log_probability")
        - _trace_auc(baseline_trace, "truth_log_probability")
        for strategy_trace, baseline_trace in zip(
            traces["strategy_eig"], traces[baseline]
        )
    ]
    return entropy, truth


def _stratified_bootstrap(
    values_by_seed: list[list[float]], *, seed: int
) -> tuple[float, list[float]]:
    arrays = [np.asarray(values, dtype=np.float64) for values in values_by_seed]
    assert arrays and all(array.shape == (30,) for array in arrays)
    rng = np.random.default_rng(seed)
    samples = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for array in arrays:
        indices = rng.integers(
            0, len(array), size=(BOOTSTRAP_REPLICATES, len(array))
        )
        samples += np.mean(array[indices], axis=1) / len(arrays)
    estimate = statistics.fmean(statistics.fmean(values) for values in values_by_seed)
    return estimate, [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def analyze(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(payloads) == len(RUN_KEYS)
    runs = {
        run_key: analyze_run(payload, run_key)
        for run_key, payload in zip(RUN_KEYS, payloads)
    }
    pooled: dict[str, Any] = {}
    for baseline in BASELINES:
        entropy_by_seed: list[list[float]] = []
        truth_by_seed: list[list[float]] = []
        for payload in payloads:
            entropy, truth = _paired_values(payload, baseline)
            entropy_by_seed.append(entropy)
            truth_by_seed.append(truth)
        entropy_mean, entropy_ci = _stratified_bootstrap(
            entropy_by_seed,
            seed=_stable_seed(BOOTSTRAP_SEED, baseline, "entropy-auc"),
        )
        truth_mean, truth_ci = _stratified_bootstrap(
            truth_by_seed,
            seed=_stable_seed(BOOTSTRAP_SEED, baseline, "truth-log-auc"),
        )
        entropy_values = [value for values in entropy_by_seed for value in values]
        wins = sum(value > 1e-12 for value in entropy_values)
        losses = sum(value < -1e-12 for value in entropy_values)
        pooled[baseline] = {
            "entropy_auc_gain": entropy_mean,
            "entropy_auc_ci95": entropy_ci,
            "truth_log_auc_gain": truth_mean,
            "truth_log_auc_ci95": truth_ci,
            "entropy_auc_wins_ties_losses": [
                wins,
                len(entropy_values) - wins - losses,
                losses,
            ],
        }

    fresh_runs = [runs[run_key] for run_key in FRESH_RUN_KEYS]
    fresh_primary = all(run["primary_gate_passed"] for run in fresh_runs)
    fresh_truth = all(run["truth_log_corroboration_passed"] for run in fresh_runs)
    return {
        "schema_version": 1,
        "claim": "direct_vllm_15_rock_gain_replicates_across_two_fresh_seeds",
        "all_fresh_seed_primary_gates_passed": fresh_primary,
        "all_fresh_seed_truth_log_gates_passed": fresh_truth,
        "all_12_fresh_seed_intervals_passed": fresh_primary and fresh_truth,
        "all_three_direct_vllm_seeds_passed": all(
            run["primary_gate_passed"] and run["truth_log_corroboration_passed"]
            for run in runs.values()
        ),
        "runs": runs,
        "pooled_90_pair_comparisons": pooled,
        "bootstrap": {
            "method": "equal_seed_weight_stratified_paired_trial_bootstrap",
            "base_seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
        },
        "total_cost_usd": sum(run["usage"]["run_cost_usd"] for run in runs.values()),
        "total_requests": sum(run["usage"]["requests"] for run in runs.values()),
    }


def render_summary(audit: dict[str, Any]) -> str:
    lines = [
        "# RockSample[15,15] Direct-vLLM Multi-Seed Robustness",
        "",
        (
            "The preregistered two-fresh-seed robustness gate "
            + ("passes." if audit["all_12_fresh_seed_intervals_passed"] else "fails.")
        ),
        "",
        "| Seed | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for run in audit["runs"].values():
        seed = run["label"].split()[-1]
        for baseline in BASELINES:
            comparison = run["comparisons"][baseline]
            e_ci = comparison["entropy_auc_ci95"]
            t_ci = comparison["truth_log_auc_ci95"]
            wtl = comparison["wins_ties_losses"]
            lines.append(
                f"| {seed} | {ARM_LABELS[baseline]} | "
                f"{comparison['entropy_auc_gain']:+.4f} "
                f"[{e_ci[0]:+.4f}, {e_ci[1]:+.4f}] | "
                f"{comparison['truth_log_auc_gain']:+.4f} "
                f"[{t_ci[0]:+.4f}, {t_ci[1]:+.4f}] | "
                f"{wtl[0]}/{wtl[1]}/{wtl[2]} |"
            )
    lines.extend(
        [
            "",
            "## Pooled Three-Seed Estimates",
            "",
            "| Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |",
            "| --- | --- | --- | --- |",
        ]
    )
    for baseline in BASELINES:
        comparison = audit["pooled_90_pair_comparisons"][baseline]
        e_ci = comparison["entropy_auc_ci95"]
        t_ci = comparison["truth_log_auc_ci95"]
        wtl = comparison["entropy_auc_wins_ties_losses"]
        lines.append(
            f"| {ARM_LABELS[baseline]} | {comparison['entropy_auc_gain']:+.4f} "
            f"[{e_ci[0]:+.4f}, {e_ci[1]:+.4f}] | "
            f"{comparison['truth_log_auc_gain']:+.4f} "
            f"[{t_ci[0]:+.4f}, {t_ci[1]:+.4f}] | "
            f"{wtl[0]}/{wtl[1]}/{wtl[2]} |"
        )
    lines.extend(
        [
            "",
            f"The three runs made {audit['total_requests']:,} requests at "
            f"${audit['total_cost_usd']:.8f} API cost. Pooled intervals are "
            "secondary; the robustness claim requires both fresh seeds to pass "
            "independently.",
        ]
    )
    return "\n".join(lines)


def plot_entropy(audit: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharex=True, sharey=True)
    rounds = range(1, 16)
    for axis, run in zip(axes, audit["runs"].values()):
        for arm in ARMS:
            axis.plot(
                rounds,
                run["arms"][arm]["round_entropy_mean"],
                label=ARM_LABELS[arm],
                markersize=3.0,
                **ARM_STYLES[arm],
            )
        axis.set_title(f"Seed {run['label'].split()[-1]}")
        axis.set_xlabel("Round")
        axis.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean posterior entropy (nats)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.0)
    fig.suptitle("RockSample[15,15]: direct-vLLM seed robustness")
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs=3, type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    audit = analyze(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.summary_output.write_text(render_summary(audit), encoding="utf-8")
    plot_entropy(audit, args.plot_output)
    print(
        json.dumps(
            {
                "all_12_fresh_seed_intervals_passed": audit[
                    "all_12_fresh_seed_intervals_passed"
                ],
                "all_three_direct_vllm_seeds_passed": audit[
                    "all_three_direct_vllm_seeds_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
