"""Audit the preregistered K2/K4/K6 RockSample[11,11] proposal-width test."""

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
from scripts.analyze_nonmyopic_rocksample_11_11_llm import analyze_run


RUN_KEYS = ("width_k2", "width_k4", "width_k6")
WIDTHS = (2, 4, 6)
BOOTSTRAP_SEED = 24086
BOOTSTRAP_REPLICATES = 10_000


def _auc(trace: dict[str, Any], field: str) -> float:
    return statistics.fmean(float(step[field]) for step in trace["steps"])


def _paired_ci(values: list[float], *, rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(
        0, len(array), size=(BOOTSTRAP_REPLICATES, len(array))
    )
    means = np.mean(array[indices], axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _strategy_values(payload: dict[str, Any]) -> tuple[list[float], list[float]]:
    traces = payload["traces"]["11-11"]["strategy_eig"]
    return (
        [_auc(trace, "entropy_after") for trace in traces],
        [_auc(trace, "truth_log_probability") for trace in traces],
    )


def _arm_summary(payload: dict[str, Any], arm: str) -> dict[str, float]:
    traces = payload["traces"]["11-11"][arm]
    steps = [step for trace in traces for step in trace["steps"]]
    return {
        "entropy_auc_mean": statistics.fmean(
            _auc(trace, "entropy_after") for trace in traces
        ),
        "truth_log_auc_mean": statistics.fmean(
            _auc(trace, "truth_log_probability") for trace in traces
        ),
        "scorer_units_per_decision": statistics.fmean(
            float(step["scorer_units"]) for step in steps
        ),
    }


def analyze(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(payloads) == len(RUN_KEYS)
    runs = {
        run_key: analyze_run(payload, run_key)
        for run_key, payload in zip(RUN_KEYS, payloads)
    }
    summaries = {
        run_key: {
            arm: _arm_summary(payload, arm)
            for arm in ARMS
        }
        for run_key, payload in zip(RUN_KEYS, payloads)
    }
    values = {
        run_key: _strategy_values(payload)
        for run_key, payload in zip(RUN_KEYS, payloads)
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cross_width: dict[str, Any] = {}
    for high_key, low_key in (
        ("width_k4", "width_k2"),
        ("width_k6", "width_k4"),
        ("width_k6", "width_k2"),
    ):
        high_entropy, high_truth = values[high_key]
        low_entropy, low_truth = values[low_key]
        entropy_gain = [
            low - high for high, low in zip(high_entropy, low_entropy)
        ]
        truth_gain = [
            high - low for high, low in zip(high_truth, low_truth)
        ]
        label = f"{high_key}_minus_{low_key}"
        cross_width[label] = {
            "entropy_auc_gain": statistics.fmean(entropy_gain),
            "entropy_auc_ci95": _paired_ci(entropy_gain, rng=rng),
            "truth_log_auc_gain": statistics.fmean(truth_gain),
            "truth_log_auc_ci95": _paired_ci(truth_gain, rng=rng),
        }

    primary = all(run["primary_gate_passed"] for run in runs.values())
    truth = all(run["truth_log_corroboration_passed"] for run in runs.values())
    return {
        "schema_version": 1,
        "claim": "strategy_eig_gain_is_robust_from_k2_to_k6",
        "all_width_primary_gates_passed": primary,
        "all_width_truth_log_gates_passed": truth,
        "all_18_intervals_passed": primary and truth,
        "runs": runs,
        "arm_summaries": summaries,
        "secondary_cross_width_comparisons": cross_width,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "method": "paired_trial_bootstrap",
        },
        "total_cost_usd": sum(run["usage"]["run_cost_usd"] for run in runs.values()),
        "total_requests": sum(run["usage"]["requests"] for run in runs.values()),
    }


def render_summary(audit: dict[str, Any]) -> str:
    lines = [
        "# RockSample[11,11] Proposal-Width Robustness",
        "",
        "The preregistered all-width gate "
        + ("passes." if audit["all_18_intervals_passed"] else "fails."),
        "",
        "| K | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for width, run in zip(WIDTHS, audit["runs"].values()):
        for baseline in BASELINES:
            comparison = run["comparisons"][baseline]
            e_ci = comparison["entropy_auc_ci95"]
            t_ci = comparison["truth_log_auc_ci95"]
            wtl = comparison["wins_ties_losses"]
            lines.append(
                f"| {width} | {ARM_LABELS[baseline]} | "
                f"{comparison['entropy_auc_gain']:+.4f} "
                f"[{e_ci[0]:+.4f}, {e_ci[1]:+.4f}] | "
                f"{comparison['truth_log_auc_gain']:+.4f} "
                f"[{t_ci[0]:+.4f}, {t_ci[1]:+.4f}] | "
                f"{wtl[0]}/{wtl[1]}/{wtl[2]} |"
            )
    lines.extend(
        [
            "",
            "## Secondary Width Comparisons",
            "",
            "Positive gains favor the larger K.",
            "",
            "| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] |",
            "| --- | --- | --- |",
        ]
    )
    for label, comparison in audit["secondary_cross_width_comparisons"].items():
        e_ci = comparison["entropy_auc_ci95"]
        t_ci = comparison["truth_log_auc_ci95"]
        lines.append(
            f"| {label.removeprefix('width_').replace('_minus_width_', ' - ')} | "
            f"{comparison['entropy_auc_gain']:+.4f} "
            f"[{e_ci[0]:+.4f}, {e_ci[1]:+.4f}] | "
            f"{comparison['truth_log_auc_gain']:+.4f} "
            f"[{t_ci[0]:+.4f}, {t_ci[1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            f"The three runs made {audit['total_requests']:,} requests and cost "
            f"${audit['total_cost_usd']:.8f}. Cross-width differences are secondary; "
            "the primary claim is positive gain at every K.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_width(audit: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    for arm in ARMS:
        axes[0].plot(
            WIDTHS,
            [
                audit["arm_summaries"][run_key][arm]["entropy_auc_mean"]
                for run_key in RUN_KEYS
            ],
            label=ARM_LABELS[arm],
            markersize=4.2,
            **ARM_STYLES[arm],
        )
    axes[0].set_xlabel("Proposed strategies (K)")
    axes[0].set_ylabel("Mean posterior entropy AUC (nats)")
    axes[0].set_xticks(WIDTHS)
    axes[0].grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axes[0].spines[["top", "right"]].set_visible(False)

    for arm in ("strategy_eig", "exhaustive_d2"):
        axes[1].plot(
            WIDTHS,
            [
                audit["arm_summaries"][run_key][arm]["scorer_units_per_decision"]
                for run_key in RUN_KEYS
            ],
            label=ARM_LABELS[arm],
            markersize=4.2,
            **ARM_STYLES[arm],
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Proposed strategies (K)")
    axes[1].set_ylabel("Exact scorer nodes / decision")
    axes[1].set_xticks(WIDTHS)
    axes[1].grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8.0)
    fig.suptitle("RockSample[11,11]: proposal quality and verifier width")
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
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
    plot_width(audit, args.plot_output)
    print(
        json.dumps(
            {
                "all_18_intervals_passed": audit["all_18_intervals_passed"],
                "total_cost_usd": audit["total_cost_usd"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
