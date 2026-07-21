"""Audit the two-model RockSample[11,11] root-slot confirmations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from scripts.analyze_nonmyopic_rocksample_11_11_llm import analyze_run


def analyze(
    gemma_payload: dict[str, Any], gpt_payload: dict[str, Any]
) -> dict[str, Any]:
    runs = {
        "gemma": analyze_run(gemma_payload, "gemma"),
        "gpt54_mini": analyze_run(gpt_payload, "gpt54_mini"),
    }
    assert all(run["primary_gate_passed"] for run in runs.values())
    assert all(run["truth_log_corroboration_passed"] for run in runs.values())
    return {
        "schema_version": 1,
        "claim": "positive_11_rock_nonmyopic_gain_replicates_across_model_families",
        "all_primary_gates_passed": True,
        "all_truth_log_gates_passed": True,
        "runs": runs,
        "descriptive_gpt_minus_gemma_entropy_auc_gain": {
            baseline: (
                runs["gpt54_mini"]["comparisons"][baseline]["entropy_auc_gain"]
                - runs["gemma"]["comparisons"][baseline]["entropy_auc_gain"]
            )
            for baseline in BASELINES
        },
        "descriptive_gpt_minus_gemma_exact_gap": (
            runs["gpt54_mini"]["exact_d2_entropy_auc_gap"]
            - runs["gemma"]["exact_d2_entropy_auc_gap"]
        ),
    }


def render_summary(audit: dict[str, Any]) -> str:
    lines = [
        "# RockSample[11,11] Cross-Model Root-Slot Replication",
        "",
        "Both preregistered model runs pass all entropy-AUC and truth-log-AUC gates "
        "under the identical ordered root-slot interface. Positive gains favor "
        "StrategyEIG.",
        "",
        "| Model | Control | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in audit["runs"].values():
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
            "All twelve cross-model intervals exclude zero. Effect-size differences "
            "are descriptive because the models use different fresh policy seeds.",
            "",
            "| Model | Movement | h2 exhaustive fraction | Exact-d2 AUC gap | Requests | Rejects | Cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in audit["runs"].values():
        strategy = run["arms"]["strategy_eig"]
        lines.append(
            f"| {run['label']} | {strategy['move_count']}/{strategy['decision_count']} "
            f"({strategy['move_rate']:.3f}) | "
            f"{strategy['mean_h2_exhaustive_fraction']:.3f} | "
            f"{run['exact_d2_entropy_auc_gap']:+.4f} | "
            f"{run['usage']['requests']} | "
            f"{run['mechanics']['raw_rejected_responses']} | "
            f"${run['usage']['run_cost_usd']:.8f} |"
        )
    lines.extend(
        [
            "",
            "GPT's smaller gains, lower exhaustive fraction, and larger exact-d2 gap "
            "show model-dependent proposal quality, but not a model-dependent sign. "
            "Both runs use one proposal call per StrategyEIG decision and zero "
            "rollout-scoring LLM calls.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_entropy(audit: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharex=True, sharey=True)
    rounds = range(1, 13)
    for axis, run in zip(axes, audit["runs"].values()):
        for arm in ARMS:
            axis.plot(
                rounds,
                run["arms"][arm]["round_entropy_mean"],
                label=ARM_LABELS[arm],
                markersize=3.2,
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
    fig.suptitle("RockSample[11,11]: cross-model root-slot replication")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gemma_result", type=Path)
    parser.add_argument("gpt_result", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    audit = analyze(
        json.loads(args.gemma_result.read_text(encoding="utf-8")),
        json.loads(args.gpt_result.read_text(encoding="utf-8")),
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
                "all_primary_gates_passed": audit["all_primary_gates_passed"],
                "all_truth_log_gates_passed": audit["all_truth_log_gates_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
