"""Audit the AUC-aligned RockSample[7,8] exact depth qualification."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis.core import EPSILON


EXPECTED_CONFIG = {
    "map_name": "7-8",
    "num_trials": 500,
    "num_rounds": 10,
    "max_depth": 3,
    "seed": 24076,
    "bootstrap_replicates": 10_000,
    "trial_concurrency": 16,
    "half_efficiency_distance": math.log(2.0),
}


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _round_entropy(traces: list[dict[str, Any]]) -> list[float]:
    return [
        _mean([trace["steps"][round_index]["entropy"] for trace in traces])
        for round_index in range(10)
    ]


def analyze(
    payload: dict[str, Any],
    terminal_payload: dict[str, Any],
    reference_payload: dict[str, Any],
) -> dict[str, Any]:
    assert payload["schema_version"] == 1
    assert payload["stage"] == "auc_aligned_depth3_exact_qualification"
    assert payload["planning_utility"] == "entropy_auc"
    assert payload["config"] == EXPECTED_CONFIG
    assert payload["source"]["map"] == "7-8"
    assert all(payload["mechanics"].values())
    assert payload["primary_gate_passed"] is True
    assert payload["truth_log_corroboration_passed"] is True

    traces = payload["traces"]
    reference_pairs = [
        (trace["trial_index"], trace["truth_index"]) for trace in traces["3"]
    ]
    assert [trial for trial, _truth in reference_pairs] == list(range(500))
    for depth in (1, 2, 3):
        assert len(traces[str(depth)]) == 500
        assert [
            (trace["trial_index"], trace["truth_index"]) for trace in traces[str(depth)]
        ] == reference_pairs
        assert all(len(trace["steps"]) == 10 for trace in traces[str(depth)])

    comparisons: dict[str, Any] = {}
    for label, deeper_depth, shallower_depth in (
        ("d2_minus_d1", "2", "1"),
        ("d3_minus_d2", "3", "2"),
        ("d3_minus_d1", "3", "1"),
    ):
        deeper = traces[deeper_depth]
        shallower = traces[shallower_depth]
        entropy_gains = np.asarray(
            [shallower[index]["entropy_auc"] - deeper[index]["entropy_auc"] for index in range(500)]
        )
        truth_gains = np.asarray(
            [
                deeper[index]["truth_log_probability_auc"]
                - shallower[index]["truth_log_probability_auc"]
                for index in range(500)
            ]
        )
        stored = payload["comparisons"][label]
        assert math.isclose(float(np.mean(entropy_gains)), stored["entropy_auc_gain_mean"])
        assert math.isclose(
            float(np.mean(truth_gains)), stored["truth_log_probability_auc_gain_mean"]
        )
        wtl = [
            int(np.count_nonzero(entropy_gains > EPSILON)),
            int(np.count_nonzero(np.abs(entropy_gains) <= EPSILON)),
            int(np.count_nonzero(entropy_gains < -EPSILON)),
        ]
        assert wtl == stored["entropy_auc_wins_ties_losses"]
        comparisons[label] = {
            "entropy_auc_gain_mean": float(np.mean(entropy_gains)),
            "entropy_auc_gain_ci95": stored["entropy_auc_gain_ci95"],
            "truth_log_auc_gain_mean": float(np.mean(truth_gains)),
            "truth_log_auc_gain_ci95": stored["truth_log_probability_auc_gain_ci95"],
            "wins_ties_losses": wtl,
        }

    aligned_summaries = {
        depth: {
            "entropy_auc_mean": _mean([trace["entropy_auc"] for trace in depth_traces]),
            "final_entropy_mean": _mean([trace["final_entropy"] for trace in depth_traces]),
            "truth_log_auc_mean": _mean(
                [trace["truth_log_probability_auc"] for trace in depth_traces]
            ),
            "round_entropy_mean": _round_entropy(depth_traces),
            "unique_action_sequences": len(
                Counter(
                    tuple(step["action"] for step in trace["steps"])
                    for trace in depth_traces
                )
            ),
        }
        for depth, depth_traces in traces.items()
    }
    terminal_summaries = {
        depth: {
            "entropy_auc_mean": _mean([trace["entropy_auc"] for trace in depth_traces]),
            "final_entropy_mean": _mean([trace["final_entropy"] for trace in depth_traces]),
            "round_entropy_mean": _round_entropy(depth_traces),
        }
        for depth, depth_traces in terminal_payload["traces"].items()
    }

    aligned_d3_sequence = Counter(
        tuple(step["action"] for step in trace["steps"]) for trace in traces["3"]
    )
    terminal_d2_sequence = Counter(
        tuple(step["action"] for step in trace["steps"])
        for trace in terminal_payload["traces"]["2"]
    )
    assert len(aligned_d3_sequence) == 1
    assert len(terminal_d2_sequence) == 1
    assert next(iter(aligned_d3_sequence)) == next(iter(terminal_d2_sequence))
    assert np.allclose(
        aligned_summaries["3"]["round_entropy_mean"],
        terminal_summaries["2"]["round_entropy_mean"],
        atol=1e-12,
        rtol=0.0,
    )
    external_d2 = reference_payload["maps"]["7-8"]["summary"]["exhaustive_d2"]
    assert math.isclose(
        aligned_summaries["3"]["entropy_auc_mean"], external_d2["entropy_auc_mean"]
    )
    assert math.isclose(
        aligned_summaries["3"]["final_entropy_mean"], external_d2["final_entropy_mean"]
    )

    return {
        "schema_version": 1,
        "audit_passed": True,
        "same_utility_depth_monotonicity_passed": True,
        "truth_log_corroboration_passed": True,
        "strict_gain_over_prior_best_d2": False,
        "comparisons": comparisons,
        "aligned_summaries": aligned_summaries,
        "terminal_summaries": terminal_summaries,
        "mechanism": {
            "aligned_d3_matches_terminal_d2_sequence": True,
            "aligned_d3_matches_external_d2_curve": True,
            "aligned_d3_action_sequence": list(next(iter(aligned_d3_sequence))),
            "interpretation": "AUC weighting removes d3 procrastination but recovers, rather than surpasses, the prior best exact d2 policy",
        },
    }


def render_summary(audit: dict[str, Any]) -> str:
    lines = [
        "# RockSample[7,8] AUC-Aligned Depth Result",
        "",
        "AUC-aligned exact planning restores monotonic depth under the same utility. Positive paired gains favor the deeper arm.",
        "",
        "| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |",
        "| --- | --- | --- | --- |",
    ]
    for label in ("d2_minus_d1", "d3_minus_d2", "d3_minus_d1"):
        row = audit["comparisons"][label]
        entropy_ci = row["entropy_auc_gain_ci95"]
        truth_ci = row["truth_log_auc_gain_ci95"]
        wtl = row["wins_ties_losses"]
        lines.append(
            f"| {label.replace('_', ' ')} | {row['entropy_auc_gain_mean']:+.4f} "
            f"[{entropy_ci[0]:+.4f}, {entropy_ci[1]:+.4f}] | "
            f"{row['truth_log_auc_gain_mean']:+.4f} "
            f"[{truth_ci[0]:+.4f}, {truth_ci[1]:+.4f}] | {wtl[0]}/{wtl[1]}/{wtl[2]} |"
        )
    lines.extend(
        [
            "",
            "The repair is real but bounded: aligned d3 exactly matches the previous "
            "terminal-EIG d2 action sequence, round-entropy curve, entropy AUC "
            "(`4.6440861`), and final entropy (`3.4657359`). It fixes the d3 "
            "receding-horizon failure but creates no strict oracle gain over the strongest d2 policy.",
            "",
            f"- Same-utility monotonicity gate: **{audit['same_utility_depth_monotonicity_passed']}**.",
            f"- Truth-log corroboration: **{audit['truth_log_corroboration_passed']}**.",
            f"- Strict gain over prior best d2: **{audit['strict_gain_over_prior_best_d2']}**.",
            f"- Independent trace audit: **{audit['audit_passed']}**.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_entropy(audit: dict[str, Any], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), sharex=True, sharey=True)
    rounds = range(1, 11)
    colors = {"1": "#2a6f97", "2": "#2a9d62", "3": "#c83e37"}
    markers = {"1": "o", "2": "s", "3": "D"}
    for axis, title, summaries in (
        (axes[0], "Terminal-EIG planning", audit["terminal_summaries"]),
        (axes[1], "Entropy-AUC-aligned planning", audit["aligned_summaries"]),
    ):
        for depth in ("1", "2", "3"):
            axis.plot(
                rounds,
                summaries[depth]["round_entropy_mean"],
                color=colors[depth],
                marker=markers[depth],
                linewidth=2.0,
                markersize=3.8,
                label=f"Exact d{depth}",
            )
        axis.set_title(title)
        axis.set_xlabel("Round")
        axis.set_xticks(list(rounds))
        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean posterior entropy (nats)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Endpoint alignment removes the rolling-horizon reversal")
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("terminal_result", type=Path)
    parser.add_argument("reference_result", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    audit = analyze(
        json.loads(args.result.read_text(encoding="utf-8")),
        json.loads(args.terminal_result.read_text(encoding="utf-8")),
        json.loads(args.reference_result.read_text(encoding="utf-8")),
    )
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.summary_output.write_text(render_summary(audit), encoding="utf-8")
    plot_entropy(audit, args.plot_output)
    print(json.dumps({"audit_passed": audit["audit_passed"]}, indent=2))


if __name__ == "__main__":
    main()
