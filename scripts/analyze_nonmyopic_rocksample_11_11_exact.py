"""Audit the preregistered RockSample[11,11] exact scale qualification."""

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
    "map_name": "11-11",
    "num_trials": 500,
    "num_rounds": 12,
    "max_depth": 2,
    "seed": 24078,
    "bootstrap_replicates": 10_000,
    "trial_concurrency": 16,
    "half_efficiency_distance": math.log(2.0),
}
EXPECTED_ROCKS = [
    [0, 3],
    [0, 7],
    [1, 8],
    [2, 4],
    [3, 3],
    [3, 8],
    [4, 3],
    [5, 8],
    [6, 1],
    [9, 3],
    [9, 9],
]


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    assert payload["schema_version"] == 1
    assert payload["stage"] == "depth2_exact_qualification"
    assert payload["config"] == EXPECTED_CONFIG
    assert payload["primary_comparison"] == "d2_minus_d1"
    assert payload["source"]["map"] == "11-11"
    assert payload["source"]["map_spec"]["rock_positions"] == EXPECTED_ROCKS
    assert payload["source"]["map_spec"]["start_position"] == [0, 5]
    assert payload["source"]["map_spec"]["source_url"].endswith(
        "examples/POMDPX/RockSample_11_11.pomdpx"
    )
    assert all(payload["mechanics"].values())
    assert payload["primary_gate_passed"] is True
    assert payload["truth_log_corroboration_passed"] is True

    traces = payload["traces"]
    assert set(traces) == {"1", "2"}
    reference_pairs = [
        (trace["trial_index"], trace["truth_index"]) for trace in traces["2"]
    ]
    assert [trial for trial, _truth in reference_pairs] == list(range(500))
    for depth in (1, 2):
        depth_traces = traces[str(depth)]
        assert len(depth_traces) == 500
        assert [
            (trace["trial_index"], trace["truth_index"])
            for trace in depth_traces
        ] == reference_pairs
        assert all(trace["depth"] == depth for trace in depth_traces)
        assert all(len(trace["steps"]) == 12 for trace in depth_traces)

    d1 = traces["1"]
    d2 = traces["2"]
    entropy_gains = np.asarray(
        [d1[index]["entropy_auc"] - d2[index]["entropy_auc"] for index in range(500)]
    )
    truth_gains = np.asarray(
        [
            d2[index]["truth_log_probability_auc"]
            - d1[index]["truth_log_probability_auc"]
            for index in range(500)
        ]
    )
    stored = payload["comparisons"]["d2_minus_d1"]
    assert np.allclose(entropy_gains, stored["entropy_auc_paired_values"], atol=1e-12)
    assert np.allclose(
        truth_gains, stored["truth_log_probability_auc_paired_values"], atol=1e-12
    )
    assert math.isclose(float(np.mean(entropy_gains)), stored["entropy_auc_gain_mean"])
    assert math.isclose(
        float(np.mean(truth_gains)), stored["truth_log_probability_auc_gain_mean"]
    )
    wins_ties_losses = [
        int(np.count_nonzero(entropy_gains > EPSILON)),
        int(np.count_nonzero(np.abs(entropy_gains) <= EPSILON)),
        int(np.count_nonzero(entropy_gains < -EPSILON)),
    ]
    assert wins_ties_losses == stored["entropy_auc_wins_ties_losses"]

    depth_summaries: dict[str, Any] = {}
    for depth, depth_traces in traces.items():
        steps = [step for trace in depth_traces for step in trace["steps"]]
        sequence_counts = Counter(
            tuple(step["action"] for step in trace["steps"]) for trace in depth_traces
        )
        depth_summaries[depth] = {
            "entropy_auc_mean": _mean(
                [float(trace["entropy_auc"]) for trace in depth_traces]
            ),
            "truth_log_auc_mean": _mean(
                [float(trace["truth_log_probability_auc"]) for trace in depth_traces]
            ),
            "final_entropy_mean": _mean(
                [float(trace["final_entropy"]) for trace in depth_traces]
            ),
            "move_count": sum(step["action"].startswith("move-") for step in steps),
            "decision_count": len(steps),
            "unique_action_sequences": len(sequence_counts),
            "most_common_sequence": list(sequence_counts.most_common(1)[0][0]),
            "most_common_sequence_count": sequence_counts.most_common(1)[0][1],
            "round_entropy_mean": [
                _mean(
                    [
                        float(trace["steps"][round_index]["entropy"])
                        for trace in depth_traces
                    ]
                )
                for round_index in range(12)
            ],
        }
    assert depth_summaries["1"]["move_count"] == 0
    assert depth_summaries["2"]["move_count"] > 0

    return {
        "schema_version": 1,
        "audit_passed": True,
        "primary_gate_passed": True,
        "truth_log_corroboration_passed": True,
        "comparison": {
            "entropy_auc_gain_mean": float(np.mean(entropy_gains)),
            "entropy_auc_gain_ci95": stored["entropy_auc_gain_ci95"],
            "truth_log_auc_gain_mean": float(np.mean(truth_gains)),
            "truth_log_auc_gain_ci95": stored[
                "truth_log_probability_auc_gain_ci95"
            ],
            "final_entropy_gain_mean": stored["final_entropy_gain_mean"],
            "final_entropy_gain_ci95": stored["final_entropy_gain_ci95"],
            "wins_ties_losses": wins_ties_losses,
        },
        "initial_exact_values": payload["initial_exact_values"],
        "depth_summaries": depth_summaries,
    }


def render_summary(audit: dict[str, Any]) -> str:
    row = audit["comparison"]
    entropy_ci = row["entropy_auc_gain_ci95"]
    truth_ci = row["truth_log_auc_gain_ci95"]
    final_ci = row["final_entropy_gain_ci95"]
    wtl = row["wins_ties_losses"]
    d1 = audit["depth_summaries"]["1"]
    d2 = audit["depth_summaries"]["2"]
    return "\n".join(
        [
            "# RockSample[11,11] Exact Scale Result",
            "",
            "The preregistered zero-LLM structural gate passed. Positive gains favor "
            "exhaustive receding-horizon d2 over exhaustive d1.",
            "",
            "| Endpoint | Paired gain | 95% CI | W/T/L |",
            "| --- | ---: | --- | --- |",
            f"| Entropy AUC | {row['entropy_auc_gain_mean']:+.4f} | "
            f"[{entropy_ci[0]:+.4f}, {entropy_ci[1]:+.4f}] | "
            f"{wtl[0]}/{wtl[1]}/{wtl[2]} |",
            f"| Truth-log AUC | {row['truth_log_auc_gain_mean']:+.4f} | "
            f"[{truth_ci[0]:+.4f}, {truth_ci[1]:+.4f}] | - |",
            f"| Final entropy | {row['final_entropy_gain_mean']:+.4f} | "
            f"[{final_ci[0]:+.4f}, {final_ci[1]:+.4f}] | - |",
            "",
            "## Mechanism",
            "",
            f"Greedy d1 moves on `{d1['move_count']}/{d1['decision_count']}` decisions. "
            f"Exact d2 moves on `{d2['move_count']}/{d2['decision_count']}` decisions and "
            f"uses `{d2['unique_action_sequences']}` action sequence across all 500 truths:",
            "",
            f"`{' -> '.join(d2['most_common_sequence'])}`",
            "",
            f"Initial exact values are `{audit['initial_exact_values']}`. The independent "
            "audit reconstructed every paired AUC value from the raw traces. This result "
            "authorizes a separately preregistered root-slot serving smoke; it is not yet "
            "an LLM-policy result.",
            "",
        ]
    )


def plot_entropy(audit: dict[str, Any], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 3.8))
    rounds = range(1, 13)
    styles = {
        "1": {"color": "#3366a8", "marker": "o", "label": "Exhaustive d1"},
        "2": {"color": "#c43c39", "marker": "s", "label": "Exhaustive d2"},
    }
    for depth in ("1", "2"):
        axis.plot(
            rounds,
            audit["depth_summaries"][depth]["round_entropy_mean"],
            linewidth=2.2,
            markersize=4,
            **styles[depth],
        )
    axis.set_title("RockSample[11,11] exact structural qualification")
    axis.set_xlabel("Round")
    axis.set_ylabel("Mean posterior entropy (nats)")
    axis.set_xticks(list(rounds))
    axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
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
    print(json.dumps({"audit_passed": audit["audit_passed"]}, indent=2))


if __name__ == "__main__":
    main()
