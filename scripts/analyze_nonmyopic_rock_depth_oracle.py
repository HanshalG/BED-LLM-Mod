"""Audit and explain the RockSample[7,8] exact depth-three result."""

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

from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
from environments.rock_diagnosis.core import EPSILON
from scripts.nonmyopic_rock_depth_oracle import exhaustive_action_values


EXPECTED_CONFIG = {
    "map_name": "7-8",
    "num_trials": 500,
    "num_rounds": 10,
    "max_depth": 3,
    "seed": 24075,
    "bootstrap_replicates": 10_000,
    "trial_concurrency": 16,
    "half_efficiency_distance": math.log(2.0),
}


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def analyze(payload: dict[str, Any], reference_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    assert payload["schema_version"] == 1
    assert payload["stage"] == "depth3_exact_qualification"
    assert payload["config"] == EXPECTED_CONFIG
    assert payload["source"]["map"] == "7-8"
    assert payload["source"]["map_spec"]["rock_positions"] == [
        [1, 0],
        [5, 1],
        [2, 2],
        [3, 2],
        [6, 3],
        [0, 5],
        [3, 5],
        [2, 6],
    ]
    assert all(payload["mechanics"].values())
    assert payload["primary_gate_passed"] is False
    assert payload["truth_log_corroboration_passed"] is False

    traces = payload["traces"]
    reference_pairs = [
        (trace["trial_index"], trace["truth_index"]) for trace in traces["3"]
    ]
    assert [trial for trial, _truth in reference_pairs] == list(range(500))
    for depth in (1, 2, 3):
        depth_traces = traces[str(depth)]
        assert len(depth_traces) == 500
        assert [
            (trace["trial_index"], trace["truth_index"]) for trace in depth_traces
        ] == reference_pairs
        assert all(trace["depth"] == depth for trace in depth_traces)
        assert all(len(trace["steps"]) == 10 for trace in depth_traces)

    recomputed: dict[str, Any] = {}
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
        recomputed[label] = {
            "entropy_auc_gain_mean": float(np.mean(entropy_gains)),
            "entropy_auc_gain_ci95": stored["entropy_auc_gain_ci95"],
            "truth_log_auc_gain_mean": float(np.mean(truth_gains)),
            "truth_log_auc_gain_ci95": stored["truth_log_probability_auc_gain_ci95"],
            "wins_ties_losses": wtl,
        }

    d2_sequences = Counter(
        tuple(step["action"] for step in trace["steps"]) for trace in traces["2"]
    )
    d3_sequences = Counter(
        tuple(step["action"] for step in trace["steps"]) for trace in traces["3"]
    )
    assert len(d2_sequences) == 1
    assert len(d3_sequences) == 1
    d2_sequence = next(iter(d2_sequences))
    d3_sequence = next(iter(d3_sequences))
    first_divergence = next(
        index for index, pair in enumerate(zip(d2_sequence, d3_sequence), start=1) if pair[0] != pair[1]
    )
    assert first_divergence == 6

    model = RockDiagnosisModel(get_paper_map("7-8"))
    representative = traces["3"][0]
    belief = model.initial_belief.copy()
    position = model.map_spec.start_position
    for step in representative["steps"][: first_divergence - 1]:
        belief = model.posterior(position, belief, step["action"], step["observation"])
        position = model.next_position(position, step["action"])
    d2_values, _ = exhaustive_action_values(model, position=position, belief=belief, depth=2)
    d3_values, _ = exhaustive_action_values(model, position=position, belief=belief, depth=3)
    assert max(d2_values, key=d2_values.get) == "move-EAST"
    assert max(d3_values, key=d3_values.get) == "check-6"

    depth_summaries = {
        depth: {
            "entropy_auc_mean": _mean([trace["entropy_auc"] for trace in depth_traces]),
            "final_entropy_mean": _mean([trace["final_entropy"] for trace in depth_traces]),
            "truth_log_auc_mean": _mean(
                [trace["truth_log_probability_auc"] for trace in depth_traces]
            ),
            "round_entropy_mean": [
                _mean([trace["steps"][round_index]["entropy"] for trace in depth_traces])
                for round_index in range(10)
            ],
        }
        for depth, depth_traces in traces.items()
    }
    if reference_payload is not None:
        reference = reference_payload["maps"]["7-8"]["summary"]["exhaustive_d2"]
        assert math.isclose(
            depth_summaries["2"]["entropy_auc_mean"], reference["entropy_auc_mean"]
        )
        assert math.isclose(
            depth_summaries["2"]["final_entropy_mean"], reference["final_entropy_mean"]
        )

    return {
        "schema_version": 1,
        "audit_passed": True,
        "primary_gate_passed": False,
        "truth_log_corroboration_passed": False,
        "comparisons": recomputed,
        "depth_summaries": depth_summaries,
        "mechanism": {
            "first_divergence_round": first_divergence,
            "shared_position_before_divergence": list(position),
            "d2_action_sequence": list(d2_sequence),
            "d3_action_sequence": list(d3_sequence),
            "d2_top_values": sorted(d2_values.items(), key=lambda item: -item[1])[:4],
            "d3_top_values": sorted(d3_values.items(), key=lambda item: -item[1])[:4],
            "interpretation": "constant-horizon replanning repeats an immediate noisy check and postpones its promised movement continuation",
        },
        "prior_d2_reference_matched": reference_payload is not None,
    }


def render_summary(audit: dict[str, Any]) -> str:
    lines = [
        "# RockSample[7,8] Exact Depth-Three Result",
        "",
        "The preregistered incremental d3-over-d2 gate failed. Positive paired gains favor the deeper arm.",
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
    mechanism = audit["mechanism"]
    d2_top = dict(mechanism["d2_top_values"])
    d3_top = dict(mechanism["d3_top_values"])
    lines.extend(
        [
            "",
            "## Mechanism",
            "",
            "D2 and d3 take the same first five actions, then diverge at round 6 from position "
            f"`{tuple(mechanism['shared_position_before_divergence'])}`. D2 values `move-EAST` "
            f"at `{d2_top['move-EAST']:.4f}` nats, ahead of immediate `check-6` at "
            f"`{d2_top['check-6']:.4f}`. D3 instead values `check-6` at "
            f"`{d3_top['check-6']:.4f}`, ahead of `move-EAST` at "
            f"`{d3_top['move-EAST']:.4f}`: check now, then move and check perfectly.",
            "",
            "After the noisy check, constant-horizon replanning restores a three-step window and "
            "makes the same promise again. D3 repeats `check-6` for three rounds instead of "
            "executing the promised move. D2 moves immediately and obtains a perfect check. This "
            "is a deterministic receding-horizon commitment failure: all 500 d2 trajectories and "
            "all 500 d3 trajectories use their respective fixed action sequences.",
            "",
            f"- Primary gate: **{audit['primary_gate_passed']}**.",
            f"- Independent trace audit: **{audit['audit_passed']}**.",
            f"- Prior exact-d2 result reproduced: **{audit['prior_d2_reference_matched']}**.",
            "- Consequence: the frozen gate stops paid h3 LLM-policy engineering.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_entropy(audit: dict[str, Any], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 3.8))
    colors = {"1": "#2a6f97", "2": "#2a9d62", "3": "#c83e37"}
    markers = {"1": "o", "2": "s", "3": "D"}
    rounds = range(1, 11)
    for depth in ("1", "2", "3"):
        axis.plot(
            rounds,
            audit["depth_summaries"][depth]["round_entropy_mean"],
            color=colors[depth],
            marker=markers[depth],
            linewidth=2.1,
            markersize=4,
            label=f"Exact d{depth}",
        )
    axis.axvline(6, color="#777777", linestyle=":", linewidth=1.2)
    axis.text(6.12, 5.36, "first d2/d3 divergence", color="#555555", fontsize=8.5)
    axis.set_title("Receding-horizon depth is not monotonic")
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
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    audit = analyze(
        json.loads(args.result.read_text(encoding="utf-8")),
        json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None,
    )
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.summary_output.write_text(render_summary(audit), encoding="utf-8")
    plot_entropy(audit, args.plot_output)
    print(json.dumps({"audit_passed": audit["audit_passed"]}, indent=2))


if __name__ == "__main__":
    main()
