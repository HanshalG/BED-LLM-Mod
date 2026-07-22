"""Audit the preregistered frozen RockSample[15,15] exact qualification."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis.core import EPSILON
from scripts.nonmyopic_rock_depth_oracle import _bootstrap_mean_ci, _stable_seed


EXPECTED_CONFIG = {
    "map_name": "15-15",
    "num_trials": 100,
    "num_rounds": 15,
    "max_depth": 2,
    "seed": 24099,
    "bootstrap_replicates": 5_000,
    "trial_concurrency": 2,
    "half_efficiency_distance": math.log(2.0),
}
EXPECTED_ROCKS = [
    [13, 9], [4, 2], [13, 8], [2, 6], [2, 10],
    [10, 1], [14, 10], [9, 5], [5, 12], [13, 7],
    [4, 5], [3, 9], [0, 0], [14, 2], [3, 7],
]


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    assert payload["schema_version"] == 1
    assert payload["stage"] == "depth2_exact_qualification"
    assert payload["config"] == EXPECTED_CONFIG
    assert payload["primary_comparison"] == "d2_minus_d1"
    assert payload["source"]["map"] == "15-15"
    assert payload["source"]["map_spec"]["rock_positions"] == EXPECTED_ROCKS
    assert payload["source"]["map_spec"]["start_position"] == [0, 7]
    assert "a5e1d62d14e4efe783885b9d4f19cffa2a568eec" in payload["source"]["url"]
    assert all(payload["mechanics"].values())

    traces = payload["traces"]
    assert set(traces) == {"1", "2"}
    reference_pairs = [(trace["trial_index"], trace["truth_index"]) for trace in traces["2"]]
    assert [trial for trial, _truth in reference_pairs] == list(range(100))
    for depth in (1, 2):
        depth_traces = traces[str(depth)]
        assert len(depth_traces) == 100
        assert [(trace["trial_index"], trace["truth_index"]) for trace in depth_traces] == reference_pairs
        assert all(trace["depth"] == depth for trace in depth_traces)
        assert all(len(trace["steps"]) == 15 for trace in depth_traces)

    d1 = traces["1"]
    d2 = traces["2"]
    entropy_gains = np.asarray(
        [d1[index]["entropy_auc"] - d2[index]["entropy_auc"] for index in range(100)]
    )
    truth_gains = np.asarray(
        [
            d2[index]["truth_log_probability_auc"]
            - d1[index]["truth_log_probability_auc"]
            for index in range(100)
        ]
    )
    final_gains = np.asarray(
        [d1[index]["final_entropy"] - d2[index]["final_entropy"] for index in range(100)]
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
    assert math.isclose(float(np.mean(final_gains)), stored["final_entropy_gain_mean"])

    ci_specs = (
        (entropy_gains, "entropy-auc-bootstrap", "entropy_auc_gain_ci95"),
        (truth_gains, "truth-log-auc-bootstrap", "truth_log_probability_auc_gain_ci95"),
        (final_gains, "final-entropy-bootstrap", "final_entropy_gain_ci95"),
    )
    for values, suffix, field in ci_specs:
        expected_ci = _bootstrap_mean_ci(
            values,
            seed=_stable_seed(EXPECTED_CONFIG["seed"], "d2-minus-d1", suffix),
            replicates=EXPECTED_CONFIG["bootstrap_replicates"],
        )
        assert np.allclose(expected_ci, stored[field], atol=1e-12)

    wins_ties_losses = [
        int(np.count_nonzero(entropy_gains > EPSILON)),
        int(np.count_nonzero(np.abs(entropy_gains) <= EPSILON)),
        int(np.count_nonzero(entropy_gains < -EPSILON)),
    ]
    assert wins_ties_losses == stored["entropy_auc_wins_ties_losses"]

    depth_summaries: dict[str, Any] = {}
    for depth, depth_traces in traces.items():
        steps = [step for trace in depth_traces for step in trace["steps"]]
        sequences = Counter(tuple(step["action"] for step in trace["steps"]) for trace in depth_traces)
        depth_summaries[depth] = {
            "entropy_auc_mean": _mean([trace["entropy_auc"] for trace in depth_traces]),
            "truth_log_auc_mean": _mean(
                [trace["truth_log_probability_auc"] for trace in depth_traces]
            ),
            "final_entropy_mean": _mean([trace["final_entropy"] for trace in depth_traces]),
            "move_count": sum(step["action"].startswith("move-") for step in steps),
            "decision_count": len(steps),
            "unique_action_sequences": len(sequences),
            "most_common_sequence": list(sequences.most_common(1)[0][0]),
            "most_common_sequence_count": sequences.most_common(1)[0][1],
        }
    assert depth_summaries["1"]["move_count"] == 0
    assert depth_summaries["2"]["move_count"] > 0

    primary_passed = stored["entropy_auc_gain_ci95"][0] > 0.0
    truth_passed = stored["truth_log_probability_auc_gain_ci95"][0] > 0.0
    assert payload["primary_gate_passed"] is primary_passed
    assert payload["truth_log_corroboration_passed"] is truth_passed
    return {
        "schema_version": 1,
        "audit_passed": True,
        "primary_gate_passed": primary_passed,
        "truth_log_corroboration_passed": truth_passed,
        "comparison": {
            "entropy_auc_gain_mean": float(np.mean(entropy_gains)),
            "entropy_auc_gain_ci95": stored["entropy_auc_gain_ci95"],
            "truth_log_auc_gain_mean": float(np.mean(truth_gains)),
            "truth_log_auc_gain_ci95": stored["truth_log_probability_auc_gain_ci95"],
            "final_entropy_gain_mean": float(np.mean(final_gains)),
            "final_entropy_gain_ci95": stored["final_entropy_gain_ci95"],
            "wins_ties_losses": wins_ties_losses,
        },
        "initial_exact_values": payload["initial_exact_values"],
        "depth_summaries": depth_summaries,
    }


def render_summary(audit: dict[str, Any]) -> str:
    row = audit["comparison"]
    d1 = audit["depth_summaries"]["1"]
    d2 = audit["depth_summaries"]["2"]
    return "\n".join(
        [
            "# Frozen RockSample[15,15] Exact Scale Result",
            "",
            "The preregistered zero-LLM structural gate passed on 100 paired trials.",
            "",
            "| Endpoint | d2 gain over d1 | Paired 95% CI |",
            "| --- | ---: | --- |",
            f"| Entropy AUC | {row['entropy_auc_gain_mean']:+.4f} | {row['entropy_auc_gain_ci95']} |",
            f"| Truth-log AUC | {row['truth_log_auc_gain_mean']:+.4f} | {row['truth_log_auc_gain_ci95']} |",
            f"| Final entropy | {row['final_entropy_gain_mean']:+.4f} | {row['final_entropy_gain_ci95']} |",
            "",
            f"Entropy-AUC wins/ties/losses were `{row['wins_ties_losses']}`. Greedy d1 moved on "
            f"`{d1['move_count']}/{d1['decision_count']}` decisions; exact d2 moved on "
            f"`{d2['move_count']}/{d2['decision_count']}`.",
            "",
            f"The independent audit reconstructed every paired value and bootstrap interval. "
            f"Initial exact values were `{audit['initial_exact_values']}`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    audit = analyze(json.loads(args.result.read_text(encoding="utf-8")))
    args.audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_output.write_text(render_summary(audit), encoding="utf-8")
    print(json.dumps({"audit_passed": audit["audit_passed"]}, indent=2))


if __name__ == "__main__":
    main()
