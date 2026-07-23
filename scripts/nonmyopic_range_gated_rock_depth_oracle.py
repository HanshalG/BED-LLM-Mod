"""Exact depth-three qualification for range-gated Rock Diagnosis."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel, get_paper_map
from scripts.nonmyopic_rock_depth_oracle import (
    _comparison,
    _run_policy,
    _stable_seed,
    exhaustive_action_values,
)


@dataclass(frozen=True)
class RangeGatedDepthConfig:
    map_name: str = "7-8"
    num_trials: int = 500
    num_rounds: int = 8
    seed: int = 24_141
    bootstrap_replicates: int = 10_000
    trial_concurrency: int = 16
    remote_accuracy: float = 0.55
    onsite_accuracy: float = 0.95

    def validate(self) -> None:
        get_paper_map(self.map_name)
        if min(
            self.num_trials,
            self.num_rounds,
            self.bootstrap_replicates,
            self.trial_concurrency,
        ) <= 0:
            raise ValueError("trial, round, bootstrap, and concurrency counts must be positive")
        if self.num_rounds < 3:
            raise ValueError("range-gated depth qualification requires at least three rounds")
        if not 0.5 <= self.remote_accuracy < self.onsite_accuracy <= 1.0:
            raise ValueError("accuracies must satisfy 0.5 <= remote < onsite <= 1")


def _model(config: RangeGatedDepthConfig) -> RangeGatedRockDiagnosisModel:
    return RangeGatedRockDiagnosisModel(
        get_paper_map(config.map_name),
        remote_accuracy=config.remote_accuracy,
        onsite_accuracy=config.onsite_accuracy,
    )


def _run_trial(
    trial_index: int, config: RangeGatedDepthConfig
) -> dict[int, dict[str, Any]]:
    model = _model(config)
    truth_rng = np.random.default_rng(
        _stable_seed(config.seed, "range-gated-rock-truth", trial_index)
    )
    truth_index = int(truth_rng.integers(len(model.hidden_states)))
    return {
        depth: _run_policy(
            model,
            trial_index=trial_index,
            truth_index=truth_index,
            depth=depth,
            config=config,
        )
        for depth in (1, 2, 3)
    }


def run_qualification(config: RangeGatedDepthConfig) -> dict[str, Any]:
    config.validate()
    with ThreadPoolExecutor(max_workers=config.trial_concurrency) as executor:
        rows = list(executor.map(lambda index: _run_trial(index, config), range(config.num_trials)))
    traces = {str(depth): [row[depth] for row in rows] for depth in (1, 2, 3)}
    comparisons = {
        "d2_minus_d1": _comparison(
            traces["2"], traces["1"], config=config, label="range-gated-d2-minus-d1"
        ),
        "d3_minus_d2": _comparison(
            traces["3"], traces["2"], config=config, label="range-gated-d3-minus-d2"
        ),
        "d3_minus_d1": _comparison(
            traces["3"], traces["1"], config=config, label="range-gated-d3-minus-d1"
        ),
    }
    model = _model(config)
    initial_values = {
        str(depth): exhaustive_action_values(
            model,
            position=model.map_spec.start_position,
            belief=model.initial_belief,
            depth=depth,
        )[0]
        for depth in (1, 2, 3)
    }
    primary = comparisons["d3_minus_d2"]
    paired = all(
        [
            (trace["trial_index"], trace["truth_index"])
            for trace in traces[str(depth)]
        ]
        == [
            (trace["trial_index"], trace["truth_index"])
            for trace in traces["3"]
        ]
        for depth in (1, 2)
    )
    all_steps = [
        step
        for depth_traces in traces.values()
        for trace in depth_traces
        for step in trace["steps"]
    ]
    mechanics = {
        "paired_trial_truths": paired,
        "all_selected_actions_legal": all(
            step["action"] in model.legal_actions(tuple(step["position_before"]))
            for step in all_steps
        ),
        "all_traces_have_registered_rounds": all(
            len(trace["steps"]) == config.num_rounds
            for depth_traces in traces.values()
            for trace in depth_traces
        ),
        "d2_initial_roots_are_checks": all(
            trace["steps"][0]["action"].startswith("check-") for trace in traces["2"]
        ),
        "d3_initial_roots_are_moves": all(
            trace["steps"][0]["action"].startswith("move-") for trace in traces["3"]
        ),
        "d3_reaches_onsite_inspection": all(
            any(
                step["action"].startswith("check-")
                and tuple(step["position_before"])
                == model.map_spec.rock_positions[int(step["action"].split("-")[1])]
                for step in trace["steps"]
            )
            for trace in traces["3"]
        ),
        "no_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_depth3_exact_qualification",
        "config": asdict(config),
        "source": {
            "map": config.map_name,
            "map_spec": asdict(model.map_spec),
            "adaptation": "weak remote assay; high-fidelity assay only at the rock coordinate",
        },
        "initial_action_values": initial_values,
        "comparisons": comparisons,
        "primary_comparison": "d3_minus_d2",
        "mechanics": mechanics,
        "primary_gate_passed": (
            primary["entropy_auc_gain_ci95"][0] > 0.0 and all(mechanics.values())
        ),
        "truth_log_corroboration_passed": (
            primary["truth_log_probability_auc_gain_ci95"][0] > 0.0
        ),
        "traces": traces,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Range-Gated RockSample[7,8] Exact Depth-Three Qualification",
        "",
        "Positive paired gains favor the deeper exact policy.",
        "",
        "| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |",
        "| --- | --- | --- | --- |",
    ]
    for label, comparison in summary["comparisons"].items():
        entropy_ci = comparison["entropy_auc_gain_ci95"]
        truth_ci = comparison["truth_log_probability_auc_gain_ci95"]
        wtl = comparison["entropy_auc_wins_ties_losses"]
        lines.append(
            f"| {label.replace('_', ' ')} | {comparison['entropy_auc_gain_mean']:+.6f} "
            f"[{entropy_ci[0]:+.6f}, {entropy_ci[1]:+.6f}] | "
            f"{comparison['truth_log_probability_auc_gain_mean']:+.6f} "
            f"[{truth_ci[0]:+.6f}, {truth_ci[1]:+.6f}] | "
            f"{wtl[0]}/{wtl[1]}/{wtl[2]} |"
        )
    lines.extend(
        [
            "",
            f"- Primary d3-over-d2 gate: **{summary['primary_gate_passed']}**.",
            f"- Truth-log corroboration: **{summary['truth_log_corroboration_passed']}**.",
            f"- Mechanics: `{summary['mechanics']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="7-8")
    parser.add_argument("--num-trials", type=int, default=500)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=24_141)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--trial-concurrency", type=int, default=16)
    parser.add_argument("--remote-accuracy", type=float, default=0.55)
    parser.add_argument("--onsite-accuracy", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/range_gated_rock_depth3_qualification_20260723"),
    )
    args = parser.parse_args()
    config = RangeGatedDepthConfig(
        map_name=args.map_name,
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        trial_concurrency=args.trial_concurrency,
        remote_accuracy=args.remote_accuracy,
        onsite_accuracy=args.onsite_accuracy,
    )
    summary = run_qualification(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "primary_gate_passed": summary["primary_gate_passed"],
                "truth_log_corroboration_passed": summary[
                    "truth_log_corroboration_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
