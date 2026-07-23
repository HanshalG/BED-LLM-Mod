"""Exact depth-four qualification for corner-start range-gated Rock Diagnosis."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
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
class RangeGatedDepth4Config:
    map_name: str = "7-8"
    start_position: tuple[int, int] = (6, 6)
    num_trials: int = 500
    num_rounds: int = 8
    seed: int = 24_201
    bootstrap_replicates: int = 10_000
    trial_concurrency: int = 16
    remote_accuracy: float = 0.55
    onsite_accuracy: float = 0.95

    def validate(self) -> None:
        spec = get_paper_map(self.map_name)
        if self.start_position != (6, 6):
            raise ValueError("the frozen h4 qualification starts at (6, 6)")
        if not (
            0 <= self.start_position[0] < spec.grid_size
            and 0 <= self.start_position[1] < spec.grid_size
        ):
            raise ValueError("start position must lie inside the map")
        if self.num_trials != 500 or self.num_rounds != 8:
            raise ValueError("the frozen h4 qualification uses 500 trials and 8 rounds")
        if self.bootstrap_replicates != 10_000:
            raise ValueError("the frozen h4 qualification uses 10,000 bootstraps")
        if self.trial_concurrency <= 0:
            raise ValueError("trial concurrency must be positive")
        if not 0.5 <= self.remote_accuracy < self.onsite_accuracy <= 1.0:
            raise ValueError("accuracies must satisfy 0.5 <= remote < onsite <= 1")


def _model(config: RangeGatedDepth4Config) -> RangeGatedRockDiagnosisModel:
    map_spec = replace(
        get_paper_map(config.map_name),
        start_position=config.start_position,
    )
    return RangeGatedRockDiagnosisModel(
        map_spec,
        remote_accuracy=config.remote_accuracy,
        onsite_accuracy=config.onsite_accuracy,
    )


def _run_trial(
    trial_index: int, config: RangeGatedDepth4Config
) -> dict[int, dict[str, Any]]:
    model = _model(config)
    truth_index = int(
        np.random.default_rng(
            _stable_seed(config.seed, "range-gated-h4-truth", trial_index)
        ).integers(len(model.hidden_states))
    )
    return {
        depth: _run_policy(
            model,
            trial_index=trial_index,
            truth_index=truth_index,
            depth=depth,
            config=config,
        )
        for depth in (3, 4)
    }


def run_qualification(config: RangeGatedDepth4Config) -> dict[str, Any]:
    config.validate()
    with ThreadPoolExecutor(max_workers=config.trial_concurrency) as executor:
        rows = list(
            executor.map(
                lambda index: _run_trial(index, config),
                range(config.num_trials),
            )
        )
    traces = {
        str(depth): [row[depth] for row in rows] for depth in (3, 4)
    }
    comparison = _comparison(
        traces["4"],
        traces["3"],
        config=config,
        label="range-gated-d4-minus-d3",
    )
    model = _model(config)
    initial_action_values = {
        str(depth): exhaustive_action_values(
            model,
            position=model.map_spec.start_position,
            belief=model.initial_belief,
            depth=depth,
        )[0]
        for depth in (3, 4)
    }
    paired = [
        (trace["trial_index"], trace["truth_index"])
        for trace in traces["3"]
    ] == [
        (trace["trial_index"], trace["truth_index"])
        for trace in traces["4"]
    ]
    mechanics = {
        "paired_trial_truths": paired,
        "all_traces_have_eight_rounds": all(
            len(trace["steps"]) == config.num_rounds
            for depth_traces in traces.values()
            for trace in depth_traces
        ),
        "all_actions_legal": all(
            step["action"]
            in model.legal_actions(tuple(step["position_before"]))
            for depth_traces in traces.values()
            for trace in depth_traces
            for step in trace["steps"]
        ),
        "all_d3_initial_roots_are_checks": all(
            trace["steps"][0]["action"].startswith("check-")
            for trace in traces["3"]
        ),
        "all_d4_initial_roots_are_move_north": all(
            trace["steps"][0]["action"] == "move-NORTH"
            for trace in traces["4"]
        ),
        "all_d4_traces_reach_onsite_inspection": all(
            any(
                step["action"].startswith("check-")
                and tuple(step["position_before"])
                == model.map_spec.rock_positions[
                    int(step["action"].split("-")[1])
                ]
                for step in trace["steps"]
            )
            for trace in traces["4"]
        ),
        "no_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_depth4_exact_qualification",
        "config": asdict(config),
        "source": {
            "map": config.map_name,
            "map_spec": asdict(model.map_spec),
            "adaptation": (
                "corner start; weak remote assay; high-fidelity assay only "
                "at the rock coordinate"
            ),
        },
        "initial_action_values": initial_action_values,
        "comparison": comparison,
        "mechanics": mechanics,
        "primary_gate_passed": (
            comparison["entropy_auc_gain_ci95"][0] > 0.0
            and all(mechanics.values())
        ),
        "truth_log_corroboration_passed": (
            comparison["truth_log_probability_auc_gain_ci95"][0] > 0.0
        ),
        "traces": traces,
    }


def render_report(result: dict[str, Any]) -> str:
    comparison = result["comparison"]
    entropy_ci = comparison["entropy_auc_gain_ci95"]
    truth_ci = comparison["truth_log_probability_auc_gain_ci95"]
    wtl = comparison["entropy_auc_wins_ties_losses"]
    return "\n".join(
        [
            "# Range-Gated RockSample[7,8] Exact Depth-Four Qualification",
            "",
            f"Primary gate: **{result['primary_gate_passed']}**.",
            f"Truth-log corroboration: **{result['truth_log_corroboration_passed']}**.",
            "",
            "| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |",
            "| --- | --- | --- | --- |",
            (
                f"| d4 minus d3 | {comparison['entropy_auc_gain_mean']:+.6f} "
                f"[{entropy_ci[0]:+.6f}, {entropy_ci[1]:+.6f}] | "
                f"{comparison['truth_log_probability_auc_gain_mean']:+.6f} "
                f"[{truth_ci[0]:+.6f}, {truth_ci[1]:+.6f}] | "
                f"{wtl[0]}/{wtl[1]}/{wtl[2]} |"
            ),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=24_201)
    parser.add_argument("--trial-concurrency", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = RangeGatedDepth4Config(
        seed=args.seed,
        trial_concurrency=args.trial_concurrency,
    )
    result = run_qualification(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(
        render_report(result), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "primary_gate_passed": result["primary_gate_passed"],
                "truth_log_corroboration_passed": result[
                    "truth_log_corroboration_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
