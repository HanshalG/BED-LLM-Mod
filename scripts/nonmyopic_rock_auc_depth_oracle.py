"""Exact paired Rock Diagnosis depth oracle using entropy-AUC-aligned planning."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
from scripts.nonmyopic_rock_depth_oracle import (
    DepthOracleConfig,
    _comparison,
    _stable_seed,
    _uniform,
    exhaustive_action_values,
)


def _choose_action(values: dict[str, float]) -> str:
    actions = tuple(values)
    return max(actions, key=lambda action: (values[action], -actions.index(action)))


def _run_policy(
    model: RockDiagnosisModel,
    *,
    trial_index: int,
    truth_index: int,
    depth: int,
    config: DepthOracleConfig,
) -> dict[str, Any]:
    belief = model.initial_belief.copy()
    position = model.map_spec.start_position
    check_counts: dict[tuple[tuple[int, int], int], int] = {}
    steps: list[dict[str, Any]] = []

    for round_index in range(config.num_rounds):
        horizon = min(depth, config.num_rounds - round_index)
        values, scorer_units = exhaustive_action_values(
            model,
            position=position,
            belief=belief,
            depth=horizon,
            planning_utility="entropy_auc",
        )
        action = _choose_action(values)
        check_id = model.check_id(action)
        if check_id is None:
            outcome: str | None = None
        else:
            key = (position, check_id)
            repeat_index = check_counts.get(key, 0)
            check_counts[key] = repeat_index + 1
            probability_good = float(model.likelihood_vector(position, action, "good")[truth_index])
            outcome = (
                "good"
                if _uniform(
                    config.seed,
                    "rock-auc-depth-observation",
                    trial_index,
                    position,
                    check_id,
                    repeat_index,
                )
                < probability_good
                else "bad"
            )
        immediate_eig = model.expected_information_gain(position, belief, action)
        belief = model.posterior(position, belief, action, outcome)
        steps.append(
            {
                "round": round_index + 1,
                "horizon": horizon,
                "position_before": list(position),
                "action": action,
                "observation": outcome,
                "planning_value": values[action],
                "immediate_eig": immediate_eig,
                "scorer_units": scorer_units,
                "entropy": model.entropy(belief),
                "truth_log_probability": math.log(
                    max(float(belief[truth_index]), np.finfo(float).tiny)
                ),
            }
        )
        position = model.next_position(position, action)

    entropy_values = [float(step["entropy"]) for step in steps]
    truth_log_values = [float(step["truth_log_probability"]) for step in steps]
    return {
        "depth": depth,
        "trial_index": trial_index,
        "truth_index": truth_index,
        "entropy_auc": float(np.mean(entropy_values)),
        "truth_log_probability_auc": float(np.mean(truth_log_values)),
        "final_entropy": entropy_values[-1],
        "final_truth_log_probability": truth_log_values[-1],
        "final_map_accuracy": float(model.decode_map_index(belief) == truth_index),
        "steps": steps,
    }


def _run_trial(trial_index: int, config: DepthOracleConfig) -> dict[int, dict[str, Any]]:
    model = RockDiagnosisModel(
        get_paper_map(config.map_name),
        half_efficiency_distance=config.half_efficiency_distance,
    )
    truth_rng = np.random.default_rng(_stable_seed(config.seed, "rock-auc-depth-truth", trial_index))
    truth_index = int(truth_rng.integers(len(model.hidden_states)))
    return {
        depth: _run_policy(
            model,
            trial_index=trial_index,
            truth_index=truth_index,
            depth=depth,
            config=config,
        )
        for depth in range(1, config.max_depth + 1)
    }


def run_auc_depth_oracle(config: DepthOracleConfig) -> dict[str, Any]:
    config.validate()
    with ThreadPoolExecutor(max_workers=config.trial_concurrency) as executor:
        trial_rows = list(executor.map(lambda index: _run_trial(index, config), range(config.num_trials)))
    traces = {
        str(depth): [trial[depth] for trial in trial_rows]
        for depth in range(1, config.max_depth + 1)
    }
    reference_pairs = [
        (trace["trial_index"], trace["truth_index"]) for trace in traces[str(config.max_depth)]
    ]
    comparisons = {
        "d2_minus_d1": _comparison(traces["2"], traces["1"], config=config, label="auc-d2-minus-d1"),
        "d3_minus_d2": _comparison(traces["3"], traces["2"], config=config, label="auc-d3-minus-d2"),
        "d3_minus_d1": _comparison(traces["3"], traces["1"], config=config, label="auc-d3-minus-d1"),
    }
    primary = comparisons["d3_minus_d2"]
    model = RockDiagnosisModel(
        get_paper_map(config.map_name),
        half_efficiency_distance=config.half_efficiency_distance,
    )
    mechanics = {
        "paired_trial_truths": all(
            [(trace["trial_index"], trace["truth_index"]) for trace in traces[str(depth)]]
            == reference_pairs
            for depth in range(1, config.max_depth + 1)
        ),
        "all_selected_actions_legal": all(
            step["action"] in model.legal_actions(tuple(step["position_before"]))
            for depth_traces in traces.values()
            for trace in depth_traces
            for step in trace["steps"]
        ),
        "all_traces_have_registered_rounds": all(
            len(trace["steps"]) == config.num_rounds
            for depth_traces in traces.values()
            for trace in depth_traces
        ),
        "no_llm_calls": True,
    }
    initial_values = {
        str(depth): max(
            exhaustive_action_values(
                model,
                position=model.map_spec.start_position,
                belief=model.initial_belief,
                depth=depth,
                planning_utility="entropy_auc",
            )[0].values()
        )
        for depth in range(1, config.max_depth + 1)
    }
    return {
        "schema_version": 1,
        "stage": "auc_aligned_depth3_exact_qualification",
        "planning_utility": "entropy_auc",
        "config": asdict(config),
        "source": {
            "paper": model.map_spec.source_citation,
            "url": model.map_spec.source_url,
            "page": model.map_spec.source_page,
            "map": config.map_name,
            "map_spec": asdict(model.map_spec),
            "pomdp_py_version": "1.3.5.1",
        },
        "initial_exact_values": initial_values,
        "comparisons": comparisons,
        "mechanics": mechanics,
        "primary_gate_passed": primary["entropy_auc_gain_ci95"][0] > 0.0
        and all(mechanics.values()),
        "truth_log_corroboration_passed": primary[
            "truth_log_probability_auc_gain_ci95"
        ][0]
        > 0.0,
        "traces": traces,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# RockSample[7,8] AUC-Aligned Exact Depth Qualification",
        "",
        "Positive paired gains favor the deeper AUC-aligned policy.",
        "",
        "| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |",
        "| --- | --- | --- | --- |",
    ]
    for label in ("d2_minus_d1", "d3_minus_d2", "d3_minus_d1"):
        row = summary["comparisons"][label]
        entropy_ci = row["entropy_auc_gain_ci95"]
        truth_ci = row["truth_log_probability_auc_gain_ci95"]
        wtl = row["entropy_auc_wins_ties_losses"]
        lines.append(
            f"| {label.replace('_', ' ')} | {row['entropy_auc_gain_mean']:+.4f} "
            f"[{entropy_ci[0]:+.4f}, {entropy_ci[1]:+.4f}] | "
            f"{row['truth_log_probability_auc_gain_mean']:+.4f} "
            f"[{truth_ci[0]:+.4f}, {truth_ci[1]:+.4f}] | {wtl[0]}/{wtl[1]}/{wtl[2]} |"
        )
    lines.extend(
        [
            "",
            f"- Primary d3-over-d2 gate: **{summary['primary_gate_passed']}**.",
            f"- Truth-log corroboration: **{summary['truth_log_corroboration_passed']}**.",
            f"- Initial AUC-aligned values: `{summary['initial_exact_values']}`.",
            f"- Mechanics: `{summary['mechanics']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="7-8", choices=("3-6", "5-7", "7-8"))
    parser.add_argument("--num-trials", type=int, default=500)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=24_076)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--trial-concurrency", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/rocksample_7_8_auc_depth3_oracle_20260721"),
    )
    args = parser.parse_args()
    config = DepthOracleConfig(
        map_name=args.map_name,
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        max_depth=args.max_depth,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        trial_concurrency=args.trial_concurrency,
    )
    summary = run_auc_depth_oracle(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "primary_gate_passed": summary["primary_gate_passed"],
                "truth_log_corroboration_passed": summary["truth_log_corroboration_passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
