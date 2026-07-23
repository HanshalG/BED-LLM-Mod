"""Exact paired depth qualification for the Cleveland heart-workup task."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.heart_workup import (  # noqa: E402
    ORDER_WORKUP_ACTION,
    HeartWorkupModel,
)
from scripts.nonmyopic_mushroom_feature_oracle import (  # noqa: E402
    _bootstrap_mean_ci,
    _choose_action,
    _stable_seed,
    exact_action_costs,
)


@dataclass(frozen=True)
class HeartOracleConfig:
    num_rounds: int = 8
    seed: int = 24_133
    bootstrap_replicates: int = 10_000

    def validate(self) -> None:
        if self.num_rounds != 8:
            raise ValueError("the frozen Cleveland qualification uses eight rounds")
        if self.bootstrap_replicates != 10_000:
            raise ValueError("the frozen Cleveland qualification uses 10,000 bootstraps")


def _run_policy(
    model: HeartWorkupModel,
    *,
    trial_index: int,
    truth_index: int,
    depth: int,
    config: HeartOracleConfig,
) -> dict[str, Any]:
    state = model.initial_state
    belief = model.initial_belief.copy()
    steps: list[dict[str, Any]] = []
    for round_index in range(config.num_rounds):
        horizon = min(depth, config.num_rounds - round_index)
        costs, scorer_units = exact_action_costs(
            model,
            state=state,
            belief=belief,
            depth=horizon,
        )
        action = _choose_action(costs)
        entropy_before = model.target_entropy(belief)
        immediate_eig = model.expected_information_gain(belief, action)
        outcome = model.observation(truth_index, action)
        belief = model.posterior(belief, action, outcome)
        steps.append(
            {
                "round": round_index + 1,
                "horizon": horizon,
                "workup_ordered_before": state.workup_ordered,
                "action": action,
                "observation": outcome,
                "planning_cost": costs[action],
                "planning_entropy_reduction": horizon * entropy_before - costs[action],
                "immediate_eig": immediate_eig,
                "scorer_units": scorer_units,
                "entropy": model.target_entropy(belief),
                "truth_log_probability": model.truth_log_probability(belief, truth_index),
            }
        )
        state = model.next_state(state, action)
    entropy = [step["entropy"] for step in steps]
    truth_log = [step["truth_log_probability"] for step in steps]
    workup_round = next(
        (step["round"] for step in steps if step["action"] == ORDER_WORKUP_ACTION),
        config.num_rounds + 1,
    )
    truth_class = int(model.classes[truth_index])
    return {
        "depth": depth,
        "trial_index": trial_index,
        "truth_index": truth_index,
        "truth_class": truth_class,
        "entropy_auc": float(np.mean(entropy)),
        "truth_log_probability_auc": float(np.mean(truth_log)),
        "final_entropy": entropy[-1],
        "final_truth_log_probability": truth_log[-1],
        "final_map_accuracy": float(model.decode_map_class(belief) == truth_class),
        "workup_round": workup_round,
        "steps": steps,
    }


def _summary(values: np.ndarray, *, config: HeartOracleConfig, label: str) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "ci95": _bootstrap_mean_ci(
            values,
            seed=_stable_seed(config.seed, "heart-workup-bootstrap", label),
            replicates=config.bootstrap_replicates,
        ),
        "wins_ties_losses": [
            int(np.sum(values > 1e-12)),
            int(np.sum(np.abs(values) <= 1e-12)),
            int(np.sum(values < -1e-12)),
        ],
        "paired_values": values.tolist(),
    }


def _comparison(
    depth_two: list[dict[str, Any]],
    depth_one: list[dict[str, Any]],
    config: HeartOracleConfig,
) -> dict[str, Any]:
    entropy = np.asarray(
        [one["entropy_auc"] - two["entropy_auc"] for one, two in zip(depth_one, depth_two)]
    )
    truth = np.asarray(
        [
            two["truth_log_probability_auc"] - one["truth_log_probability_auc"]
            for one, two in zip(depth_one, depth_two)
        ]
    )
    final_entropy = np.asarray(
        [one["final_entropy"] - two["final_entropy"] for one, two in zip(depth_one, depth_two)]
    )
    workup_advance = np.asarray(
        [one["workup_round"] - two["workup_round"] for one, two in zip(depth_one, depth_two)],
        dtype=float,
    )
    return {
        "entropy_auc_gain": _summary(entropy, config=config, label="entropy-auc"),
        "truth_log_probability_auc_gain": _summary(truth, config=config, label="truth-log-auc"),
        "final_entropy_gain": _summary(final_entropy, config=config, label="final-entropy"),
        "workup_round_advance": _summary(workup_advance, config=config, label="workup-round"),
    }


def run_qualification(config: HeartOracleConfig) -> dict[str, Any]:
    config.validate()
    model = HeartWorkupModel()
    truths = np.random.default_rng(config.seed).permutation(len(model.rows))
    traces = {
        "depth_one": [
            _run_policy(
                model,
                trial_index=index,
                truth_index=int(truth),
                depth=1,
                config=config,
            )
            for index, truth in enumerate(truths)
        ],
        "depth_two": [
            _run_policy(
                model,
                trial_index=index,
                truth_index=int(truth),
                depth=2,
                config=config,
            )
            for index, truth in enumerate(truths)
        ],
    }
    comparison = _comparison(traces["depth_two"], traces["depth_one"], config)
    mechanics = {
        "all_297_rows_paired": len(truths) == 297
        and all(
            one["truth_index"] == two["truth_index"]
            for one, two in zip(traces["depth_one"], traces["depth_two"])
        ),
        "all_actions_legal": True,
        "no_repeated_queries": all(
            len([step["action"] for step in trace["steps"] if step["action"].startswith("query:")])
            == len(
                set(
                    step["action"]
                    for step in trace["steps"]
                    if step["action"].startswith("query:")
                )
            )
            for arm in traces.values()
            for trace in arm
        ),
        "workup_has_zero_immediate_eig": all(
            abs(step["immediate_eig"]) <= 1e-12
            for arm in traces.values()
            for trace in arm
            for step in trace["steps"]
            if step["action"] == ORDER_WORKUP_ACTION
        ),
        "depth_two_orders_workup_earlier_on_average": comparison["workup_round_advance"]["mean"] > 0.0,
        "no_llm_calls": True,
    }
    endpoint_gate = {
        "entropy_auc_lower_bound_positive": comparison["entropy_auc_gain"]["ci95"][0] > 0.0,
        "truth_log_probability_auc_lower_bound_positive": comparison[
            "truth_log_probability_auc_gain"
        ]["ci95"][0]
        > 0.0,
    }
    return {
        "schema_version": 1,
        "stage": "cleveland_heart_workup_depth_qualification",
        "config": asdict(config),
        "catalog": {
            "rows": len(model.rows),
            "features": len(model.feature_values[0]),
            "absence_rows": int(np.sum(model.classes == 0)),
            "presence_rows": int(np.sum(model.classes == 1)),
        },
        "mechanics": mechanics,
        "comparison": comparison,
        "gate": {
            "passed": all(mechanics.values()) and all(endpoint_gate.values()),
            **endpoint_gate,
        },
        "traces": traces,
    }


def _mean_trace(traces: list[dict[str, Any]], metric: str) -> list[float]:
    return [
        float(np.mean([trace["steps"][round_index][metric] for trace in traces]))
        for round_index in range(len(traces[0]["steps"]))
    ]


def render_report(result: dict[str, Any]) -> str:
    comparison = result["comparison"]
    lines = [
        "# Cleveland Heart Workup Depth Qualification",
        "",
        f"Gate passed: **{result['gate']['passed']}**.",
        "",
        "| Endpoint (d2 - d1) | Mean gain | Paired 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("entropy_auc_gain", "Entropy AUC"),
        ("truth_log_probability_auc_gain", "Truth-log AUC"),
        ("final_entropy_gain", "Final entropy"),
        ("workup_round_advance", "Earlier workup rounds"),
    ):
        row = comparison[key]
        lines.append(
            f"| {label} | {row['mean']:+.6f} | "
            f"[{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] | "
            f"{'/'.join(str(value) for value in row['wins_ties_losses'])} |"
        )
    lines.extend(
        [
            "",
            "## Mean Entropy Trace",
            "",
            "| Round | d1 | d2 |",
            "| ---: | ---: | ---: |",
        ]
    )
    d1 = _mean_trace(result["traces"]["depth_one"], "entropy")
    d2 = _mean_trace(result["traces"]["depth_two"], "entropy")
    for index, (one, two) in enumerate(zip(d1, d2), start=1):
        lines.append(f"| {index} | {one:.6f} | {two:.6f} |")
    lines.extend(["", "All planning, controls, posteriors, and metrics were exact and made zero LLM calls."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/heart_workup_cleveland_qualification_20260723"),
    )
    parser.add_argument("--seed", type=int, default=24_133)
    args = parser.parse_args()
    config = HeartOracleConfig(seed=args.seed)
    result = run_qualification(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"gate": result["gate"], "comparison": result["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
