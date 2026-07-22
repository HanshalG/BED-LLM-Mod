"""Exact paired depth qualification for semantic mushroom feature acquisition."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.mushroom_feature_acquisition import (  # noqa: E402
    COLLECT_ACTION,
    AcquisitionState,
    MushroomFeatureModel,
)
from environments.mushroom_feature_acquisition.model import EPSILON  # noqa: E402


@dataclass(frozen=True)
class OracleConfig:
    num_trials: int = 1000
    num_rounds: int = 8
    seed: int = 24_123
    bootstrap_replicates: int = 10_000

    def validate(self, *, catalog_size: int) -> None:
        if min(self.num_trials, self.num_rounds, self.bootstrap_replicates) <= 0:
            raise ValueError("trial, round, and bootstrap counts must be positive")
        if self.num_trials > catalog_size:
            raise ValueError("num_trials cannot exceed the catalog for sampling without replacement")


def _stable_seed(*parts: Any) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def exact_action_costs(
    model: MushroomFeatureModel,
    *,
    state: AcquisitionState,
    belief: np.ndarray,
    depth: int,
) -> tuple[dict[str, float], int]:
    """Expected cumulative post-action target entropy; lower is better."""

    if depth not in (1, 2):
        raise ValueError("mushroom oracle supports depth one or two")
    values: dict[str, float] = {}
    scorer_units = 0
    for action in model.legal_actions(state):
        expected_cost = 0.0
        for outcome in model.outcomes(action):
            probability = model.outcome_probability(belief, action, outcome)
            if probability <= EPSILON:
                continue
            posterior = model.posterior(belief, action, outcome)
            next_state = model.next_state(state, action)
            branch_cost = model.target_entropy(posterior)
            scorer_units += 1
            if depth > 1:
                next_values, next_units = exact_action_costs(
                    model,
                    state=next_state,
                    belief=posterior,
                    depth=depth - 1,
                )
                branch_cost += min(next_values.values())
                scorer_units += next_units
            expected_cost += probability * branch_cost
        values[action] = expected_cost
    return values, scorer_units


def _choose_action(costs: dict[str, float]) -> str:
    actions = tuple(costs)
    return min(actions, key=lambda action: (costs[action], actions.index(action)))


def _run_policy(
    model: MushroomFeatureModel,
    *,
    trial_index: int,
    truth_index: int,
    depth: int,
    config: OracleConfig,
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
        if action not in model.legal_actions(state):
            raise AssertionError("oracle selected an illegal mushroom action")
        outcome = model.observation(truth_index, action)
        entropy_before = model.target_entropy(belief)
        immediate_eig = model.expected_information_gain(belief, action)
        belief = model.posterior(belief, action, outcome)
        next_state = model.next_state(state, action)
        steps.append(
            {
                "round": round_index + 1,
                "horizon": horizon,
                "specimen_collected_before": state.specimen_collected,
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
        state = next_state

    entropy = [step["entropy"] for step in steps]
    truth_log = [step["truth_log_probability"] for step in steps]
    truth_class = str(model.classes[truth_index])
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
        "steps": steps,
    }


def _bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, replicates: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    batches: list[np.ndarray] = []
    for start in range(0, replicates, 512):
        size = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        batches.append(np.mean(values[indices], axis=1))
    samples = np.concatenate(batches)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _comparison(
    depth_two: list[dict[str, Any]],
    depth_one: list[dict[str, Any]],
    config: OracleConfig,
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

    def summary(values: np.ndarray, label: str) -> dict[str, Any]:
        return {
            "mean": float(np.mean(values)),
            "ci95": _bootstrap_mean_ci(
                values,
                seed=_stable_seed(config.seed, "mushroom-bootstrap", label),
                replicates=config.bootstrap_replicates,
            ),
            "paired_values": values.tolist(),
            "wins_ties_losses": [
                int(np.sum(values > 1e-12)),
                int(np.sum(np.abs(values) <= 1e-12)),
                int(np.sum(values < -1e-12)),
            ],
        }

    return {
        "entropy_auc_gain": summary(entropy, "entropy-auc"),
        "truth_log_probability_auc_gain": summary(truth, "truth-log-auc"),
        "final_entropy_gain": summary(final_entropy, "final-entropy"),
    }


def run_qualification(config: OracleConfig) -> dict[str, Any]:
    model = MushroomFeatureModel()
    config.validate(catalog_size=len(model.rows))
    rng = np.random.default_rng(config.seed)
    truths = rng.choice(len(model.rows), size=config.num_trials, replace=False)
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
        "paired_trials_and_truths": all(
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
        "collection_has_zero_immediate_eig": all(
            abs(step["immediate_eig"]) <= 1e-12
            for arm in traces.values()
            for trace in arm
            for step in trace["steps"]
            if step["action"] == COLLECT_ACTION
        ),
        "depth_two_collects_first": all(
            trace["steps"][0]["action"] == COLLECT_ACTION for trace in traces["depth_two"]
        ),
        "depth_one_does_not_collect_first": all(
            trace["steps"][0]["action"] != COLLECT_ACTION for trace in traces["depth_one"]
        ),
        "no_llm_calls": True,
    }
    entropy_pass = comparison["entropy_auc_gain"]["ci95"][0] > 0.0
    truth_pass = comparison["truth_log_probability_auc_gain"]["ci95"][0] > 0.0
    return {
        "schema_version": 1,
        "stage": "mushroom_feature_acquisition_depth_qualification",
        "config": asdict(config),
        "catalog": {
            "rows": len(model.rows),
            "features": 22,
            "edible_rows": int(np.sum(model.classes == "e")),
            "poisonous_rows": int(np.sum(model.classes == "p")),
        },
        "mechanics": mechanics,
        "comparison": comparison,
        "gate": {
            "passed": all(mechanics.values()) and entropy_pass and truth_pass,
            "entropy_auc_lower_bound_positive": entropy_pass,
            "truth_log_probability_auc_lower_bound_positive": truth_pass,
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
        "# Mushroom Feature Acquisition Depth Qualification",
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
    ):
        row = comparison[key]
        lines.append(
            f"| {label} | {row['mean']:+.6f} | "
            f"[{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] | "
            f"{'/'.join(str(value) for value in row['wins_ties_losses'])} |"
        )
    lines.extend(["", "## Mean Entropy Trace", "", "| Round | d1 | d2 |", "| ---: | ---: | ---: |"])
    one = _mean_trace(result["traces"]["depth_one"], "entropy")
    two = _mean_trace(result["traces"]["depth_two"], "entropy")
    for index, (d1, d2) in enumerate(zip(one, two), start=1):
        lines.append(f"| {index} | {d1:.6f} | {d2:.6f} |")
    lines.extend(["", "All planning, posterior updates, controls, and metrics were exact and made zero LLM calls."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/mushroom_feature_depth_qualification_20260723"),
    )
    parser.add_argument("--num-trials", type=int, default=1000)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=24_123)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    config = OracleConfig(
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    result = run_qualification(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"gate": result["gate"], "comparison": result["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
