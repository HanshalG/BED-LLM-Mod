"""Exact paired depth-one/depth-two oracle for gated sensor diagnosis."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.gated_sensor import GatedSensorModel, SensorState
from environments.gated_sensor.model import EPSILON


@dataclass(frozen=True)
class OracleConfig:
    num_trials: int = 500
    num_rounds: int = 8
    seed: int = 24_091
    bootstrap_replicates: int = 10_000
    screen_accuracy: float = 0.65
    precise_accuracy: float = 0.95

    def validate(self) -> None:
        if min(self.num_trials, self.num_rounds, self.bootstrap_replicates) <= 0:
            raise ValueError("trial, round, and bootstrap counts must be positive")
        GatedSensorModel(
            screen_accuracy=self.screen_accuracy,
            precise_accuracy=self.precise_accuracy,
        )


def _stable_seed(*parts: Any) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def _uniform(*parts: Any) -> float:
    return _stable_seed(*parts) / 2**64


def exact_action_values(
    model: GatedSensorModel,
    *,
    state: SensorState,
    belief: np.ndarray,
    depth: int,
) -> tuple[dict[str, float], int]:
    """Return exact terminal-EIG values and the number of action evaluations."""

    if depth not in (1, 2):
        raise ValueError("gated-sensor oracle supports depth one or two")
    legal = model.legal_actions(state)
    values: dict[str, float] = {}
    scorer_units = len(legal)
    for action in legal:
        value = model.expected_information_gain(belief, action)
        if depth == 2:
            next_state = model.next_state(state, action)
            for outcome in model.outcomes(action):
                probability = model.outcome_probability(belief, action, outcome)
                if probability <= EPSILON:
                    continue
                posterior = model.posterior(belief, action, outcome)
                next_values, next_units = exact_action_values(
                    model,
                    state=next_state,
                    belief=posterior,
                    depth=1,
                )
                value += probability * max(next_values.values())
                scorer_units += next_units
        values[action] = value
    return values, scorer_units


def _choose_action(values: dict[str, float]) -> str:
    actions = tuple(values)
    return max(actions, key=lambda action: (values[action], -actions.index(action)))


def _run_policy(
    model: GatedSensorModel,
    *,
    trial_index: int,
    truth_index: int,
    depth: int,
    config: OracleConfig,
) -> dict[str, Any]:
    state = model.initial_state
    belief = model.initial_belief.copy()
    action_counts: dict[str, int] = {}
    steps: list[dict[str, Any]] = []

    for round_index in range(config.num_rounds):
        horizon = min(depth, config.num_rounds - round_index)
        values, scorer_units = exact_action_values(model, state=state, belief=belief, depth=horizon)
        action = _choose_action(values)
        kind = model.action_kind(action)
        if kind == "activate":
            outcome: str | None = None
        else:
            repeat_index = action_counts.get(action, 0)
            action_counts[action] = repeat_index + 1
            probability_positive = float(model.likelihood_vector(action, "positive")[truth_index])
            outcome = (
                "positive"
                if _uniform(config.seed, "gated-sensor-observation", trial_index, action, repeat_index)
                < probability_positive
                else "negative"
            )
        immediate_eig = model.expected_information_gain(belief, action)
        belief = model.posterior(belief, action, outcome)
        steps.append(
            {
                "round": round_index + 1,
                "horizon": horizon,
                "state_before": state.active_panel,
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
        state = model.next_state(state, action)

    entropy_values = [step["entropy"] for step in steps]
    truth_log_values = [step["truth_log_probability"] for step in steps]
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


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, replicates: int) -> list[float]:
    rng = np.random.default_rng(seed)
    batches: list[np.ndarray] = []
    for start in range(0, replicates, 512):
        size = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        batches.append(np.mean(values[indices], axis=1))
    samples = np.concatenate(batches)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _comparison(d2: list[dict[str, Any]], d1: list[dict[str, Any]], config: OracleConfig) -> dict[str, Any]:
    entropy_gains = np.asarray(
        [d1[index]["entropy_auc"] - d2[index]["entropy_auc"] for index in range(len(d2))]
    )
    truth_gains = np.asarray(
        [
            d2[index]["truth_log_probability_auc"] - d1[index]["truth_log_probability_auc"]
            for index in range(len(d2))
        ]
    )
    final_gains = np.asarray(
        [d1[index]["final_entropy"] - d2[index]["final_entropy"] for index in range(len(d2))]
    )
    return {
        "entropy_auc_gain_mean": float(np.mean(entropy_gains)),
        "entropy_auc_gain_ci95": _bootstrap_mean_ci(
            entropy_gains,
            seed=_stable_seed(config.seed, "entropy-auc-bootstrap"),
            replicates=config.bootstrap_replicates,
        ),
        "truth_log_probability_auc_gain_mean": float(np.mean(truth_gains)),
        "truth_log_probability_auc_gain_ci95": _bootstrap_mean_ci(
            truth_gains,
            seed=_stable_seed(config.seed, "truth-log-auc-bootstrap"),
            replicates=config.bootstrap_replicates,
        ),
        "final_entropy_gain_mean": float(np.mean(final_gains)),
        "final_entropy_gain_ci95": _bootstrap_mean_ci(
            final_gains,
            seed=_stable_seed(config.seed, "final-entropy-bootstrap"),
            replicates=config.bootstrap_replicates,
        ),
        "entropy_auc_wins_ties_losses": [
            int(np.count_nonzero(entropy_gains > EPSILON)),
            int(np.count_nonzero(np.abs(entropy_gains) <= EPSILON)),
            int(np.count_nonzero(entropy_gains < -EPSILON)),
        ],
        "entropy_auc_paired_values": entropy_gains.tolist(),
        "truth_log_probability_auc_paired_values": truth_gains.tolist(),
    }


def run_oracle(config: OracleConfig) -> dict[str, Any]:
    config.validate()
    model = GatedSensorModel(
        screen_accuracy=config.screen_accuracy,
        precise_accuracy=config.precise_accuracy,
    )
    traces: dict[str, list[dict[str, Any]]] = {"1": [], "2": []}
    for trial_index in range(config.num_trials):
        truth_rng = np.random.default_rng(_stable_seed(config.seed, "gated-sensor-truth", trial_index))
        truth_index = int(truth_rng.integers(len(model.hidden_states)))
        for depth in (1, 2):
            traces[str(depth)].append(
                _run_policy(
                    model,
                    trial_index=trial_index,
                    truth_index=truth_index,
                    depth=depth,
                    config=config,
                )
            )
    comparison = _comparison(traces["2"], traces["1"], config)
    paired = all(
        (d1["trial_index"], d1["truth_index"]) == (d2["trial_index"], d2["truth_index"])
        for d1, d2 in zip(traces["1"], traces["2"], strict=True)
    )
    mechanics = {
        "paired_trials_and_truths": paired,
        "d1_initial_actions_are_measurements": all(
            trace["steps"][0]["action"].startswith("screen:") for trace in traces["1"]
        ),
        "d2_initial_actions_are_zero_eig_activations": all(
            trace["steps"][0]["action"].startswith("activate:")
            and abs(trace["steps"][0]["immediate_eig"]) <= EPSILON
            for trace in traces["2"]
        ),
        "all_selected_actions_legal": all(
            step["action"] in model.legal_actions(SensorState(step["state_before"]))
            for depth_traces in traces.values()
            for trace in depth_traces
            for step in trace["steps"]
        ),
    }
    gate_passed = (
        mechanics["paired_trials_and_truths"]
        and mechanics["d1_initial_actions_are_measurements"]
        and mechanics["d2_initial_actions_are_zero_eig_activations"]
        and mechanics["all_selected_actions_legal"]
        and comparison["entropy_auc_gain_ci95"][0] > 0.0
        and comparison["truth_log_probability_auc_gain_ci95"][0] > 0.0
    )
    return {
        "schema_version": 1,
        "no_llm_calls": True,
        "config": asdict(config),
        "environment": {
            "num_hidden_states": len(model.hidden_states),
            "panels": model.panels,
            "predicates": [predicate.name for predicate in model.predicates],
        },
        "comparison": comparison,
        "mechanics": mechanics,
        "gate": {
            "passed": gate_passed,
            "rule": "paired lower 95% bounds are positive for entropy AUC and truth-log AUC",
        },
        "traces": traces,
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Remove the full trace tree while retaining paired inferential inputs."""

    compact = {key: value for key, value in summary.items() if key != "traces"}
    compact["arm_summaries"] = {
        depth: {
            "entropy_auc_mean": float(np.mean([trace["entropy_auc"] for trace in traces])),
            "truth_log_probability_auc_mean": float(
                np.mean([trace["truth_log_probability_auc"] for trace in traces])
            ),
            "final_entropy_mean": float(np.mean([trace["final_entropy"] for trace in traces])),
            "final_truth_log_probability_mean": float(
                np.mean([trace["final_truth_log_probability"] for trace in traces])
            ),
            "final_map_accuracy_mean": float(
                np.mean([trace["final_map_accuracy"] for trace in traces])
            ),
            "initial_action_counts": dict(
                sorted(Counter(trace["steps"][0]["action"] for trace in traces).items())
            ),
        }
        for depth, traces in summary["traces"].items()
    }
    return compact


def write_full_traces(summary: dict[str, Any], output_path: Path) -> None:
    """Write one compressed record per policy trajectory."""

    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for depth, traces in summary["traces"].items():
            for trace in traces:
                handle.write(json.dumps({"depth": int(depth), **trace}, sort_keys=True))
                handle.write("\n")


def render_report(summary: dict[str, Any]) -> str:
    comparison = summary["comparison"]
    entropy_ci = comparison["entropy_auc_gain_ci95"]
    truth_ci = comparison["truth_log_probability_auc_gain_ci95"]
    return "\n".join(
        [
            "# Gated Sensor Exact Depth Qualification",
            "",
            "This zero-LLM-call benchmark tests whether exact two-step planning exploits a setup action that has no immediate information but unlocks precise measurements.",
            "",
            "| Comparison | Mean gain | Paired 95% CI |",
            "| --- | ---: | --- |",
            f"| d2 - d1 entropy AUC | {comparison['entropy_auc_gain_mean']:+.4f} | [{entropy_ci[0]:+.4f}, {entropy_ci[1]:+.4f}] |",
            f"| d2 - d1 truth-log AUC | {comparison['truth_log_probability_auc_gain_mean']:+.4f} | [{truth_ci[0]:+.4f}, {truth_ci[1]:+.4f}] |",
            "",
            f"**Gate passed:** `{summary['gate']['passed']}`",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=500)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=24_091)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--screen-accuracy", type=float, default=0.65)
    parser.add_argument("--precise-accuracy", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/gated_sensor_depth_qualification"),
    )
    args = parser.parse_args()
    config = OracleConfig(
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        screen_accuracy=args.screen_accuracy,
        precise_accuracy=args.precise_accuracy,
    )
    summary = run_oracle(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(compact_summary(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    write_full_traces(summary, args.output_dir / "TRACES.jsonl.gz")
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
