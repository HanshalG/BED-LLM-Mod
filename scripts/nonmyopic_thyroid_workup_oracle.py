"""Exact paired qualification for the UCI thyroid blood-workup task."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.thyroid_workup import (
    COLLECT_BLOOD_ACTION,
    ThyroidWorkupModel,
    ThyroidWorkupState,
)


@dataclass(frozen=True)
class ThyroidQualificationConfig:
    num_trials: int = 1_000
    num_rounds: int = 8
    seed: int = 24_148
    bootstrap_replicates: int = 10_000
    trial_concurrency: int = 16
    minimum_first_collection_rate: float = 0.90

    def validate(self, cohort_size: int) -> None:
        if min(
            self.num_trials,
            self.num_rounds,
            self.bootstrap_replicates,
            self.trial_concurrency,
        ) <= 0:
            raise ValueError("trial, round, bootstrap, and concurrency counts must be positive")
        if self.num_trials > cohort_size:
            raise ValueError("trials cannot exceed the finite thyroid cohort")
        if self.num_rounds < 2:
            raise ValueError("thyroid qualification requires at least two rounds")
        if not 0.0 <= self.minimum_first_collection_rate <= 1.0:
            raise ValueError("collection threshold must be a probability")


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def exact_action_costs(
    model: ThyroidWorkupModel,
    *,
    state: ThyroidWorkupState,
    belief: np.ndarray,
    depth: int,
) -> dict[str, float]:
    """Expected cumulative post-action target entropy."""

    if depth <= 0:
        raise ValueError("depth must be positive")
    costs: dict[str, float] = {}
    for action in model.legal_actions(state):
        next_state = model.next_state(state, action)
        expected = 0.0
        for outcome in model.outcomes(action):
            probability = model.outcome_probability(belief, action, outcome)
            if probability <= 0.0:
                continue
            posterior = model.posterior(belief, action, outcome)
            branch_cost = model.target_entropy(posterior)
            if depth > 1:
                branch_cost += min(
                    exact_action_costs(
                        model,
                        state=next_state,
                        belief=posterior,
                        depth=depth - 1,
                    ).values()
                )
            expected += probability * branch_cost
        costs[action] = float(expected)
    return costs


def _choose_action(costs: dict[str, float]) -> str:
    actions = tuple(costs)
    return min(actions, key=lambda action: (costs[action], actions.index(action)))


def _run_policy(
    model: ThyroidWorkupModel,
    *,
    trial_index: int,
    truth_index: int,
    depth: int,
    config: ThyroidQualificationConfig,
) -> dict[str, Any]:
    state = model.initial_state
    belief = model.initial_belief
    steps: list[dict[str, Any]] = []
    for round_index in range(config.num_rounds):
        horizon = min(depth, config.num_rounds - round_index)
        costs = exact_action_costs(model, state=state, belief=belief, depth=horizon)
        action = _choose_action(costs)
        outcome = model.observation(truth_index, action)
        entropy_before = model.target_entropy(belief)
        immediate_eig = model.expected_information_gain(belief, action)
        belief = model.posterior(belief, action, outcome)
        steps.append(
            {
                "round": round_index + 1,
                "horizon": horizon,
                "blood_collected_before": state.blood_collected,
                "action": action,
                "observation": outcome,
                "planning_cost": costs[action],
                "planning_entropy_reduction": horizon * entropy_before - costs[action],
                "immediate_eig": immediate_eig,
                "entropy": model.target_entropy(belief),
                "truth_log_probability": model.truth_log_probability(belief, truth_index),
            }
        )
        state = model.next_state(state, action)
    entropies = [float(step["entropy"]) for step in steps]
    truth_logs = [float(step["truth_log_probability"]) for step in steps]
    return {
        "depth": depth,
        "trial_index": trial_index,
        "truth_index": truth_index,
        "truth_class": int(model.targets[truth_index]),
        "entropy_auc": float(np.mean(entropies)),
        "truth_log_probability_auc": float(np.mean(truth_logs)),
        "final_entropy": entropies[-1],
        "final_truth_log_probability": truth_logs[-1],
        "final_class_accuracy": float(model.decode_class(belief) == model.targets[truth_index]),
        "steps": steps,
    }


def _bootstrap(values: np.ndarray, *, seed: int, replicates: int) -> list[float]:
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        chunks.append(values[indices].mean(axis=1))
    samples = np.concatenate(chunks)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def run_qualification(config: ThyroidQualificationConfig) -> dict[str, Any]:
    model = ThyroidWorkupModel()
    config.validate(len(model.targets))
    truths = np.random.default_rng(config.seed).choice(
        len(model.targets), size=config.num_trials, replace=False
    )

    def run_trial(item: tuple[int, int]) -> dict[int, dict[str, Any]]:
        trial_index, truth_index = item
        return {
            depth: _run_policy(
                model,
                trial_index=trial_index,
                truth_index=int(truth_index),
                depth=depth,
                config=config,
            )
            for depth in (1, 2)
        }

    with ThreadPoolExecutor(max_workers=config.trial_concurrency) as executor:
        rows = list(executor.map(run_trial, enumerate(truths)))
    traces = {"depth_one": [row[1] for row in rows], "depth_two": [row[2] for row in rows]}
    entropy_gain = np.asarray(
        [one["entropy_auc"] - two["entropy_auc"] for one, two in zip(traces["depth_one"], traces["depth_two"])],
        dtype=float,
    )
    truth_gain = np.asarray(
        [two["truth_log_probability_auc"] - one["truth_log_probability_auc"] for one, two in zip(traces["depth_one"], traces["depth_two"])],
        dtype=float,
    )
    collection_rate = float(
        np.mean(
            [trace["steps"][0]["action"] == COLLECT_BLOOD_ACTION for trace in traces["depth_two"]]
        )
    )
    mechanics = {
        "paired_truths_without_replacement": len(set(int(value) for value in truths)) == config.num_trials,
        "all_traces_complete": all(
            len(trace["steps"]) == config.num_rounds for arm in traces.values() for trace in arm
        ),
        "all_actions_legal": all(
            step["action"]
            in model.legal_actions(
                ThyroidWorkupState(
                    bool(step["blood_collected_before"]),
                    tuple(
                        sorted(
                            model.action_feature(previous["action"])
                            for previous in trace["steps"][: step["round"] - 1]
                            if previous["action"].startswith("query:")
                        )
                    ),
                )
            )
            for arm in traces.values()
            for trace in arm
            for step in trace["steps"]
        ),
        "blood_collection_zero_immediate_eig": model.expected_information_gain(
            model.initial_belief, COLLECT_BLOOD_ACTION
        )
        == 0.0,
        "d2_first_collection_rate_at_least_threshold": collection_rate
        >= config.minimum_first_collection_rate,
        "no_llm_calls": True,
    }
    comparison = {
        "entropy_auc_gain": {
            "mean": float(np.mean(entropy_gain)),
            "ci95": _bootstrap(
                entropy_gain,
                seed=_stable_seed(config.seed, "thyroid-entropy-bootstrap"),
                replicates=config.bootstrap_replicates,
            ),
            "wins_ties_losses": [
                int(np.sum(entropy_gain > 1e-12)),
                int(np.sum(np.abs(entropy_gain) <= 1e-12)),
                int(np.sum(entropy_gain < -1e-12)),
            ],
        },
        "truth_log_probability_auc_gain": {
            "mean": float(np.mean(truth_gain)),
            "ci95": _bootstrap(
                truth_gain,
                seed=_stable_seed(config.seed, "thyroid-truth-bootstrap"),
                replicates=config.bootstrap_replicates,
            ),
        },
        "d2_first_collection_rate": collection_rate,
    }
    return {
        "schema_version": 1,
        "stage": "uci_thyroid_workup_depth_qualification",
        "config": asdict(config),
        "source": {
            "dataset": "UCI Thyroid Disease ann-thyroid",
            "doi": "10.24432/C5D010",
            "rows": len(model.targets),
            "classes": {str(value): int(np.sum(model.targets == value)) for value in (1, 2, 3)},
            "bin_edges": {str(key): value.tolist() for key, value in model.bin_edges.items()},
        },
        "mechanics": mechanics,
        "comparison": comparison,
        "primary_gate_passed": comparison["entropy_auc_gain"]["ci95"][0] > 0.0
        and all(mechanics.values()),
        "truth_log_corroboration_passed": comparison["truth_log_probability_auc_gain"]["ci95"][0]
        > 0.0,
        "traces": traces,
    }


def render(summary: dict[str, Any]) -> str:
    entropy = summary["comparison"]["entropy_auc_gain"]
    truth = summary["comparison"]["truth_log_probability_auc_gain"]
    return "\n".join(
        [
            "# UCI Thyroid Blood-Workup Exact Qualification",
            "",
            f"Primary gate passed: **{summary['primary_gate_passed']}**.",
            f"Truth-log corroboration passed: **{summary['truth_log_corroboration_passed']}**.",
            "",
            f"- Entropy-AUC d2-over-d1: `{entropy['mean']:+.6f}` "
            f"`[{entropy['ci95'][0]:+.6f}, {entropy['ci95'][1]:+.6f}]`, "
            f"W/T/L `{entropy['wins_ties_losses'][0]}/{entropy['wins_ties_losses'][1]}/{entropy['wins_ties_losses'][2]}`.",
            f"- Truth-log-AUC d2-over-d1: `{truth['mean']:+.6f}` "
            f"`[{truth['ci95'][0]:+.6f}, {truth['ci95'][1]:+.6f}]`.",
            f"- D2 first-round blood collection: `{summary['comparison']['d2_first_collection_rate']:.1%}`.",
            f"- Mechanics: `{summary['mechanics']}`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=1_000)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=24_148)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--trial-concurrency", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/thyroid_workup_qualification_20260723"),
    )
    args = parser.parse_args()
    config = ThyroidQualificationConfig(
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        trial_concurrency=args.trial_concurrency,
    )
    summary = run_qualification(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render(summary), encoding="utf-8")
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
