"""Exact h5-over-h4 qualification for a focused-prior range-gated Rock task."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import cached_property
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import (
    RangeGatedRockDiagnosisModel,
    get_paper_map,
)
from environments.rock_diagnosis.core import (
    EPSILON,
    RockType,
    _FactorizedRockBelief,
)
from scripts.nonmyopic_rock_depth_oracle import (
    _comparison,
    _stable_seed,
    _uniform,
    exhaustive_action_values,
)


H5_ROUTE = (
    "move-NORTH",
    "move-WEST",
    "move-WEST",
    "move-WEST",
    "check-6",
)


@dataclass(frozen=True)
class FocusedDepth5Config:
    map_name: str = "7-8"
    start_position: tuple[int, int] = (6, 6)
    uncertain_rock: int = 6
    uncertain_good_probability: float = 0.5
    secondary_good_probability: float = 0.005
    num_trials: int = 500
    num_rounds: int = 8
    seed: int = 24_235
    bootstrap_replicates: int = 10_000
    remote_accuracy: float = 0.55
    onsite_accuracy: float = 0.95

    def validate(self) -> None:
        spec = get_paper_map(self.map_name)
        if self.map_name != "7-8" or self.start_position != (6, 6):
            raise ValueError("the focused h5 protocol uses map 7-8 from start (6, 6)")
        if not 0 <= self.uncertain_rock < len(spec.rock_positions):
            raise ValueError("uncertain rock is out of range")
        if self.uncertain_rock != 6:
            raise ValueError("the focused h5 protocol targets rock 6")
        if self.uncertain_good_probability != 0.5:
            raise ValueError("the focused h5 protocol uses p_good=.5 for rock 6")
        if self.secondary_good_probability != 0.005:
            raise ValueError("the focused h5 protocol uses p_good=.005 elsewhere")
        if min(self.num_trials, self.num_rounds, self.bootstrap_replicates) <= 0:
            raise ValueError("trial, round, and bootstrap counts must be positive")
        if self.num_rounds < 5:
            raise ValueError("the focused h5 protocol needs at least five rounds")
        if not 0.5 <= self.remote_accuracy < self.onsite_accuracy <= 1.0:
            raise ValueError("accuracies must satisfy 0.5 <= remote < onsite <= 1")


class FocusedPriorRangeGatedModel(RangeGatedRockDiagnosisModel):
    def __init__(
        self,
        config: FocusedDepth5Config,
    ) -> None:
        self.focused_config = config
        super().__init__(
            replace(
                get_paper_map(config.map_name),
                start_position=config.start_position,
            ),
            remote_accuracy=config.remote_accuracy,
            onsite_accuracy=config.onsite_accuracy,
        )

    @cached_property
    def prior_good_probabilities(self) -> np.ndarray:
        probabilities = np.full(
            self.num_rocks,
            self.focused_config.secondary_good_probability,
            dtype=float,
        )
        probabilities[self.focused_config.uncertain_rock] = (
            self.focused_config.uncertain_good_probability
        )
        return probabilities

    @cached_property
    def initial_belief(self) -> np.ndarray:
        probabilities = self.prior_good_probabilities
        values: list[float] = []
        for state in self.hidden_states:
            probability = 1.0
            for rock_index, rock_type in enumerate(state):
                p_good = float(probabilities[rock_index])
                probability *= (
                    p_good if rock_type == RockType.GOOD else 1.0 - p_good
                )
            values.append(probability)
        belief = _FactorizedRockBelief(
            np.asarray(values, dtype=float),
            probabilities,
        )
        if not np.isclose(float(np.sum(belief)), 1.0, atol=1e-12, rtol=0.0):
            raise RuntimeError("focused prior does not sum to one")
        return belief


def build_focused_depth5_model(
    config: FocusedDepth5Config,
) -> FocusedPriorRangeGatedModel:
    config.validate()
    return FocusedPriorRangeGatedModel(config)


def _choose_action(values: dict[str, float]) -> str:
    actions = tuple(values)
    return max(actions, key=lambda action: (values[action], -actions.index(action)))


def _sample_truth_index(
    model: FocusedPriorRangeGatedModel,
    *,
    trial_index: int,
    seed: int,
) -> int:
    rng = np.random.default_rng(
        _stable_seed(seed, "focused-range-gated-h5-truth", trial_index)
    )
    truth = tuple(
        RockType.GOOD if rng.random() < probability else RockType.BAD
        for probability in model.prior_good_probabilities
    )
    return model.hidden_states.index(truth)


def _belief_key(belief: np.ndarray) -> bytes:
    return np.ascontiguousarray(belief, dtype=np.float64).tobytes()


def _run_policy_cached(
    model: FocusedPriorRangeGatedModel,
    *,
    trial_index: int,
    truth_index: int,
    depth: int,
    config: FocusedDepth5Config,
    decision_cache: dict[
        tuple[int, tuple[int, int], int, bytes],
        tuple[dict[str, float], int],
    ],
) -> dict[str, Any]:
    belief = model.initial_belief.copy()
    position = model.map_spec.start_position
    check_counts: dict[tuple[tuple[int, int], int], int] = {}
    steps: list[dict[str, Any]] = []
    for round_index in range(config.num_rounds):
        horizon = min(depth, config.num_rounds - round_index)
        key = (depth, position, horizon, _belief_key(belief))
        cached = decision_cache.get(key)
        if cached is None:
            cached = exhaustive_action_values(
                model,
                position=position,
                belief=belief,
                depth=horizon,
            )
            decision_cache[key] = cached
        values, scorer_units = cached
        action = _choose_action(values)
        check_id = model.check_id(action)
        if check_id is None:
            outcome: str | None = None
        else:
            check_key = (position, check_id)
            repeat_index = check_counts.get(check_key, 0)
            check_counts[check_key] = repeat_index + 1
            probability_good = float(
                model.likelihood_vector(position, action, RockType.GOOD)[
                    truth_index
                ]
            )
            outcome = (
                RockType.GOOD
                if _uniform(
                    config.seed,
                    "focused-h5-observation",
                    trial_index,
                    position,
                    check_id,
                    repeat_index,
                )
                < probability_good
                else RockType.BAD
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


def run_qualification(config: FocusedDepth5Config) -> dict[str, Any]:
    model = build_focused_depth5_model(config)
    decision_cache: dict[
        tuple[int, tuple[int, int], int, bytes],
        tuple[dict[str, float], int],
    ] = {}
    traces: dict[str, list[dict[str, Any]]] = {"4": [], "5": []}
    truth_indices: list[int] = []
    for trial_index in range(config.num_trials):
        truth_index = _sample_truth_index(
            model,
            trial_index=trial_index,
            seed=config.seed,
        )
        truth_indices.append(truth_index)
        for depth in (4, 5):
            traces[str(depth)].append(
                _run_policy_cached(
                    model,
                    trial_index=trial_index,
                    truth_index=truth_index,
                    depth=depth,
                    config=config,
                    decision_cache=decision_cache,
                )
            )
    comparison = _comparison(
        traces["5"],
        traces["4"],
        config=config,
        label="focused-range-gated-d5-minus-d4",
    )
    initial_action_values = {
        str(depth): exhaustive_action_values(
            model,
            position=model.map_spec.start_position,
            belief=model.initial_belief,
            depth=depth,
        )[0]
        for depth in (4, 5)
    }
    mechanics = {
        "paired_trial_truths": [
            (row["trial_index"], row["truth_index"]) for row in traces["4"]
        ]
        == [
            (row["trial_index"], row["truth_index"]) for row in traces["5"]
        ],
        "all_traces_have_registered_rounds": all(
            len(row["steps"]) == config.num_rounds
            for depth_rows in traces.values()
            for row in depth_rows
        ),
        "all_actions_legal": all(
            step["action"]
            in model.legal_actions(tuple(step["position_before"]))
            for depth_rows in traces.values()
            for row in depth_rows
            for step in row["steps"]
        ),
        "all_d4_initial_roots_are_remote_check6": all(
            row["steps"][0]["action"] == "check-6" for row in traces["4"]
        ),
        "all_d5_initial_roots_are_move_north": all(
            row["steps"][0]["action"] == "move-NORTH" for row in traces["5"]
        ),
        "all_d5_prefixes_follow_registered_route": all(
            tuple(step["action"] for step in row["steps"][:5]) == H5_ROUTE
            for row in traces["5"]
        ),
        "all_d5_checks_rock6_onsite_at_round5": all(
            row["steps"][4]["action"] == "check-6"
            and tuple(row["steps"][4]["position_before"])
            == model.map_spec.rock_positions[6]
            for row in traces["5"]
        ),
        "at_least_ninety_percent_entropy_auc_wins": (
            comparison["entropy_auc_wins_ties_losses"][0]
            >= math.ceil(0.9 * config.num_trials)
        ),
        "no_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "stage": "focused_range_gated_rock_depth5_exact_qualification",
        "config": asdict(config),
        "source": {
            "map_spec": asdict(model.map_spec),
            "prior_good_probabilities": model.prior_good_probabilities.tolist(),
            "adaptation": (
                "corner start; rock 6 maximally uncertain; secondary rocks "
                "retain small nonzero uncertainty; accurate checks require on-site"
            ),
        },
        "initial_entropy": model.entropy(model.initial_belief),
        "initial_action_values": initial_action_values,
        "truth_indices": truth_indices,
        "comparison": comparison,
        "mechanics": mechanics,
        "planner_cache_entries": len(decision_cache),
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
            "# Focused-Prior Range-Gated Rock Exact Depth-Five Qualification",
            "",
            f"Primary gate: **{result['primary_gate_passed']}**.",
            f"Truth-log corroboration: **{result['truth_log_corroboration_passed']}**.",
            "",
            "| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | W/T/L |",
            "| --- | --- | --- | --- |",
            (
                f"| d5 minus d4 | {comparison['entropy_auc_gain_mean']:+.6f} "
                f"[{entropy_ci[0]:+.6f}, {entropy_ci[1]:+.6f}] | "
                f"{comparison['truth_log_probability_auc_gain_mean']:+.6f} "
                f"[{truth_ci[0]:+.6f}, {truth_ci[1]:+.6f}] | "
                f"{wtl[0]}/{wtl[1]}/{wtl[2]} |"
            ),
            "",
            f"- Registered d5 route: `{list(H5_ROUTE)}`.",
            f"- Initial entropy: `{result['initial_entropy']:.6f}` nats.",
            f"- Exact planner cache entries: `{result['planner_cache_entries']}`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=24_235)
    parser.add_argument("--num-trials", type=int, default=500)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = FocusedDepth5Config(
        seed=args.seed,
        num_trials=args.num_trials,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    result = run_qualification(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(
        render_report(result),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "primary_gate_passed": result["primary_gate_passed"],
                "truth_log_corroboration_passed": result[
                    "truth_log_corroboration_passed"
                ],
                "comparison": result["comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
