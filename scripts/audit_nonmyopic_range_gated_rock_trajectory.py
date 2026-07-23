"""Independent replay audit for cached range-gated h3 trajectories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel, get_paper_map
from scripts.nonmyopic_range_gated_rock_fixed_tail_proposal_gate import (
    matched_random_fixed_root_plans,
)
from scripts.nonmyopic_range_gated_rock_proposal_gate import _stable_seed
from scripts.nonmyopic_range_gated_rock_stable_controls import stable_best_index
from scripts.nonmyopic_rock_depth_oracle import _uniform, exhaustive_action_values


def _plan_value(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    plan: tuple[str, ...],
) -> float:
    if not plan:
        return 0.0
    action = plan[0]
    if action not in model.legal_actions(position):
        raise ValueError(f"illegal audited action {action} from {position}")
    value = model.expected_information_gain(position, belief, action)
    next_position = model.next_position(position, action)
    for outcome in model.outcomes(action):
        probability = model.outcome_probability(position, belief, action, outcome)
        if probability <= 0.0:
            continue
        posterior = model.posterior(position, belief, action, outcome)
        value += probability * _plan_value(
            model,
            position=next_position,
            belief=posterior,
            plan=plan[1:],
        )
    return float(value)


def _best_plan(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    plans: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], list[float]]:
    values = [
        _plan_value(model, position=position, belief=belief, plan=plan)
        for plan in plans
    ]
    return plans[stable_best_index(values)], values


def _exact_action(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    depth: int,
) -> str:
    values, _ = exhaustive_action_values(
        model, position=position, belief=belief, depth=depth
    )
    actions = tuple(values)
    return actions[stable_best_index([values[action] for action in actions])]


def _bootstrap(values: np.ndarray, *, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(10_000)
    for start in range(0, len(samples), 256):
        size = min(256, len(samples) - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        samples[start : start + size] = values[indices].mean(axis=1)
    return [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def _isclose(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("stage") != (
        "range_gated_rock_cached_h3_paired_trajectory_confirmation"
    ):
        raise ValueError("unexpected range-gated trajectory stage")
    config = payload["config"]
    if (
        config["num_trials"] != 50
        or config["num_rounds"] != 8
        or config["bootstrap_replicates"] != 10_000
        or config["max_unique_llm_cells"] != 64
    ):
        raise ValueError("trajectory payload does not use the frozen protocol")
    model = RangeGatedRockDiagnosisModel(
        get_paper_map("7-8"),
        remote_accuracy=float(config["remote_accuracy"]),
        onsite_accuracy=float(config["onsite_accuracy"]),
    )
    logical_by_cell = {
        int(request["cell_index"]): request
        for request in payload["logical_requests"]
    }
    physical_by_cell = {
        int(request["cell_index"]): request
        for request in payload["candidate_requests"]
    }
    cache_states: dict[str, tuple[Any, Any, Any]] = {}
    cache_consistent = True
    misses = 0
    for request in payload["logical_requests"]:
        key = request["cache_key"]
        state = (
            request["position"],
            request["history"],
            request["compiled_plans"],
        )
        prior = cache_states.setdefault(key, state)
        cache_consistent &= prior == state
        if not request["cache_hit"]:
            misses += 1
            physical = physical_by_cell.get(int(request["cell_index"]))
            cache_consistent &= (
                physical is not None
                and physical["compiled_plans"] == request["compiled_plans"]
                and physical["position"] == request["position"]
                and physical["history"] == request["history"]
            )

    checks = {
        "all_cache_keys_are_state_and_plan_consistent": cache_consistent,
        "all_unique_cache_misses_match_physical_requests": (
            misses == len(cache_states) == len(payload["candidate_requests"])
        ),
        "all_histories_positions_and_observations_replayed": True,
        "all_llm_plans_values_and_stable_selections_match": True,
        "all_random_plans_values_and_stable_selections_match": True,
        "all_exact_control_actions_match": True,
        "all_posterior_metrics_match": True,
        "all_aggregate_means_match": True,
        "paired_truths_match": True,
        "no_llm_calls": True,
    }
    replayed: dict[str, list[dict[str, float]]] = {
        arm: [] for arm in payload["traces"]
    }
    seed = int(config["seed"])
    rounds = int(config["num_rounds"])
    truths = [int(value) for value in payload["truth_indices"]]

    for arm, traces in payload["traces"].items():
        checks["paired_truths_match"] &= (
            [int(trace["truth_index"]) for trace in traces] == truths
        )
        for trace in traces:
            trial_index = int(trace["trial_index"])
            truth_index = int(trace["truth_index"])
            position = model.map_spec.start_position
            belief = model.initial_belief.copy()
            history: list[list[Any]] = []
            check_counts: dict[tuple[tuple[int, int], int], int] = {}
            entropies: list[float] = []
            truth_logs: list[float] = []

            for round_index, step in enumerate(trace["steps"]):
                action = str(step["action"])
                checks["all_histories_positions_and_observations_replayed"] &= (
                    list(position) == step["position_before"]
                    and action in model.legal_actions(position)
                )
                remaining = rounds - round_index
                if arm == "llm_h3" and remaining >= 3:
                    cell_index = trial_index * rounds + round_index
                    request = logical_by_cell[cell_index]
                    plans = tuple(
                        tuple(plan) for plan in request["compiled_plans"]
                    )
                    selected, values = _best_plan(
                        model,
                        position=position,
                        belief=belief,
                        plans=plans,
                    )
                    policy = step["policy"]
                    checks[
                        "all_llm_plans_values_and_stable_selections_match"
                    ] &= (
                        request["position"] == list(position)
                        and request["history"] == history
                        and policy["plans"] == request["compiled_plans"]
                        and policy["selected_plan"] == list(selected)
                        and np.allclose(
                            policy["plan_values"], values, atol=1e-12, rtol=0.0
                        )
                        and action == selected[0]
                    )
                elif arm == "random_h3" and remaining >= 3:
                    plans = matched_random_fixed_root_plans(
                        model,
                        position=position,
                        belief=belief,
                        seed=_stable_seed(
                            seed,
                            "trajectory-random",
                            trial_index,
                            round_index,
                        ),
                    )
                    selected, values = _best_plan(
                        model,
                        position=position,
                        belief=belief,
                        plans=plans,
                    )
                    policy = step["policy"]
                    checks[
                        "all_random_plans_values_and_stable_selections_match"
                    ] &= (
                        policy["plans"] == [list(plan) for plan in plans]
                        and policy["selected_plan"] == list(selected)
                        and np.allclose(
                            policy["plan_values"], values, atol=1e-12, rtol=0.0
                        )
                        and action == selected[0]
                    )
                else:
                    requested_depth = 3 if arm == "exact_d3" else 2
                    depth = min(requested_depth, remaining)
                    checks["all_exact_control_actions_match"] &= (
                        _exact_action(
                            model,
                            position=position,
                            belief=belief,
                            depth=depth,
                        )
                        == action
                    )

                check_id = model.check_id(action)
                if check_id is None:
                    repeat_index = 0
                    observation = None
                else:
                    key = (position, check_id)
                    repeat_index = check_counts.get(key, 0)
                    check_counts[key] = repeat_index + 1
                    probability_good = float(
                        model.likelihood_vector(position, action, "good")[
                            truth_index
                        ]
                    )
                    observation = (
                        "good"
                        if _uniform(
                            seed,
                            "range-trajectory-observation",
                            trial_index,
                            position,
                            check_id,
                            repeat_index,
                        )
                        < probability_good
                        else "bad"
                    )
                checks["all_histories_positions_and_observations_replayed"] &= (
                    observation == step["observation"]
                    and repeat_index == step["repeat_index"]
                )
                belief = model.posterior(
                    position, belief, action, observation
                )
                position = model.next_position(position, action)
                history.append([action, observation])
                entropy = model.entropy(belief)
                truth_log = math.log(
                    max(float(belief[truth_index]), np.finfo(float).tiny)
                )
                entropies.append(entropy)
                truth_logs.append(truth_log)
                checks["all_posterior_metrics_match"] &= (
                    _isclose(entropy, step["entropy"])
                    and _isclose(truth_log, step["truth_log_probability"])
                )

            entropy_auc = float(np.mean(entropies))
            truth_auc = float(np.mean(truth_logs))
            checks["all_posterior_metrics_match"] &= (
                _isclose(entropy_auc, trace["entropy_auc"])
                and _isclose(
                    truth_auc, trace["truth_log_probability_auc"]
                )
            )
            replayed[arm].append(
                {"entropy_auc": entropy_auc, "truth_auc": truth_auc}
            )

    comparisons: dict[str, dict[str, Any]] = {}
    independent_seeds = {
        "entropy_auc_gain_vs_exact_d2": 24_189,
        "truth_log_auc_gain_vs_exact_d2": 24_190,
        "entropy_auc_gain_vs_random_h3": 24_191,
        "truth_log_auc_gain_vs_random_h3": 24_192,
    }
    for baseline, suffix in (
        ("exact_d2", "exact_d2"),
        ("random_h3", "random_h3"),
    ):
        entropy = np.asarray(
            [
                base["entropy_auc"] - llm["entropy_auc"]
                for base, llm in zip(
                    replayed[baseline], replayed["llm_h3"], strict=True
                )
            ]
        )
        truth = np.asarray(
            [
                llm["truth_auc"] - base["truth_auc"]
                for base, llm in zip(
                    replayed[baseline], replayed["llm_h3"], strict=True
                )
            ]
        )
        for name, values in (
            (f"entropy_auc_gain_vs_{suffix}", entropy),
            (f"truth_log_auc_gain_vs_{suffix}", truth),
        ):
            comparisons[name] = {
                "mean": float(values.mean()),
                "independent_ci95": _bootstrap(
                    values, seed=independent_seeds[name]
                ),
            }
            checks["all_aggregate_means_match"] &= _isclose(
                values.mean(), payload["comparisons"][name]["mean"]
            )

    exact_gain = float(
        np.mean(
            [
                d2["entropy_auc"] - d3["entropy_auc"]
                for d2, d3 in zip(
                    replayed["exact_d2"], replayed["exact_d3"], strict=True
                )
            ]
        )
    )
    recovery = (
        comparisons["entropy_auc_gain_vs_exact_d2"]["mean"] / exact_gain
    )
    first_two_south = float(
        np.mean(
            [
                trace["steps"][0]["action"] == "move-SOUTH"
                and trace["steps"][1]["action"] == "move-SOUTH"
                for trace in payload["traces"]["llm_h3"]
            ]
        )
    )
    onsite_by_three = float(
        np.mean(
            [
                any(
                    step["round"] <= 3
                    and str(step["action"]).startswith("check-")
                    and tuple(step["position_before"])
                    == model.map_spec.rock_positions[
                        int(str(step["action"]).split("-")[1])
                    ]
                    for step in trace["steps"]
                )
                for trace in payload["traces"]["llm_h3"]
            ]
        )
    )
    scientific_gate = (
        comparisons["entropy_auc_gain_vs_exact_d2"]["independent_ci95"][0]
        > 0.0
        and comparisons["truth_log_auc_gain_vs_exact_d2"][
            "independent_ci95"
        ][0]
        > 0.0
        and comparisons["entropy_auc_gain_vs_random_h3"][
            "independent_ci95"
        ][0]
        > 0.0
        and comparisons["truth_log_auc_gain_vs_random_h3"][
            "independent_ci95"
        ][0]
        > 0.0
        and recovery >= float(config["recovery_threshold"])
        and first_two_south >= float(config["route_rate_threshold"])
        and onsite_by_three >= float(config["route_rate_threshold"])
    )
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_cached_h3_trajectory_audit",
        "mechanics": checks,
        "comparisons": comparisons,
        "exact_d3_entropy_gain_vs_exact_d2": exact_gain,
        "llm_recovery_fraction_of_exact_d3_gain": recovery,
        "llm_first_two_south_rate": first_two_south,
        "llm_onsite_by_round_three_rate": onsite_by_three,
        "audit_valid": all(checks.values()),
        "registered_scientific_gate_recomputed": scientific_gate,
        "gate": {
            "passed": all(checks.values())
            and scientific_gate
            and bool(payload["gate"]["passed"])
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        json.loads(args.confirmation_json.read_text(encoding="utf-8"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["gate"], indent=2))


if __name__ == "__main__":
    main()
