"""Independent replay audit for focused-prior cached h5 Rock trajectories."""

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

from scripts.audit_nonmyopic_range_gated_rock_depth4_fixed_tail import _close
from scripts.nonmyopic_range_gated_rock_depth5_goal_compiler import (
    build_depth5_model,
    compile_target_plan,
    matched_random_goal_plans,
)
from scripts.nonmyopic_range_gated_rock_depth5_oracle import (
    H5_ROUTE,
    _sample_truth_index,
)
from scripts.nonmyopic_range_gated_rock_depth5_trajectory import (
    _belief_key,
    _bootstrap,
)
from scripts.nonmyopic_range_gated_rock_proposal_gate import _stable_seed
from scripts.nonmyopic_range_gated_rock_stable_controls import (
    stable_best_index,
    stable_best_plan,
)
from scripts.nonmyopic_rock_depth_oracle import _uniform, exhaustive_action_values


def _isclose(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def audit(
    payload: dict[str, Any],
    *,
    audit_bootstrap_seed: int = 24_246,
) -> dict[str, Any]:
    if payload.get("stage") != (
        "focused_range_gated_rock_cached_h5_trajectory_confirmation"
    ):
        raise ValueError("unexpected focused h5 trajectory stage")
    config = payload["config"]
    if (
        config["num_trials"] != 50
        or config["num_rounds"] != 8
        or config["bootstrap_replicates"] != 10_000
        or config["max_unique_llm_cells"] != 64
    ):
        raise ValueError("trajectory payload does not use the frozen protocol")
    model = build_depth5_model()
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
    compiler_consistent = True
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
            if physical is not None:
                roots = tuple(physical["roots"])
                targets = tuple(
                    int(value) for value in physical["target_assignments"]
                )
                plans = tuple(
                    compile_target_plan(
                        model,
                        position=tuple(physical["position"]),
                        root=root,
                        target_rock=target,
                    )
                    for root, target in zip(roots, targets, strict=True)
                )
                compiler_consistent &= [list(plan) for plan in plans] == physical[
                    "compiled_plans"
                ]

    checks = {
        "all_cache_keys_are_state_and_plan_consistent": cache_consistent,
        "all_unique_cache_misses_match_physical_requests": (
            misses == len(cache_states) == len(payload["candidate_requests"])
        ),
        "all_target_assignments_recompiled": compiler_consistent,
        "source_prior_recomputed": _close(
            model.prior_good_probabilities.tolist(),
            [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.5, 0.005],
        ),
        "all_histories_positions_and_observations_replayed": True,
        "all_llm_plan_values_and_selections_match": True,
        "all_random_plan_values_and_selections_match": True,
        "all_exact_control_actions_match": True,
        "all_posterior_metrics_match": True,
        "all_aggregate_means_match": True,
        "paired_truths_match": True,
        "no_llm_calls": True,
    }
    traces = payload["traces"]
    replayed: dict[str, list[dict[str, float]]] = {
        arm: [] for arm in traces
    }
    seed = int(config["seed"])
    rounds = int(config["num_rounds"])
    truths = [int(value) for value in payload["truth_indices"]]
    checks["paired_truths_match"] &= truths == [
        _sample_truth_index(model, trial_index=index, seed=seed)
        for index in range(int(config["num_trials"]))
    ]
    exact_cache: dict[
        tuple[int, tuple[int, int], bytes], tuple[str, float]
    ] = {}

    def exact_action(
        position: tuple[int, int],
        belief: np.ndarray,
        depth: int,
    ) -> tuple[str, float]:
        key = (depth, position, _belief_key(belief))
        cached = exact_cache.get(key)
        if cached is not None:
            return cached
        values, _ = exhaustive_action_values(
            model, position=position, belief=belief, depth=depth
        )
        actions = tuple(values)
        action = actions[
            stable_best_index([values[candidate] for candidate in actions])
        ]
        result = (action, float(values[action]))
        exact_cache[key] = result
        return result

    for arm, rows in traces.items():
        checks["paired_truths_match"] &= (
            [int(trace["truth_index"]) for trace in rows] == truths
        )
        for trace in rows:
            trial_index = int(trace["trial_index"])
            truth_index = int(trace["truth_index"])
            position = model.map_spec.start_position
            belief = model.initial_belief.copy()
            history: list[list[Any]] = []
            check_counts: dict[tuple[tuple[int, int], int], int] = {}
            entropies: list[float] = []
            truth_logs: list[float] = []
            for round_index, step in enumerate(trace["steps"]):
                remaining = rounds - round_index
                action = str(step["action"])
                checks["all_histories_positions_and_observations_replayed"] &= (
                    list(position) == step["position_before"]
                    and action in model.legal_actions(position)
                )
                if arm in ("llm_h5", "shared_h4") and remaining >= 5:
                    arm_offset = (
                        0
                        if arm == "llm_h5"
                        else int(config["num_trials"]) * rounds
                    )
                    cell_index = arm_offset + trial_index * rounds + round_index
                    request = logical_by_cell[cell_index]
                    plans = tuple(
                        tuple(plan) for plan in request["compiled_plans"]
                    )
                    scored = (
                        plans
                        if arm == "llm_h5"
                        else tuple(plan[:4] for plan in plans)
                    )
                    selected, value, values = stable_best_plan(
                        model,
                        position=position,
                        belief=belief,
                        plans=scored,
                    )
                    selected_index = scored.index(selected)
                    full_plan = plans[selected_index]
                    policy = step["policy"]
                    checks["all_llm_plan_values_and_selections_match"] &= (
                        request["position"] == list(position)
                        and request["history"] == history
                        and policy["plans"] == [list(plan) for plan in plans]
                        and policy["scored_plans"]
                        == [list(plan) for plan in scored]
                        and policy["selected_full_plan"] == list(full_plan)
                        and np.allclose(
                            policy["plan_values"],
                            values,
                            atol=1e-12,
                            rtol=0.0,
                        )
                        and _isclose(policy["selected_value"], value)
                        and action == full_plan[0]
                    )
                elif arm == "random_h5" and remaining >= 5:
                    targets, plans = matched_random_goal_plans(
                        model,
                        position=position,
                        belief=belief,
                        seed=_stable_seed(
                            seed,
                            "focused-h5-trajectory-random",
                            trial_index,
                            round_index,
                        ),
                    )
                    selected, value, values = stable_best_plan(
                        model, position=position, belief=belief, plans=plans
                    )
                    policy = step["policy"]
                    checks["all_random_plan_values_and_selections_match"] &= (
                        policy["targets"] == list(targets)
                        and policy["plans"] == [list(plan) for plan in plans]
                        and policy["selected_full_plan"] == list(selected)
                        and np.allclose(
                            policy["plan_values"],
                            values,
                            atol=1e-12,
                            rtol=0.0,
                        )
                        and _isclose(policy["selected_value"], value)
                        and action == selected[0]
                    )
                else:
                    requested_depth = (
                        5
                        if arm in ("llm_h5", "random_h5", "exact_d5")
                        else 4
                    )
                    depth = min(requested_depth, remaining)
                    expected_action, expected_value = exact_action(
                        position, belief, depth
                    )
                    checks["all_exact_control_actions_match"] &= (
                        action == expected_action
                        and int(step["policy"]["depth"]) == depth
                        and _isclose(
                            step["policy"]["selected_value"], expected_value
                        )
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
                            "focused-h5-trajectory-observation",
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
    for baseline in ("shared_h4", "random_h5", "exact_d4"):
        entropy = np.asarray(
            [
                base["entropy_auc"] - llm["entropy_auc"]
                for base, llm in zip(
                    replayed[baseline], replayed["llm_h5"], strict=True
                )
            ]
        )
        truth = np.asarray(
            [
                llm["truth_auc"] - base["truth_auc"]
                for base, llm in zip(
                    replayed[baseline], replayed["llm_h5"], strict=True
                )
            ]
        )
        for metric, values in (("entropy", entropy), ("truth_log", truth)):
            name = f"{metric}_auc_gain_vs_{baseline}"
            comparisons[name] = {
                "mean": float(values.mean()),
                "independent_ci95": _bootstrap(
                    values,
                    seed=_stable_seed(
                        audit_bootstrap_seed, "h5-trajectory-audit", name
                    ),
                    replicates=int(config["bootstrap_replicates"]),
                ),
            }
            checks["all_aggregate_means_match"] &= _isclose(
                values.mean(), payload["comparisons"][name]["mean"]
            )

    exact_gain = float(
        np.mean(
            [
                d4["entropy_auc"] - d5["entropy_auc"]
                for d4, d5 in zip(
                    replayed["exact_d4"], replayed["exact_d5"], strict=True
                )
            ]
        )
    )
    recovery = (
        comparisons["entropy_auc_gain_vs_exact_d4"]["mean"] / exact_gain
    )
    route_rate = float(
        np.mean(
            [
                tuple(step["action"] for step in trace["steps"][:5])
                == H5_ROUTE
                for trace in payload["traces"]["llm_h5"]
            ]
        )
    )
    onsite_rate = float(
        np.mean(
            [
                any(
                    step["round"] <= 5
                    and step["action"] == "check-6"
                    and tuple(step["position_before"])
                    == model.map_spec.rock_positions[6]
                    for step in trace["steps"]
                )
                for trace in payload["traces"]["llm_h5"]
            ]
        )
    )
    checks["all_aggregate_means_match"] &= all(
        (
            _isclose(
                exact_gain, payload["exact_h5_entropy_gain_vs_exact_d4"]
            ),
            _isclose(
                recovery,
                payload["llm_recovery_fraction_of_exact_h5_gain"],
            ),
            _isclose(route_rate, payload["llm_registered_route_rate"]),
            _isclose(
                onsite_rate, payload["llm_onsite_by_round_five_rate"]
            ),
        )
    )
    endpoint_gate = {
        f"{metric}_vs_{baseline}_lower_bound_positive": comparisons[
            f"{metric}_auc_gain_vs_{baseline}"
        ]["independent_ci95"][0]
        > 0.0
        for baseline in ("shared_h4", "random_h5", "exact_d4")
        for metric in ("entropy", "truth_log")
    }
    endpoint_gate.update(
        {
            "mean_exact_h5_recovery_at_least_threshold": recovery
            >= float(config["recovery_threshold"]),
            "registered_route_rate_at_least_threshold": route_rate
            >= float(config["route_rate_threshold"]),
            "onsite_by_round_five_at_least_threshold": onsite_rate
            >= float(config["route_rate_threshold"]),
        }
    )
    return {
        "schema_version": 1,
        "stage": "focused_range_gated_rock_cached_h5_trajectory_audit",
        "source_stage": payload["stage"],
        "audit_bootstrap_seed": audit_bootstrap_seed,
        "mechanics": checks,
        "comparisons": comparisons,
        "exact_h5_entropy_gain_vs_exact_d4": exact_gain,
        "llm_recovery_fraction_of_exact_h5_gain": recovery,
        "llm_registered_route_rate": route_rate,
        "llm_onsite_by_round_five_rate": onsite_rate,
        "endpoint_gate": endpoint_gate,
        "gate": {
            "passed": all(checks.values()) and all(endpoint_gate.values())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation_json", type=Path)
    parser.add_argument("--audit-bootstrap-seed", type=int, default=24_246)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(
        args.confirmation_json.read_text(encoding="utf-8")
    )
    result = audit(
        payload, audit_bootstrap_seed=args.audit_bootstrap_seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "AUDIT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate": result["gate"],
                "endpoint_gate": result["endpoint_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
