"""Independent replay audit for the Mushroom trajectory confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.mushroom_feature_acquisition import (  # noqa: E402
    COLLECT_ACTION,
    FIELD_FEATURES,
    MushroomFeatureModel,
)
from environments.mushroom_feature_acquisition.model import EPSILON  # noqa: E402


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _bootstrap(
    values: np.ndarray, *, seed: int, replicates: int = 10_000
) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        samples[start : start + size] = values[indices].mean(axis=1)
    return [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def _roots(
    model: MushroomFeatureModel, state: Any, belief: np.ndarray
) -> tuple[str, ...]:
    legal = model.legal_actions(state)
    count = min(4, len(legal))
    queries = [action for action in legal if action != COLLECT_ACTION]
    queries.sort(
        key=lambda action: (
            model.expected_target_entropy(belief, action),
            legal.index(action),
        )
    )
    setup = (COLLECT_ACTION,) if COLLECT_ACTION in legal else ()
    return (*setup, *queries[: count - len(setup)])


def _menus(
    model: MushroomFeatureModel,
    *,
    state: Any,
    belief: np.ndarray,
    roots: tuple[str, ...],
) -> list[dict[str, tuple[str, ...]]] | None:
    menus = []
    for root in roots:
        child_state = model.next_state(state, root)
        choices = tuple(
            action
            for action in model.legal_actions(child_state)
            if action.startswith("query:")
        )
        if root == COLLECT_ACTION:
            choices = tuple(
                action
                for action in choices
                if model.action_feature(action) not in FIELD_FEATURES
            )
        if not choices:
            return None
        menus.append(
            {
                "none" if outcome is None else str(outcome): choices
                for outcome in model.outcomes(root)
                if model.outcome_probability(belief, root, outcome) > EPSILON
            }
        )
    return menus


def _policy_cost(
    model: MushroomFeatureModel,
    state: Any,
    belief: np.ndarray,
    root: str,
    followups: dict[str, str],
) -> float:
    child_state = model.next_state(state, root)
    total = 0.0
    for outcome in model.outcomes(root):
        probability = model.outcome_probability(belief, root, outcome)
        if probability <= EPSILON:
            continue
        posterior = model.posterior(belief, root, outcome)
        key = "none" if outcome is None else str(outcome)
        followup = followups[key]
        if followup not in model.legal_actions(child_state):
            raise ValueError(f"illegal audited follow-up {followup}")
        total += probability * (
            model.target_entropy(posterior)
            + model.expected_target_entropy(posterior, followup)
        )
    return float(total)


def _exact_costs(
    model: MushroomFeatureModel,
    *,
    state: Any,
    belief: np.ndarray,
    depth: int,
) -> dict[str, float]:
    costs = {}
    for action in model.legal_actions(state):
        child_state = model.next_state(state, action)
        total = 0.0
        for outcome in model.outcomes(action):
            probability = model.outcome_probability(belief, action, outcome)
            if probability <= EPSILON:
                continue
            posterior = model.posterior(belief, action, outcome)
            branch = model.target_entropy(posterior)
            if depth > 1:
                branch += min(
                    _exact_costs(
                        model,
                        state=child_state,
                        belief=posterior,
                        depth=depth - 1,
                    ).values()
                )
            total += probability * branch
        costs[action] = total
    return costs


def _choose(costs: dict[str, float]) -> str:
    legal = tuple(costs)
    return min(legal, key=lambda action: (costs[action], legal.index(action)))


def _random_followups(
    *,
    menus: list[dict[str, tuple[str, ...]]],
    seed: int,
) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    return [
        {
            outcome: choices[int(rng.integers(0, len(choices)))]
            for outcome, choices in root_menus.items()
        }
        for root_menus in menus
    ]


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("stage") != (
        "mushroom_feature_acquisition_projected_utility_paired_trajectory_confirmation"
    ):
        raise ValueError("unexpected Mushroom confirmation stage")
    model = MushroomFeatureModel()
    checks = {
        "all_histories_states_and_observations_replayed": True,
        "all_posterior_metrics_match": True,
        "all_llm_policies_and_selections_match": True,
        "all_random_policies_and_selections_match": True,
        "all_exact_control_actions_match": True,
        "all_aggregate_metrics_match": True,
        "paired_truths_distinct_and_match": True,
        "all_projections_are_exact_legal_minima": True,
        "projection_aggregates_match": True,
        "no_llm_calls": True,
    }
    requests = {
        int(request["cell_index"]): request
        for request in payload["candidate_requests"]
    }
    projection = {
        "projected_cells": len(payload["projected_responses"]),
        "projected_cell_rate": len(payload["projected_responses"])
        / len(payload["candidate_requests"]),
        "projected_branches": sum(
            len(request["projection_events"])
            for request in payload["projected_responses"]
        ),
        "total_branches": sum(
            sum(len(root_menus) for root_menus in request["menus"])
            for request in payload["candidate_requests"]
        ),
        "projected_branch_rate": 0.0,
        "max_expected_entropy_excess": 0.0,
    }
    projection["projected_branch_rate"] = (
        projection["projected_branches"] / projection["total_branches"]
    )
    for request in payload["projected_responses"]:
        state = model.initial_state
        belief = model.initial_belief.copy()
        for item in request["history"]:
            action = item["action"]
            outcome = item["outcome"]
            belief = model.posterior(belief, action, outcome)
            state = model.next_state(state, action)
        for event in request["projection_events"]:
            slot = int(event["slot"])
            outcome = event["outcome"]
            raw_outcome = None if outcome == "none" else outcome
            posterior = model.posterior(belief, event["root"], raw_outcome)
            choices = request["menus"][slot][outcome]
            entropies = [
                model.expected_target_entropy(posterior, choice)
                for choice in choices
            ]
            minimum = min(entropies)
            best_index = next(
                index
                for index, value in enumerate(entropies)
                if value <= minimum + EPSILON
            )
            replacement_index = int(event["replacement_index"])
            excess = entropies[replacement_index] - minimum
            projection["max_expected_entropy_excess"] = max(
                projection["max_expected_entropy_excess"], excess
            )
            checks["all_projections_are_exact_legal_minima"] &= (
                replacement_index == best_index
                and event["replacement"] == choices[best_index]
                and excess <= EPSILON
                and event["replacement"]
                in model.legal_actions(model.next_state(state, event["root"]))
            )
    checks["projection_aggregates_match"] &= all(
        math.isclose(
            float(projection[key]),
            float(payload["projection"][key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for key in payload["projection"]
    )

    replayed: dict[str, list[dict[str, float]]] = {
        arm: [] for arm in payload["traces"]
    }
    for arm, traces in payload["traces"].items():
        for trace in traces:
            state = model.initial_state
            belief = model.initial_belief.copy()
            truth = int(trace["truth_index"])
            history: list[dict[str, Any]] = []
            entropies = []
            truth_logs = []
            for round_index, step in enumerate(trace["steps"]):
                action = str(step["action"])
                checks["all_histories_states_and_observations_replayed"] &= (
                    state.specimen_collected
                    == step["specimen_collected_before"]
                    and action in model.legal_actions(state)
                    and model.observation(truth, action) == step["observation"]
                )
                remaining = int(payload["config"]["num_rounds"]) - round_index
                handled = False
                if arm in ("llm", "random") and remaining > 1:
                    roots = _roots(model, state, belief)
                    menus = _menus(
                        model, state=state, belief=belief, roots=roots
                    )
                    policy = step["policy"]
                    if arm == "random" and menus is None:
                        costs = _exact_costs(
                            model, state=state, belief=belief, depth=1
                        )
                        checks["all_random_policies_and_selections_match"] &= (
                            policy.get("exact_depth") == 1
                            and _choose(costs) == action
                        )
                        handled = True
                    else:
                        if menus is None:
                            checks["all_llm_policies_and_selections_match"] = False
                            followups = []
                        elif arm == "llm":
                            request = requests[
                                int(trace["trial_index"])
                                * int(payload["config"]["num_rounds"])
                                + round_index
                            ]
                            followups = [
                                item["followups"]
                                for item in request["compiled_strategies"]
                            ]
                            checks["all_llm_policies_and_selections_match"] &= (
                                request["history"] == history
                                and [
                                    item["root_action"]
                                    for item in request["compiled_strategies"]
                                ]
                                == list(roots)
                                and followups == policy["followups"]
                            )
                        else:
                            followups = _random_followups(
                                menus=menus,
                                seed=_stable_seed(
                                    payload["config"]["seed"],
                                    "trajectory-random",
                                    trace["trial_index"],
                                    round_index,
                                ),
                            )
                            checks[
                                "all_random_policies_and_selections_match"
                            ] &= followups == policy["followups"]
                        if menus is not None:
                            costs = [
                                _policy_cost(
                                    model,
                                    state,
                                    belief,
                                    root,
                                    branch_map,
                                )
                                for root, branch_map in zip(
                                    roots, followups, strict=True
                                )
                            ]
                            slot = min(
                                range(len(roots)),
                                key=lambda index: (costs[index], index),
                            )
                            checks[
                                f"all_{arm}_policies_and_selections_match"
                            ] &= (
                                list(roots) == policy["roots"]
                                and np.allclose(
                                    costs,
                                    policy["policy_costs"],
                                    rtol=0.0,
                                    atol=1e-12,
                                )
                                and slot == policy["selected_slot"]
                                and roots[slot] == action
                            )
                            handled = True
                if not handled:
                    depth = min(2 if arm == "depth_two" else 1, remaining)
                    costs = _exact_costs(
                        model, state=state, belief=belief, depth=depth
                    )
                    checks["all_exact_control_actions_match"] &= (
                        _choose(costs) == action
                    )
                belief = model.posterior(belief, action, step["observation"])
                state = model.next_state(state, action)
                history.append({"action": action, "outcome": step["observation"]})
                entropies.append(model.target_entropy(belief))
                truth_logs.append(model.truth_log_probability(belief, truth))
                checks["all_posterior_metrics_match"] &= math.isclose(
                    entropies[-1],
                    step["entropy"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ) and math.isclose(
                    truth_logs[-1],
                    step["truth_log_probability"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            entropy_auc = float(np.mean(entropies))
            truth_auc = float(np.mean(truth_logs))
            checks["all_posterior_metrics_match"] &= math.isclose(
                entropy_auc,
                trace["entropy_auc"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ) and math.isclose(
                truth_auc,
                trace["truth_log_probability_auc"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            replayed[arm].append(
                {"entropy_auc": entropy_auc, "truth_auc": truth_auc}
            )

    truths = payload["truth_indices"]
    checks["paired_truths_distinct_and_match"] &= (
        len(set(truths)) == len(truths) == 50
    )
    checks["paired_truths_distinct_and_match"] &= all(
        [trace["truth_index"] for trace in traces] == truths
        for traces in payload["traces"].values()
    )
    comparisons: dict[str, dict[str, Any]] = {}
    seeds = {
        "entropy_auc_gain_vs_depth_one": 24_176,
        "truth_log_auc_gain_vs_depth_one": 24_177,
        "entropy_auc_gain_vs_random": 24_178,
        "truth_log_auc_gain_vs_random": 24_179,
    }
    for baseline, suffix in (("depth_one", "depth_one"), ("random", "random")):
        entropy = np.asarray(
            [
                base["entropy_auc"] - llm["entropy_auc"]
                for base, llm in zip(replayed[baseline], replayed["llm"])
            ]
        )
        truth = np.asarray(
            [
                llm["truth_auc"] - base["truth_auc"]
                for base, llm in zip(replayed[baseline], replayed["llm"])
            ]
        )
        for metric, values in (
            (f"entropy_auc_gain_vs_{suffix}", entropy),
            (f"truth_log_auc_gain_vs_{suffix}", truth),
        ):
            comparisons[metric] = {
                "mean": float(values.mean()),
                "independent_ci95": _bootstrap(values, seed=seeds[metric]),
            }
            checks["all_aggregate_metrics_match"] &= math.isclose(
                values.mean(),
                payload["comparisons"][metric]["mean"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
    exact_gain = float(
        np.mean(
            [
                d1["entropy_auc"] - d2["entropy_auc"]
                for d1, d2 in zip(
                    replayed["depth_one"], replayed["depth_two"]
                )
            ]
        )
    )
    recovery = (
        comparisons["entropy_auc_gain_vs_depth_one"]["mean"] / exact_gain
    )
    collection_rate = float(
        np.mean(
            [
                trace["steps"][0]["action"] == COLLECT_ACTION
                for trace in payload["traces"]["llm"]
            ]
        )
    )
    scientific_gate = (
        comparisons["entropy_auc_gain_vs_depth_one"]["independent_ci95"][0] > 0.0
        and comparisons["truth_log_auc_gain_vs_depth_one"]["independent_ci95"][0]
        > 0.0
        and comparisons["entropy_auc_gain_vs_random"]["independent_ci95"][0] > 0.0
        and comparisons["truth_log_auc_gain_vs_random"]["independent_ci95"][0]
        > 0.0
        and recovery >= 0.60
        and collection_rate >= 0.75
        and projection["projected_cell_rate"]
        <= float(payload["config"]["projection_cell_rate_threshold"])
        and projection["projected_branch_rate"]
        <= float(payload["config"]["projection_branch_rate_threshold"])
    )
    return {
        "schema_version": 1,
        "stage": "mushroom_projected_utility_confirmation_audit",
        "mechanics": checks,
        "comparisons": comparisons,
        "exact_depth_two_entropy_gain_vs_depth_one": exact_gain,
        "llm_recovery_fraction_of_exact_depth_two_gain": recovery,
        "llm_first_collection_rate": collection_rate,
        "projection": projection,
        "audit_valid": all(checks.values()),
        "registered_scientific_gate_recomputed": scientific_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.confirmation.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
