"""Independent replay audit for thyroid paired trajectory confirmations."""

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

from environments.thyroid_workup import COLLECT_BLOOD_ACTION, ThyroidWorkupModel  # noqa: E402
from scripts.audit_nonmyopic_thyroid_workup_oracle import independent_action_costs  # noqa: E402


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _bootstrap(values: np.ndarray, *, seed: int, replicates: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        samples[start : start + size] = values[indices].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _roots(model: ThyroidWorkupModel, state: Any, belief: np.ndarray) -> tuple[str, ...]:
    legal = model.legal_actions(state)
    queries = [action for action in legal if action.startswith("query:")]
    queries.sort(
        key=lambda action: (model.expected_target_entropy(belief, action), legal.index(action))
    )
    setup = (COLLECT_BLOOD_ACTION,) if COLLECT_BLOOD_ACTION in legal else ()
    return (*setup, *queries[: 4 - len(setup)])


def _policy_cost(
    model: ThyroidWorkupModel,
    state: Any,
    belief: np.ndarray,
    root: str,
    followups: dict[str, str],
) -> float:
    child_state = model.next_state(state, root)
    total = 0.0
    for outcome in model.outcomes(root):
        probability = model.outcome_probability(belief, root, outcome)
        if probability <= 0.0:
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


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("stage") not in {
        "uci_thyroid_workup_26b_paired_trajectory_confirmation",
        "uci_thyroid_workup_utility_grounded_paired_trajectory_confirmation",
    }:
        raise ValueError("unexpected thyroid confirmation stage")
    model = ThyroidWorkupModel()
    checks = {
        "all_histories_and_observations_replayed": True,
        "all_posterior_metrics_match": True,
        "all_llm_policies_and_selections_match": True,
        "all_random_policies_and_selections_match": True,
        "all_exact_control_actions_match": True,
        "all_aggregate_metrics_match": True,
        "paired_truths_distinct_and_match": True,
        "no_llm_calls": True,
    }
    replayed: dict[str, list[dict[str, float]]] = {arm: [] for arm in payload["traces"]}
    cache: dict[tuple[bool, tuple[int, ...], int, bytes], dict[str, float]] = {}
    for arm, traces in payload["traces"].items():
        for trace in traces:
            state = model.initial_state
            belief = model.initial_belief
            truth = int(trace["truth_index"])
            entropies = []
            truth_logs = []
            for round_index, step in enumerate(trace["steps"]):
                action = str(step["action"])
                checks["all_histories_and_observations_replayed"] &= (
                    state.blood_collected == step["blood_collected_before"]
                    and action in model.legal_actions(state)
                    and model.observation(truth, action) == step["observation"]
                )
                remaining = int(payload["config"]["num_rounds"]) - round_index
                if arm in ("llm", "random") and remaining > 1:
                    policy = step["policy"]
                    roots = _roots(model, state, belief)
                    checks[f"all_{arm}_policies_and_selections_match"] &= list(roots) == policy[
                        "roots"
                    ]
                    if arm == "llm":
                        followups = policy["followups"]
                    else:
                        rng = np.random.default_rng(
                            _stable_seed(
                                payload["config"]["seed"],
                                "trajectory-random",
                                trace["trial_index"],
                                round_index,
                            )
                        )
                        followups = []
                        for root in roots:
                            choices = tuple(model.legal_actions(model.next_state(state, root)))
                            followups.append(
                                {
                                    "none" if outcome is None else str(outcome): choices[
                                        int(rng.integers(0, len(choices)))
                                    ]
                                    for outcome in model.outcomes(root)
                                    if model.outcome_probability(belief, root, outcome) > 0.0
                                }
                            )
                        checks["all_random_policies_and_selections_match"] &= followups == policy[
                            "followups"
                        ]
                    costs = [
                        _policy_cost(model, state, belief, root, branch_map)
                        for root, branch_map in zip(roots, followups, strict=True)
                    ]
                    slot = min(range(4), key=lambda index: (costs[index], index))
                    checks[f"all_{arm}_policies_and_selections_match"] &= (
                        np.allclose(costs, policy["policy_costs"], rtol=0.0, atol=1e-12)
                        and slot == policy["selected_slot"]
                        and roots[slot] == action
                    )
                else:
                    depth = min(2 if arm == "depth_two" else 1, remaining)
                    costs = independent_action_costs(
                        model, state=state, belief=belief, depth=depth, cache=cache
                    )
                    legal = model.legal_actions(state)
                    selected = min(
                        legal, key=lambda candidate: (costs[candidate], legal.index(candidate))
                    )
                    checks["all_exact_control_actions_match"] &= selected == action
                belief = model.posterior(belief, action, step["observation"])
                state = model.next_state(state, action)
                entropies.append(model.target_entropy(belief))
                truth_logs.append(model.truth_log_probability(belief, truth))
                checks["all_posterior_metrics_match"] &= math.isclose(
                    entropies[-1], step["entropy"], rel_tol=0.0, abs_tol=1e-12
                ) and math.isclose(
                    truth_logs[-1], step["truth_log_probability"], rel_tol=0.0, abs_tol=1e-12
                )
            entropy_auc = float(np.mean(entropies))
            truth_auc = float(np.mean(truth_logs))
            checks["all_posterior_metrics_match"] &= math.isclose(
                entropy_auc, trace["entropy_auc"], rel_tol=0.0, abs_tol=1e-12
            ) and math.isclose(
                truth_auc, trace["truth_log_probability_auc"], rel_tol=0.0, abs_tol=1e-12
            )
            replayed[arm].append({"entropy_auc": entropy_auc, "truth_auc": truth_auc})

    truths = payload["truth_indices"]
    checks["paired_truths_distinct_and_match"] &= len(set(truths)) == len(truths) == 50
    checks["paired_truths_distinct_and_match"] &= all(
        [trace["truth_index"] for trace in traces] == truths
        for traces in payload["traces"].values()
    )
    comparisons: dict[str, dict[str, Any]] = {}
    for baseline, suffix, entropy_seed, truth_seed in (
        ("depth_one", "depth_one", 24_159, 24_160),
        ("random", "random", 24_161, 24_162),
    ):
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
        comparisons[f"entropy_auc_gain_vs_{suffix}"] = {
            "mean": float(entropy.mean()),
            "independent_ci95": _bootstrap(entropy, seed=entropy_seed),
        }
        comparisons[f"truth_log_auc_gain_vs_{suffix}"] = {
            "mean": float(truth.mean()),
            "independent_ci95": _bootstrap(truth, seed=truth_seed),
        }
        checks["all_aggregate_metrics_match"] &= math.isclose(
            entropy.mean(),
            payload["comparisons"][f"entropy_auc_gain_vs_{suffix}"]["mean"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) and math.isclose(
            truth.mean(),
            payload["comparisons"][f"truth_log_auc_gain_vs_{suffix}"]["mean"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    exact_gain = float(
        np.mean(
            [
                d1["entropy_auc"] - d2["entropy_auc"]
                for d1, d2 in zip(replayed["depth_one"], replayed["depth_two"])
            ]
        )
    )
    recovery = comparisons["entropy_auc_gain_vs_depth_one"]["mean"] / exact_gain
    collection = float(
        np.mean(
            [trace["steps"][0]["action"] == COLLECT_BLOOD_ACTION for trace in payload["traces"]["llm"]]
        )
    )
    scientific_gate = (
        comparisons["entropy_auc_gain_vs_depth_one"]["independent_ci95"][0] > 0.0
        and comparisons["truth_log_auc_gain_vs_depth_one"]["independent_ci95"][0] > 0.0
        and comparisons["entropy_auc_gain_vs_random"]["independent_ci95"][0] > 0.0
        and comparisons["truth_log_auc_gain_vs_random"]["independent_ci95"][0] > 0.0
        and recovery >= 0.60
        and collection >= 0.75
    )
    return {
        "schema_version": 1,
        "stage": "uci_thyroid_workup_paired_trajectory_confirmation_audit",
        "mechanics": checks,
        "unique_exact_subtrees_recomputed": len(cache),
        "comparisons": comparisons,
        "exact_depth_two_entropy_gain_vs_depth_one": exact_gain,
        "llm_recovery_fraction_of_exact_depth_two_gain": recovery,
        "llm_first_collection_rate": collection,
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
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
