"""Independent replay audit for the UCI thyroid named-proposal gate."""

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

from environments.thyroid_workup import (  # noqa: E402
    COLLECT_BLOOD_ACTION,
    ThyroidWorkupModel,
)
from scripts.audit_nonmyopic_thyroid_workup_oracle import (  # noqa: E402
    independent_action_costs,
)


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _bootstrap(values: np.ndarray, *, seed: int, replicates: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        samples[start : start + size] = values[indices].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _fixed_roots(model: ThyroidWorkupModel, state: Any, belief: np.ndarray) -> tuple[str, ...]:
    legal = model.legal_actions(state)
    queries = [action for action in legal if action.startswith("query:")]
    queries.sort(
        key=lambda action: (model.expected_target_entropy(belief, action), legal.index(action))
    )
    setup = (COLLECT_BLOOD_ACTION,) if COLLECT_BLOOD_ACTION in legal else ()
    return (*setup, *queries[: 4 - len(setup)])


def _policy_cost(
    model: ThyroidWorkupModel,
    *,
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
            raise ValueError(f"illegal audited follow-up {followup} under {root}")
        total += probability * (
            model.target_entropy(posterior)
            + model.expected_target_entropy(posterior, followup)
        )
    return float(total)


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("stage") != "uci_thyroid_workup_26b_named_proposal_gate":
        raise ValueError("unexpected thyroid proposal stage")
    model = ThyroidWorkupModel()
    checks = {
        "all_histories_replayed": True,
        "all_stored_observations_match_truth": True,
        "all_roots_independently_match": True,
        "all_named_followups_legal_and_complete": True,
        "all_llm_policy_costs_match": True,
        "all_random_controls_match": True,
        "all_exact_controls_match": True,
        "all_selected_slots_and_roots_match": True,
        "all_cells_are_positive_collection_opportunities": True,
        "all_histories_distinct": True,
        "no_llm_calls": True,
    }
    random_minus_llm: list[float] = []
    shared_minus_llm: list[float] = []
    recovery: list[float] = []
    collection_selected: list[bool] = []
    histories = []
    planning_cache: dict[tuple[bool, tuple[int, ...], int, bytes], dict[str, float]] = {}
    for record in payload["records"]:
        state = model.initial_state
        belief = model.initial_belief
        truth = int(record["truth_index"])
        history = tuple((str(action), outcome) for action, outcome in record["history"])
        histories.append(history)
        for action, stored_outcome in history:
            checks["all_histories_replayed"] &= action in model.legal_actions(state)
            outcome = model.observation(truth, action)
            checks["all_stored_observations_match_truth"] &= outcome == stored_outcome
            belief = model.posterior(belief, action, outcome)
            state = model.next_state(state, action)

        roots = _fixed_roots(model, state, belief)
        checks["all_roots_independently_match"] &= list(roots) == record["roots"]
        llm_costs = []
        for root, followups in zip(roots, record["llm_followups"], strict=True):
            expected_keys = {
                "none" if outcome is None else str(outcome)
                for outcome in model.outcomes(root)
                if model.outcome_probability(belief, root, outcome) > 0.0
            }
            checks["all_named_followups_legal_and_complete"] &= (
                set(followups) == expected_keys
            )
            llm_costs.append(
                _policy_cost(
                    model,
                    state=state,
                    belief=belief,
                    root=root,
                    followups=followups,
                )
            )
        checks["all_llm_policy_costs_match"] &= np.allclose(
            llm_costs, record["llm_costs"], rtol=0.0, atol=1e-12
        )
        llm_slot = min(range(4), key=lambda index: (llm_costs[index], index))

        rng = np.random.default_rng(
            _stable_seed(payload["config"]["seed"], "matched-random", record["cell_index"])
        )
        random_costs = []
        for root in roots:
            choices = tuple(model.legal_actions(model.next_state(state, root)))
            followups = {
                "none" if outcome is None else str(outcome): choices[
                    int(rng.integers(0, len(choices)))
                ]
                for outcome in model.outcomes(root)
                if model.outcome_probability(belief, root, outcome) > 0.0
            }
            random_costs.append(
                _policy_cost(
                    model,
                    state=state,
                    belief=belief,
                    root=root,
                    followups=followups,
                )
            )
        checks["all_random_controls_match"] &= np.allclose(
            random_costs, record["random_costs"], rtol=0.0, atol=1e-12
        )
        random_slot = min(range(4), key=lambda index: (random_costs[index], index))

        d1 = independent_action_costs(
            model, state=state, belief=belief, depth=1, cache=planning_cache
        )
        d2 = independent_action_costs(
            model, state=state, belief=belief, depth=2, cache=planning_cache
        )
        legal = model.legal_actions(state)
        d1_root = min(legal, key=lambda action: (d1[action], legal.index(action)))
        d2_root = min(legal, key=lambda action: (d2[action], legal.index(action)))
        llm_cost = llm_costs[llm_slot]
        random_cost = random_costs[random_slot]
        shared_cost = d2[d1_root]
        exhaustive_cost = d2[d2_root]
        opportunity = shared_cost - exhaustive_cost
        fraction = (shared_cost - llm_cost) / opportunity
        checks["all_exact_controls_match"] &= all(
            [
                d1_root == record["shared_d1_root"],
                d2_root == record["exhaustive_d2_root"],
                math.isclose(shared_cost, record["shared_d1_exact_continuation_cost"], abs_tol=1e-12),
                math.isclose(exhaustive_cost, record["exhaustive_d2_cost"], abs_tol=1e-12),
                math.isclose(opportunity, record["d2_opportunity"], abs_tol=1e-12),
                math.isclose(fraction, record["recovery_fraction"], abs_tol=1e-12),
            ]
        )
        checks["all_selected_slots_and_roots_match"] &= all(
            [
                llm_slot == record["llm_selected_slot"],
                roots[llm_slot] == record["llm_selected_root"],
                random_slot == record["random_selected_slot"],
                math.isclose(llm_cost, record["llm_cost"], abs_tol=1e-12),
                math.isclose(random_cost, record["random_cost"], abs_tol=1e-12),
            ]
        )
        checks["all_cells_are_positive_collection_opportunities"] &= (
            d2_root == COLLECT_BLOOD_ACTION and opportunity > 0.0
        )
        random_minus_llm.append(random_cost - llm_cost)
        shared_minus_llm.append(shared_cost - llm_cost)
        recovery.append(fraction)
        collection_selected.append(roots[llm_slot] == COLLECT_BLOOD_ACTION)

    checks["all_histories_distinct"] &= len(set(histories)) == len(histories) == 32
    random_values = np.asarray(random_minus_llm)
    shared_values = np.asarray(shared_minus_llm)
    recovery_values = np.asarray(recovery)
    random_ci = _bootstrap(random_values, seed=24_153)
    shared_ci = _bootstrap(shared_values, seed=24_154)
    recovery_ci = _bootstrap(recovery_values, seed=24_155)
    collection_rate = float(np.mean(collection_selected))
    summary = {
        "schema_version": 1,
        "stage": "uci_thyroid_workup_26b_named_proposal_gate_audit",
        "mechanics": checks,
        "unique_planning_subtrees_recomputed": len(planning_cache),
        "matched_random_minus_llm_cost": {
            "mean": float(np.mean(random_values)),
            "independent_ci95": random_ci,
        },
        "shared_d1_minus_llm_cost": {
            "mean": float(np.mean(shared_values)),
            "independent_ci95": shared_ci,
        },
        "recovery_fraction": {
            "mean": float(np.mean(recovery_values)),
            "independent_ci95": recovery_ci,
        },
        "collection_selection_rate": collection_rate,
    }
    summary["passed"] = (
        all(checks.values())
        and random_ci[0] > 0.0
        and shared_ci[0] > 0.0
        and float(np.mean(recovery_values)) >= 0.60
        and collection_rate >= 0.75
    )
    return summary


def render(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# UCI Thyroid 26B Named-Proposal Gate Audit",
            "",
            f"Audit passed: **{summary['passed']}**.",
            "",
            f"- Matched-random-minus-LLM: `{summary['matched_random_minus_llm_cost']['mean']:+.6f}`, independent 95% CI `{summary['matched_random_minus_llm_cost']['independent_ci95']}`.",
            f"- Shared-d1-minus-LLM: `{summary['shared_d1_minus_llm_cost']['mean']:+.6f}`, independent 95% CI `{summary['shared_d1_minus_llm_cost']['independent_ci95']}`.",
            f"- Recovery: `{summary['recovery_fraction']['mean']:.3%}`; collection selection: `{summary['collection_selection_rate']:.1%}`.",
            "- Every history, named branch policy, matched-random draw, exact control, and selected root was independently replayed.",
            "- The audit made zero LLM calls.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(json.loads(args.gate.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_output.write_text(render(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
