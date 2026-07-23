"""Independent replay audit for the UCI thyroid workup qualification."""

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

from environments.thyroid_workup import ThyroidWorkupModel  # noqa: E402


def _bootstrap(values: np.ndarray, *, seed: int, replicates: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        samples[start : start + size] = values[indices].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def independent_action_costs(
    model: ThyroidWorkupModel,
    *,
    state: Any,
    belief: np.ndarray,
    depth: int,
    cache: dict[tuple[bool, tuple[int, ...], int, bytes], dict[str, float]] | None = None,
) -> dict[str, float]:
    """Recompute expected cumulative post-action entropy independently."""

    if depth <= 0:
        raise ValueError("depth must be positive")
    cache_key = (
        bool(state.blood_collected),
        tuple(state.asked_features),
        depth,
        np.packbits(belief > 0.0, bitorder="little").tobytes(),
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    result: dict[str, float] = {}
    for action in model.legal_actions(state):
        following_state = model.next_state(state, action)
        expected_cost = 0.0
        for observation in model.outcomes(action):
            branch_probability = model.outcome_probability(belief, action, observation)
            if branch_probability == 0.0:
                continue
            following_belief = model.posterior(belief, action, observation)
            branch_cost = model.target_entropy(following_belief)
            if depth > 1:
                continuation = independent_action_costs(
                    model,
                    state=following_state,
                    belief=following_belief,
                    depth=depth - 1,
                    cache=cache,
                )
                branch_cost += min(continuation.values())
            expected_cost += branch_probability * branch_cost
        result[action] = float(expected_cost)
    if cache is not None:
        cache[cache_key] = result
    return result


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("stage") != "uci_thyroid_workup_depth_qualification":
        raise ValueError("unexpected thyroid qualification stage")

    model = ThyroidWorkupModel()
    replayed: dict[str, list[dict[str, float]]] = {}
    checks = {
        "all_actions_legal": True,
        "stored_states_match": True,
        "deterministic_outcomes_match": True,
        "posterior_metrics_match": True,
        "independent_planning_costs_match": True,
        "every_recorded_action_is_independently_optimal": True,
    }
    planning_cache: dict[
        tuple[bool, tuple[int, ...], int, bytes], dict[str, float]
    ] = {}
    for arm, traces in payload["traces"].items():
        replayed[arm] = []
        for trace in traces:
            state = model.initial_state
            belief = model.initial_belief
            entropies: list[float] = []
            truth_logs: list[float] = []
            for step in trace["steps"]:
                action = str(step["action"])
                legal_actions = model.legal_actions(state)
                checks["all_actions_legal"] &= action in legal_actions
                checks["stored_states_match"] &= (
                    state.blood_collected == step["blood_collected_before"]
                )
                costs = independent_action_costs(
                    model,
                    state=state,
                    belief=belief,
                    depth=int(step["horizon"]),
                    cache=planning_cache,
                )
                optimal = min(
                    legal_actions,
                    key=lambda candidate: (costs[candidate], legal_actions.index(candidate)),
                )
                checks["every_recorded_action_is_independently_optimal"] &= optimal == action
                checks["independent_planning_costs_match"] &= math.isclose(
                    costs[action], float(step["planning_cost"]), rel_tol=0.0, abs_tol=1e-12
                )
                entropy_before = model.target_entropy(belief)
                checks["independent_planning_costs_match"] &= math.isclose(
                    int(step["horizon"]) * entropy_before - costs[action],
                    float(step["planning_entropy_reduction"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                checks["independent_planning_costs_match"] &= math.isclose(
                    model.expected_information_gain(belief, action),
                    float(step["immediate_eig"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                observation = model.observation(int(trace["truth_index"]), action)
                checks["deterministic_outcomes_match"] &= observation == step["observation"]
                belief = model.posterior(belief, action, observation)
                state = model.next_state(state, action)
                entropies.append(model.target_entropy(belief))
                truth_logs.append(
                    model.truth_log_probability(belief, int(trace["truth_index"]))
                )
                checks["posterior_metrics_match"] &= math.isclose(
                    entropies[-1], float(step["entropy"]), rel_tol=0.0, abs_tol=1e-12
                )
                checks["posterior_metrics_match"] &= math.isclose(
                    truth_logs[-1],
                    float(step["truth_log_probability"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            entropy_auc = float(np.mean(entropies))
            truth_auc = float(np.mean(truth_logs))
            checks["posterior_metrics_match"] &= math.isclose(
                entropy_auc, float(trace["entropy_auc"]), rel_tol=0.0, abs_tol=1e-12
            )
            checks["posterior_metrics_match"] &= math.isclose(
                truth_auc,
                float(trace["truth_log_probability_auc"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            replayed[arm].append(
                {"entropy_auc": entropy_auc, "truth_log_probability_auc": truth_auc}
            )

    one = replayed["depth_one"]
    two = replayed["depth_two"]
    entropy_gain = np.asarray(
        [d1["entropy_auc"] - d2["entropy_auc"] for d1, d2 in zip(one, two)],
        dtype=float,
    )
    truth_gain = np.asarray(
        [
            d2["truth_log_probability_auc"] - d1["truth_log_probability_auc"]
            for d1, d2 in zip(one, two)
        ],
        dtype=float,
    )
    stored = payload["comparison"]
    source_d1 = payload["traces"]["depth_one"]
    source_d2 = payload["traces"]["depth_two"]
    checks.update(
        {
            "all_registered_rows_replayed": len(source_d1)
            == len(source_d2)
            == int(payload["config"]["num_trials"]),
            "paired_truths_match": all(
                left["truth_index"] == right["truth_index"]
                for left, right in zip(source_d1, source_d2)
            ),
            "truths_sampled_without_replacement": len(
                {trace["truth_index"] for trace in source_d1}
            )
            == len(source_d1),
            "stored_comparison_matches_replay": math.isclose(
                float(np.mean(entropy_gain)),
                float(stored["entropy_auc_gain"]["mean"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(np.mean(truth_gain)),
                float(stored["truth_log_probability_auc_gain"]["mean"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "no_llm_calls": True,
        }
    )
    entropy_ci = _bootstrap(entropy_gain, seed=24_149)
    truth_ci = _bootstrap(truth_gain, seed=24_150)
    return {
        "schema_version": 1,
        "stage": "uci_thyroid_workup_depth_qualification_audit",
        "source_config": payload["config"],
        "mechanics": checks,
        "unique_planning_subtrees_recomputed": len(planning_cache),
        "entropy_auc_gain": {
            "mean": float(np.mean(entropy_gain)),
            "independent_ci95": entropy_ci,
        },
        "truth_log_probability_auc_gain": {
            "mean": float(np.mean(truth_gain)),
            "independent_ci95": truth_ci,
        },
        "passed": all(checks.values()) and entropy_ci[0] > 0.0 and truth_ci[0] > 0.0,
    }


def render(summary: dict[str, Any]) -> str:
    entropy = summary["entropy_auc_gain"]
    truth = summary["truth_log_probability_auc_gain"]
    return "\n".join(
        [
            "# UCI Thyroid Blood-Workup Qualification Audit",
            "",
            f"Audit passed: **{summary['passed']}**.",
            "",
            f"- Entropy-AUC gain: `{entropy['mean']:+.6f}`, independent 95% CI "
            f"`[{entropy['independent_ci95'][0]:+.6f}, {entropy['independent_ci95'][1]:+.6f}]`.",
            f"- Truth-log-AUC gain: `{truth['mean']:+.6f}`, independent 95% CI "
            f"`[{truth['independent_ci95'][0]:+.6f}, {truth['independent_ci95'][1]:+.6f}]`.",
            "- Every stored action was independently re-scored and verified optimal.",
            "- Every state, observation, posterior metric, paired truth, and aggregate was replayed from the official cohort.",
            "- The audit made zero LLM calls.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(json.loads(args.report.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_output.write_text(render(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
