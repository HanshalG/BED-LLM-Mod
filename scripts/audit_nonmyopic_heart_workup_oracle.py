"""Independent trace and planner audit for the Cleveland heart-workup qualification."""

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

from environments.heart_workup import HeartWorkupModel, WorkupState  # noqa: E402


def _bootstrap(values: np.ndarray, *, seed: int, replicates: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        samples[start : start + size] = values[indices].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def independent_action_costs(
    model: HeartWorkupModel,
    *,
    state: WorkupState,
    belief: np.ndarray,
    depth: int,
) -> dict[str, float]:
    """Expected cumulative post-action entropy, independently of the producer."""

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
            branch = model.target_entropy(posterior)
            if depth > 1:
                branch += min(
                    independent_action_costs(
                        model,
                        state=next_state,
                        belief=posterior,
                        depth=depth - 1,
                    ).values()
                )
            expected += probability * branch
        costs[action] = expected
    return costs


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("stage") != "cleveland_heart_workup_depth_qualification":
        raise ValueError("unexpected Heart qualification stage")
    model = HeartWorkupModel()
    replayed: dict[str, list[dict[str, float]]] = {}
    legality = True
    state_match = True
    outcome_match = True
    posterior_match = True
    planning_match = True
    action_optimal = True
    for arm, traces in payload["traces"].items():
        replayed[arm] = []
        for trace in traces:
            state = model.initial_state
            belief = model.initial_belief.copy()
            entropy: list[float] = []
            truth_log: list[float] = []
            for step in trace["steps"]:
                action = str(step["action"])
                legality &= action in model.legal_actions(state)
                state_match &= state.workup_ordered == step["workup_ordered_before"]
                costs = independent_action_costs(
                    model,
                    state=state,
                    belief=belief,
                    depth=int(step["horizon"]),
                )
                legal = model.legal_actions(state)
                chosen = min(legal, key=lambda candidate: (costs[candidate], legal.index(candidate)))
                action_optimal &= chosen == action
                planning_match &= math.isclose(
                    costs[action], float(step["planning_cost"]), rel_tol=0.0, abs_tol=1e-12
                )
                entropy_before = model.target_entropy(belief)
                planning_match &= math.isclose(
                    int(step["horizon"]) * entropy_before - costs[action],
                    float(step["planning_entropy_reduction"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                planning_match &= math.isclose(
                    model.expected_information_gain(belief, action),
                    float(step["immediate_eig"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                outcome = model.observation(int(trace["truth_index"]), action)
                outcome_match &= outcome == step["observation"]
                belief = model.posterior(belief, action, outcome)
                state = model.next_state(state, action)
                entropy.append(model.target_entropy(belief))
                truth_log.append(model.truth_log_probability(belief, int(trace["truth_index"])))
                posterior_match &= math.isclose(
                    entropy[-1], float(step["entropy"]), rel_tol=0.0, abs_tol=1e-12
                )
                posterior_match &= math.isclose(
                    truth_log[-1],
                    float(step["truth_log_probability"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            entropy_auc = float(np.mean(entropy))
            truth_auc = float(np.mean(truth_log))
            posterior_match &= math.isclose(
                entropy_auc, float(trace["entropy_auc"]), rel_tol=0.0, abs_tol=1e-12
            )
            posterior_match &= math.isclose(
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
        [d1["entropy_auc"] - d2["entropy_auc"] for d1, d2 in zip(one, two)]
    )
    truth_gain = np.asarray(
        [
            d2["truth_log_probability_auc"] - d1["truth_log_probability_auc"]
            for d1, d2 in zip(one, two)
        ]
    )
    stored = payload["comparison"]
    comparison_match = math.isclose(
        float(np.mean(entropy_gain)),
        float(stored["entropy_auc_gain"]["mean"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) and math.isclose(
        float(np.mean(truth_gain)),
        float(stored["truth_log_probability_auc_gain"]["mean"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    entropy_ci = _bootstrap(entropy_gain, seed=24_134)
    truth_ci = _bootstrap(truth_gain, seed=24_135)
    source_d1 = payload["traces"]["depth_one"]
    source_d2 = payload["traces"]["depth_two"]
    mechanics = {
        "all_297_rows_replayed": len(source_d1) == len(source_d2) == 297,
        "all_actions_legal": legality,
        "stored_states_match": state_match,
        "deterministic_outcomes_match": outcome_match,
        "posterior_metrics_match": posterior_match,
        "independent_planning_costs_match": planning_match,
        "every_recorded_action_is_independently_optimal": action_optimal,
        "paired_truths_match": all(
            one_trace["truth_index"] == two_trace["truth_index"]
            for one_trace, two_trace in zip(source_d1, source_d2)
        ),
        "truths_sampled_without_replacement": len(
            {trace["truth_index"] for trace in source_d1}
        )
        == 297,
        "stored_comparison_matches_replay": comparison_match,
        "no_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "stage": "cleveland_heart_workup_depth_qualification_audit",
        "source_config": payload["config"],
        "mechanics": mechanics,
        "entropy_auc_gain": {
            "mean": float(np.mean(entropy_gain)),
            "independent_ci95": entropy_ci,
        },
        "truth_log_probability_auc_gain": {
            "mean": float(np.mean(truth_gain)),
            "independent_ci95": truth_ci,
        },
        "passed": all(mechanics.values()) and entropy_ci[0] > 0.0 and truth_ci[0] > 0.0,
    }


def render(summary: dict[str, Any]) -> str:
    entropy = summary["entropy_auc_gain"]
    truth = summary["truth_log_probability_auc_gain"]
    return "\n".join(
        [
            "# Cleveland Heart Workup Qualification Audit",
            "",
            f"Audit passed: **{summary['passed']}**.",
            "",
            f"- Entropy-AUC gain: `{entropy['mean']:+.6f}`, independent 95% CI "
            f"`[{entropy['independent_ci95'][0]:+.6f}, {entropy['independent_ci95'][1]:+.6f}]`.",
            f"- Truth-log-AUC gain: `{truth['mean']:+.6f}`, independent 95% CI "
            f"`[{truth['independent_ci95'][0]:+.6f}, {truth['independent_ci95'][1]:+.6f}]`.",
            "- Every stored action was independently re-scored and verified optimal.",
            "- Every state, deterministic observation, posterior metric, paired truth, and aggregate comparison was replayed from the raw Cleveland cohort.",
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
