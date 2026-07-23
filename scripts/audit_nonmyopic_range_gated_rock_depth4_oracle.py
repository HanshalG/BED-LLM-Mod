"""Independent audit for the corner-start range-gated depth-four gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel, get_paper_map


def _seed(*parts: Any) -> int:
    encoded = json.dumps(
        parts, sort_keys=True, separators=(",", ":"), default=list
    ).encode()
    return int.from_bytes(
        hashlib.sha256(encoded).digest()[:8], "big", signed=False
    )


def _uniform(*parts: Any) -> float:
    return _seed(*parts) / 2**64


def _bootstrap(
    values: np.ndarray, *, seed: int, replicates: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 256):
        size = min(256, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        samples[start : start + size] = values[indices].mean(axis=1)
    return [
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    ]


def independent_action_values(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    depth: int,
) -> dict[str, float]:
    """Re-solve terminal-history EIG without the producer planner."""

    memo: dict[tuple[tuple[int, int], int, bytes], dict[str, float]] = {}

    def solve(
        current_position: tuple[int, int],
        current_belief: np.ndarray,
        horizon: int,
    ) -> dict[str, float]:
        key = (
            current_position,
            horizon,
            np.ascontiguousarray(current_belief, dtype=np.float64).tobytes(),
        )
        cached = memo.get(key)
        if cached is not None:
            return cached
        values: dict[str, float] = {}
        for action in model.legal_actions(current_position):
            value = model.expected_information_gain(
                current_position, current_belief, action
            )
            if horizon > 1:
                next_position = model.next_position(
                    current_position, action
                )
                for outcome in model.outcomes(action):
                    probability = model.outcome_probability(
                        current_position,
                        current_belief,
                        action,
                        outcome,
                    )
                    if probability <= 0.0:
                        continue
                    posterior = model.posterior(
                        current_position,
                        current_belief,
                        action,
                        outcome,
                    )
                    value += probability * max(
                        solve(next_position, posterior, horizon - 1).values()
                    )
            values[action] = value
        memo[key] = values
        return values

    if depth <= 0:
        raise ValueError("depth must be positive")
    return solve(position, belief, depth)


def audit(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("stage") != "range_gated_rock_depth4_exact_qualification":
        raise ValueError("unexpected range-gated depth-four stage")
    config = payload["config"]
    map_spec = replace(
        get_paper_map(str(config["map_name"])),
        start_position=tuple(config["start_position"]),
    )
    model = RangeGatedRockDiagnosisModel(
        map_spec,
        remote_accuracy=float(config["remote_accuracy"]),
        onsite_accuracy=float(config["onsite_accuracy"]),
    )
    replayed: dict[str, list[dict[str, float]]] = {
        depth: [] for depth in ("3", "4")
    }
    legality = True
    state_match = True
    outcome_match = True
    posterior_match = True
    planning_match = True
    action_optimal = True

    for depth, traces in payload["traces"].items():
        for trace in traces:
            position = model.map_spec.start_position
            belief = model.initial_belief.copy()
            check_counts: dict[tuple[tuple[int, int], int], int] = {}
            entropies: list[float] = []
            truth_logs: list[float] = []
            for step in trace["steps"]:
                action = str(step["action"])
                state_match &= list(position) == step["position_before"]
                legal = model.legal_actions(position)
                legality &= action in legal
                values = independent_action_values(
                    model,
                    position=position,
                    belief=belief,
                    depth=int(step["horizon"]),
                )
                chosen = max(
                    legal,
                    key=lambda candidate: (
                        values[candidate],
                        -legal.index(candidate),
                    ),
                )
                action_optimal &= chosen == action
                planning_match &= math.isclose(
                    values[action],
                    float(step["planning_value"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                planning_match &= math.isclose(
                    model.expected_information_gain(
                        position, belief, action
                    ),
                    float(step["immediate_eig"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                rock_id = model.check_id(action)
                if rock_id is None:
                    outcome: str | None = None
                else:
                    key = (position, rock_id)
                    repeat = check_counts.get(key, 0)
                    check_counts[key] = repeat + 1
                    probability_good = float(
                        model.likelihood_vector(position, action, "good")[
                            int(trace["truth_index"])
                        ]
                    )
                    outcome = (
                        "good"
                        if _uniform(
                            int(config["seed"]),
                            "rock-depth-observation",
                            int(trace["trial_index"]),
                            position,
                            rock_id,
                            repeat,
                        )
                        < probability_good
                        else "bad"
                    )
                outcome_match &= outcome == step["observation"]
                belief = model.posterior(
                    position, belief, action, outcome
                )
                position = model.next_position(position, action)
                entropy = model.entropy(belief)
                truth_log = math.log(
                    max(
                        float(belief[int(trace["truth_index"])]),
                        np.finfo(float).tiny,
                    )
                )
                entropies.append(entropy)
                truth_logs.append(truth_log)
                posterior_match &= math.isclose(
                    entropy,
                    float(step["entropy"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                posterior_match &= math.isclose(
                    truth_log,
                    float(step["truth_log_probability"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            entropy_auc = float(np.mean(entropies))
            truth_auc = float(np.mean(truth_logs))
            posterior_match &= math.isclose(
                entropy_auc,
                float(trace["entropy_auc"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            posterior_match &= math.isclose(
                truth_auc,
                float(trace["truth_log_probability_auc"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            replayed[depth].append(
                {
                    "entropy_auc": entropy_auc,
                    "truth_log_probability_auc": truth_auc,
                }
            )

    entropy_gain = np.asarray(
        [
            d3["entropy_auc"] - d4["entropy_auc"]
            for d3, d4 in zip(
                replayed["3"], replayed["4"], strict=True
            )
        ]
    )
    truth_gain = np.asarray(
        [
            d4["truth_log_probability_auc"]
            - d3["truth_log_probability_auc"]
            for d3, d4 in zip(
                replayed["3"], replayed["4"], strict=True
            )
        ]
    )
    stored = payload["comparison"]
    comparison_match = math.isclose(
        float(entropy_gain.mean()),
        float(stored["entropy_auc_gain_mean"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) and math.isclose(
        float(truth_gain.mean()),
        float(stored["truth_log_probability_auc_gain_mean"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    replicates = int(config["bootstrap_replicates"])
    entropy_ci = _bootstrap(
        entropy_gain, seed=24_202, replicates=replicates
    )
    truth_ci = _bootstrap(
        truth_gain, seed=24_203, replicates=replicates
    )
    reference = [
        (trace["trial_index"], trace["truth_index"])
        for trace in payload["traces"]["4"]
    ]
    mechanics = {
        "all_registered_traces_replayed": all(
            len(payload["traces"][depth]) == int(config["num_trials"])
            for depth in ("3", "4")
        ),
        "all_actions_legal": legality,
        "stored_positions_match": state_match,
        "seeded_observations_match": outcome_match,
        "posterior_metrics_match": posterior_match,
        "independent_planning_values_match": planning_match,
        "every_recorded_action_is_independently_optimal": action_optimal,
        "paired_truths_match": [
            (trace["trial_index"], trace["truth_index"])
            for trace in payload["traces"]["3"]
        ]
        == reference,
        "stored_comparison_matches_replay": comparison_match,
        "no_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_depth4_exact_qualification_audit",
        "source_config": config,
        "mechanics": mechanics,
        "entropy_auc_gain": {
            "mean": float(entropy_gain.mean()),
            "independent_ci95": entropy_ci,
        },
        "truth_log_probability_auc_gain": {
            "mean": float(truth_gain.mean()),
            "independent_ci95": truth_ci,
        },
        "passed": (
            all(mechanics.values())
            and entropy_ci[0] > 0.0
            and truth_ci[0] > 0.0
        ),
    }


def render(result: dict[str, Any]) -> str:
    entropy = result["entropy_auc_gain"]
    truth = result["truth_log_probability_auc_gain"]
    return "\n".join(
        [
            "# Range-Gated RockSample[7,8] Depth-Four Audit",
            "",
            f"Audit passed: **{result['passed']}**.",
            "",
            (
                f"- Entropy-AUC d4-over-d3: `{entropy['mean']:+.6f}`, "
                f"independent 95% CI "
                f"`[{entropy['independent_ci95'][0]:+.6f}, "
                f"{entropy['independent_ci95'][1]:+.6f}]`."
            ),
            (
                f"- Truth-log-AUC d4-over-d3: `{truth['mean']:+.6f}`, "
                f"independent 95% CI "
                f"`[{truth['independent_ci95'][0]:+.6f}, "
                f"{truth['independent_ci95'][1]:+.6f}]`."
            ),
            "- Every stored action, state, observation, posterior, and aggregate was replayed.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        json.loads(args.report_json.read_text(encoding="utf-8"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary_output.write_text(render(result), encoding="utf-8")
    print(json.dumps({"passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
