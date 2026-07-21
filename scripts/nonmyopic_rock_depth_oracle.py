"""Exact paired depth-one/two/three oracle for Rock Diagnosis."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Literal

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
from environments.rock_diagnosis.core import EPSILON


@dataclass(frozen=True)
class DepthOracleConfig:
    map_name: str = "7-8"
    num_trials: int = 500
    num_rounds: int = 10
    max_depth: int = 3
    seed: int = 24_075
    bootstrap_replicates: int = 10_000
    trial_concurrency: int = 16
    half_efficiency_distance: float = math.log(2.0)

    def validate(self) -> None:
        get_paper_map(self.map_name)
        if min(
            self.num_trials,
            self.num_rounds,
            self.max_depth,
            self.bootstrap_replicates,
            self.trial_concurrency,
        ) <= 0:
            raise ValueError("trial, round, depth, bootstrap, and concurrency counts must be positive")
        if self.max_depth != 3:
            raise ValueError("the registered qualification requires max_depth=3")
        if self.num_rounds < self.max_depth:
            raise ValueError("num_rounds must be at least max_depth")
        if self.half_efficiency_distance <= 0.0:
            raise ValueError("half_efficiency_distance must be positive")


def _stable_seed(*parts: Any) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def _uniform(*parts: Any) -> float:
    return _stable_seed(*parts) / 2**64


def _belief_key(belief: np.ndarray) -> bytes:
    return np.ascontiguousarray(belief, dtype=np.float64).tobytes()


def exhaustive_action_values(
    model: RockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    depth: int,
    planning_utility: Literal["terminal_eig", "entropy_auc"] = "terminal_eig",
) -> tuple[dict[str, float], int]:
    """Return exact total-EIG values and expanded action evaluations."""

    if depth <= 0:
        raise ValueError("depth must be positive")
    if planning_utility not in {"terminal_eig", "entropy_auc"}:
        raise ValueError("planning_utility must be terminal_eig or entropy_auc")
    memo: dict[tuple[tuple[int, int], int, bytes], tuple[float, int]] = {}

    def best_value(current_position: tuple[int, int], current_belief: np.ndarray, horizon: int) -> tuple[float, int]:
        key = (current_position, horizon, _belief_key(current_belief))
        cached = memo.get(key)
        if cached is not None:
            return cached
        values, units = action_values(current_position, current_belief, horizon)
        result = (max(values.values()), units)
        memo[key] = result
        return result

    def action_values(
        current_position: tuple[int, int], current_belief: np.ndarray, horizon: int
    ) -> tuple[dict[str, float], int]:
        legal = model.legal_actions(current_position)
        values: dict[str, float] = {}
        units = len(legal)
        for action in legal:
            immediate_weight = horizon if planning_utility == "entropy_auc" else 1
            value = immediate_weight * model.expected_information_gain(
                current_position, current_belief, action
            )
            if horizon > 1:
                next_position = model.next_position(current_position, action)
                for outcome in model.outcomes(action):
                    probability = model.outcome_probability(
                        current_position, current_belief, action, outcome
                    )
                    if probability <= EPSILON:
                        continue
                    posterior = model.posterior(current_position, current_belief, action, outcome)
                    continuation, continuation_units = best_value(
                        next_position, posterior, horizon - 1
                    )
                    value += probability * continuation
                    units += continuation_units
            values[action] = value
        return values, units

    return action_values(position, belief, depth)


def _choose_action(values: dict[str, float]) -> str:
    actions = tuple(values)
    return max(actions, key=lambda action: (values[action], -actions.index(action)))


def _run_policy(
    model: RockDiagnosisModel,
    *,
    trial_index: int,
    truth_index: int,
    depth: int,
    config: DepthOracleConfig,
) -> dict[str, Any]:
    belief = model.initial_belief.copy()
    position = model.map_spec.start_position
    check_counts: dict[tuple[tuple[int, int], int], int] = {}
    steps: list[dict[str, Any]] = []

    for round_index in range(config.num_rounds):
        horizon = min(depth, config.num_rounds - round_index)
        values, scorer_units = exhaustive_action_values(
            model, position=position, belief=belief, depth=horizon
        )
        action = _choose_action(values)
        check_id = model.check_id(action)
        if check_id is None:
            outcome: str | None = None
        else:
            key = (position, check_id)
            repeat_index = check_counts.get(key, 0)
            check_counts[key] = repeat_index + 1
            probability_good = float(model.likelihood_vector(position, action, "good")[truth_index])
            outcome = (
                "good"
                if _uniform(
                    config.seed,
                    "rock-depth-observation",
                    trial_index,
                    position,
                    check_id,
                    repeat_index,
                )
                < probability_good
                else "bad"
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


def _run_trial(trial_index: int, config: DepthOracleConfig) -> dict[int, dict[str, Any]]:
    model = RockDiagnosisModel(
        get_paper_map(config.map_name),
        half_efficiency_distance=config.half_efficiency_distance,
    )
    truth_rng = np.random.default_rng(_stable_seed(config.seed, "rock-depth-truth", trial_index))
    truth_index = int(truth_rng.integers(len(model.hidden_states)))
    return {
        depth: _run_policy(
            model,
            trial_index=trial_index,
            truth_index=truth_index,
            depth=depth,
            config=config,
        )
        for depth in range(1, config.max_depth + 1)
    }


def _bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, replicates: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    for start in range(0, replicates, 512):
        batch = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        means.append(np.mean(values[indices], axis=1))
    samples = np.concatenate(means)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _comparison(
    deeper: list[dict[str, Any]],
    shallower: list[dict[str, Any]],
    *,
    config: DepthOracleConfig,
    label: str,
) -> dict[str, Any]:
    entropy_gains = np.asarray(
        [shallower[index]["entropy_auc"] - deeper[index]["entropy_auc"] for index in range(len(deeper))],
        dtype=float,
    )
    truth_log_gains = np.asarray(
        [
            deeper[index]["truth_log_probability_auc"]
            - shallower[index]["truth_log_probability_auc"]
            for index in range(len(deeper))
        ],
        dtype=float,
    )
    final_entropy_gains = np.asarray(
        [shallower[index]["final_entropy"] - deeper[index]["final_entropy"] for index in range(len(deeper))],
        dtype=float,
    )
    entropy_ci = _bootstrap_mean_ci(
        entropy_gains,
        seed=_stable_seed(config.seed, label, "entropy-auc-bootstrap"),
        replicates=config.bootstrap_replicates,
    )
    truth_ci = _bootstrap_mean_ci(
        truth_log_gains,
        seed=_stable_seed(config.seed, label, "truth-log-auc-bootstrap"),
        replicates=config.bootstrap_replicates,
    )
    final_ci = _bootstrap_mean_ci(
        final_entropy_gains,
        seed=_stable_seed(config.seed, label, "final-entropy-bootstrap"),
        replicates=config.bootstrap_replicates,
    )
    return {
        "entropy_auc_gain_mean": float(np.mean(entropy_gains)),
        "entropy_auc_gain_ci95": list(entropy_ci),
        "entropy_auc_wins_ties_losses": [
            int(np.count_nonzero(entropy_gains > EPSILON)),
            int(np.count_nonzero(np.abs(entropy_gains) <= EPSILON)),
            int(np.count_nonzero(entropy_gains < -EPSILON)),
        ],
        "entropy_auc_paired_values": entropy_gains.tolist(),
        "truth_log_probability_auc_gain_mean": float(np.mean(truth_log_gains)),
        "truth_log_probability_auc_gain_ci95": list(truth_ci),
        "truth_log_probability_auc_paired_values": truth_log_gains.tolist(),
        "final_entropy_gain_mean": float(np.mean(final_entropy_gains)),
        "final_entropy_gain_ci95": list(final_ci),
    }


def run_depth_oracle(config: DepthOracleConfig) -> dict[str, Any]:
    config.validate()
    with ThreadPoolExecutor(max_workers=config.trial_concurrency) as executor:
        trial_rows = list(executor.map(lambda index: _run_trial(index, config), range(config.num_trials)))
    traces = {
        str(depth): [trial[depth] for trial in trial_rows]
        for depth in range(1, config.max_depth + 1)
    }
    reference_pairs = [
        (trace["trial_index"], trace["truth_index"]) for trace in traces[str(config.max_depth)]
    ]
    paired = all(
        [(trace["trial_index"], trace["truth_index"]) for trace in traces[str(depth)]]
        == reference_pairs
        for depth in range(1, config.max_depth + 1)
    )
    comparisons = {
        "d2_minus_d1": _comparison(traces["2"], traces["1"], config=config, label="d2-minus-d1"),
        "d3_minus_d2": _comparison(traces["3"], traces["2"], config=config, label="d3-minus-d2"),
        "d3_minus_d1": _comparison(traces["3"], traces["1"], config=config, label="d3-minus-d1"),
    }
    primary = comparisons["d3_minus_d2"]
    initial_model = RockDiagnosisModel(
        get_paper_map(config.map_name),
        half_efficiency_distance=config.half_efficiency_distance,
    )
    initial_values = {
        str(depth): max(
            exhaustive_action_values(
                initial_model,
                position=initial_model.map_spec.start_position,
                belief=initial_model.initial_belief,
                depth=depth,
            )[0].values()
        )
        for depth in range(1, config.max_depth + 1)
    }
    mechanics = {
        "paired_trial_truths": paired,
        "all_selected_actions_legal": all(
            step["action"]
            in initial_model.legal_actions(tuple(step["position_before"]))
            for depth_traces in traces.values()
            for trace in depth_traces
            for step in trace["steps"]
        ),
        "all_traces_have_registered_rounds": all(
            len(trace["steps"]) == config.num_rounds
            for depth_traces in traces.values()
            for trace in depth_traces
        ),
        "no_llm_calls": True,
    }
    gate_passed = (
        primary["entropy_auc_gain_ci95"][0] > 0.0
        and all(mechanics.values())
    )
    corroboration_passed = primary["truth_log_probability_auc_gain_ci95"][0] > 0.0
    return {
        "schema_version": 1,
        "stage": "depth3_exact_qualification",
        "config": asdict(config),
        "source": {
            "paper": initial_model.map_spec.source_citation,
            "url": initial_model.map_spec.source_url,
            "page": initial_model.map_spec.source_page,
            "map": config.map_name,
            "map_spec": asdict(initial_model.map_spec),
            "pomdp_py_version": "1.3.5.1",
        },
        "initial_exact_values": initial_values,
        "comparisons": comparisons,
        "mechanics": mechanics,
        "primary_gate_passed": gate_passed,
        "truth_log_corroboration_passed": corroboration_passed,
        "traces": traces,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# RockSample[7,8] Exact Depth-Three Qualification",
        "",
        "Positive paired gains favor the deeper exact policy.",
        "",
        "| Comparison | Entropy-AUC gain [95% CI] | Truth-log-AUC gain [95% CI] | AUC W/T/L |",
        "| --- | --- | --- | --- |",
    ]
    for label in ("d2_minus_d1", "d3_minus_d2", "d3_minus_d1"):
        comparison = summary["comparisons"][label]
        entropy_ci = comparison["entropy_auc_gain_ci95"]
        truth_ci = comparison["truth_log_probability_auc_gain_ci95"]
        wtl = comparison["entropy_auc_wins_ties_losses"]
        lines.append(
            f"| {label.replace('_', ' ')} | {comparison['entropy_auc_gain_mean']:+.4f} "
            f"[{entropy_ci[0]:+.4f}, {entropy_ci[1]:+.4f}] | "
            f"{comparison['truth_log_probability_auc_gain_mean']:+.4f} "
            f"[{truth_ci[0]:+.4f}, {truth_ci[1]:+.4f}] | {wtl[0]}/{wtl[1]}/{wtl[2]} |"
        )
    lines.extend(
        [
            "",
            f"- Primary d3-over-d2 gate: **{summary['primary_gate_passed']}**.",
            f"- Truth-log corroboration: **{summary['truth_log_corroboration_passed']}**.",
            f"- Initial exact values: `{summary['initial_exact_values']}`.",
            f"- Mechanics: `{summary['mechanics']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="7-8", choices=("3-6", "5-7", "7-8"))
    parser.add_argument("--num-trials", type=int, default=500)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=24_075)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--trial-concurrency", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/rocksample_7_8_depth3_oracle_20260721"),
    )
    args = parser.parse_args()
    config = DepthOracleConfig(
        map_name=args.map_name,
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        max_depth=args.max_depth,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        trial_concurrency=args.trial_concurrency,
    )
    summary = run_depth_oracle(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
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
