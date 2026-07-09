from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import BeliefState
from core.defaults import register_defaults
from environments.location_finding.beliefs import build_location_belief_state_unpruned
from environments.location_finding.env import LocationBEDEnvironment
from environments.location_finding.physics import round_positive_observation, signal_intensity_for_hypothesis, source_rmse
from environments.location_finding.task_loss import posterior_expected_source_rmse
from environments.location_finding.strategy import (
    evaluate_location_strategies_by_rollout_many,
    generate_location_strategy_roots_many,
    generate_strategy_locations_many,
)
from environments.location_finding.types import (
    Location,
    LocationObservation,
    LocationStrategyCandidate,
    LocationStrategyEvaluation,
    LocationStrategyLibrary,
    SourceConfig,
    _StrategyEvaluationRequest,
    _StrategyLocationRequest,
    _dedupe_source_configs,
)
from helpers import Config, load_config, resolve_run_id
from scripts.llm_token_usage import summarize_llm_token_usage, token_usage_report_lines


@dataclass
class _ProbeState:
    trial_index: int
    round_index: int
    hidden_state: np.ndarray
    belief_state: BeliefState[SourceConfig]
    history: list[tuple[Location, LocationObservation]]
    library: LocationStrategyLibrary = field(default_factory=LocationStrategyLibrary)


@dataclass
class _DeploymentBranch:
    candidate_index: int
    replicate_index: int
    depth: int
    strategy: str
    hidden_state: np.ndarray
    start_belief_state: BeliefState[SourceConfig]
    belief_state: BeliefState[SourceConfig]
    start_history: list[tuple[Location, LocationObservation]]
    history: list[tuple[Location, LocationObservation]]
    rng: np.random.Generator
    root_query: Location | None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, LocationObservation):
        return {"query": list(value.query), "value": float(value.value)}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _entropy(probabilities: list[float] | np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0.0]
    if len(values) == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return None
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= 0.0:
        return None
    return float(np.sum(x * y) / denom)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_rankdata(xs), _rankdata(ys))


def _source_config_from_hidden_state(hidden_state: np.ndarray) -> SourceConfig:
    return tuple(tuple(float(coord) for coord in row) for row in np.asarray(hidden_state, dtype=float))


def _copy_config_with_depth(config: Config, depth: int) -> Config:
    depth_config = copy.copy(config)
    depth_config.location_strategy_planning_depth = int(depth)
    return depth_config


def _diagnostic_num_rounds(state_rounds: list[int], depths: list[int]) -> int:
    max_state_round = max(state_rounds, default=0)
    max_depth = max(depths, default=1)
    return int(max_state_round + max_depth)


def _copy_config_for_score_variant(config: Config, variant: str) -> Config:
    variant_config = copy.copy(config)
    if variant == "baseline":
        variant_config.location_strategy_rollout_scoring_support_mode = "union"
        variant_config.location_strategy_rollout_final_refresh_enabled = True
        variant_config.location_strategy_rollout_refresh_hypotheses_each_step = False
        return variant_config
    if variant == "configured":
        return variant_config
    raise ValueError(f"Unknown score variant: {variant}")


def _append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"{datetime.now().isoformat()} {message}\n")
        log_handle.flush()


def _best_rmse(belief_state: BeliefState[SourceConfig], hidden_state: np.ndarray) -> float:
    if not belief_state.hypotheses:
        return float("inf")
    return source_rmse(belief_state.hypotheses[0], hidden_state)


def _fixed_support_entropy_drop(
    support: list[SourceConfig],
    start_observations: list[LocationObservation],
    end_observations: list[LocationObservation],
    config: Config,
) -> float:
    if len(support) <= 1:
        return 0.0
    start_state = build_location_belief_state_unpruned(support, start_observations, config)
    end_state = build_location_belief_state_unpruned(support, end_observations, config)
    return float(_entropy(start_state.probabilities) - _entropy(end_state.probabilities))


def _fixed_support_truth_log_probability(
    support: list[SourceConfig],
    observations: list[LocationObservation],
    truth: SourceConfig,
    config: Config,
) -> float:
    state = build_location_belief_state_unpruned(support, observations, config)
    for hypothesis, probability in zip(state.hypotheses, state.probabilities):
        if hypothesis == truth:
            return math.log(max(float(probability), 1e-300))
    return math.log(1e-300)


def _posterior_expected_rmse(belief_state: BeliefState[SourceConfig]) -> float | None:
    value = posterior_expected_source_rmse(belief_state)
    return float(value) if math.isfinite(value) else None


def _posterior_state_record(
    belief_state: BeliefState[SourceConfig],
    *,
    candidate_index: int,
    replicate_index: int,
) -> dict[str, Any]:
    return {
        "candidate_index": int(candidate_index),
        "replicate_index": int(replicate_index),
        "expected_rmse": _posterior_expected_rmse(belief_state),
        "hypotheses": [
            [[float(value) for value in source] for source in hypothesis]
            for hypothesis in belief_state.hypotheses
        ],
        "probabilities": [float(probability) for probability in belief_state.probabilities],
    }


def _trajectory_distance(first: list[Location], second: list[Location]) -> float:
    max_len = max(len(first), len(second))
    if max_len == 0:
        return 0.0
    total = 0.0
    missing_penalty = 10.0
    for index in range(max_len):
        if index >= len(first) or index >= len(second):
            total += missing_penalty
            continue
        total += float(np.linalg.norm(np.asarray(first[index], dtype=float) - np.asarray(second[index], dtype=float)))
    return total / float(max_len)


def _strategy_execution_fidelity(branches: list[_DeploymentBranch]) -> dict[str, float | None]:
    trajectories: list[tuple[int, int, list[Location]]] = []
    for branch in branches:
        start_len = len(branch.start_history)
        trajectory = [action for action, _observation in branch.history[start_len:]]
        trajectories.append((branch.candidate_index, branch.replicate_index, trajectory))

    within: list[float] = []
    between: list[float] = []
    for left_idx, left in enumerate(trajectories):
        for right in trajectories[left_idx + 1 :]:
            distance = _trajectory_distance(left[2], right[2])
            if left[0] == right[0]:
                within.append(distance)
            else:
                between.append(distance)

    within_mean = float(np.mean(within)) if within else None
    between_mean = float(np.mean(between)) if between else None
    ratio = None
    if within_mean is not None and between_mean is not None and within_mean > 0.0:
        ratio = float(between_mean / within_mean)
    return {
        "within_strategy_query_distance_mean": within_mean,
        "between_strategy_query_distance_mean": between_mean,
        "between_over_within_query_distance": ratio,
    }


def _bootstrap_ci(values: list[float], rng: np.random.Generator, *, samples: int = 2000) -> list[float | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return [None, None]
    if len(clean) == 1:
        return [clean[0], clean[0]]
    arr = np.asarray(clean, dtype=float)
    means = np.empty(samples, dtype=float)
    for sample_idx in range(samples):
        means[sample_idx] = float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def _generate_probe_states(
    env: LocationBEDEnvironment,
    questioner: Any,
    config: Config,
    *,
    num_trials: int,
    trial_offset: int,
    state_rounds: list[int],
    rng: np.random.Generator,
) -> list[_ProbeState]:
    target_rounds = sorted(set(int(round_idx) for round_idx in state_rounds))
    if not target_rounds or target_rounds[0] < 0:
        raise ValueError("state rounds must be non-negative")
    max_round = max(target_rounds)
    probes: list[_ProbeState] = []
    for local_trial_index in range(num_trials):
        trial_index = int(trial_offset) + local_trial_index
        if getattr(config, "location_seed", None) is None:
            trial_rng = rng
        else:
            trial_rng = np.random.default_rng(
                np.random.SeedSequence([int(config.location_seed), int(trial_index)])
            )
        hidden_state = env.sample_hidden_state_for_trial(trial_index, trial_rng)
        belief_state = env.initial_belief_state(questioner, config)
        history: list[tuple[Location, LocationObservation]] = []
        if 0 in target_rounds:
            probes.append(_ProbeState(trial_index, 0, hidden_state, belief_state, list(history)))
        for round_index in range(max_round):
            action = env.generate_naive_action(
                belief_state,
                history,
                questioner,
                config,
                method_name="naive+belief",
            )
            observation = env.observe(action, hidden_state, trial_rng)
            history.append((action, observation))
            belief_state = env.update_belief_state(belief_state, history, questioner, config)
            completed_round = round_index + 1
            if completed_round in target_rounds:
                probes.append(
                    _ProbeState(
                        trial_index,
                        completed_round,
                        hidden_state,
                        belief_state,
                        list(history),
                    )
                )
    return probes


def _score_candidates_by_depth(
    questioner: Any,
    config: Config,
    probe: _ProbeState,
    candidates: list[LocationStrategyCandidate],
    depths: list[int],
    *,
    seed: int,
    log_path: Path | None = None,
    log_prefix: str = "",
) -> dict[int, list[LocationStrategyEvaluation]]:
    result: dict[int, list[LocationStrategyEvaluation]] = {}
    strategies = [candidate.strategy for candidate in candidates]
    roots = [candidate.root_query for candidate in candidates]
    observations = [observation for _action, observation in probe.history]
    for depth in depths:
        if log_path is not None:
            _append_log(log_path, f"{log_prefix}scoring depth={depth}")
        eval_config = _copy_config_with_depth(config, depth)
        request = _StrategyEvaluationRequest(
            strategies=strategies,
            belief_state=probe.belief_state,
            observations=observations,
            rng=np.random.default_rng(seed + 10_007 * depth),
            root_queries=roots,
        )
        result[depth] = evaluate_location_strategies_by_rollout_many(questioner, [request], eval_config)[0]
        if log_path is not None:
            _append_log(log_path, f"{log_prefix}finished scoring depth={depth}")
    return result


def _deploy_candidates_for_depth(
    env: LocationBEDEnvironment,
    questioner: Any,
    config: Config,
    probe: _ProbeState,
    candidates: list[LocationStrategyCandidate],
    *,
    depth: int,
    deployments: int,
    seed: int,
) -> dict[str, Any]:
    truth = _source_config_from_hidden_state(probe.hidden_state)
    fixed_support = _dedupe_source_configs(list(probe.belief_state.hypotheses) + [truth])
    start_observations = [observation for _action, observation in probe.history]
    start_rmse = _best_rmse(probe.belief_state, probe.hidden_state)
    start_expected_posterior_rmse = _posterior_expected_rmse(probe.belief_state)
    branches: list[_DeploymentBranch] = []
    for candidate_index, candidate in enumerate(candidates):
        if candidate.root_query is None:
            continue
        for replicate_index in range(deployments):
            branch_seed = seed + 1_000_003 * (candidate_index + 1) + 9_176 * (replicate_index + 1)
            branches.append(
                _DeploymentBranch(
                    candidate_index=candidate_index,
                    replicate_index=replicate_index,
                    depth=depth,
                    strategy=candidate.strategy,
                    hidden_state=probe.hidden_state,
                    start_belief_state=probe.belief_state,
                    belief_state=probe.belief_state,
                    start_history=list(probe.history),
                    history=list(probe.history),
                    rng=np.random.default_rng(branch_seed),
                    root_query=candidate.root_query,
                )
            )

    for step_idx in range(depth):
        active = [branch for branch in branches if branch.belief_state.hypotheses]
        if not active:
            break
        locations: list[Location | None]
        if step_idx == 0:
            locations = [branch.root_query for branch in active]
        else:
            requests = [
                _StrategyLocationRequest(
                    strategy=branch.strategy,
                    belief_state=branch.belief_state,
                    observations=[observation for _action, observation in branch.history],
                )
                for branch in active
            ]
            locations = generate_strategy_locations_many(questioner, requests, config)

        updated_beliefs: list[BeliefState[SourceConfig]] = []
        updated_histories: list[list[tuple[Location, LocationObservation]]] = []
        updated_branches: list[_DeploymentBranch] = []
        for branch, location in zip(active, locations):
            if location is None:
                continue
            mean = signal_intensity_for_hypothesis(truth, location)
            value = round_positive_observation(
                mean * math.exp(float(branch.rng.normal(0.0, config.location_noise_sd))),
                2,
            )
            observation = LocationObservation(query=location, value=value)
            branch.history.append((location, observation))
            updated_beliefs.append(branch.belief_state)
            updated_histories.append(branch.history)
            updated_branches.append(branch)
        if updated_branches:
            new_beliefs = env.update_belief_states(updated_beliefs, updated_histories, questioner, config)
            for branch, new_belief in zip(updated_branches, new_beliefs):
                branch.belief_state = new_belief

    entropy_drops: list[list[float]] = [[] for _candidate in candidates]
    rmse_drops: list[list[float]] = [[] for _candidate in candidates]
    truth_log_probs: list[list[float]] = [[] for _candidate in candidates]
    expected_posterior_rmses: list[list[float]] = [[] for _candidate in candidates]
    expected_posterior_rmse_drops: list[list[float]] = [[] for _candidate in candidates]
    final_posterior_states: list[list[dict[str, Any]]] = [[] for _candidate in candidates]
    for branch in branches:
        end_observations = [observation for _action, observation in branch.history]
        entropy_drops[branch.candidate_index].append(
            _fixed_support_entropy_drop(fixed_support, start_observations, end_observations, config)
        )
        rmse_drops[branch.candidate_index].append(start_rmse - _best_rmse(branch.belief_state, probe.hidden_state))
        truth_log_probs[branch.candidate_index].append(
            _fixed_support_truth_log_probability(fixed_support, end_observations, truth, config)
        )
        posterior_expected_rmse = _posterior_expected_rmse(branch.belief_state)
        if posterior_expected_rmse is not None:
            expected_posterior_rmses[branch.candidate_index].append(posterior_expected_rmse)
            if start_expected_posterior_rmse is not None:
                expected_posterior_rmse_drops[branch.candidate_index].append(
                    start_expected_posterior_rmse - posterior_expected_rmse
                )
        final_posterior_states[branch.candidate_index].append(
            _posterior_state_record(
                branch.belief_state,
                candidate_index=branch.candidate_index,
                replicate_index=branch.replicate_index,
            )
        )

    return {
        "fixed_support_size": len(fixed_support),
        "start_expected_posterior_rmse": start_expected_posterior_rmse,
        "strategy_execution_fidelity": _strategy_execution_fidelity(branches),
        "entropy_drop_mean": [
            float(np.mean(values)) if values else None
            for values in entropy_drops
        ],
        "entropy_drop_std": [
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
            for values in entropy_drops
        ],
        "rmse_drop_mean": [
            float(np.mean(values)) if values else None
            for values in rmse_drops
        ],
        "rmse_drop_std": [
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
            for values in rmse_drops
        ],
        "truth_log_prob_mean": [
            float(np.mean(values)) if values else None
            for values in truth_log_probs
        ],
        "truth_log_prob_std": [
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
            for values in truth_log_probs
        ],
        "expected_posterior_rmse_mean": [
            float(np.mean(values)) if values else None
            for values in expected_posterior_rmses
        ],
        "expected_posterior_rmse_std": [
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
            for values in expected_posterior_rmses
        ],
        "expected_posterior_rmse_drop_mean": [
            float(np.mean(values)) if values else None
            for values in expected_posterior_rmse_drops
        ],
        "expected_posterior_rmse_drop_std": [
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
            for values in expected_posterior_rmse_drops
        ],
        "final_posterior_states": final_posterior_states,
        "deployments_per_candidate": [
            len(values)
            for values in entropy_drops
        ],
    }


def _state_depth_metrics(
    evaluations: list[LocationStrategyEvaluation],
    realized_entropy: list[float | None],
    realized_rmse: list[float | None],
    realized_truth_log_prob: list[float | None] | None = None,
    execution_fidelity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    estimated: list[float] = []
    entropy: list[float] = []
    rmse: list[float] = []
    truth_log_prob: list[float] = []
    rollout_variances: list[float] = []
    for index, evaluation in enumerate(evaluations):
        if index >= len(realized_entropy) or realized_entropy[index] is None:
            continue
        estimated.append(float(evaluation.mean_score))
        entropy.append(float(realized_entropy[index]))
        if index < len(realized_rmse) and realized_rmse[index] is not None:
            rmse.append(float(realized_rmse[index]))
        if (
            realized_truth_log_prob is not None
            and index < len(realized_truth_log_prob)
            and realized_truth_log_prob[index] is not None
        ):
            truth_log_prob.append(float(realized_truth_log_prob[index]))
        if evaluation.rollout_scores:
            rollout_variances.append(float(np.var(evaluation.rollout_scores, ddof=1)) if len(evaluation.rollout_scores) > 1 else 0.0)

    best_estimated_idx = int(np.argmax(estimated)) if estimated else None
    best_realized_idx = int(np.argmax(entropy)) if entropy else None
    top1_regret = None
    if best_estimated_idx is not None and best_realized_idx is not None:
        top1_regret = float(entropy[best_realized_idx] - entropy[best_estimated_idx])
    between = float(np.var(estimated, ddof=1)) if len(estimated) > 1 else 0.0
    within = float(np.mean(rollout_variances)) if rollout_variances else 0.0
    return {
        "n": len(estimated),
        "spearman_entropy": _spearman(estimated, entropy),
        "pearson_entropy": _pearson(estimated, entropy),
        "spearman_rmse": _spearman(estimated, rmse) if len(rmse) == len(estimated) else None,
        "pearson_rmse": _pearson(estimated, rmse) if len(rmse) == len(estimated) else None,
        "spearman_truth_log_prob": _spearman(estimated, truth_log_prob) if len(truth_log_prob) == len(estimated) else None,
        "pearson_truth_log_prob": _pearson(estimated, truth_log_prob) if len(truth_log_prob) == len(estimated) else None,
        "top1_regret_entropy": top1_regret,
        "score_var_between_strategies": between,
        "score_var_within_strategy": within,
        "snr_between_over_within": None if within <= 0.0 else float(between / within),
        "within_strategy_query_distance_mean": None if execution_fidelity is None else execution_fidelity.get("within_strategy_query_distance_mean"),
        "between_strategy_query_distance_mean": None if execution_fidelity is None else execution_fidelity.get("between_strategy_query_distance_mean"),
        "between_over_within_query_distance": None if execution_fidelity is None else execution_fidelity.get("between_over_within_query_distance"),
    }


def _aggregate_depth_metrics(records: list[dict[str, Any]], depths: list[int], rng: np.random.Generator) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for depth in depths:
        key = str(depth)
        depth_records = [
            record["depth_metrics"][key]
            for record in records
            if key in record["depth_metrics"]
        ]
        metric_names = [
            "spearman_entropy",
            "pearson_entropy",
            "spearman_rmse",
            "pearson_rmse",
            "spearman_truth_log_prob",
            "pearson_truth_log_prob",
            "top1_regret_entropy",
            "score_var_between_strategies",
            "score_var_within_strategy",
            "snr_between_over_within",
            "within_strategy_query_distance_mean",
            "between_strategy_query_distance_mean",
            "between_over_within_query_distance",
        ]
        depth_summary: dict[str, Any] = {"state_count": len(depth_records)}
        for metric_name in metric_names:
            values = [
                float(record[metric_name])
                for record in depth_records
                if record.get(metric_name) is not None and math.isfinite(float(record[metric_name]))
            ]
            depth_summary[metric_name] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
                "bootstrap_ci95": _bootstrap_ci(values, rng) if values else [None, None],
                "n": len(values),
            }
        aggregate[key] = depth_summary
    return aggregate


def _aggregate_variant_metrics(
    records: list[dict[str, Any]],
    depths: list[int],
    score_variants: list[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for variant in score_variants:
        variant_records = [
            {"depth_metrics": record["score_variant_metrics"][variant]}
            for record in records
            if variant in record.get("score_variant_metrics", {})
        ]
        aggregate[variant] = _aggregate_depth_metrics(variant_records, depths, rng)
    return aggregate


def _gate_assessment(summary: dict[str, Any], *, threshold: float = 0.4) -> dict[str, Any]:
    variants = [str(variant) for variant in summary.get("score_variants", [])]
    variant = "configured" if "configured" in variants else variants[-1] if variants else None
    result: dict[str, Any] = {
        "variant": variant,
        "threshold": threshold,
        "status": "inconclusive",
        "best_depth": None,
        "best_spearman_entropy": None,
        "best_spearman_truth_log_prob": None,
        "message": "No score variants are available for the gate assessment.",
    }
    if variant is None:
        return result

    best_depth: int | None = None
    best_value: float | None = None
    best_truth_value: float | None = None
    for depth in summary.get("depths", []):
        metrics = summary.get("aggregate", {}).get(variant, {}).get(str(depth), {})
        mean = metrics.get("spearman_entropy", {}).get("mean")
        if mean is None:
            continue
        value = float(mean)
        if not math.isfinite(value):
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_depth = int(depth)
            truth_mean = metrics.get("spearman_truth_log_prob", {}).get("mean")
            best_truth_value = None if truth_mean is None else float(truth_mean)

    result["best_depth"] = best_depth
    result["best_spearman_entropy"] = best_value
    result["best_spearman_truth_log_prob"] = best_truth_value
    if best_value is None:
        result["message"] = f"Variant `{variant}` has no finite Spearman entropy estimates yet."
        return result
    if best_value >= threshold and best_truth_value is not None and best_truth_value > 0.0:
        result["status"] = "proceed"
        result["message"] = (
            f"Gate passed: `{variant}` reaches Spearman entropy {best_value:.3f} "
            f"and truth-log-prob Spearman {best_truth_value:.3f} at depth {best_depth}."
        )
    elif best_value >= threshold and (best_truth_value is None or abs(best_truth_value) < 0.1):
        result["status"] = "calibration_warning"
        result["message"] = (
            f"Entropy gate passes at depth {best_depth} ({best_value:.3f}), but truth-log-prob "
            f"Spearman is {best_truth_value}; diagnose confident-but-wrong posteriors before Phase 4 spend."
        )
    else:
        result["status"] = "stop"
        result["message"] = (
            f"Gate not passed: best `{variant}` Spearman entropy is {best_value:.3f} "
            f"and truth-log-prob Spearman is {best_truth_value} at depth {best_depth}."
        )
    return result


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    gate = summary.get("gate_assessment") or _gate_assessment(summary)
    lines = [
        "# Strategy Ranking Fidelity",
        "",
        f"- Generated: {datetime.now().isoformat()}",
        f"- Config: `{summary['config_path']}`",
        f"- Questioner model: `{summary.get('questioner_model', 'unknown')}`",
        f"- Host: `{summary.get('hostname', 'unknown')}`",
        f"- Slurm job: `{summary.get('slurm_job_id', 'unknown')}`",
        f"- Probe states: {summary['num_probe_states']}",
        f"- Trials: {summary['num_trials']}",
        f"- State rounds: {summary['state_rounds']}",
        f"- Depths: {summary['depths']}",
        f"- Deployments per strategy/depth: {summary['deployments']}",
        f"- Target candidates per probe: {summary.get('target_num_candidates', 'config default')}",
        "",
        f"- Score variants: {summary['score_variants']}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Variant: `{gate['variant']}`",
        f"- Threshold: {gate['threshold']}",
        f"- Best depth: {gate['best_depth']}",
        f"- Best Spearman entropy: {gate['best_spearman_entropy']}",
        f"- Best Spearman truth log-prob: {gate.get('best_spearman_truth_log_prob')}",
        f"- Decision note: {gate['message']}",
        "",
        *token_usage_report_lines(summary.get("token_usage")),
        "",
        "| Variant | Depth | Spearman entropy | 95% CI | Spearman truth log-prob | Spearman RMSE | Top-1 regret | SNR | query distance ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in summary["score_variants"]:
        for depth in summary["depths"]:
            metrics = summary["aggregate"][variant][str(depth)]
            spearman = metrics["spearman_entropy"]
            spearman_truth = metrics["spearman_truth_log_prob"]
            spearman_rmse = metrics["spearman_rmse"]
            regret = metrics["top1_regret_entropy"]
            snr = metrics["snr_between_over_within"]
            query_ratio = metrics["between_over_within_query_distance"]
            ci = spearman["bootstrap_ci95"]
            lines.append(
                "| "
                f"{variant} | "
                f"{depth} | "
                f"{spearman['mean'] if spearman['mean'] is not None else 'NA'} | "
                f"[{ci[0]}, {ci[1]}] | "
                f"{spearman_truth['mean'] if spearman_truth['mean'] is not None else 'NA'} | "
                f"{spearman_rmse['mean'] if spearman_rmse['mean'] is not None else 'NA'} | "
                f"{regret['mean'] if regret['mean'] is not None else 'NA'} | "
                f"{snr['mean'] if snr['mean'] is not None else 'NA'} | "
                f"{query_ratio['mean'] if query_ratio['mean'] is not None else 'NA'} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, summary: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    depths = [int(depth) for depth in summary["depths"]]
    plt.figure(figsize=(6, 4))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axhline(0.4, color="tab:green", linestyle="--", linewidth=1.0)
    for variant in summary["score_variants"]:
        means = [
            summary["aggregate"][variant][str(depth)]["spearman_entropy"]["mean"]
            for depth in depths
        ]
        plt.plot(depths, means, marker="o", label=variant)
    plt.xlabel("Estimated EIG depth")
    plt.ylabel("Spearman vs realized entropy drop")
    plt.title("Strategy ranking fidelity")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def run_strategy_ranking_fidelity(
    config: Config,
    *,
    config_path: Path,
    output_root: Path,
    run_name: str | None,
    num_trials: int,
    trial_offset: int,
    state_rounds: list[int],
    depths: list[int],
    deployments: int,
    score_variants: list[str],
    num_candidates: int | None = None,
) -> Path:
    if config.task != "location_finding":
        raise ValueError("strategy_ranking_fidelity only supports task=location_finding")
    if not config.model_pairs:
        raise ValueError("Config must include at least one model pair")
    if num_trials <= 0:
        raise ValueError("num_trials must be positive")
    if trial_offset < 0:
        raise ValueError("trial_offset must be non-negative")
    if deployments <= 0:
        raise ValueError("deployments must be positive")
    if num_candidates is not None and num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    if any(depth <= 0 for depth in depths):
        raise ValueError("depths must be positive")
    if not score_variants:
        raise ValueError("score_variants must not be empty")
    for variant in score_variants:
        _copy_config_for_score_variant(config, variant)

    register_defaults()
    config.run_id = resolve_run_id()
    config.location_num_trials = int(num_trials)
    config.location_num_rounds = _diagnostic_num_rounds(state_rounds, depths)
    if num_candidates is not None:
        config.location_target_num_candidates = int(num_candidates)
    config.method_names = ["StrategyEIG+ranking-fidelity"]

    stem = run_name or f"{config.run_id}_strategy_ranking_fidelity"
    run_dir = output_root / stem
    run_dir.mkdir(parents=True, exist_ok=False)
    config.log_path = run_dir / "run.log"
    config.log_path.write_text(
        (
            f"START TIME: {datetime.now().isoformat()}\n"
            f"Config file: {config_path}\n"
            f"Strategy ranking fidelity depths={depths}, state_rounds={state_rounds}, deployments={deployments}, trial_offset={trial_offset}\n"
            f"Target candidates={config.location_target_num_candidates}\n"
            f"Score variants={score_variants}\n"
            f"Questioner model={config.model_pairs[0].questioner.model}\n"
            f"Host={os.uname().nodename}\n"
            f"Slurm job={os.environ.get('SLURM_JOB_ID', '')}\n"
        ),
        encoding="utf-8",
    )

    try:
        import wandb

        if wandb.run is None:
            wandb.init(
                project="BED-LLM-reproduction",
                name=stem,
                mode="disabled",
        config={
                    "run_name": stem,
                    "config_path": str(config_path),
                    "diagnostic": "strategy_ranking_fidelity",
                    "depths": depths,
                    "state_rounds": state_rounds,
                    "deployments": deployments,
                    "trial_offset": trial_offset,
                    "target_num_candidates": config.location_target_num_candidates,
                    "score_variants": score_variants,
                },
            )
    except Exception as exc:
        with config.log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"Warning: could not initialize disabled wandb run: {type(exc).__name__}: {exc}\n")

    from model import build_model_adapter

    pair = config.model_pairs[0]
    questioner = build_model_adapter(pair.questioner, config=config)
    env = LocationBEDEnvironment(config=config)
    env.validate_config(config)
    env.rng = np.random.default_rng(config.location_seed)

    seed = int(config.location_seed or 0)
    master_rng = np.random.default_rng(config.location_seed)
    _append_log(config.log_path, "generating probe states")
    probes = _generate_probe_states(
        env,
        questioner,
        config,
        num_trials=num_trials,
        trial_offset=trial_offset,
        state_rounds=state_rounds,
        rng=master_rng,
    )
    _append_log(config.log_path, f"generated {len(probes)} probe state(s)")

    records_path = run_dir / "strategy_ranking_fidelity_records.jsonl"
    records: list[dict[str, Any]] = []
    with records_path.open("w", encoding="utf-8") as records_handle:
        for probe_index, probe in enumerate(probes):
            probe_start = time.perf_counter()
            _append_log(
                config.log_path,
                f"probe {probe_index + 1}/{len(probes)} trial={probe.trial_index} round={probe.round_index}: generating candidates",
            )
            candidates = generate_location_strategy_roots_many(
                questioner,
                [(probe.belief_state, [obs for _action, obs in probe.history], probe.library)],
                config,
            )[0]
            _append_log(
                config.log_path,
                f"probe {probe_index + 1}/{len(probes)}: generated {len(candidates)} candidate strategy/root pair(s)",
            )
            score_seed = seed + 100_003 * (probe.trial_index + 1) + 10_007 * (probe.round_index + 1)
            evaluations_by_variant = {}
            for variant in score_variants:
                _append_log(
                    config.log_path,
                    f"probe {probe_index + 1}/{len(probes)}: scoring variant={variant} depths={depths}",
                )
                evaluations_by_variant[variant] = _score_candidates_by_depth(
                    questioner,
                    _copy_config_for_score_variant(config, variant),
                    probe,
                    candidates,
                    depths,
                    seed=score_seed,
                    log_path=config.log_path,
                    log_prefix=f"probe {probe_index + 1}/{len(probes)} variant={variant}: ",
                )
                _append_log(
                    config.log_path,
                    f"probe {probe_index + 1}/{len(probes)}: finished scoring variant={variant}",
                )
            realized_by_depth = {}
            for depth in depths:
                _append_log(
                    config.log_path,
                    f"probe {probe_index + 1}/{len(probes)}: deploying candidates depth={depth} deployments={deployments}",
                )
                realized_by_depth[depth] = _deploy_candidates_for_depth(
                    env,
                    questioner,
                    config,
                    probe,
                    candidates,
                    depth=depth,
                    deployments=deployments,
                    seed=seed + 1_000_003 * (probe.trial_index + 1) + 97_409 * (probe.round_index + 1) + depth,
                )
                _append_log(
                    config.log_path,
                    f"probe {probe_index + 1}/{len(probes)}: finished deployments depth={depth}",
                )
            score_variant_metrics = {
                variant: {
                    str(depth): _state_depth_metrics(
                        evaluations_by_variant[variant][depth],
                        realized_by_depth[depth]["entropy_drop_mean"],
                        realized_by_depth[depth]["rmse_drop_mean"],
                        realized_by_depth[depth]["truth_log_prob_mean"],
                        realized_by_depth[depth].get("strategy_execution_fidelity"),
                    )
                    for depth in depths
                }
                for variant in score_variants
            }
            record = {
                "probe_index": probe_index,
                "trial_index": probe.trial_index,
                "round_index": probe.round_index,
                "hidden_state": _jsonable(probe.hidden_state),
                "history": [
                    {"action": list(action), "observation": _jsonable(observation)}
                    for action, observation in probe.history
                ],
                "candidate_count": len(candidates),
                "candidates": [
                    {
                        "index": index,
                        "strategy": candidate.strategy,
                        "root_query": None if candidate.root_query is None else list(candidate.root_query),
                    }
                    for index, candidate in enumerate(candidates)
                ],
                "estimated_by_variant": {
                    variant: {
                        str(depth): [
                            {
                                "index": index,
                                "mean_score": float(evaluation.mean_score),
                                "score_variance": float(evaluation.score_variance),
                                "rollout_scores": [float(score) for score in evaluation.rollout_scores],
                                "root_query": None if evaluation.root_query is None else list(evaluation.root_query),
                                "strategy": evaluation.strategy,
                            }
                            for index, evaluation in enumerate(evaluations)
                        ]
                        for depth, evaluations in evaluations_by_depth.items()
                    }
                    for variant, evaluations_by_depth in evaluations_by_variant.items()
                },
                "realized_by_depth": realized_by_depth,
                "score_variant_metrics": score_variant_metrics,
            }
            records.append(record)
            records_handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            records_handle.flush()
            _append_log(
                config.log_path,
                f"completed probe {probe_index + 1}/{len(probes)} elapsed_sec={time.perf_counter() - probe_start:.1f}",
            )

    summary = {
        "config_path": str(config_path),
        "questioner_model": pair.questioner.model,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "num_trials": num_trials,
        "trial_offset": trial_offset,
        "num_probe_states": len(probes),
        "state_rounds": state_rounds,
        "depths": depths,
        "deployments": deployments,
        "target_num_candidates": config.location_target_num_candidates,
        "score_variants": score_variants,
        "location_seed": config.location_seed,
        "location_num_rounds": config.location_num_rounds,
        "location_strategy_num_rollouts": config.location_strategy_num_rollouts,
        "location_strategy_rollout_scoring_support_mode": config.location_strategy_rollout_scoring_support_mode,
        "location_strategy_rollout_score_mode": config.location_strategy_rollout_score_mode,
        "location_strategy_rollout_refresh_hypotheses_each_step": config.location_strategy_rollout_refresh_hypotheses_each_step,
        "location_strategy_rollout_final_refresh_enabled": config.location_strategy_rollout_final_refresh_enabled,
        "token_usage": summarize_llm_token_usage(config.log_path),
        "aggregate": _aggregate_variant_metrics(records, depths, score_variants, np.random.default_rng(seed + 67)),
        "records_path": str(records_path),
    }
    summary["gate_assessment"] = _gate_assessment(summary)
    summary_path = run_dir / "strategy_ranking_fidelity_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_dir = Path("results") / "ranking_fidelity"
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_report(report_dir / "REPORT.md", summary)
    _write_plot(Path("plots") / "strategy_ranking_fidelity.png", summary)
    _append_log(config.log_path, f"Summary: {summary_path}")
    _append_log(config.log_path, f"Records: {records_path}")
    _append_log(config.log_path, f"Report: {report_dir / 'REPORT.md'}")
    _append_log(config.log_path, "END")
    return run_dir


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def main() -> None:
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Measure StrategyEIG ranking fidelity against deployed strategies.")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--trial-offset", type=int, default=0)
    parser.add_argument("--state-rounds", type=_parse_int_list, default=[0, 3, 6])
    parser.add_argument("--depths", type=_parse_int_list, default=[2, 3, 5])
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--deployments", type=int, default=8)
    parser.add_argument(
        "--score-variants",
        default="baseline,configured",
        help="Comma-separated scoring variants to compare: baseline, configured",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    run_dir = run_strategy_ranking_fidelity(
        config,
        config_path=config_path,
        output_root=args.output_root,
        run_name=args.run_name,
        num_trials=args.num_trials,
        trial_offset=args.trial_offset,
        state_rounds=sorted(set(args.state_rounds)),
        depths=sorted(set(args.depths)),
        deployments=args.deployments,
        score_variants=[item.strip() for item in args.score_variants.split(",") if item.strip()],
        num_candidates=args.num_candidates,
    )
    print(f"[ranking-fidelity] Run directory: {run_dir.resolve()}")
    print(f"[ranking-fidelity] Total time: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
