from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import socket
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
from environments.location_finding.physics import round_positive_observation, signal_intensity_for_hypothesis
from environments.location_finding.strategy import (
    _strategy_entries_from_evaluations,
    evaluate_location_strategies_by_rollout_many,
    generate_location_strategy_roots_many,
)
from environments.location_finding.types import (
    Location,
    LocationObservation,
    LocationStrategyCandidate,
    LocationStrategyEvaluation,
    LocationStrategyLibrary,
    SourceConfig,
    _StrategyEvaluationRequest,
)
from helpers import Config, load_config, resolve_run_id
from scripts.llm_token_usage import summarize_llm_token_usage, token_usage_report_lines
from methods.eig import build_eig_method


@dataclass
class _DepthBranch:
    trial_index: int
    policy_label: str
    policy_kind: str
    policy_depth: int | None
    selection_depth: int | None
    hidden_state: np.ndarray
    belief_state: BeliefState[SourceConfig]
    history: list[tuple[Location, LocationObservation]] = field(default_factory=list)
    library: LocationStrategyLibrary = field(default_factory=LocationStrategyLibrary)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)


def _rounded_tuple(values: Any, *, digits: int = 10) -> Any:
    if isinstance(values, (list, tuple)):
        return tuple(_rounded_tuple(value, digits=digits) for value in values)
    if isinstance(values, np.ndarray):
        return _rounded_tuple(values.tolist(), digits=digits)
    if isinstance(values, (float, np.floating)):
        return round(float(values), digits)
    if isinstance(values, (int, np.integer)):
        return int(values)
    return values


def _belief_fingerprint(belief_state: BeliefState[SourceConfig]) -> tuple[Any, ...]:
    return tuple(
        (
            _rounded_tuple(hypothesis),
            round(float(probability), 12),
        )
        for hypothesis, probability in zip(belief_state.hypotheses, belief_state.probabilities)
    )


def _history_fingerprint(history: list[tuple[Location, LocationObservation]]) -> tuple[Any, ...]:
    return tuple(
        (
            _rounded_tuple(action),
            _rounded_tuple(observation.query),
            round(float(observation.value), 12),
        )
        for action, observation in history
    )


def _library_fingerprint(library: LocationStrategyLibrary) -> tuple[Any, ...]:
    return tuple(
        (
            entry.strategy,
            round(float(entry.mean_score), 12),
            round(float(entry.score_variance), 12),
            entry.root_query_fingerprint,
            int(entry.round_index),
            None if entry.root_query is None else _rounded_tuple(entry.root_query),
        )
        for entry in library.entries
    )


def _strategy_state_cache_key(branch: _DepthBranch) -> tuple[Any, ...]:
    return (
        int(branch.trial_index),
        _belief_fingerprint(branch.belief_state),
        _history_fingerprint(branch.history),
        _library_fingerprint(branch.library),
    )


def _group_strategy_branch_indices(
    branches: list[_DepthBranch],
    strategy_indices: list[int],
) -> list[list[int]]:
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for branch_idx in strategy_indices:
        grouped.setdefault(_strategy_state_cache_key(branches[branch_idx]), []).append(branch_idx)
    return list(grouped.values())


def _policy_specs(
    max_depth: int,
    *,
    include_myopic_controls: bool = False,
    strategy_depths: list[int] | None = None,
    myopic_control_depths: list[int] | None = None,
) -> list[tuple[str, str, int | None, int | None]]:
    selected_strategy_depths = strategy_depths or list(range(1, max_depth + 1))
    if not selected_strategy_depths:
        raise ValueError("strategy_depths must contain at least one depth")
    for depth in selected_strategy_depths:
        if depth <= 0 or depth > max_depth:
            raise ValueError("strategy_depths must be between 1 and max_depth")

    selected_myopic_depths = (
        myopic_control_depths
        if myopic_control_depths is not None
        else list(range(2, max_depth + 1))
    )
    for depth in selected_myopic_depths:
        if depth <= 1 or depth > max_depth:
            raise ValueError("myopic_control_depths must be between 2 and max_depth")

    specs: list[tuple[str, str, int | None, int | None]] = [
        ("naive", "naive", None, None),
        ("naive+belief", "naive+belief", None, None),
        ("EIG", "EIG", None, None),
    ]
    specs.extend(
        (f"StrategyEIG-d{depth}", "StrategyEIG", depth, depth)
        for depth in selected_strategy_depths
    )
    if include_myopic_controls:
        specs.extend(
            (f"StrategyEIG-myopic-d{depth}", "StrategyEIG-myopic-control", depth, 1)
            for depth in selected_myopic_depths
        )
    return specs


def _parse_depth_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    depths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not depths:
        raise ValueError("depth list must contain at least one integer")
    if len(set(depths)) != len(depths):
        raise ValueError("depth list must not contain duplicate depths")
    return depths


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


def _candidate_payload(candidate: LocationStrategyCandidate, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "strategy": candidate.strategy,
        "root_query": None if candidate.root_query is None else list(candidate.root_query),
    }


def _evaluation_payload(evaluation: LocationStrategyEvaluation, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "strategy": evaluation.strategy,
        "mean_score": float(evaluation.mean_score),
        "score_variance": float(evaluation.score_variance),
        "root_query_fingerprint": evaluation.root_query_fingerprint,
        "root_query": None if evaluation.root_query is None else list(evaluation.root_query),
        "rollout_scores": [float(score) for score in evaluation.rollout_scores],
    }


def _entropy(probabilities: list[float] | np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0.0]
    if len(values) == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def _truth_augmented_state(
    belief_state: BeliefState[SourceConfig],
    history: list[tuple[Location, LocationObservation]],
    hidden_state: np.ndarray,
    config: Config,
) -> BeliefState[SourceConfig]:
    truth = _source_config_from_hidden_state(hidden_state)
    support = list(belief_state.hypotheses)
    if truth not in support:
        support.append(truth)
    observations = [observation for _action, observation in history]
    return build_location_belief_state_unpruned(support, observations, config)


def _truth_log_probability(
    belief_state: BeliefState[SourceConfig],
    hidden_state: np.ndarray,
) -> float:
    truth = _source_config_from_hidden_state(hidden_state)
    for hypothesis, probability in zip(belief_state.hypotheses, belief_state.probabilities):
        if hypothesis == truth:
            return math.log(max(float(probability), 1e-300))
    return math.log(1e-300)


def _observe_with_noise_z(
    action: Location,
    hidden_state: np.ndarray,
    config: Config,
    *,
    noise_z: float,
) -> LocationObservation:
    truth = _source_config_from_hidden_state(hidden_state)
    mean = signal_intensity_for_hypothesis(truth, action, config=config)
    value = round_positive_observation(mean * math.exp(config.location_noise_sd * noise_z), 2)
    return LocationObservation(query=action, value=float(value))


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        average_rank = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            ranks[indexed[k][0]] = average_rank
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
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _source_config_from_hidden_state(hidden_state: np.ndarray) -> SourceConfig:
    return tuple(tuple(float(coord) for coord in row) for row in np.asarray(hidden_state, dtype=float))


def _candidate_realized_entropy_drops(
    branch: _DepthBranch,
    candidates: list[LocationStrategyCandidate],
    config: Config,
    *,
    noise_z: float,
) -> list[float | None]:
    support = list(branch.belief_state.hypotheses)
    if not support:
        return [None for _candidate in candidates]
    previous_observations = [observation for _action, observation in branch.history]
    previous_state = build_location_belief_state_unpruned(support, previous_observations, config)
    previous_entropy = _entropy(previous_state.probabilities)
    drops: list[float | None] = []
    for candidate in candidates:
        if candidate.root_query is None:
            drops.append(None)
            continue
        observation = _observe_with_noise_z(
            candidate.root_query,
            branch.hidden_state,
            config,
            noise_z=noise_z,
        )
        next_state = build_location_belief_state_unpruned(
            support,
            previous_observations + [observation],
            config,
        )
        drops.append(float(previous_entropy - _entropy(next_state.probabilities)))
    return drops


def _fidelity_by_eval_depth(
    evaluations_by_depth: dict[int, list[LocationStrategyEvaluation]],
    realized_drops: list[float | None],
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for eval_depth, evaluations in evaluations_by_depth.items():
        estimated: list[float] = []
        realized: list[float] = []
        for index, evaluation in enumerate(evaluations):
            if index >= len(realized_drops) or realized_drops[index] is None:
                continue
            estimated.append(float(evaluation.mean_score))
            realized.append(float(realized_drops[index]))
        result[str(eval_depth)] = {
            "n": len(estimated),
            "spearman": _spearman(estimated, realized),
            "pearson": _pearson(estimated, realized),
        }
    return result


def _metric_summary(per_trial: list[dict[str, Any]], max_depth: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for depth in range(1, max_depth + 1):
        depth_records = [record for record in per_trial if record["policy_depth"] == depth]
        if not depth_records:
            continue
        metric_names = sorted(
            {
                metric_name
                for record in depth_records
                for round_metrics in record["round_metrics"]
                for metric_name in round_metrics
            }
        )
        depth_summary: dict[str, Any] = {}
        for metric_name in metric_names:
            max_rounds = max(len(record["round_metrics"]) for record in depth_records)
            means: list[float] = []
            stds: list[float] = []
            for round_index in range(max_rounds):
                values = [
                    float(record["round_metrics"][round_index][metric_name])
                    for record in depth_records
                    if round_index < len(record["round_metrics"])
                    and metric_name in record["round_metrics"][round_index]
                ]
                means.append(float(np.mean(values)) if values else float("nan"))
                stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            finals = [
                float(record["round_metrics"][-1][metric_name])
                for record in depth_records
                if record["round_metrics"] and metric_name in record["round_metrics"][-1]
            ]
            depth_summary[metric_name] = {
                "mean_trace": means,
                "std_trace": stds,
                "final_mean": float(np.mean(finals)) if finals else float("nan"),
                "final_std": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
            }
        summary[str(depth)] = depth_summary
    return summary


def _metric_summary_by_policy(per_trial: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    policy_labels = sorted({str(record["policy_label"]) for record in per_trial})
    for policy_label in policy_labels:
        policy_records = [record for record in per_trial if record["policy_label"] == policy_label]
        metric_names = sorted(
            {
                metric_name
                for record in policy_records
                for round_metrics in record["round_metrics"]
                for metric_name in round_metrics
            }
        )
        policy_summary: dict[str, Any] = {}
        for metric_name in metric_names:
            max_rounds = max((len(record["round_metrics"]) for record in policy_records), default=0)
            means: list[float] = []
            stds: list[float] = []
            for round_index in range(max_rounds):
                values = [
                    float(record["round_metrics"][round_index][metric_name])
                    for record in policy_records
                    if round_index < len(record["round_metrics"])
                    and metric_name in record["round_metrics"][round_index]
                ]
                means.append(float(np.mean(values)) if values else float("nan"))
                stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            finals = [
                float(record["round_metrics"][-1][metric_name])
                for record in policy_records
                if record["round_metrics"] and metric_name in record["round_metrics"][-1]
            ]
            policy_summary[metric_name] = {
                "mean_trace": means,
                "std_trace": stds,
                "final_mean": float(np.mean(finals)) if finals else float("nan"),
                "final_std": float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0,
            }
        summary[policy_label] = policy_summary
    return summary


def _paired_delta_summary_by_policy(
    per_trial: list[dict[str, Any]],
    *,
    baseline_label: str = "EIG",
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    rng = rng or np.random.default_rng(0)
    baseline_by_trial = {
        int(record["trial_index"]): record
        for record in per_trial
        if record["policy_label"] == baseline_label and record["round_metrics"]
    }
    result: dict[str, Any] = {}
    for policy_label in sorted({str(record["policy_label"]) for record in per_trial}):
        if policy_label == baseline_label:
            continue
        policy_records = [
            record
            for record in per_trial
            if record["policy_label"] == policy_label
            and record["round_metrics"]
            and int(record["trial_index"]) in baseline_by_trial
        ]
        metric_deltas: dict[str, Any] = {}
        for metric_name in ("source_rmse", "posterior_entropy", "truth_log_probability"):
            deltas = []
            auc_deltas = []
            for record in policy_records:
                baseline = baseline_by_trial[int(record["trial_index"])]
                if (
                    metric_name not in record["round_metrics"][-1]
                    or metric_name not in baseline["round_metrics"][-1]
                ):
                    continue
                final_delta = (
                    float(record["round_metrics"][-1][metric_name])
                    - float(baseline["round_metrics"][-1][metric_name])
                )
                policy_trace = [
                    float(round_metrics[metric_name])
                    for round_metrics in record["round_metrics"]
                    if metric_name in round_metrics
                ]
                baseline_trace = [
                    float(round_metrics[metric_name])
                    for round_metrics in baseline["round_metrics"]
                    if metric_name in round_metrics
                ]
                trace_len = min(len(policy_trace), len(baseline_trace))
                deltas.append(final_delta)
                auc_deltas.append(float(np.sum(policy_trace[:trace_len]) - np.sum(baseline_trace[:trace_len])))
            values = np.asarray(deltas, dtype=float)
            auc_values = np.asarray(auc_deltas, dtype=float)
            metric_deltas[metric_name] = {
                "n": int(values.size),
                "final_delta_mean": float(np.mean(values)) if values.size else float("nan"),
                "final_delta_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "final_delta_ci95": _bootstrap_mean_ci(values, rng=rng),
                "wilcoxon_signed_rank_p": _wilcoxon_signed_rank_pvalue(values),
                "auc_delta_mean": float(np.mean(auc_values)) if auc_values.size else float("nan"),
                "auc_delta_std": float(np.std(auc_values, ddof=1)) if auc_values.size > 1 else 0.0,
                "deltas": [float(value) for value in values],
            }
        result[policy_label] = metric_deltas
    return result


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    confidence: float = 0.95,
    resamples: int = 2000,
) -> list[float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [None, None]
    if values.size == 1:
        value = float(values[0])
        return [value, value]
    draws = rng.choice(values, size=(resamples, values.size), replace=True)
    means = np.mean(draws, axis=1)
    alpha = 1.0 - confidence
    return [
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    ]


def _wilcoxon_signed_rank_pvalue(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > 1e-12]
    n = int(values.size)
    if n == 0:
        return None
    ranks = np.asarray(_rankdata(np.abs(values).tolist()), dtype=float)
    w_plus = float(np.sum(ranks[values > 0.0]))
    w_minus = float(np.sum(ranks[values < 0.0]))
    statistic = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    if variance <= 0.0:
        return None
    z = (statistic - mean + 0.5) / math.sqrt(variance)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _fidelity_summary(decisions: list[dict[str, Any]], max_depth: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for policy_depth in range(1, max_depth + 1):
        depth_decisions = [record for record in decisions if record["policy_depth"] == policy_depth]
        by_eval_depth: dict[str, Any] = {}
        for eval_depth in range(1, max_depth + 1):
            key = str(eval_depth)
            spearman_values = [
                float(record["ranking_fidelity_by_eval_depth"][key]["spearman"])
                for record in depth_decisions
                if key in record["ranking_fidelity_by_eval_depth"]
                and record["ranking_fidelity_by_eval_depth"][key]["spearman"] is not None
            ]
            pearson_values = [
                float(record["ranking_fidelity_by_eval_depth"][key]["pearson"])
                for record in depth_decisions
                if key in record["ranking_fidelity_by_eval_depth"]
                and record["ranking_fidelity_by_eval_depth"][key]["pearson"] is not None
            ]
            by_eval_depth[key] = {
                "n": len(spearman_values),
                "spearman_mean": float(np.mean(spearman_values)) if spearman_values else None,
                "pearson_mean": float(np.mean(pearson_values)) if pearson_values else None,
            }
        summary[str(policy_depth)] = by_eval_depth
    return summary


def _copy_config_with_depth(config: Config, depth: int) -> Config:
    depth_config = copy.copy(config)
    depth_config.location_strategy_planning_depth = int(depth)
    return depth_config


def _select_best(evaluations: list[LocationStrategyEvaluation]) -> tuple[int, LocationStrategyEvaluation]:
    if not evaluations:
        raise ValueError("Strategy root diagnostic received no evaluations")
    best_index, best_evaluation = max(
        enumerate(evaluations),
        key=lambda item: item[1].mean_score,
    )
    if best_evaluation.root_query is None:
        raise ValueError("Strategy root diagnostic selected an evaluation without a root query")
    return best_index, best_evaluation


def _write_depth_sweep_report(report_path: Path, summary: dict[str, Any]) -> None:
    run_metadata = summary.get("run_metadata", {})
    paired_trial_delta_plot_path = (
        summary.get("public_paired_trial_delta_plot_path")
        or summary.get("paired_trial_delta_plot_path")
    )
    lines = [
        "# Fixed-Root Location Depth Sweep Report",
        "",
        f"- Config: `{summary['config_path']}`",
        f"- Trials: {summary['num_trials']}",
        f"- Rounds: {summary['num_rounds']}",
        f"- Seed: {summary['location_seed']}",
        f"- Source prior: `{summary['location_source_prior']}`",
        f"- Signal model: `{summary['location_signal_model']}`",
        f"- Max step radius: {summary['location_max_step_radius']}",
        f"- Matched-compute myopic controls: {bool(summary.get('include_myopic_controls', False))}",
        f"- Questioner model: `{run_metadata.get('questioner_model')}`",
        f"- Host: `{run_metadata.get('hostname')}`",
        f"- SLURM job: `{run_metadata.get('slurm_job_id')}` on partition `{run_metadata.get('slurm_partition')}`",
        f"- CUDA visible devices: `{run_metadata.get('cuda_visible_devices')}`",
        "",
        *token_usage_report_lines(summary.get("token_usage")),
        "",
        *(
            [f"- Paired trial delta plot: `{paired_trial_delta_plot_path}`", ""]
            if paired_trial_delta_plot_path
            else []
        ),
        "## Final RMSE",
        "",
        "| policy | final mean | final std |",
        "|---|---:|---:|",
    ]
    aggregate = summary.get("aggregate", {})
    for policy_label in sorted(aggregate):
        rmse = aggregate[policy_label].get("source_rmse", {})
        lines.append(
            f"| `{policy_label}` | {float(rmse.get('final_mean', float('nan'))):.4f} | "
            f"{float(rmse.get('final_std', float('nan'))):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Paired Delta vs EIG",
            "",
            "Negative RMSE deltas mean the policy beat greedy EIG on that paired trial set.",
            "",
            "| policy | metric | n | final delta mean | 95% CI | Wilcoxon p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    paired = summary.get("paired_delta_vs_eig", {})
    for policy_label in sorted(paired):
        for metric_name in ("source_rmse", "posterior_entropy", "truth_log_probability"):
            metric = paired[policy_label].get(metric_name, {})
            ci = metric.get("final_delta_ci95", [None, None])
            ci_text = (
                "n/a"
                if ci[0] is None or ci[1] is None
                else f"[{float(ci[0]):.4f}, {float(ci[1]):.4f}]"
            )
            p_value = metric.get("wilcoxon_signed_rank_p")
            p_text = "n/a" if p_value is None else f"{float(p_value):.4g}"
            lines.append(
                f"| `{policy_label}` | `{metric_name}` | {int(metric.get('n', 0))} | "
                f"{float(metric.get('final_delta_mean', float('nan'))):.4f} | {ci_text} | {p_text} |"
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_depth_sweep(summary: dict[str, Any], plot_path: Path) -> None:
    import matplotlib.pyplot as plt

    aggregate = summary.get("aggregate", {})
    if not aggregate:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for policy_label in sorted(aggregate):
        rmse = aggregate[policy_label].get("source_rmse", {})
        trace = rmse.get("mean_trace", [])
        if not trace:
            continue
        rounds = np.arange(1, len(trace) + 1)
        axes[0].plot(rounds, trace, marker="o", linewidth=1.5, label=policy_label)
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("mean RMSE")
    axes[0].set_title("RMSE trace")
    axes[0].legend(fontsize=7)

    paired = summary.get("paired_delta_vs_eig", {})
    labels = []
    means = []
    lows = []
    highs = []
    for policy_label in sorted(paired):
        metric = paired[policy_label].get("source_rmse", {})
        if int(metric.get("n", 0)) <= 0:
            continue
        mean = float(metric.get("final_delta_mean", float("nan")))
        ci = metric.get("final_delta_ci95", [None, None])
        if not math.isfinite(mean) or ci[0] is None or ci[1] is None:
            continue
        labels.append(policy_label)
        means.append(mean)
        lows.append(mean - float(ci[0]))
        highs.append(float(ci[1]) - mean)
    if labels:
        x = np.arange(len(labels))
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].bar(x, means, yerr=np.asarray([lows, highs]), capsize=3)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("final RMSE delta vs EIG")
    axes[1].set_title("Paired final delta")

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _paired_trial_delta_rows(
    per_trial: list[dict[str, Any]],
    *,
    baseline_label: str = "EIG",
    metric_name: str = "source_rmse",
) -> list[dict[str, Any]]:
    baseline_by_trial = {
        int(record["trial_index"]): record
        for record in per_trial
        if record.get("policy_label") == baseline_label
        and record.get("round_metrics")
        and metric_name in record["round_metrics"][-1]
    }
    rows: list[dict[str, Any]] = []
    for record in per_trial:
        policy_label = str(record.get("policy_label"))
        if policy_label == baseline_label or not record.get("round_metrics"):
            continue
        trial_index = int(record["trial_index"])
        baseline = baseline_by_trial.get(trial_index)
        if baseline is None or metric_name not in record["round_metrics"][-1]:
            continue
        policy_value = float(record["round_metrics"][-1][metric_name])
        baseline_value = float(baseline["round_metrics"][-1][metric_name])
        rows.append(
            {
                "trial_index": trial_index,
                "policy_label": policy_label,
                "metric_name": metric_name,
                "policy_value": policy_value,
                "baseline_label": baseline_label,
                "baseline_value": baseline_value,
                "delta": policy_value - baseline_value,
            }
        )
    return sorted(rows, key=lambda row: (str(row["policy_label"]), int(row["trial_index"])))


def _plot_paired_trial_differences(summary: dict[str, Any], plot_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _paired_trial_delta_rows(summary.get("per_trial", []))
    if not rows:
        return
    labels = sorted({str(row["policy_label"]) for row in rows})
    label_to_x = {label: index for index, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(labels)), 4.5))
    ax.axhline(0.0, color="black", linewidth=0.8)
    rng = np.random.default_rng(0)
    for label in labels:
        label_rows = [row for row in rows if row["policy_label"] == label]
        x = np.asarray([label_to_x[label]] * len(label_rows), dtype=float)
        jitter = rng.uniform(-0.15, 0.15, size=len(label_rows))
        y = np.asarray([float(row["delta"]) for row in label_rows], dtype=float)
        ax.scatter(x + jitter, y, s=24, alpha=0.75)
        ax.hlines(float(np.mean(y)), label_to_x[label] - 0.25, label_to_x[label] + 0.25, linewidth=2.0)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("per-trial final RMSE delta vs EIG")
    ax.set_title("Paired per-trial final RMSE differences")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _run_metadata(config: Config) -> dict[str, Any]:
    questioner = config.model_pairs[0].questioner if config.model_pairs else None
    return {
        "questioner_model": getattr(questioner, "model", None),
        "questioner_thinking": getattr(questioner, "thinking", None),
        "questioner_thinking_max_new_tokens": getattr(questioner, "thinking_max_new_tokens", None),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
    }


def run_fixed_root_depth_sweep(
    config: Config,
    *,
    config_path: Path,
    output_root: Path,
    report_dir: Path | None,
    plot_dir: Path | None,
    max_depth: int,
    num_trials: int | None,
    num_rounds: int | None,
    run_name: str | None,
    include_myopic_controls: bool = False,
    strategy_depths: list[int] | None = None,
    eval_depths: list[int] | None = None,
    myopic_control_depths: list[int] | None = None,
) -> Path:
    if config.task != "location_finding":
        raise ValueError("location_fixed_root_depth_sweep only supports task=location_finding")
    if not config.model_pairs:
        raise ValueError("Config must include at least one model pair")
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    selected_eval_depths = eval_depths or list(range(1, max_depth + 1))
    if not selected_eval_depths:
        raise ValueError("eval_depths must contain at least one depth")
    for depth in selected_eval_depths:
        if depth <= 0 or depth > max_depth:
            raise ValueError("eval_depths must be between 1 and max_depth")

    register_defaults()
    config.run_id = resolve_run_id()
    config.location_num_trials = int(num_trials or config.location_num_trials)
    config.location_num_rounds = int(num_rounds or config.location_num_rounds)
    config.method_names = ["naive", "naive+belief", "EIG", "StrategyEIG+root"]

    stem = run_name or f"{config.run_id}_fixed_root_depth_sweep_d{max_depth}"
    run_dir = output_root / stem
    run_dir.mkdir(parents=True, exist_ok=False)
    config.log_path = run_dir / "run.log"
    config.log_path.write_text(
        f"START TIME: {datetime.now().isoformat()}\n"
        f"Config file: {config_path}\n"
        f"Fixed-root depth sweep max_depth={max_depth}\n",
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
                    "diagnostic": "fixed_root_depth_sweep",
                    "max_depth": max_depth,
                },
            )
    except Exception as exc:
        with config.log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"Warning: could not initialize disabled wandb run: {type(exc).__name__}: {exc}\n")

    pair = config.model_pairs[0]
    from model import build_model_adapter

    questioner = build_model_adapter(pair.questioner, config=config)
    env = LocationBEDEnvironment(config=config)
    env.validate_config(config)
    env.rng = np.random.default_rng(config.location_seed)

    seed = config.location_seed
    master_rng = np.random.default_rng(seed)
    paired_noise_zs = master_rng.normal(
        0.0,
        1.0,
        size=(config.location_num_trials, config.location_num_rounds),
    )
    branches: list[_DepthBranch] = []
    per_trial: list[dict[str, Any]] = []
    policy_specs = _policy_specs(
        max_depth,
        include_myopic_controls=include_myopic_controls,
        strategy_depths=strategy_depths,
        myopic_control_depths=myopic_control_depths,
    )
    missing_selection_depths = sorted(
        {
            selection_depth
            for _label, kind, _policy_depth, selection_depth in policy_specs
            if kind in {"StrategyEIG", "StrategyEIG-myopic-control"}
            and selection_depth is not None
            and selection_depth not in selected_eval_depths
        }
    )
    if missing_selection_depths:
        raise ValueError(
            "eval_depths must include every StrategyEIG selection depth; missing "
            + ",".join(str(depth) for depth in missing_selection_depths)
        )
    for trial_index in range(config.location_num_trials):
        hidden_state = env.sample_hidden_state_for_trial(trial_index, master_rng)
        initial_belief = env.initial_belief_state(questioner, config)
        for policy_label, policy_kind, policy_depth, selection_depth in policy_specs:
            branch_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
            branches.append(
                _DepthBranch(
                    trial_index=trial_index,
                    policy_label=policy_label,
                    policy_kind=policy_kind,
                    policy_depth=policy_depth,
                    selection_depth=selection_depth,
                    hidden_state=hidden_state,
                    belief_state=initial_belief,
                    rng=np.random.default_rng(branch_seed),
                )
            )
            per_trial.append(
                {
                    "trial_index": trial_index,
                    "policy_label": policy_label,
                    "policy_kind": policy_kind,
                    "policy_depth": policy_depth,
                    "selection_depth": selection_depth,
                    "hidden_state": _jsonable(hidden_state),
                    "round_metrics": [],
                    "history": [],
                }
            )

    decisions_path = run_dir / "fixed_root_depth_sweep_decisions.jsonl"
    decision_records: list[dict[str, Any]] = []
    with decisions_path.open("w", encoding="utf-8") as decisions_handle:
        for round_index in range(config.location_num_rounds):
            selected_scores = [0.0 for _branch in branches]

            for method_name in ("naive", "naive+belief"):
                branch_indices = [
                    idx for idx, branch in enumerate(branches) if branch.policy_kind == method_name
                ]
                if not branch_indices:
                    continue
                actions = env.generate_naive_actions_many(
                    [branches[idx].belief_state for idx in branch_indices],
                    [branches[idx].history for idx in branch_indices],
                    questioner,
                    config,
                    method_name=method_name,
                )
                for branch_idx, action in zip(branch_indices, actions):
                    branch = branches[branch_idx]
                    paired_noise_z = float(paired_noise_zs[branch.trial_index, round_index])
                    observation = _observe_with_noise_z(
                        action,
                        branch.hidden_state,
                        config,
                        noise_z=paired_noise_z,
                    )
                    branch.history.append((action, observation))
                    decision_record = {
                        "trial_index": branch.trial_index,
                        "policy_label": branch.policy_label,
                        "policy_kind": branch.policy_kind,
                        "policy_depth": branch.policy_depth,
                        "selection_depth": branch.selection_depth,
                        "round_index": round_index,
                        "selected_index": None,
                        "selected_eig": 0.0,
                        "selected_strategy": None,
                        "selected_root_query": list(action),
                        "observation_noise_z": paired_noise_z,
                        "observation": _jsonable(observation),
                        "candidates": [],
                        "evaluations_by_depth": {},
                    }
                    decision_records.append(decision_record)
                    decisions_handle.write(json.dumps(decision_record, sort_keys=True) + "\n")
                    decisions_handle.flush()

            eig_indices = [
                idx for idx, branch in enumerate(branches) if branch.policy_kind == "EIG"
            ]
            if eig_indices:
                eig_candidates_many = env.generate_candidate_actions_many(
                    [branches[idx].belief_state for idx in eig_indices],
                    [branches[idx].history for idx in eig_indices],
                    questioner,
                    config,
                )
                eig_method = build_eig_method(config, env)
                eig_chosen_many = eig_method.select_actions(
                    eig_candidates_many,
                    [branches[idx].belief_state for idx in eig_indices],
                    env,
                    questioner,
                    [branches[idx].history for idx in eig_indices],
                    config,
                )
                for branch_idx, candidates, chosen in zip(eig_indices, eig_candidates_many, eig_chosen_many):
                    branch = branches[branch_idx]
                    paired_noise_z = float(paired_noise_zs[branch.trial_index, round_index])
                    observation = _observe_with_noise_z(
                        chosen.action,
                        branch.hidden_state,
                        config,
                        noise_z=paired_noise_z,
                    )
                    branch.history.append((chosen.action, observation))
                    selected_scores[branch_idx] = float(chosen.score)
                    all_scores = []
                    if chosen.extras and "all_scores" in chosen.extras:
                        all_scores = [float(score) for score in chosen.extras["all_scores"]]
                    selected_index = int(np.argmax(all_scores)) if all_scores else None
                    decision_record = {
                        "trial_index": branch.trial_index,
                        "policy_label": branch.policy_label,
                        "policy_kind": branch.policy_kind,
                        "policy_depth": branch.policy_depth,
                        "selection_depth": branch.selection_depth,
                        "round_index": round_index,
                        "selected_index": selected_index,
                        "selected_eig": float(chosen.score),
                        "selected_strategy": None,
                        "selected_root_query": list(chosen.action),
                        "observation_noise_z": paired_noise_z,
                        "observation": _jsonable(observation),
                        "candidates": [
                            {
                                "index": index,
                                "location": list(candidate),
                                "score": all_scores[index] if index < len(all_scores) else None,
                            }
                            for index, candidate in enumerate(candidates)
                        ],
                        "evaluations_by_depth": {},
                    }
                    decision_records.append(decision_record)
                    decisions_handle.write(json.dumps(decision_record, sort_keys=True) + "\n")
                    decisions_handle.flush()

            strategy_indices = [
                idx
                for idx, branch in enumerate(branches)
                if branch.policy_kind in {"StrategyEIG", "StrategyEIG-myopic-control"}
            ]
            candidates_by_branch: dict[int, list[LocationStrategyCandidate]] = {}
            evaluations_by_branch_and_depth: dict[int, dict[int, list[LocationStrategyEvaluation]]] = {}
            if strategy_indices:
                strategy_groups = _group_strategy_branch_indices(branches, strategy_indices)
                representative_indices = [group[0] for group in strategy_groups]
                strategy_requests = [
                    (
                        branches[idx].belief_state,
                        [obs for _action, obs in branches[idx].history],
                        branches[idx].library,
                    )
                    for idx in representative_indices
                ]
                strategy_candidates_many = generate_location_strategy_roots_many(questioner, strategy_requests, config)
                for group, candidates in zip(strategy_groups, strategy_candidates_many):
                    for branch_idx in group:
                        candidates_by_branch[branch_idx] = candidates
                evaluations_by_branch_and_depth = {idx: {} for idx in strategy_indices}
                for eval_depth in selected_eval_depths:
                    eval_config = _copy_config_with_depth(config, eval_depth)
                    eval_requests: list[_StrategyEvaluationRequest] = []
                    request_groups: list[list[int]] = []
                    for group in strategy_groups:
                        branch_idx = group[0]
                        branch = branches[branch_idx]
                        candidates = candidates_by_branch[branch_idx]
                        strategies = [candidate.strategy for candidate in candidates]
                        root_queries = [candidate.root_query for candidate in candidates]
                        eval_seed = (
                            int(config.location_seed or 0)
                            + 1_000_003 * (branch.trial_index + 1)
                            + 10_007 * (eval_depth + 1)
                            + 101 * (round_index + 1)
                        )
                        eval_requests.append(
                            _StrategyEvaluationRequest(
                                strategies=strategies,
                                belief_state=branch.belief_state,
                                observations=[obs for _action, obs in branch.history],
                                rng=np.random.default_rng(eval_seed),
                                root_queries=root_queries,
                            )
                        )
                        request_groups.append(group)
                    evaluations_many = evaluate_location_strategies_by_rollout_many(
                        questioner,
                        eval_requests,
                        eval_config,
                    )
                    for group, evaluations in zip(request_groups, evaluations_many):
                        for branch_idx in group:
                            evaluations_by_branch_and_depth[branch_idx][eval_depth] = evaluations

            for branch_idx in strategy_indices:
                branch = branches[branch_idx]
                assert branch.policy_depth is not None
                assert branch.selection_depth is not None
                selected_depth_evals = evaluations_by_branch_and_depth[branch_idx][branch.selection_depth]
                selected_index, selected_eval = _select_best(selected_depth_evals)
                selected_action = selected_eval.root_query
                assert selected_action is not None

                diagnostic_noise_seed = (
                    int(config.location_seed or 0)
                    + 2_000_003 * (branch.trial_index + 1)
                    + 20_011 * ((branch.policy_depth or 0) + 1)
                    + 503 * (round_index + 1)
                )
                diagnostic_noise_z = float(np.random.default_rng(diagnostic_noise_seed).normal(0.0, 1.0))
                realized_drops = _candidate_realized_entropy_drops(
                    branch,
                    candidates_by_branch[branch_idx],
                    config,
                    noise_z=diagnostic_noise_z,
                )
                fidelity_by_depth = _fidelity_by_eval_depth(
                    evaluations_by_branch_and_depth[branch_idx],
                    realized_drops,
                )

                branch.library.replace_entries(_strategy_entries_from_evaluations(selected_depth_evals, round_index))
                paired_noise_z = float(paired_noise_zs[branch.trial_index, round_index])
                observation = _observe_with_noise_z(
                    selected_action,
                    branch.hidden_state,
                    config,
                    noise_z=paired_noise_z,
                )
                branch.history.append((selected_action, observation))
                selected_scores[branch_idx] = float(selected_eval.mean_score)

                selected_by_eval_depth = {
                    str(eval_depth): _select_best(evaluations)[0]
                    for eval_depth, evaluations in evaluations_by_branch_and_depth[branch_idx].items()
                    if evaluations
                }
                decision_record = {
                    "trial_index": branch.trial_index,
                    "policy_label": branch.policy_label,
                        "policy_kind": branch.policy_kind,
                        "policy_depth": branch.policy_depth,
                        "selection_depth": branch.selection_depth,
                        "round_index": round_index,
                        "selected_index": selected_index,
                        "selected_eig": float(selected_eval.mean_score),
                    "selected_strategy": selected_eval.strategy,
                    "selected_root_query": list(selected_action),
                    "observation_noise_z": paired_noise_z,
                    "observation": _jsonable(observation),
                    "candidate_realized_entropy_drop": [
                        None if value is None else float(value)
                        for value in realized_drops
                    ],
                    "ranking_fidelity_by_eval_depth": fidelity_by_depth,
                    "selected_by_eval_depth": selected_by_eval_depth,
                    "candidates": [
                        _candidate_payload(candidate, index)
                        for index, candidate in enumerate(candidates_by_branch[branch_idx])
                    ],
                    "evaluations_by_depth": {
                        str(eval_depth): [
                            _evaluation_payload(evaluation, index)
                            for index, evaluation in enumerate(evaluations)
                        ]
                        for eval_depth, evaluations in evaluations_by_branch_and_depth[branch_idx].items()
                    },
                }
                decision_records.append(decision_record)
                decisions_handle.write(json.dumps(decision_record, sort_keys=True) + "\n")
                decisions_handle.flush()

            new_beliefs = env.update_belief_states(
                [branch.belief_state for branch in branches],
                [branch.history for branch in branches],
                questioner,
                config,
            )
            for branch, new_belief, selected_score in zip(branches, new_beliefs, selected_scores):
                branch.belief_state = new_belief
                metrics = env.round_metrics(branch.belief_state, branch.history, branch.hidden_state)
                truth_augmented_state = _truth_augmented_state(
                    branch.belief_state,
                    branch.history,
                    branch.hidden_state,
                    config,
                )
                metrics["posterior_entropy"] = _entropy(truth_augmented_state.probabilities)
                metrics["truth_log_probability"] = _truth_log_probability(
                    truth_augmented_state,
                    branch.hidden_state,
                )
                metrics["selected_eig"] = float(selected_score)
                branch_record = next(
                    record
                    for record in per_trial
                    if record["trial_index"] == branch.trial_index
                    and record["policy_label"] == branch.policy_label
                )
                branch_record["round_metrics"].append({key: float(value) for key, value in metrics.items()})
                branch_record["history"] = [
                    {
                        "action": list(action),
                        "observation": _jsonable(observation),
                    }
                    for action, observation in branch.history
                ]

            with config.log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"completed round {round_index + 1}/{config.location_num_rounds}\n")

    metrics_path = run_dir / "fixed_root_depth_sweep_metrics.json"
    summary = {
        "config_path": str(config_path),
        "max_depth": max_depth,
        "strategy_depths": strategy_depths or list(range(1, max_depth + 1)),
        "eval_depths": selected_eval_depths,
        "myopic_control_depths": (
            myopic_control_depths
            if myopic_control_depths is not None
            else (list(range(2, max_depth + 1)) if include_myopic_controls else [])
        ),
        "num_trials": config.location_num_trials,
        "num_rounds": config.location_num_rounds,
        "location_seed": config.location_seed,
        "location_source_prior": config.location_source_prior,
        "location_source_radius": config.location_source_radius,
        "location_signal_model": config.location_signal_model,
        "location_signal_lengthscale": config.location_signal_lengthscale,
        "location_signal_amplitude": config.location_signal_amplitude,
        "location_noise_sd": config.location_noise_sd,
        "location_max_step_radius": config.location_max_step_radius,
        "location_strategy_num_rollouts": config.location_strategy_num_rollouts,
        "location_strategy_num_candidates": config.location_strategy_num_candidates,
        "location_strategy_discount_factor": config.location_strategy_discount_factor,
        "location_strategy_rollout_score_mode": config.location_strategy_rollout_score_mode,
        "location_strategy_rollout_scoring_support_mode": config.location_strategy_rollout_scoring_support_mode,
        "location_strategy_rollout_refresh_hypotheses_each_step": config.location_strategy_rollout_refresh_hypotheses_each_step,
        "include_myopic_controls": include_myopic_controls,
        "token_usage": summarize_llm_token_usage(config.log_path),
        "run_metadata": _run_metadata(config),
        "per_trial": per_trial,
        "aggregate": _metric_summary_by_policy(per_trial),
        "aggregate_by_strategy_depth": _metric_summary(
            [record for record in per_trial if record["policy_kind"] == "StrategyEIG"],
            max_depth,
        ),
        "paired_delta_vs_eig": _paired_delta_summary_by_policy(
            per_trial,
            baseline_label="EIG",
            rng=np.random.default_rng(int(config.location_seed or 0) + 99_991),
        ),
        "ranking_fidelity": _fidelity_summary(decision_records, max_depth),
    }
    report_path = run_dir / "REPORT.md"
    plot_path = run_dir / "fixed_root_depth_sweep.png"
    paired_trial_delta_plot_path = run_dir / "paired_trial_rmse_deltas.png"
    try:
        _plot_depth_sweep(summary, plot_path)
        summary["plot_path"] = str(plot_path)
    except Exception as exc:
        summary["plot_error"] = repr(exc)
    try:
        _plot_paired_trial_differences(summary, paired_trial_delta_plot_path)
        if paired_trial_delta_plot_path.exists():
            summary["paired_trial_delta_plot_path"] = str(paired_trial_delta_plot_path)
    except Exception as exc:
        summary["paired_trial_delta_plot_error"] = repr(exc)
    _write_depth_sweep_report(report_path, summary)
    if plot_dir is not None:
        public_plot_path = plot_dir / f"{stem}_depth_sweep.png"
        try:
            _plot_depth_sweep(summary, public_plot_path)
            summary["public_plot_path"] = str(public_plot_path)
        except Exception as exc:
            summary["public_plot_error"] = repr(exc)
        public_paired_plot_path = plot_dir / f"{stem}_paired_trial_rmse_deltas.png"
        try:
            _plot_paired_trial_differences(summary, public_paired_plot_path)
            if public_paired_plot_path.exists():
                summary["public_paired_trial_delta_plot_path"] = str(public_paired_plot_path)
        except Exception as exc:
            summary["public_paired_trial_delta_plot_error"] = repr(exc)
    if report_dir is not None:
        public_report_path = report_dir / f"{stem}_REPORT.md"
        _write_depth_sweep_report(public_report_path, summary)
        summary["public_report_path"] = str(public_report_path)
    metrics_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with config.log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"Metrics: {metrics_path}\n")
        log_handle.write(f"Decisions: {decisions_path}\n")
        log_handle.write(f"END TIME: {datetime.now().isoformat()}\n")
    return run_dir


def main() -> None:
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run fixed-root StrategyEIG depth sweep diagnostics.")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--report-dir", type=Path, default=Path("results/location_fixed_root_depth_sweep"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plots/location_fixed_root_depth_sweep"))
    parser.add_argument("--run-name")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument(
        "--strategy-depths",
        help="Comma-separated StrategyEIG policy depths to run, default 1..max-depth.",
    )
    parser.add_argument(
        "--eval-depths",
        help="Comma-separated rollout evaluation depths to compute, default 1..max-depth.",
    )
    parser.add_argument(
        "--myopic-control-depths",
        help="Comma-separated matched-compute myopic control depths, default 2..max-depth.",
    )
    parser.add_argument("--num-trials", type=int)
    parser.add_argument("--num-rounds", type=int)
    parser.add_argument(
        "--include-myopic-controls",
        action="store_true",
        help=(
            "Add matched-compute StrategyEIG-myopic-dN arms that generate/evaluate "
            "strategies like the depth sweep but select using depth-1 scores."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(str(config_path))
    strategy_depths = _parse_depth_list(args.strategy_depths)
    eval_depths = _parse_depth_list(args.eval_depths)
    myopic_control_depths = _parse_depth_list(args.myopic_control_depths)
    run_dir = run_fixed_root_depth_sweep(
        config,
        config_path=config_path,
        output_root=args.output_root,
        report_dir=args.report_dir,
        plot_dir=args.plot_dir,
        max_depth=args.max_depth,
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        run_name=args.run_name,
        include_myopic_controls=args.include_myopic_controls,
        strategy_depths=strategy_depths,
        eval_depths=eval_depths,
        myopic_control_depths=myopic_control_depths,
    )
    print(f"[fixed-root] Run directory: {run_dir.resolve()}")
    print(f"[fixed-root] Total time: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
