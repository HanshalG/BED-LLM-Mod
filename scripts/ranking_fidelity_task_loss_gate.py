from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from itertools import permutations
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.location_finding.physics import (
    round_positive_observation,
    signal_intensities_for_hypotheses,
    source_rmse,
)
from environments.location_finding.types import Location
from helpers import Config, load_config
from scripts.strategy_ranking_fidelity import _spearman


@dataclass(frozen=True)
class ReplaySettings:
    prior_particles: int = 100_000
    posterior_particles: int = 256
    estimated_rollouts: int = 32
    realized_deployments: int = 32
    mcmc_steps: int = 8
    mcmc_scale: float = 0.08
    seed: int = 731_941


def _load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonicalize_particles(particles: np.ndarray) -> np.ndarray:
    values = np.asarray(particles, dtype=float)
    if values.ndim != 3:
        raise ValueError("particles must have shape (count, num_sources, dim)")
    order = np.argsort(values[:, :, 0], axis=1, kind="stable")
    return np.take_along_axis(values, order[:, :, None], axis=1)


def _normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(log_weights, dtype=float)
    maximum = float(np.max(values))
    if not math.isfinite(maximum):
        return np.full(len(values), 1.0 / len(values), dtype=float), 0.0
    weights = np.exp(values - maximum)
    total = float(np.sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        return np.full(len(values), 1.0 / len(values), dtype=float), 0.0
    weights /= total
    ess = float(1.0 / np.sum(weights * weights))
    return weights, ess


def _observation_log_likelihoods(
    particles: np.ndarray,
    action: Location,
    value: float,
    config: Config,
) -> np.ndarray:
    means = signal_intensities_for_hypotheses(particles, action, config=config)
    if value <= 0.0:
        return np.full(len(particles), -np.inf, dtype=float)
    z = (math.log(value) - np.log(means)) / float(config.location_noise_sd)
    return -0.5 * z * z - math.log(float(config.location_noise_sd)) - 0.5 * math.log(2.0 * math.pi)


def _history_log_likelihoods(
    particles: np.ndarray,
    history: list[dict[str, Any]],
    config: Config,
) -> np.ndarray:
    result = np.zeros(len(particles), dtype=float)
    for item in history:
        observation = item["observation"]
        result += _observation_log_likelihoods(
            particles,
            tuple(float(value) for value in item["action"]),
            float(observation["value"]),
            config,
        )
    return result


def _normal_log_prior(particles: np.ndarray) -> np.ndarray:
    flat_dimension = particles.shape[1] * particles.shape[2]
    return -0.5 * np.sum(particles * particles, axis=(1, 2)) - 0.5 * flat_dimension * math.log(2.0 * math.pi)


def _log_posterior(
    particles: np.ndarray,
    history: list[dict[str, Any]],
    config: Config,
) -> np.ndarray:
    if str(getattr(config, "location_source_prior", "normal")) != "normal":
        raise ValueError("Gate 0 local replay currently supports the standard normal source prior only")
    return _normal_log_prior(particles) + _history_log_likelihoods(particles, history, config)


def _posterior_support_for_record(
    record: dict[str, Any],
    config: Config,
    settings: ReplaySettings,
) -> tuple[np.ndarray, dict[str, float]]:
    prior_rng = np.random.default_rng(
        np.random.SeedSequence([settings.seed, int(config.location_seed or 0)])
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([settings.seed, int(config.location_seed or 0), int(record["trial_index"]), int(record["round_index"])])
    )
    prior = _canonicalize_particles(
        prior_rng.normal(
            0.0,
            1.0,
            size=(settings.prior_particles, int(config.location_num_sources), int(config.location_dim)),
        )
    )
    importance_weights, importance_ess = _normalize_log_weights(
        _history_log_likelihoods(prior, record.get("history", []), config)
    )
    cumulative_weights = np.cumsum(importance_weights)
    cumulative_weights[-1] = 1.0
    systematic_positions = (
        np.arange(settings.posterior_particles, dtype=float) + 0.5
    ) / float(settings.posterior_particles)
    support = prior[np.searchsorted(cumulative_weights, systematic_positions, side="left")].copy()

    accepted = 0
    proposed = 0
    current_log_posterior = _log_posterior(support, record.get("history", []), config)
    for _step in range(settings.mcmc_steps if record.get("history") else 0):
        proposal = _canonicalize_particles(
            support + rng.normal(0.0, settings.mcmc_scale, size=support.shape)
        )
        proposal_log_posterior = _log_posterior(proposal, record.get("history", []), config)
        log_acceptance = proposal_log_posterior - current_log_posterior
        accept = np.log(rng.uniform(size=len(support))) < np.minimum(log_acceptance, 0.0)
        support[accept] = proposal[accept]
        current_log_posterior[accept] = proposal_log_posterior[accept]
        accepted += int(np.sum(accept))
        proposed += len(support)
    return support, {
        "importance_ess": importance_ess,
        "importance_ess_fraction": importance_ess / float(settings.prior_particles),
        "mcmc_acceptance_rate": float(accepted / proposed) if proposed else 0.0,
    }


def _posterior_mean(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(particles * weights[:, None, None], axis=0)


def _posterior_risk(particles: np.ndarray, weights: np.ndarray) -> float:
    estimate = _posterior_mean(particles, weights)
    matched_mses = []
    for ordering in permutations(range(particles.shape[1])):
        differences = particles[:, ordering, :] - estimate[None, :, :]
        matched_mses.append(np.mean(differences * differences, axis=(1, 2)))
    rmses = np.sqrt(np.min(np.stack(matched_mses, axis=1), axis=1))
    return float(np.sum(weights * rmses))


def _update_weights(
    particles: np.ndarray,
    weights: np.ndarray,
    action: Location,
    value: float,
    config: Config,
) -> np.ndarray:
    return _normalize_log_weights(
        np.log(np.maximum(weights, 1e-300))
        + _observation_log_likelihoods(particles, action, value, config)
    )[0]


def _unique_root_actions(record: dict[str, Any]) -> list[Location]:
    actions: list[Location] = []
    seen: set[Location] = set()
    for candidate in record.get("candidates", []):
        raw = candidate.get("root_query")
        if raw is None:
            continue
        action = tuple(float(value) for value in raw)
        if action not in seen:
            seen.add(action)
            actions.append(action)
    return actions


def _predictive_variance_action(
    particles: np.ndarray,
    weights: np.ndarray,
    actions: list[Location],
    config: Config,
) -> Location:
    best_action = actions[0]
    best_variance = -float("inf")
    for action in actions:
        log_means = np.log(signal_intensities_for_hypotheses(particles, action, config=config))
        center = float(np.sum(weights * log_means))
        variance = float(np.sum(weights * (log_means - center) ** 2))
        if variance > best_variance:
            best_variance = variance
            best_action = action
    return best_action


def _simulate_plan(
    particles: np.ndarray,
    start_weights: np.ndarray,
    truth: np.ndarray,
    root_action: Location,
    action_pool: list[Location],
    noise_zs: np.ndarray,
    config: Config,
) -> np.ndarray:
    weights = start_weights.copy()
    for step_index, noise_z in enumerate(noise_zs):
        action = root_action if step_index == 0 else _predictive_variance_action(
            particles, weights, action_pool, config
        )
        mean = float(signal_intensities_for_hypotheses(truth[None, :, :], action, config=config)[0])
        value = round_positive_observation(
            mean * math.exp(float(config.location_noise_sd) * float(noise_z)),
            2,
        )
        weights = _update_weights(particles, weights, action, value, config)
    return weights


def _candidate_scores_for_depth(
    record: dict[str, Any],
    particles: np.ndarray,
    depth: int,
    config: Config,
    settings: ReplaySettings,
) -> dict[str, list[float | None]]:
    candidate_actions = [
        None if item.get("root_query") is None else tuple(float(value) for value in item["root_query"])
        for item in record.get("candidates", [])
    ]
    action_pool = _unique_root_actions(record)
    count = len(particles)
    start_weights = np.full(count, 1.0 / count, dtype=float)
    start_risk = _posterior_risk(particles, start_weights)
    hidden_truth = _canonicalize_particles(np.asarray(record["hidden_state"], dtype=float)[None, :, :])[0]
    start_point_rmse = source_rmse(
        tuple(tuple(float(value) for value in source) for source in _posterior_mean(particles, start_weights)),
        hidden_truth,
    )

    rng = np.random.default_rng(
        np.random.SeedSequence(
            [settings.seed, int(config.location_seed or 0), int(record["trial_index"]), int(record["round_index"]), int(depth)]
        )
    )
    estimated_truth_indices = rng.integers(0, count, size=settings.estimated_rollouts)
    estimated_noise = rng.normal(size=(settings.estimated_rollouts, depth))
    realized_noise = rng.normal(size=(settings.realized_deployments, depth))

    scores_by_root: dict[Location, tuple[float, float, float, float]] = {}
    for root_action in action_pool:
        estimated_scores = []
        for rollout_index, truth_index in enumerate(estimated_truth_indices):
            final_weights = _simulate_plan(
                particles,
                start_weights,
                particles[int(truth_index)],
                root_action,
                action_pool,
                estimated_noise[rollout_index],
                config,
            )
            estimated_scores.append(start_risk - _posterior_risk(particles, final_weights))
        estimated_mean = float(np.mean(estimated_scores))
        estimated_se = (
            float(np.std(estimated_scores, ddof=1) / math.sqrt(len(estimated_scores)))
            if len(estimated_scores) > 1
            else 0.0
        )

        realized_risk_scores = []
        realized_point_scores = []
        for deployment_noise in realized_noise:
            final_weights = _simulate_plan(
                particles,
                start_weights,
                hidden_truth,
                root_action,
                action_pool,
                deployment_noise,
                config,
            )
            realized_risk_scores.append(start_risk - _posterior_risk(particles, final_weights))
            final_estimate = _posterior_mean(particles, final_weights)
            final_point_rmse = source_rmse(
                tuple(tuple(float(value) for value in source) for source in final_estimate),
                hidden_truth,
            )
            realized_point_scores.append(start_point_rmse - final_point_rmse)
        scores_by_root[root_action] = (
            estimated_mean,
            estimated_se,
            float(np.mean(realized_risk_scores)),
            float(np.mean(realized_point_scores)),
        )

    candidate_scores = [None if action is None else scores_by_root.get(action) for action in candidate_actions]
    return {
        "estimated_task_loss_drop_mean": [None if score is None else score[0] for score in candidate_scores],
        "estimated_task_loss_drop_se": [None if score is None else score[1] for score in candidate_scores],
        "realized_posterior_risk_drop_mean": [None if score is None else score[2] for score in candidate_scores],
        "realized_point_rmse_drop_mean": [None if score is None else score[3] for score in candidate_scores],
    }


def _clean_pairs(first: list[Any], second: list[Any]) -> tuple[list[float], list[float]]:
    left: list[float] = []
    right: list[float] = []
    for first_value, second_value in zip(first, second):
        if first_value is None or second_value is None:
            continue
        a = float(first_value)
        b = float(second_value)
        if math.isfinite(a) and math.isfinite(b):
            left.append(a)
            right.append(b)
    return left, right


def _correlation(first: list[Any], second: list[Any]) -> float | None:
    left, right = _clean_pairs(first, second)
    return _spearman(left, right)


def _bootstrap_mean_ci(values: list[float], rng: np.random.Generator, samples: int = 5_000) -> list[float | None]:
    if not values:
        return [None, None]
    array = np.asarray(values, dtype=float)
    boot = np.mean(rng.choice(array, size=(samples, len(array)), replace=True), axis=1)
    return [float(value) for value in np.percentile(boot, [2.5, 97.5])]


def _summary(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    if not values:
        return {"mean": None, "se": None, "bootstrap_ci95": [None, None], "n": 0}
    return {
        "mean": float(np.mean(values)),
        "se": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
        "bootstrap_ci95": _bootstrap_mean_ci(values, rng),
        "n": len(values),
    }


def run_gate(
    records: list[dict[str, Any]],
    config: Config,
    settings: ReplaySettings,
    *,
    depths: list[int] | None = None,
) -> dict[str, Any]:
    if depths is None:
        depths = sorted({int(depth) for record in records for depth in record.get("realized_by_depth", {})})
    if not depths or any(depth <= 0 for depth in depths):
        raise ValueError("depths must contain positive integers")
    replay_records: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        particles, posterior_diagnostics = _posterior_support_for_record(record, config, settings)
        depth_results: dict[str, Any] = {}
        for depth in depths:
            scores = _candidate_scores_for_depth(record, particles, depth, config, settings)
            entropy_evaluations = record.get("estimated_by_variant", {}).get("configured", {}).get(str(depth), [])
            entropy_scores = [item.get("mean_score") for item in entropy_evaluations]
            task = scores["estimated_task_loss_drop_mean"]
            realized_risk = scores["realized_posterior_risk_drop_mean"]
            point_rmse = scores["realized_point_rmse_drop_mean"]
            depth_results[str(depth)] = {
                **scores,
                "stored_entropy_score_mean": entropy_scores,
                "spearman_task_vs_realized_risk": _correlation(task, realized_risk),
                "spearman_entropy_vs_realized_risk": _correlation(entropy_scores, realized_risk),
                "spearman_task_vs_point_rmse": _correlation(task, point_rmse),
                "spearman_realized_risk_vs_point_rmse": _correlation(realized_risk, point_rmse),
            }
        replay_records.append(
            {
                "record_index": record_index,
                "trial_index": int(record["trial_index"]),
                "round_index": int(record["round_index"]),
                "posterior_diagnostics": posterior_diagnostics,
                "depths": depth_results,
            }
        )

    rng = np.random.default_rng(settings.seed + 99)
    aggregate: dict[str, Any] = {}
    metric_names = (
        "spearman_task_vs_realized_risk",
        "spearman_entropy_vs_realized_risk",
        "spearman_task_vs_point_rmse",
        "spearman_realized_risk_vs_point_rmse",
    )
    for depth in depths:
        aggregate[str(depth)] = {}
        for metric_name in metric_names:
            values = [
                float(record["depths"][str(depth)][metric_name])
                for record in replay_records
                if record["depths"][str(depth)].get(metric_name) is not None
            ]
            aggregate[str(depth)][metric_name] = _summary(values, rng)

    best_depth = max(
        depths,
        key=lambda depth: float(aggregate[str(depth)]["spearman_task_vs_realized_risk"]["mean"] or -math.inf),
    )
    best_value = aggregate[str(best_depth)]["spearman_task_vs_realized_risk"]["mean"]
    status = "pass" if best_value is not None and float(best_value) >= 0.3 else "fail"
    ess_fractions = [record["posterior_diagnostics"]["importance_ess_fraction"] for record in replay_records]
    acceptance_rates = [record["posterior_diagnostics"]["mcmc_acceptance_rate"] for record in replay_records]
    return {
        "generated_at": datetime.now().isoformat(),
        "gate": {
            "status": status,
            "threshold": 0.3,
            "best_depth": best_depth,
            "best_spearman_task_vs_realized_risk": best_value,
        },
        "method": {
            "estimated_utility": "posterior expected permutation-matched RMSE reduction",
            "realized_smooth_target": "posterior expected permutation-matched RMSE reduction after observations from the fixed hidden truth",
            "point_target": "posterior-mean permutation-matched RMSE reduction to fixed hidden truth",
            "executor": "candidate root followed by greedy posterior predictive log-signal variance over the stored root set",
            "truth_used_by_estimated_scorer": False,
            "uses_llm": False,
            "fallback_reason": "legacy records lack rollout posterior snapshots and query trajectories",
        },
        "settings": settings.__dict__,
        "num_records": len(records),
        "depths": depths,
        "posterior_diagnostics": {
            "importance_ess_fraction_mean": float(np.mean(ess_fractions)),
            "importance_ess_fraction_min": float(np.min(ess_fractions)),
            "mcmc_acceptance_rate_mean": float(np.mean(acceptance_rates)),
        },
        "aggregate": aggregate,
        "records": replay_records,
    }


def _format_metric(metric: dict[str, Any]) -> str:
    if metric["mean"] is None:
        return "NA"
    lo, hi = metric["bootstrap_ci95"]
    return f"{metric['mean']:.3f} [{lo:.3f}, {hi:.3f}]"


def _mean_record_metric(result: dict[str, Any], depth: int, round_index: int, metric: str) -> float | None:
    values = [
        float(record["depths"][str(depth)][metric])
        for record in result["records"]
        if record["round_index"] == round_index
        and record["depths"][str(depth)].get(metric) is not None
    ]
    return float(np.mean(values)) if values else None


def _score_noise_ratio(result: dict[str, Any], depth: int) -> float | None:
    ratios: list[float] = []
    for record in result["records"]:
        depth_record = record["depths"][str(depth)]
        means = np.asarray(
            [value for value in depth_record["estimated_task_loss_drop_mean"] if value is not None],
            dtype=float,
        )
        standard_errors = np.asarray(
            [value for value in depth_record["estimated_task_loss_drop_se"] if value is not None],
            dtype=float,
        )
        spread = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
        if spread > 0.0 and len(standard_errors):
            ratios.append(float(np.mean(standard_errors) / spread))
    return float(np.mean(ratios)) if ratios else None


def write_report(path: Path, result: dict[str, Any], records_path: Path, config_path: Path) -> None:
    lines = [
        "# Path B Gate 0: Task-Loss Ranking Fidelity",
        "",
        f"- Status: **{result['gate']['status'].upper()}**",
        f"- Best depth: {result['gate']['best_depth']}",
        f"- Best task scorer rho: {result['gate']['best_spearman_task_vs_realized_risk']:.3f}",
        f"- Gate threshold: {result['gate']['threshold']:.3f}",
        f"- Records: {result['num_records']} from `{records_path}`",
        f"- Config: `{config_path}`",
        "- LLM calls: none",
        "",
        "## Correlations",
        "",
        "Macro-average within-probe Spearman rho; brackets are a 95% bootstrap CI over probe states.",
        "",
        "| Depth | task scorer vs realized posterior-risk drop | entropy scorer vs realized posterior-risk drop | task scorer vs point-dRMSE | realized-risk vs point-dRMSE ceiling |",
        "|---:|---:|---:|---:|---:|",
    ]
    for depth in result["depths"]:
        metrics = result["aggregate"][str(depth)]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(depth),
                    _format_metric(metrics["spearman_task_vs_realized_risk"]),
                    _format_metric(metrics["spearman_entropy_vs_realized_risk"]),
                    _format_metric(metrics["spearman_task_vs_point_rmse"]),
                    _format_metric(metrics["spearman_realized_risk_vs_point_rmse"]),
                ]
            )
            + " |"
        )
    state_rounds = sorted({int(record["round_index"]) for record in result["records"]})
    lines.extend(
        [
            "",
            "## Estimator Diagnostics",
            "",
            "| Depth | mean score SE / between-candidate SD | "
            + " | ".join(f"task-vs-smooth rho at round {round_index}" for round_index in state_rounds)
            + " |",
            "|---:|---:|" + "---:|" * len(state_rounds),
        ]
    )
    for depth in result["depths"]:
        ratio = _score_noise_ratio(result, depth)
        round_values = [
            _mean_record_metric(result, depth, round_index, "spearman_task_vs_realized_risk")
            for round_index in state_rounds
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(depth),
                    "NA" if ratio is None else f"{ratio:.3f}",
                    *("NA" if value is None else f"{value:.3f}" for value in round_values),
                ]
            )
            + " |"
        )
    diagnostics = result["posterior_diagnostics"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The gating comparison is the first numeric column. The final column is the rankability ceiling: it measures how well the smooth posterior-risk endpoint itself ranks noisy point-RMSE gains.",
            "",
            "This is the specified local fallback, not a reconstruction of the original LLM rollout trajectories. The legacy records preserve candidate roots, histories, and hidden states, but not rollout posterior snapshots or future query trajectories. Each stored root is therefore replayed with the same no-LLM analytic executor: subsequent queries maximize posterior predictive log-signal variance over that probe's stored root set.",
            "",
            f"Prior importance ESS fraction: mean {diagnostics['importance_ess_fraction_mean']:.4f}, minimum {diagnostics['importance_ess_fraction_min']:.4f}. MCMC rejuvenation acceptance: mean {diagnostics['mcmc_acceptance_rate_mean']:.3f}.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(path: Path, result: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    depths = result["depths"]
    plt.figure(figsize=(7.2, 4.2))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axhline(0.3, color="tab:green", linestyle="--", linewidth=1.0, label="Gate threshold")
    series = [
        ("Task scorer\nvs smooth target", "spearman_task_vs_realized_risk", "tab:blue"),
        ("Entropy scorer\nvs smooth target", "spearman_entropy_vs_realized_risk", "tab:orange"),
        ("Task scorer\nvs point dRMSE", "spearman_task_vs_point_rmse", "tab:green"),
        ("Smooth target\nvs point dRMSE", "spearman_realized_risk_vs_point_rmse", "tab:red"),
    ]
    if len(depths) == 1:
        depth_metrics = result["aggregate"][str(depths[0])]
        available = [
            (label, depth_metrics[key], color)
            for label, key, color in series
            if depth_metrics[key]["mean"] is not None
        ]
        labels = [label for label, _metric, _color in available]
        values = [float(metric["mean"]) for _label, metric, _color in available]
        lower = [value - float(metric["bootstrap_ci95"][0]) for value, (_label, metric, _color) in zip(values, available)]
        upper = [float(metric["bootstrap_ci95"][1]) - value for value, (_label, metric, _color) in zip(values, available)]
        positions = np.arange(len(available))
        bars = plt.bar(
            positions,
            values,
            color=[color for _label, _metric, color in available],
            alpha=0.85,
            yerr=np.asarray([lower, upper]),
            capsize=4,
        )
        plt.xticks(positions, labels, fontsize=8)
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2.0, value + 0.012, f"{value:.3f}", ha="center", fontsize=8)
        plt.xlabel(f"Canonical replay depth {depths[0]}")
        plt.ylim(min(-0.02, min(values) - max(lower) - 0.02), max(0.34, max(values) + max(upper) + 0.04))
    else:
        for label, key, color in series:
            values = [result["aggregate"][str(depth)][key]["mean"] for depth in depths]
            if any(value is not None for value in values):
                plt.plot(depths, values, marker="o", color=color, label=label.replace("\n", " "))
        plt.xticks(depths)
        plt.xlabel("Replay depth")
    plt.ylabel("Mean within-state Spearman rho")
    plt.title("Path B Gate 0: task-loss ranking fidelity")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Path B task-loss ranking-fidelity Gate 0 without LLM calls.")
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("results/ranking_fidelity/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_aggregate_records.jsonl"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config_strategy_ranking_fidelity_26b_a4b.yaml"))
    parser.add_argument("--output-json", type=Path, default=Path("results/ranking_fidelity/path_b_gate0_task_loss.json"))
    parser.add_argument("--output-report", type=Path, default=Path("results/ranking_fidelity/PATH_B_GATE0_TASK_LOSS.md"))
    parser.add_argument("--output-plot", type=Path, default=Path("plots/ranking_fidelity/path_b_gate0_task_loss.png"))
    parser.add_argument("--prior-particles", type=int, default=100_000)
    parser.add_argument("--posterior-particles", type=int, default=256)
    parser.add_argument("--estimated-rollouts", type=int, default=32)
    parser.add_argument("--realized-deployments", type=int, default=32)
    parser.add_argument(
        "--depths",
        default="1,3",
        help="Comma-separated replay horizons; Path B starts with 1 and 3.",
    )
    args = parser.parse_args()
    settings = ReplaySettings(
        prior_particles=args.prior_particles,
        posterior_particles=args.posterior_particles,
        estimated_rollouts=args.estimated_rollouts,
        realized_deployments=args.realized_deployments,
    )
    config = load_config(str(args.config))
    depths = sorted({int(value.strip()) for value in args.depths.split(",") if value.strip()})
    result = run_gate(_load_records(args.records), config, settings, depths=depths)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.output_report, result, args.records, args.config)
    write_plot(args.output_plot, result)
    print(json.dumps(result["gate"], sort_keys=True))
    print(f"report: {args.output_report}")


if __name__ == "__main__":
    main()
