from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.location_finding.physics import source_rmse


@dataclass(frozen=True)
class OracleConfig:
    num_trials: int
    num_rounds: int
    num_particles: int
    grid_size: int
    arena: float
    max_step_radius: float
    noise_sd: float
    planner_depth: int
    planning_support_size: int
    source_prior: str
    source_radius: float
    seed: int
    num_sources: int
    fixed_first_query: bool
    signal_model: str = "inverse_square"
    signal_lengthscale: float = 0.75
    signal_amplitude: float = 5.0


def _entropy(probabilities: np.ndarray) -> float:
    values = probabilities[probabilities > 0.0]
    if len(values) == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def _signal(source_configs: np.ndarray, query: np.ndarray, config: OracleConfig) -> np.ndarray:
    distances_sq = np.sum((source_configs - query[None, None, :]) ** 2, axis=2)
    if config.signal_model == "inverse_square":
        return 0.1 + np.sum(1.0 / (1e-4 + distances_sq), axis=1)
    if config.signal_model == "local_bump":
        lengthscale_sq = max(config.signal_lengthscale, 1e-12) ** 2
        return 0.1 + config.signal_amplitude * np.sum(
            np.exp(-0.5 * distances_sq / lengthscale_sq),
            axis=1,
        )
    raise ValueError("signal_model must be one of: inverse_square, local_bump")


def _log_likelihood(values: float | np.ndarray, means: np.ndarray, noise_sd: float) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if np.any(y <= 0.0):
        return np.full_like(means, -np.inf, dtype=float)
    z = (np.log(y) - np.log(means)) / noise_sd
    return -0.5 * z * z - math.log(noise_sd) - 0.5 * math.log(2.0 * math.pi)


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    max_log = float(np.max(log_weights))
    if not math.isfinite(max_log):
        return np.full(len(log_weights), 1.0 / len(log_weights), dtype=float)
    weights = np.exp(log_weights - max_log)
    total = float(np.sum(weights))
    if total <= 0.0 or not math.isfinite(total):
        return np.full(len(log_weights), 1.0 / len(log_weights), dtype=float)
    return weights / total


def _update_belief(
    particles: np.ndarray,
    probabilities: np.ndarray,
    query: np.ndarray,
    value: float,
    noise_sd: float,
    config: OracleConfig,
) -> np.ndarray:
    means = _signal(particles, query, config)
    return _normalize_log_weights(np.log(np.maximum(probabilities, 1e-300)) + _log_likelihood(value, means, noise_sd))


def _feasible_actions(grid: np.ndarray, previous_query: np.ndarray | None, max_step_radius: float) -> np.ndarray:
    if previous_query is None:
        return grid
    distances = np.linalg.norm(grid - previous_query[None, :], axis=1)
    feasible = grid[distances <= max_step_radius + 1e-12]
    if len(feasible):
        return feasible
    nearest = int(np.argmin(distances))
    return grid[nearest : nearest + 1]


def _expected_value(
    particles: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray,
    previous_query: np.ndarray | None,
    config: OracleConfig,
    depth: int,
) -> tuple[float, np.ndarray]:
    actions = _feasible_actions(grid, previous_query, config.max_step_radius)
    current_entropy = _entropy(probabilities)
    best_value = -float("inf")
    best_action = actions[0]
    noise_zs = np.asarray([-1.0, 0.0, 1.0], dtype=float)
    noise_weights = np.asarray([1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0], dtype=float)

    support_indices = np.flatnonzero(probabilities > 1e-6)
    if len(support_indices) > config.planning_support_size:
        ranked = support_indices[
            np.argsort(probabilities[support_indices])[-config.planning_support_size :]
        ]
        support_indices = np.sort(ranked)

    for action in actions:
        means = _signal(particles, action, config)
        expected = 0.0
        for idx in support_indices:
            component_probability = float(probabilities[idx])
            if component_probability <= 0.0:
                continue
            for z, z_weight in zip(noise_zs, noise_weights):
                value = float(means[idx] * math.exp(config.noise_sd * float(z)))
                next_probabilities = _update_belief(particles, probabilities, action, value, config.noise_sd, config)
                immediate = current_entropy - _entropy(next_probabilities)
                future = 0.0
                if depth > 1:
                    future, _future_action = _expected_value(
                        particles,
                        next_probabilities,
                        grid,
                        action,
                        config,
                        depth - 1,
                    )
                expected += component_probability * float(z_weight) * (immediate + future)
        if expected > best_value:
            best_value = float(expected)
            best_action = action
    return best_value, np.asarray(best_action, dtype=float)


def _lawnmower_order(grid: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    y_values = sorted({float(value) for value in grid[:, 1]})
    for row_index, y_value in enumerate(y_values):
        row = grid[np.isclose(grid[:, 1], y_value)]
        row = row[np.argsort(row[:, 0])]
        rows.append(row if row_index % 2 == 0 else row[::-1])
    return np.concatenate(rows, axis=0)


def _coverage_action(
    grid: np.ndarray,
    previous_query: np.ndarray | None,
    max_step_radius: float,
    visited: list[np.ndarray],
) -> np.ndarray:
    feasible = _feasible_actions(grid, previous_query, max_step_radius)
    if not visited:
        return np.asarray(feasible[int(np.argmin(np.linalg.norm(feasible, axis=1)))], dtype=float)
    visited_array = np.asarray(visited, dtype=float)
    distances_to_visited = np.min(
        np.linalg.norm(feasible[:, None, :] - visited_array[None, :, :], axis=2),
        axis=1,
    )
    return np.asarray(feasible[int(np.argmax(distances_to_visited))], dtype=float)


def _posterior_mean_estimate(particles: np.ndarray, probabilities: np.ndarray) -> tuple[tuple[float, ...], ...]:
    weighted = np.sum(particles * probabilities[:, None, None], axis=0)
    return tuple(tuple(float(coord) for coord in row) for row in weighted)


def _sample_source_configs(
    rng: np.random.Generator,
    count: int,
    num_sources: int,
    *,
    prior: str,
    radius: float,
) -> np.ndarray:
    if prior == "normal":
        return rng.normal(0.0, 1.0, size=(count, num_sources, 2))
    if prior == "ring":
        angles = rng.uniform(0.0, 2.0 * math.pi, size=(count, num_sources))
        radial_noise = rng.normal(0.0, 0.1, size=(count, num_sources))
        radii = np.maximum(radius + radial_noise, 0.05)
        return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=2)
    if prior == "axis_endpoints":
        signs = rng.choice([-1.0, 1.0], size=(count, num_sources))
        jitter = rng.normal(0.0, 0.1, size=(count, num_sources, 2))
        configs = np.zeros((count, num_sources, 2), dtype=float)
        configs[:, :, 0] = signs * radius
        return configs + jitter
    if prior == "corners":
        corner_indices = rng.integers(0, 4, size=(count, num_sources))
        corners = np.asarray(
            [
                [-radius, -radius],
                [-radius, radius],
                [radius, -radius],
                [radius, radius],
            ],
            dtype=float,
        )
        jitter = rng.normal(0.0, 0.1, size=(count, num_sources, 2))
        return corners[corner_indices] + jitter
    if prior == "fork":
        endpoint_indices = rng.integers(0, 3, size=(count, num_sources))
        endpoints = np.asarray(
            [
                [-radius, 0.0],
                [radius, 0.75 * radius],
                [radius, -0.75 * radius],
            ],
            dtype=float,
        )
        jitter = rng.normal(0.0, 0.1, size=(count, num_sources, 2))
        return endpoints[endpoint_indices] + jitter
    if prior == "branch_decoy":
        endpoint_indices = rng.integers(0, 3, size=(count, num_sources))
        endpoints = np.asarray(
            [
                [-0.65 * radius, 0.0],
                [radius, 0.8 * radius],
                [radius, -0.8 * radius],
            ],
            dtype=float,
        )
        jitter_scales = np.asarray([0.05, 0.12, 0.12], dtype=float)
        jitter = rng.normal(0.0, jitter_scales[endpoint_indices][:, :, None], size=(count, num_sources, 2))
        return endpoints[endpoint_indices] + jitter
    raise ValueError("source_prior must be one of: normal, ring, axis_endpoints, corners, fork, branch_decoy")


def _run_policy(
    hidden_state: np.ndarray,
    initial_particles: np.ndarray,
    grid: np.ndarray,
    config: OracleConfig,
    *,
    depth: int,
    noise_zs: np.ndarray,
    policy: str = "eig",
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    particles = np.concatenate([initial_particles, hidden_state[None, :, :]], axis=0)
    probabilities = np.full(len(particles), 1.0 / len(particles), dtype=float)
    previous_query: np.ndarray | None = None
    rmse_trace: list[float] = []
    entropy_trace: list[float] = []
    action_trace: list[list[float]] = []
    visited_actions: list[np.ndarray] = []
    lawnmower_targets = _lawnmower_order(grid)
    lawnmower_index = 0

    for round_index in range(config.num_rounds):
        if config.fixed_first_query and round_index == 0:
            action = np.zeros(2, dtype=float)
        elif policy == "eig":
            _score, action = _expected_value(particles, probabilities, grid, previous_query, config, depth)
        elif policy == "lawnmower":
            feasible = _feasible_actions(grid, previous_query, config.max_step_radius)
            action = None
            for _attempt in range(len(lawnmower_targets)):
                target = lawnmower_targets[lawnmower_index % len(lawnmower_targets)]
                lawnmower_index += 1
                distances = np.linalg.norm(feasible - target[None, :], axis=1)
                if float(np.min(distances)) <= 1e-12:
                    action = np.asarray(feasible[int(np.argmin(distances))], dtype=float)
                    break
            if action is None:
                action = _coverage_action(grid, previous_query, config.max_step_radius, visited_actions)
        elif policy == "random":
            if rng is None:
                rng = np.random.default_rng(0)
            feasible = _feasible_actions(grid, previous_query, config.max_step_radius)
            action = np.asarray(feasible[int(rng.integers(0, len(feasible)))], dtype=float)
        else:
            raise ValueError("policy must be one of: eig, lawnmower, random")
        truth_config = tuple(tuple(float(coord) for coord in row) for row in hidden_state)
        mean = float(_signal(hidden_state[None, :, :], action, config)[0])
        value = mean * math.exp(config.noise_sd * float(noise_zs[round_index]))
        probabilities = _update_belief(particles, probabilities, action, value, config.noise_sd, config)
        estimate = _posterior_mean_estimate(particles, probabilities)
        rmse_trace.append(float(source_rmse(estimate, hidden_state)))
        entropy_trace.append(_entropy(probabilities))
        action_trace.append([float(coord) for coord in action])
        visited_actions.append(np.asarray(action, dtype=float))
        previous_query = action
        del truth_config

    return {
        "rmse_trace": rmse_trace,
        "entropy_trace": entropy_trace,
        "actions": action_trace,
    }


def _summarize(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    traces = np.asarray([record[key] for record in records], dtype=float)
    return {
        "mean_trace": np.mean(traces, axis=0).tolist(),
        "std_trace": np.std(traces, axis=0, ddof=1).tolist() if len(traces) > 1 else np.zeros(traces.shape[1]).tolist(),
        "final_mean": float(np.mean(traces[:, -1])),
        "final_std": float(np.std(traces[:, -1], ddof=1)) if len(traces) > 1 else 0.0,
    }


def _paired_delta_summary(
    planner_records: list[dict[str, Any]],
    greedy_records: list[dict[str, Any]],
    key: str,
) -> dict[str, float]:
    planner_finals = np.asarray([record[key][-1] for record in planner_records], dtype=float)
    greedy_finals = np.asarray([record[key][-1] for record in greedy_records], dtype=float)
    final_deltas = planner_finals - greedy_finals
    planner_auc = np.asarray([np.sum(record[key]) for record in planner_records], dtype=float)
    greedy_auc = np.asarray([np.sum(record[key]) for record in greedy_records], dtype=float)
    auc_deltas = planner_auc - greedy_auc
    return {
        "final_delta_mean": float(np.mean(final_deltas)),
        "final_delta_std": float(np.std(final_deltas, ddof=1)) if len(final_deltas) > 1 else 0.0,
        "final_planner_win_rate": float(np.mean(final_deltas < 0.0)),
        "auc_delta_mean": float(np.mean(auc_deltas)),
        "auc_delta_std": float(np.std(auc_deltas, ddof=1)) if len(auc_deltas) > 1 else 0.0,
        "auc_planner_win_rate": float(np.mean(auc_deltas < 0.0)),
    }


def _power_for_half_gap(delta_summary: dict[str, float]) -> dict[str, float | None]:
    gap = abs(float(delta_summary.get("final_delta_mean", 0.0)))
    paired_sd = float(delta_summary.get("final_delta_std", 0.0))
    target_effect = 0.5 * gap
    if target_effect <= 0.0 or paired_sd <= 0.0:
        required = None
    else:
        required = float(math.ceil(((1.96 + 0.84) * paired_sd / target_effect) ** 2))
    return {
        "target_fraction_of_gap": 0.5,
        "target_effect": target_effect,
        "paired_sd": paired_sd,
        "required_trials_80_power": required,
    }


def _write_oracle_report(path: Path, summary: dict[str, Any]) -> None:
    config = summary.get("config", {})
    rmse_delta = summary.get("paired_rmse_planner_minus_greedy", {})
    lawnmower_delta = summary.get("paired_rmse_planner_minus_lawnmower", {})
    entropy_delta = summary.get("paired_entropy_planner_minus_greedy", {})
    lawnmower_entropy_delta = summary.get("paired_entropy_planner_minus_lawnmower", {})
    power = summary.get("power_for_strategy_closing_half_oracle_gap", {})
    plot_path = summary.get("public_plot_path") or summary.get("plot_path")
    lines = [
        "# Constrained Oracle Check",
        "",
        "This is the cheap non-LLM sanity check for the locality-constrained location task.",
        "",
        "## Configuration",
        "",
        f"- Source prior: `{config.get('source_prior')}`",
        f"- Signal model: `{config.get('signal_model')}`",
        f"- Source radius: {config.get('source_radius')}",
        f"- Signal lengthscale: {config.get('signal_lengthscale')}",
        f"- Signal amplitude: {config.get('signal_amplitude')}",
        f"- Max step radius: {config.get('max_step_radius')}",
        f"- Trials: {config.get('num_trials')}",
        f"- Rounds: {config.get('num_rounds')}",
        f"- Planner depth: {config.get('planner_depth')}",
        f"- Planning support size: {config.get('planning_support_size')}",
        f"- Seed: {config.get('seed')}",
        "",
        "## Result",
        "",
        "| metric | greedy final | lawnmower final | random final | planner final | planner - greedy final | planner - lawnmower final | planner win rate vs greedy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        (
            "| RMSE | "
            f"{summary['greedy']['rmse']['final_mean']:.4f} | "
            f"{summary['lawnmower']['rmse']['final_mean']:.4f} | "
            f"{summary['random']['rmse']['final_mean']:.4f} | "
            f"{summary['planner']['rmse']['final_mean']:.4f} | "
            f"{rmse_delta.get('final_delta_mean', float('nan')):.4f} | "
            f"{lawnmower_delta.get('final_delta_mean', float('nan')):.4f} | "
            f"{rmse_delta.get('final_planner_win_rate', float('nan')):.3f} |"
        ),
        (
            "| entropy | "
            f"{summary['greedy']['entropy']['final_mean']:.4f} | "
            f"{summary['lawnmower']['entropy']['final_mean']:.4f} | "
            f"{summary['random']['entropy']['final_mean']:.4f} | "
            f"{summary['planner']['entropy']['final_mean']:.4f} | "
            f"{entropy_delta.get('final_delta_mean', float('nan')):.4f} | "
            f"{lawnmower_entropy_delta.get('final_delta_mean', float('nan')):.4f} | "
            f"{entropy_delta.get('final_planner_win_rate', float('nan')):.3f} |"
        ),
        "",
        "## Power Check",
        "",
        (
            "Estimated paired trials to detect StrategyEIG closing 50% of the greedy-to-oracle "
            f"final-RMSE gap with 80% power: `{power.get('required_trials_80_power')}` "
            f"(target effect {power.get('target_effect')}, paired SD {power.get('paired_sd')})."
        ),
        "",
    ]
    if plot_path:
        lines.extend(["## Figure", "", f"![Oracle RMSE trace]({plot_path})", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_oracle_check(config: OracleConfig, output_dir: Path, run_name: str, plot_dir: Path | None = None) -> dict[str, Any]:
    if config.grid_size < 3 or config.grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer at least 3 so the grid contains the origin")
    if config.max_step_radius <= 0.0:
        raise ValueError("max_step_radius must be positive")
    grid_spacing = 2.0 * config.arena / float(config.grid_size - 1)
    if grid_spacing > config.max_step_radius + 1e-12:
        raise ValueError(
            "grid spacing exceeds max_step_radius; increase --grid-size or decrease --arena "
            f"(spacing={grid_spacing:.3g}, radius={config.max_step_radius:.3g})"
        )
    if config.planning_support_size < 1:
        raise ValueError("planning_support_size must be positive")
    if config.source_prior not in {"normal", "ring", "axis_endpoints", "corners", "fork", "branch_decoy"}:
        raise ValueError("source_prior must be one of: normal, ring, axis_endpoints, corners, fork, branch_decoy")
    if config.source_radius <= 0.0:
        raise ValueError("source_radius must be positive")
    if config.signal_model not in {"inverse_square", "local_bump"}:
        raise ValueError("signal_model must be one of: inverse_square, local_bump")
    if config.signal_lengthscale <= 0.0:
        raise ValueError("signal_lengthscale must be positive")
    if config.signal_amplitude <= 0.0:
        raise ValueError("signal_amplitude must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    axis = np.linspace(-config.arena, config.arena, config.grid_size)
    grid = np.asarray([(float(x), float(y)) for x in axis for y in axis], dtype=float)

    greedy_records: list[dict[str, Any]] = []
    planner_records: list[dict[str, Any]] = []
    lawnmower_records: list[dict[str, Any]] = []
    random_records: list[dict[str, Any]] = []
    trial_records: list[dict[str, Any]] = []
    progress_every = max(1, config.num_trials // 10)
    for trial_index in range(config.num_trials):
        if trial_index == 0 or (trial_index + 1) % progress_every == 0 or trial_index + 1 == config.num_trials:
            print(
                f"[oracle] {run_name}: trial {trial_index + 1}/{config.num_trials}",
                file=sys.stderr,
                flush=True,
            )
        hidden_state = _sample_source_configs(
            rng,
            1,
            config.num_sources,
            prior=config.source_prior,
            radius=config.source_radius,
        )[0]
        initial_particles = _sample_source_configs(
            rng,
            config.num_particles,
            config.num_sources,
            prior=config.source_prior,
            radius=config.source_radius,
        )
        noise_zs = rng.normal(size=config.num_rounds)
        greedy = _run_policy(hidden_state, initial_particles, grid, config, depth=1, noise_zs=noise_zs)
        lawnmower = _run_policy(
            hidden_state,
            initial_particles,
            grid,
            config,
            depth=1,
            noise_zs=noise_zs,
            policy="lawnmower",
        )
        random_walk = _run_policy(
            hidden_state,
            initial_particles,
            grid,
            config,
            depth=1,
            noise_zs=noise_zs,
            policy="random",
            rng=np.random.default_rng(config.seed + 97_531 * (trial_index + 1)),
        )
        planner = _run_policy(
            hidden_state,
            initial_particles,
            grid,
            config,
            depth=config.planner_depth,
            noise_zs=noise_zs,
        )
        greedy_records.append(greedy)
        lawnmower_records.append(lawnmower)
        random_records.append(random_walk)
        planner_records.append(planner)
        trial_records.append(
            {
                "trial_index": trial_index,
                "hidden_state": hidden_state.tolist(),
                "greedy": greedy,
                "lawnmower": lawnmower,
                "random": random_walk,
                "planner": planner,
                "final_delta_rmse_planner_minus_greedy": planner["rmse_trace"][-1] - greedy["rmse_trace"][-1],
                "final_delta_rmse_planner_minus_lawnmower": planner["rmse_trace"][-1] - lawnmower["rmse_trace"][-1],
            }
        )

    summary = {
        "config": config.__dict__,
        "greedy": {
            "rmse": _summarize(greedy_records, "rmse_trace"),
            "entropy": _summarize(greedy_records, "entropy_trace"),
        },
        "planner": {
            "rmse": _summarize(planner_records, "rmse_trace"),
            "entropy": _summarize(planner_records, "entropy_trace"),
        },
        "lawnmower": {
            "rmse": _summarize(lawnmower_records, "rmse_trace"),
            "entropy": _summarize(lawnmower_records, "entropy_trace"),
        },
        "random": {
            "rmse": _summarize(random_records, "rmse_trace"),
            "entropy": _summarize(random_records, "entropy_trace"),
        },
        "paired_rmse_planner_minus_greedy": _paired_delta_summary(planner_records, greedy_records, "rmse_trace"),
        "paired_entropy_planner_minus_greedy": _paired_delta_summary(planner_records, greedy_records, "entropy_trace"),
        "paired_rmse_planner_minus_lawnmower": _paired_delta_summary(planner_records, lawnmower_records, "rmse_trace"),
        "paired_entropy_planner_minus_lawnmower": _paired_delta_summary(planner_records, lawnmower_records, "entropy_trace"),
        "paired_rmse_planner_minus_random": _paired_delta_summary(planner_records, random_records, "rmse_trace"),
    }
    summary["power_for_strategy_closing_half_oracle_gap"] = _power_for_half_gap(
        summary["paired_rmse_planner_minus_greedy"]
    )

    records_path = output_dir / f"{run_name}_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in trial_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary_path = output_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        rounds = np.arange(1, config.num_rounds + 1)
        plt.figure(figsize=(7, 4))
        plt.plot(rounds, summary["greedy"]["rmse"]["mean_trace"], label="greedy EIG")
        plt.plot(rounds, summary["lawnmower"]["rmse"]["mean_trace"], label="lawnmower")
        plt.plot(rounds, summary["random"]["rmse"]["mean_trace"], label="random walk")
        plt.plot(rounds, summary["planner"]["rmse"]["mean_trace"], label=f"depth-{config.planner_depth} planner")
        plt.xlabel("round")
        plt.ylabel("RMSE")
        plt.title("Constrained location oracle check")
        plt.legend()
        plt.tight_layout()
        plot_path = output_dir / f"{run_name}_rmse.png"
        plt.savefig(plot_path, dpi=160)
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)
            public_plot_path = plot_dir / f"{run_name}_rmse.png"
            plt.savefig(public_plot_path, dpi=160)
            summary["public_plot_path"] = str(public_plot_path)
        plt.close()
        summary["plot_path"] = str(plot_path)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        summary["plot_error"] = repr(exc)

    report_path = output_dir / f"{run_name}_REPORT.md"
    _write_oracle_report(report_path, summary)
    summary["report_path"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=200)
    parser.add_argument("--num-rounds", type=int, default=15)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--arena", type=float, default=2.0)
    parser.add_argument("--max-step-radius", type=float, default=0.5)
    parser.add_argument("--noise-sd", type=float, default=0.25)
    parser.add_argument("--planner-depth", type=int, default=3)
    parser.add_argument("--planning-support-size", type=int, default=6)
    parser.add_argument(
        "--source-prior",
        choices=["normal", "ring", "axis_endpoints", "corners", "fork", "branch_decoy"],
        default="normal",
    )
    parser.add_argument("--source-radius", type=float, default=1.5)
    parser.add_argument("--num-sources", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1304)
    parser.add_argument("--free-first-query", action="store_true")
    parser.add_argument("--signal-model", choices=["inverse_square", "local_bump"], default="inverse_square")
    parser.add_argument("--signal-lengthscale", type=float, default=0.75)
    parser.add_argument("--signal-amplitude", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/constrained_oracle"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plots/constrained_oracle"))
    parser.add_argument("--run-name", default="constrained_oracle_check")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = OracleConfig(
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        num_particles=args.num_particles,
        grid_size=args.grid_size,
        arena=args.arena,
        max_step_radius=args.max_step_radius,
        noise_sd=args.noise_sd,
        planner_depth=args.planner_depth,
        planning_support_size=args.planning_support_size,
        source_prior=args.source_prior,
        source_radius=args.source_radius,
        seed=args.seed,
        num_sources=args.num_sources,
        fixed_first_query=not args.free_first_query,
        signal_model=args.signal_model,
        signal_lengthscale=args.signal_lengthscale,
        signal_amplitude=args.signal_amplitude,
    )
    summary = run_oracle_check(config, args.output_dir, args.run_name, plot_dir=args.plot_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
