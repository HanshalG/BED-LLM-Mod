from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.constrained_oracle_check import OracleConfig, run_oracle_check


def _parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _cell_run_name(prefix: str, lengthscale: float, step_radius: float, noise_sd: float) -> str:
    def fmt(value: float) -> str:
        return str(value).replace(".", "p").replace("-", "m")

    return f"{prefix}_l{fmt(lengthscale)}_r{fmt(step_radius)}_n{fmt(noise_sd)}"


def _cell_summary(summary: dict[str, Any]) -> dict[str, Any]:
    rmse = summary["paired_rmse_planner_minus_greedy"]
    lawnmower = summary["paired_rmse_planner_minus_lawnmower"]
    return {
        "planner_minus_greedy_final_rmse": rmse["final_delta_mean"],
        "planner_minus_greedy_final_rmse_std": rmse["final_delta_std"],
        "planner_win_rate_vs_greedy": rmse["final_planner_win_rate"],
        "planner_minus_lawnmower_final_rmse": lawnmower["final_delta_mean"],
        "planner_win_rate_vs_lawnmower": lawnmower["final_planner_win_rate"],
        "greedy_final_rmse": summary["greedy"]["rmse"]["final_mean"],
        "planner_final_rmse": summary["planner"]["rmse"]["final_mean"],
        "lawnmower_final_rmse": summary["lawnmower"]["rmse"]["final_mean"],
        "required_trials_80_power_half_gap": summary["power_for_strategy_closing_half_oracle_gap"].get(
            "required_trials_80_power"
        ),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    config = payload["base_config"]
    lines = [
        "# Constrained Oracle Robustness Sweep",
        "",
        "This CPU-only sweep maps where non-myopic planning helps in the constrained",
        "branch-decoy/local-bump environment. Values are planner minus greedy final",
        "RMSE, so negative cells favor the depth planner.",
        "",
        "## Base Configuration",
        "",
        f"- Trials per cell: {config['num_trials']}",
        f"- Rounds: {config['num_rounds']}",
        f"- Particles: {config['num_particles']}",
        f"- Grid size: {config['grid_size']}",
        f"- Arena: {config['arena']}",
        f"- Source prior: `{config['source_prior']}`",
        f"- Source radius: {config['source_radius']}",
        f"- Planner depth: {config['planner_depth']}",
        f"- Planning support size: {config['planning_support_size']}",
        f"- Signal amplitude: {config['signal_amplitude']}",
        f"- Seed: {config['seed']}",
        "",
        "## Cells",
        "",
        "| lengthscale | max step radius | noise sd | planner - greedy final RMSE | planner - lawnmower final RMSE | win rate vs greedy |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in payload["cells"]:
        metrics = cell["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{cell['signal_lengthscale']:.3g}",
                    f"{cell['max_step_radius']:.3g}",
                    f"{cell['noise_sd']:.3g}",
                    f"{metrics['planner_minus_greedy_final_rmse']:.4f}",
                    f"{metrics['planner_minus_lawnmower_final_rmse']:.4f}",
                    f"{metrics['planner_win_rate_vs_greedy']:.3f}",
                ]
            )
            + " |"
        )
    if payload.get("heatmap_path"):
        lines.extend(["", "## Figure", "", f"![Robustness heatmap]({payload['heatmap_path']})", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_heatmap(path: Path, payload: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengthscales = payload["signal_lengthscales"]
    radii = payload["max_step_radii"]
    noises = payload["noise_sds"]
    fig, axes = plt.subplots(1, len(noises), figsize=(4.2 * len(noises), 3.6), squeeze=False)
    values_by_key = {
        (cell["signal_lengthscale"], cell["max_step_radius"], cell["noise_sd"]): cell["metrics"][
            "planner_minus_greedy_final_rmse"
        ]
        for cell in payload["cells"]
    }
    all_values = np.asarray(list(values_by_key.values()), dtype=float)
    vmax = float(np.nanmax(np.abs(all_values))) if all_values.size else 1.0
    vmax = max(vmax, 1e-6)
    for axis, noise_sd in zip(axes[0], noises):
        matrix = np.full((len(lengthscales), len(radii)), np.nan, dtype=float)
        for i, lengthscale in enumerate(lengthscales):
            for j, radius in enumerate(radii):
                matrix[i, j] = values_by_key.get((lengthscale, radius, noise_sd), np.nan)
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
        axis.set_title(f"noise={noise_sd:g}")
        axis.set_xlabel("max step radius")
        axis.set_xticks(range(len(radii)), [f"{value:g}" for value in radii])
        axis.set_yticks(range(len(lengthscales)), [f"{value:g}" for value in lengthscales])
        axis.set_ylabel("signal lengthscale")
        for i in range(len(lengthscales)):
            for j in range(len(radii)):
                if np.isfinite(matrix[i, j]):
                    axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85, label="planner - greedy final RMSE")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_robustness_sweep(
    *,
    base_config: OracleConfig,
    signal_lengthscales: list[float],
    max_step_radii: list[float],
    noise_sds: list[float],
    output_dir: Path,
    plot_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_dir = output_dir / f"{run_name}_cells"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
    cell_index = 0
    for lengthscale in signal_lengthscales:
        for step_radius in max_step_radii:
            for noise_sd in noise_sds:
                cell_index += 1
                config = replace(
                    base_config,
                    signal_lengthscale=float(lengthscale),
                    max_step_radius=float(step_radius),
                    noise_sd=float(noise_sd),
                    seed=int(base_config.seed + cell_index * 997),
                )
                cell_name = _cell_run_name(run_name, lengthscale, step_radius, noise_sd)
                summary = run_oracle_check(config, cell_dir, cell_name, plot_dir=None)
                cells.append(
                    {
                        "run_name": cell_name,
                        "signal_lengthscale": float(lengthscale),
                        "max_step_radius": float(step_radius),
                        "noise_sd": float(noise_sd),
                        "summary_path": str(cell_dir / f"{cell_name}_summary.json"),
                        "metrics": _cell_summary(summary),
                    }
                )

    payload = {
        "run_name": run_name,
        "base_config": base_config.__dict__,
        "signal_lengthscales": [float(value) for value in signal_lengthscales],
        "max_step_radii": [float(value) for value in max_step_radii],
        "noise_sds": [float(value) for value in noise_sds],
        "cells": cells,
    }
    heatmap_path = plot_dir / f"{run_name}_heatmap.png"
    try:
        _plot_heatmap(heatmap_path, payload)
        payload["heatmap_path"] = str(heatmap_path)
    except Exception as exc:
        payload["heatmap_error"] = repr(exc)
    summary_path = output_dir / f"{run_name}_summary.json"
    report_path = output_dir / f"{run_name}_REPORT.md"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(report_path, payload)
    payload["summary_path"] = str(summary_path)
    payload["report_path"] = str(report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep constrained oracle parameters and plot planning gap heatmaps.")
    parser.add_argument("--signal-lengthscales", default="0.35,0.5,0.75")
    parser.add_argument("--max-step-radii", default="0.4,0.5,0.7")
    parser.add_argument("--noise-sds", default="0.1,0.15,0.25")
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--num-rounds", type=int, default=6)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=13)
    parser.add_argument("--arena", type=float, default=2.2)
    parser.add_argument("--planner-depth", type=int, default=2)
    parser.add_argument("--planning-support-size", type=int, default=8)
    parser.add_argument("--source-prior", default="branch_decoy")
    parser.add_argument("--source-radius", type=float, default=2.2)
    parser.add_argument("--num-sources", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1304)
    parser.add_argument("--signal-amplitude", type=float, default=8.0)
    parser.add_argument("--free-first-query", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/constrained_oracle_robustness"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plots/constrained_oracle_robustness"))
    parser.add_argument("--run-name", default="branch_decoy_local_robustness")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    base_config = OracleConfig(
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        num_particles=args.num_particles,
        grid_size=args.grid_size,
        arena=args.arena,
        max_step_radius=1.0,
        noise_sd=0.1,
        planner_depth=args.planner_depth,
        planning_support_size=args.planning_support_size,
        source_prior=args.source_prior,
        source_radius=args.source_radius,
        seed=args.seed,
        num_sources=args.num_sources,
        fixed_first_query=not args.free_first_query,
        signal_model="local_bump",
        signal_lengthscale=0.5,
        signal_amplitude=args.signal_amplitude,
    )
    payload = run_robustness_sweep(
        base_config=base_config,
        signal_lengthscales=_parse_float_list(args.signal_lengthscales),
        max_step_radii=_parse_float_list(args.max_step_radii),
        noise_sds=_parse_float_list(args.noise_sds),
        output_dir=args.output_dir,
        plot_dir=args.plot_dir,
        run_name=args.run_name,
    )
    if args.json:
        print(json.dumps({key: value for key, value in payload.items() if key != "cells"}, indent=2, sort_keys=True))
    else:
        print(f"summary: {payload['summary_path']}")
        print(f"report: {payload['report_path']}")
        if payload.get("heatmap_path"):
            print(f"heatmap: {payload['heatmap_path']}")


if __name__ == "__main__":
    main()

