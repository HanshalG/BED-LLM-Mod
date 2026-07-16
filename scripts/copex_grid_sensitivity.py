"""Zero-LLM matched-budget grid sensitivity for a completed COPEx L3 run."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.location_finding.continuous_strategy import (
    copex_signal,
    particle_entropy,
    update_copex_belief,
)
from scripts.nonmyopic_copex_strategy_prior import (
    L3Config,
    PolicyState,
    _bootstrap_ci,
    _select,
    _stable_seed,
)


def _grid_trial(config: L3Config, trial_index: int) -> list[float]:
    rng = np.random.default_rng(_stable_seed(config.seed, "trial", trial_index))
    truth = rng.uniform(0.0, 1.0, size=2)
    particles = np.concatenate(
        [rng.uniform(0.0, 1.0, size=(config.num_particles, 2)), truth[None, :]], axis=0
    )
    state = PolicyState(
        probabilities=np.full(len(particles), 1.0 / len(particles)),
        position=rng.uniform(0.0, 1.0, size=2),
    )
    entropy: list[float] = []
    for round_index in range(config.num_rounds):
        horizon = min(config.planning_horizon, config.num_rounds - round_index)
        selection = _select(
            "grid_d2",
            None,  # type: ignore[arg-type]
            config,
            state,
            particles,
            trial_index,
            round_index,
            horizon,
        )
        action = np.asarray(selection.action, dtype=float)
        actual_z = float(
            np.random.default_rng(
                _stable_seed(config.seed, trial_index, round_index, "actual-z")
            ).normal()
        )
        observation = copex_signal(truth, action) + config.noise_sd * actual_z
        state.probabilities = update_copex_belief(
            particles,
            state.probabilities,
            action,
            observation,
            noise_sd=config.noise_sd,
        )
        state.position = action
        entropy.append(particle_entropy(state.probabilities))
    return entropy


def run_sensitivity(formal: dict[str, Any], resolutions: tuple[int, ...]) -> dict[str, Any]:
    base = L3Config(**formal["config"])
    strategy = np.asarray(
        [
            [step["entropy"] for step in trial["traces"]["strategy_eig"]]
            for trial in formal["trials"]
        ],
        dtype=float,
    )
    rows: dict[str, Any] = {}
    for resolution in resolutions:
        config = replace(base, grid_resolution=resolution)
        with ThreadPoolExecutor(max_workers=min(config.trial_concurrency, config.num_trials)) as executor:
            traces = list(executor.map(lambda index: _grid_trial(config, index), range(config.num_trials)))
        grid = np.asarray(traces, dtype=float)
        gains = grid[:, -1] - strategy[:, -1]
        ci = _bootstrap_ci(gains, config, f"grid-sensitivity-{resolution}")
        rows[str(resolution)] = {
            "full_depth2_sequence_count": resolution**2,
            "evaluated_sequence_budget_full_horizon": config.grid_sequence_budget,
            "mean_entropy_trace": np.mean(grid, axis=0).tolist(),
            "final_entropy_mean": float(np.mean(grid[:, -1])),
            "strategy_eig_final_entropy_gain_mean": float(np.mean(gains)),
            "strategy_eig_final_entropy_gain_ci95": list(ci),
            "wins_ties_losses": [
                int(np.count_nonzero(gains > 0.0)),
                int(np.count_nonzero(gains == 0.0)),
                int(np.count_nonzero(gains < 0.0)),
            ],
        }
    return {
        "schema_version": 1,
        "zero_llm_calls": True,
        "formal_run_id": formal["run_id"],
        "seed": base.seed,
        "strategy_eig_final_entropy_mean": float(np.mean(strategy[:, -1])),
        "resolutions": rows,
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# COPEx L3 Grid-Resolution Sensitivity",
        "",
        "This sensitivity makes zero LLM calls and reuses the formal seed schedule, truths, particles, initial positions, realized noise, and matched scorer budget.",
        "",
        "| Angular resolution | Full d2 sequences | Evaluated sequences | Grid final entropy | Strategy gain | 95% paired CI | W / T / L |",
        "| ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for resolution, row in summary["resolutions"].items():
        ci = row["strategy_eig_final_entropy_gain_ci95"]
        wtl = row["wins_ties_losses"]
        lines.append(
            f"| {resolution} | {row['full_depth2_sequence_count']} | "
            f"{row['evaluated_sequence_budget_full_horizon']} | {row['final_entropy_mean']:.6f} | "
            f"{row['strategy_eig_final_entropy_gain_mean']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | {wtl[0]} / {wtl[1]} / {wtl[2]} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("formal_json", type=Path)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    formal = json.loads(args.formal_json.read_text())
    summary = run_sensitivity(formal, tuple(args.resolutions))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "GRID_SENSITIVITY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "GRID_SENSITIVITY.md").write_text(render(summary))
    print(json.dumps(summary["resolutions"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
