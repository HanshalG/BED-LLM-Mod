"""Compare thinking and non-thinking direct COPEx proposal pools on fixed states."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_copex_direct_proposals import (
    DirectProposalConfig,
    DirectProposalError,
    DirectProposalProvider,
    _grid_actions,
    _immediate_eig,
    _stable_seed,
)


DEFAULT_INPUT = Path("results/nonmyopic/copex_direct_proposals_quadrature_pilot/20260718/FACTORIAL.json")
DEFAULT_OUTPUT_DIR = Path("results/nonmyopic/copex_direct_proposals_thinking_probe/20260718")


def _initial_state(config: DirectProposalConfig, trial_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_stable_seed(config.seed, "trial", trial_index))
    truth = rng.uniform(0.0, 1.0, size=2)
    particles = np.concatenate([rng.uniform(0.0, 1.0, size=(config.num_particles, 2)), truth[None, :]], axis=0)
    position = rng.uniform(0.0, 1.0, size=2)
    probabilities = np.full(len(particles), 1.0 / len(particles))
    return particles, probabilities, position, truth


def _score_pool(
    actions: tuple[tuple[float, float], ...],
    *,
    position: np.ndarray,
    particles: np.ndarray,
    probabilities: np.ndarray,
    config: DirectProposalConfig,
) -> tuple[float, list[float]]:
    scores = [
        _immediate_eig(
            action,
            position=position,
            particles=particles,
            probabilities=probabilities,
            uniforms=np.asarray([], dtype=float),
            noise_zs=np.asarray([], dtype=float),
            config=config,
        )
        for action in actions
    ]
    return float(max(scores)), scores


def _angles(raw_response: str) -> list[float]:
    normalized = raw_response.strip()
    if normalized.startswith("```json\n"):
        normalized = normalized[len("```json\n") : -len("\n```")]
    return [float(item) for item in json.loads(normalized)["angles_deg"]]


def run_probe(
    provider: DirectProposalProvider, completed: dict[str, Any], *, num_states: int | None = None
) -> dict[str, Any]:
    source_config = DirectProposalConfig(**completed["config"])
    source_config.validate()
    if source_config.one_step_scoring != "quadrature":
        raise ValueError("thinking probe requires quadrature-scored input")
    rows: list[dict[str, Any]] = []
    all_angles: list[float] = []
    trials = completed["trials"] if num_states is None else completed["trials"][:num_states]
    for trial in trials:
        trial_index = int(trial["trial_index"])
        particles, probabilities, position, truth = _initial_state(source_config, trial_index)
        if not np.allclose(truth, np.asarray(trial["truth"], dtype=float), rtol=0.0, atol=1e-14):
            raise AssertionError(f"trial {trial_index}: truth reconstruction mismatch")
        if not np.allclose(position, np.asarray(trial["initial_position"], dtype=float), rtol=0.0, atol=1e-14):
            raise AssertionError(f"trial {trial_index}: initial position reconstruction mismatch")
        cell = provider.propose(
            trial_index=trial_index,
            position=position,
            particles=particles,
            probabilities=probabilities,
            label="thinking-root",
        )
        thinking_best, thinking_scores = _score_pool(
            cell.actions, position=position, particles=particles, probabilities=probabilities, config=source_config
        )
        nonthinking_actions = tuple(
            tuple(float(value) for value in action)
            for action in trial["traces"]["llm_d1"][0]["candidate_actions"]
        )
        nonthinking_best, nonthinking_scores = _score_pool(
            nonthinking_actions, position=position, particles=particles, probabilities=probabilities, config=source_config
        )
        grid_actions = _grid_actions(position, source_config)
        grid_best, grid_scores = _score_pool(
            grid_actions, position=position, particles=particles, probabilities=probabilities, config=source_config
        )
        angles = _angles(cell.raw_response)
        all_angles.extend(angles)
        rows.append(
            {
                "trial_index": trial_index,
                "initial_position": position.tolist(),
                "thinking_actions": [list(item) for item in cell.actions],
                "thinking_angles_deg": angles,
                "thinking_scores": thinking_scores,
                "thinking_best_immediate_eig": thinking_best,
                "nonthinking_actions": [list(item) for item in nonthinking_actions],
                "nonthinking_scores": nonthinking_scores,
                "nonthinking_best_immediate_eig": nonthinking_best,
                "grid_actions": [list(item) for item in grid_actions],
                "grid_scores": grid_scores,
                "grid_best_immediate_eig": grid_best,
            }
        )
    thinking = np.asarray([row["thinking_best_immediate_eig"] for row in rows])
    nonthinking = np.asarray([row["nonthinking_best_immediate_eig"] for row in rows])
    grid = np.asarray([row["grid_best_immediate_eig"] for row in rows])
    cardinal = {0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0}
    return {
        "schema_version": 1,
        "stage": "COPEx_direct_proposal_thinking_quality_probe",
        "input_run_id": completed.get("run_id"),
        "source_config": asdict(source_config),
        "rows": rows,
        "summary": {
            "thinking_minus_nonthinking_best_immediate_eig_mean": float(np.mean(thinking - nonthinking)),
            "thinking_minus_nonthinking_wins_ties_losses": [
                int(np.count_nonzero(thinking > nonthinking)),
                int(np.count_nonzero(thinking == nonthinking)),
                int(np.count_nonzero(thinking < nonthinking)),
            ],
            "thinking_minus_grid_best_immediate_eig_mean": float(np.mean(thinking - grid)),
            "thinking_minus_grid_wins_ties_losses": [
                int(np.count_nonzero(thinking > grid)),
                int(np.count_nonzero(thinking == grid)),
                int(np.count_nonzero(thinking < grid)),
            ],
            "thinking_cardinal_angle_fraction": float(
                np.mean([round(angle, 8) in cardinal for angle in all_angles])
            ),
            "thinking_unique_angle_count": len(set(round(angle, 8) for angle in all_angles)),
        },
        "requests": provider.accepted_requests,
        "invalid_responses": provider.invalid_responses,
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# COPEx Thinking Direct-Proposal Quality Probe",
        "",
        "Eight archived initial states are replayed without policy execution. Every candidate pool is evaluated by the same full-support Gaussian-quadrature immediate-EIG function.",
        "",
        "| Comparison | Mean best-pool immediate-EIG difference | W / T / L |",
        "| --- | ---: | --- |",
        f"| thinking - nonthinking | {summary['thinking_minus_nonthinking_best_immediate_eig_mean']:+.4f} | {' / '.join(str(item) for item in summary['thinking_minus_nonthinking_wins_ties_losses'])} |",
        f"| thinking - grid | {summary['thinking_minus_grid_best_immediate_eig_mean']:+.4f} | {' / '.join(str(item) for item in summary['thinking_minus_grid_wins_ties_losses'])} |",
        "",
        f"Thinking cardinal/diagonal angle fraction: `{summary['thinking_cardinal_angle_fraction']:.3f}`; unique angles: `{summary['thinking_unique_angle_count']}`.",
        "",
    ]
    return "\n".join(lines)


def _usage(model: Any) -> dict[str, Any]:
    snapshot = getattr(model, "usage_snapshot", None)
    return snapshot() if callable(snapshot) else {"backend": "unknown"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config_nonmyopic_copex_direct_proposals_thinking_openrouter.yaml"))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default="copex-direct-proposals-thinking-quality-probe-20260718")
    parser.add_argument("--num-states", type=int, default=None)
    args = parser.parse_args()
    completed = json.loads(args.input.read_text())
    if args.num_states is not None and args.num_states <= 0:
        raise ValueError("num_states must be positive when provided")
    runtime: Config = load_config(args.config)
    runtime.run_id = args.run_id
    model = build_model_adapter(runtime.model_pairs[0].questioner, config=runtime)
    source_config = DirectProposalConfig(**completed["config"])
    provider = DirectProposalProvider(model, source_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run_probe(provider, completed, num_states=args.num_states)
    except DirectProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "COPEx_direct_proposal_thinking_quality_probe",
            "status": "failed_closed",
            "error": str(exc),
            "requests": provider.accepted_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": _usage(model),
        }
        (args.output_dir / "THINKING_PROBE_FAILURE.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise
    result["usage"] = _usage(model)
    result["run_id"] = args.run_id
    (args.output_dir / "THINKING_PROBE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "THINKING_PROBE.md").write_text(render_report(result))
    print(json.dumps({"summary": result["summary"], "usage": result["usage"]}, indent=2))


if __name__ == "__main__":
    main()
