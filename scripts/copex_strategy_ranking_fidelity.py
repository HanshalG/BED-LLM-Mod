"""Replay completed COPEx L3 histories to diagnose StrategyEIG score fidelity.

This script is deliberately offline: it consumes a completed ``L3.json`` and never
constructs a model adapter or executes the selection policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.location_finding.continuous_strategy import (
    ContinuousStrategy,
    copex_signal,
    execute_continuous_step,
    parse_continuous_strategy_cell,
    particle_entropy,
    update_copex_belief,
)
from scripts.nonmyopic_copex_strategy_prior import L3Config, _stable_seed


DEFAULT_INPUT = Path(
    "results/nonmyopic/copex_strategy_l3_confirmation_recovery1/20260716/L3.json"
)
DEFAULT_OUTPUT_DIR = Path("results/nonmyopic/copex_strategy_l3_ranking_fidelity/20260716")


def _rankdata(values: list[float]) -> np.ndarray:
    """Average ranks for tied values, matching the conventional Spearman definition."""
    array = np.asarray(values, dtype=float)
    ranks = np.empty(len(array), dtype=float)
    order = np.argsort(array, kind="stable")
    cursor = 0
    while cursor < len(array):
        end = cursor + 1
        while end < len(array) and array[order[end]] == array[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + 1 + end)
        cursor = end
    return ranks


def spearman_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _bootstrap_ci(trial_means: list[float], *, seed: int, label: str, samples: int) -> list[float | None]:
    if not trial_means:
        return [None, None]
    values = np.asarray(trial_means, dtype=float)
    rng = np.random.default_rng(_stable_seed(seed, "ranking-fidelity-bootstrap", label))
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = np.mean(values[draws], axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _summarize_trial_metric(
    values_by_trial: dict[int, list[float]], *, seed: int, label: str, bootstrap_replicates: int
) -> dict[str, Any]:
    trial_means = [float(np.mean(values)) for _, values in sorted(values_by_trial.items()) if values]
    all_values = [value for values in values_by_trial.values() for value in values]
    return {
        "mean": float(np.mean(trial_means)) if trial_means else None,
        "std_across_trial_means": float(np.std(trial_means, ddof=1)) if len(trial_means) > 1 else None,
        "bootstrap_ci95": _bootstrap_ci(
            trial_means, seed=seed, label=label, samples=bootstrap_replicates
        ),
        "num_trials": len(trial_means),
        "num_values": len(all_values),
        "median_cell_value": float(np.median(all_values)) if all_values else None,
    }


def _reconstruct_trial(config: L3Config, trial: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trial_index = int(trial["trial_index"])
    rng = np.random.default_rng(_stable_seed(config.seed, "trial", trial_index))
    reconstructed_truth = rng.uniform(0.0, 1.0, size=2)
    particles = np.concatenate(
        [rng.uniform(0.0, 1.0, size=(config.num_particles, 2)), reconstructed_truth[None, :]], axis=0
    )
    initial_position = rng.uniform(0.0, 1.0, size=2)
    truth = np.asarray(trial["truth"], dtype=float)
    if not np.allclose(truth, reconstructed_truth, rtol=0.0, atol=1e-14):
        raise AssertionError(f"trial {trial_index}: recorded truth does not match seed reconstruction")
    if not np.allclose(initial_position, np.asarray(trial["initial_position"], dtype=float), rtol=0.0, atol=1e-14):
        raise AssertionError(f"trial {trial_index}: recorded initial position does not match seed reconstruction")
    return particles, truth, initial_position


def _parse_candidates(step: dict[str, Any], *, horizon: int, max_step: float) -> tuple[ContinuousStrategy, ...]:
    raw = step["candidates"]
    if len(raw) != len(step["candidate_scores"]):
        raise AssertionError("candidate strings and candidate scores differ in length")
    response = json.dumps({"strategies": [json.loads(candidate) for candidate in raw]})
    candidates = parse_continuous_strategy_cell(
        response, expected_count=len(raw), horizon=horizon, max_step=max_step
    )
    if [candidate.canonical_json for candidate in candidates] != list(raw):
        raise AssertionError("candidate canonical JSON changed during replay")
    return candidates


def _fixed_plan_outcome(
    strategy: ContinuousStrategy,
    *,
    start_position: np.ndarray,
    start_probabilities: np.ndarray,
    particles: np.ndarray,
    truth: np.ndarray,
    noise_zs: np.ndarray,
    config: L3Config,
) -> tuple[float, float]:
    probabilities = start_probabilities.copy()
    position = start_position.copy()
    start_entropy = particle_entropy(probabilities)
    start_truth_log_probability = math.log(max(float(probabilities[-1]), 1e-300))
    for step, noise_z in zip(strategy.steps, noise_zs, strict=True):
        position = execute_continuous_step(
            step,
            position=position,
            particles=particles,
            probabilities=probabilities,
            max_step=config.max_step,
        )
        observation = copex_signal(truth, position) + config.noise_sd * float(noise_z)
        probabilities = update_copex_belief(
            particles, probabilities, position, observation, noise_sd=config.noise_sd
        )
    return (
        float(start_entropy - particle_entropy(probabilities)),
        float(math.log(max(float(probabilities[-1]), 1e-300)) - start_truth_log_probability),
    )


def analyze_l3_ranking_fidelity(
    summary: dict[str, Any], *, bootstrap_replicates: int | None = None
) -> dict[str, Any]:
    """Compute the registered, fixed-plan score-fidelity diagnostics from an L3 result."""
    if summary.get("stage") != "L3":
        raise ValueError("ranking fidelity requires an L3 result")
    config = L3Config(**summary["config"])
    config.validate()
    if bootstrap_replicates is None:
        bootstrap_replicates = config.bootstrap_replicates
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")

    metric_values: dict[str, dict[int, list[float]]] = {
        "spearman_entropy": {},
        "spearman_truth_log_probability": {},
        "predicted_top1_entropy_accuracy": {},
        "predicted_top1_truth_log_probability_accuracy": {},
        "predicted_top1_entropy_regret": {},
        "predicted_top1_truth_log_probability_regret": {},
        "score_margin": {},
        "score_std": {},
    }
    pooled_scores: list[float] = []
    pooled_entropy_drops: list[float] = []
    pooled_truth_log_probability_deltas: list[float] = []
    cells: list[dict[str, Any]] = []
    candidate_root_kinds: Counter[str] = Counter()
    selected_root_kinds: Counter[str] = Counter()
    replayed_steps = 0

    for trial in summary["trials"]:
        trial_index = int(trial["trial_index"])
        particles, truth, position = _reconstruct_trial(config, trial)
        probabilities = np.full(len(particles), 1.0 / len(particles))
        trace = trial["traces"]["strategy_eig"]
        if len(trace) != config.num_rounds:
            raise AssertionError(f"trial {trial_index}: unexpected trace length")

        for round_index, recorded_step in enumerate(trace):
            if int(recorded_step["round"]) != round_index + 1:
                raise AssertionError(f"trial {trial_index}: nonconsecutive rounds")
            horizon = min(config.planning_horizon, config.num_rounds - round_index)
            candidates = _parse_candidates(recorded_step, horizon=horizon, max_step=config.max_step)
            scores = [float(score) for score in recorded_step["candidate_scores"]]
            selected_index = _argmax(scores)
            if recorded_step["selected_strategy"] != candidates[selected_index].canonical_json:
                raise AssertionError(f"trial {trial_index}, round {round_index + 1}: selected strategy mismatch")
            selected_root_kinds[candidates[selected_index].steps[0].kind] += 1
            candidate_root_kinds.update(candidate.steps[0].kind for candidate in candidates)

            root_action = execute_continuous_step(
                candidates[selected_index].steps[0],
                position=position,
                particles=particles,
                probabilities=probabilities,
                max_step=config.max_step,
            )
            recorded_action = np.asarray(recorded_step["action"], dtype=float)
            if not np.allclose(root_action, recorded_action, rtol=0.0, atol=1e-12):
                raise AssertionError(f"trial {trial_index}, round {round_index + 1}: selected root mismatch")

            future_noise_zs = np.asarray(
                [
                    (
                        float(future_step["observation"])
                        - copex_signal(truth, np.asarray(future_step["action"], dtype=float))
                    )
                    / config.noise_sd
                    for future_step in trace[round_index : round_index + horizon]
                ],
                dtype=float,
            )
            entropy_drops: list[float] = []
            truth_log_probability_deltas: list[float] = []
            for candidate in candidates:
                entropy_drop, truth_log_probability_delta = _fixed_plan_outcome(
                    candidate,
                    start_position=position,
                    start_probabilities=probabilities,
                    particles=particles,
                    truth=truth,
                    noise_zs=future_noise_zs,
                    config=config,
                )
                entropy_drops.append(entropy_drop)
                truth_log_probability_deltas.append(truth_log_probability_delta)

            entropy_spearman = spearman_correlation(scores, entropy_drops)
            truth_spearman = spearman_correlation(scores, truth_log_probability_deltas)
            best_entropy_index = _argmax(entropy_drops)
            best_truth_index = _argmax(truth_log_probability_deltas)
            sorted_scores = sorted(scores, reverse=True)
            score_margin = float(sorted_scores[0] - sorted_scores[1])
            cell = {
                "trial_index": trial_index,
                "round": round_index + 1,
                "horizon": horizon,
                "predicted_scores": scores,
                "realized_fixed_plan_entropy_drops": entropy_drops,
                "realized_fixed_plan_truth_log_probability_deltas": truth_log_probability_deltas,
                "predicted_top1_index": selected_index,
                "realized_entropy_top1_index": best_entropy_index,
                "realized_truth_log_probability_top1_index": best_truth_index,
                "spearman_entropy": entropy_spearman,
                "spearman_truth_log_probability": truth_spearman,
                "predicted_top1_entropy_regret": float(
                    entropy_drops[best_entropy_index] - entropy_drops[selected_index]
                ),
                "predicted_top1_truth_log_probability_regret": float(
                    truth_log_probability_deltas[best_truth_index]
                    - truth_log_probability_deltas[selected_index]
                ),
                "score_margin": score_margin,
                "score_std": float(np.std(scores)),
            }
            cells.append(cell)
            if entropy_spearman is not None:
                metric_values["spearman_entropy"].setdefault(trial_index, []).append(entropy_spearman)
            if truth_spearman is not None:
                metric_values["spearman_truth_log_probability"].setdefault(trial_index, []).append(
                    truth_spearman
                )
            metric_values["predicted_top1_entropy_accuracy"].setdefault(trial_index, []).append(
                float(selected_index == best_entropy_index)
            )
            metric_values["predicted_top1_truth_log_probability_accuracy"].setdefault(trial_index, []).append(
                float(selected_index == best_truth_index)
            )
            metric_values["predicted_top1_entropy_regret"].setdefault(trial_index, []).append(
                cell["predicted_top1_entropy_regret"]
            )
            metric_values["predicted_top1_truth_log_probability_regret"].setdefault(
                trial_index, []
            ).append(cell["predicted_top1_truth_log_probability_regret"])
            metric_values["score_margin"].setdefault(trial_index, []).append(score_margin)
            metric_values["score_std"].setdefault(trial_index, []).append(float(np.std(scores)))
            pooled_scores.extend(scores)
            pooled_entropy_drops.extend(entropy_drops)
            pooled_truth_log_probability_deltas.extend(truth_log_probability_deltas)

            observation = float(recorded_step["observation"])
            probabilities = update_copex_belief(
                particles, probabilities, recorded_action, observation, noise_sd=config.noise_sd
            )
            replayed_entropy = particle_entropy(probabilities)
            if not math.isclose(replayed_entropy, float(recorded_step["entropy"]), rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(f"trial {trial_index}, round {round_index + 1}: posterior replay mismatch")
            position = recorded_action
            replayed_steps += 1

    aggregates = {
        label: _summarize_trial_metric(
            values, seed=config.seed, label=label, bootstrap_replicates=bootstrap_replicates
        )
        for label, values in metric_values.items()
    }
    return {
        "schema_version": 1,
        "diagnostic": "COPEx L3 fixed-plan ranking fidelity",
        "scope": (
            "Exploratory posthoc replay of the completed L3 histories; not a rerun, not deployed-policy "
            "regret, and not evidence that changes the failed primary gate."
        ),
        "input": {
            "stage": summary["stage"],
            "run_id": summary.get("run_id"),
            "config": asdict(config),
            "primary_gate_passed": bool(summary.get("gate_passed")),
        },
        "replay_checks": {
            "llm_calls": 0,
            "policy_reruns": 0,
            "replayed_steps": replayed_steps,
            "expected_steps": config.num_trials * config.num_rounds,
            "all_posterior_entropies_match_trace": True,
            "all_selected_roots_match_trace": True,
        },
        "counts": {
            "trials": len(summary["trials"]),
            "decision_cells": len(cells),
            "candidate_evaluations": len(pooled_scores),
            "candidate_root_kinds": dict(sorted(candidate_root_kinds.items())),
            "selected_root_kinds": dict(sorted(selected_root_kinds.items())),
        },
        "aggregate": aggregates,
        "pooled_candidate_spearman": {
            "score_vs_fixed_plan_entropy_drop": spearman_correlation(pooled_scores, pooled_entropy_drops),
            "score_vs_fixed_plan_truth_log_probability_delta": spearman_correlation(
                pooled_scores, pooled_truth_log_probability_deltas
            ),
        },
        "cells": cells,
    }


def render_report(result: dict[str, Any]) -> str:
    aggregate = result["aggregate"]
    def row(name: str, key: str) -> str:
        metric = aggregate[key]
        ci = metric["bootstrap_ci95"]
        mean = metric["mean"]
        return (
            f"| {name} | {mean:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] | "
            f"{metric['num_trials']} / {metric['num_values']} |"
        )

    lines = [
        "# COPEx L3 Fixed-Plan Ranking Fidelity",
        "",
        "Exploratory posthoc replay of the completed L3 artifact. No LLM calls, policy reruns, or new "
        "trajectories were used. The primary L3 intersection gate remains failed.",
        "",
        "## Replay",
        "",
        f"- Input run: `{result['input']['run_id']}`.",
        f"- Replayed steps: `{result['replay_checks']['replayed_steps']}` / "
        f"`{result['replay_checks']['expected_steps']}`; posterior and chosen-root assertions passed.",
        f"- Decision cells / candidates: `{result['counts']['decision_cells']}` / "
        f"`{result['counts']['candidate_evaluations']}`.",
        f"- Selected root kinds: `{result['counts']['selected_root_kinds']}`.",
        "",
        "Each candidate follows its entire original macro plan from the recorded pre-decision posterior, "
        "using the completed trial's same future Gaussian innovations. This tests the finite-horizon "
        "score's ranking; it is not receding-horizon policy regret.",
        "",
        "## Trial-Clustered Metrics",
        "",
        "| Metric | Mean of trial means | Trial-bootstrap 95% CI | Trials / cells |",
        "| --- | ---: | --- | ---: |",
        row("Spearman: score vs entropy drop", "spearman_entropy"),
        row("Spearman: score vs truth log-probability delta", "spearman_truth_log_probability"),
        row("Predicted top-1 entropy accuracy", "predicted_top1_entropy_accuracy"),
        row("Predicted top-1 truth-log-probability accuracy", "predicted_top1_truth_log_probability_accuracy"),
        row("Predicted top-1 entropy regret", "predicted_top1_entropy_regret"),
        row("Predicted top-1 truth-log-probability regret", "predicted_top1_truth_log_probability_regret"),
        row("Score margin", "score_margin"),
        row("Within-cell score standard deviation", "score_std"),
        "",
        "## Pooled Candidate Context",
        "",
        f"- Score vs fixed-plan entropy drop Spearman: "
        f"`{result['pooled_candidate_spearman']['score_vs_fixed_plan_entropy_drop']:.6f}`.",
        f"- Score vs fixed-plan truth-log-probability delta Spearman: "
        f"`{result['pooled_candidate_spearman']['score_vs_fixed_plan_truth_log_probability_delta']:.6f}`.",
        "",
        "Positive correlations indicate ranking alignment. These descriptive mechanism metrics neither "
        "rescue nor revise the preregistered primary result.",
        "",
    ]
    return "\n".join(lines)


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "REPORT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output_dir / "REPORT.md").write_text(render_report(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    args = parser.parse_args()
    result = analyze_l3_ranking_fidelity(
        json.loads(args.input.read_text()), bootstrap_replicates=args.bootstrap_replicates
    )
    write_report(result, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "entropy_spearman": result["aggregate"]["spearman_entropy"]["mean"],
                "truth_log_probability_spearman": result["aggregate"][
                    "spearman_truth_log_probability"
                ]["mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
