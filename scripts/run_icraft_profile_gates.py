"""Run preregistered iCRAFT-MD profile-support validation gates.

This script deliberately does not run a policy comparison.  It evaluates the
fixed latent-profile likelihood model at named source IDs and records the
preregistered calibration, branch-equivalence, and exact two-step diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.mediq.env import MediQEnvironment
from helpers import Config, load_config
from methods.categorical_eig import CategoricalEIG, FullTwoStepCategoricalEIG
from model_factory import build_model_adapter


PARTITIONS = {
    "smoke": ("125",),
    "calibration": ("2", "99", "132", "60", "62", "64", "23", "137", "96", "100", "117", "40"),
    "structural": ("30", "52", "103", "33", "85", "36", "94", "113", "34", "139", "15", "68"),
}


def _entropy(values: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in values if value > 0.0)


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(_rank(left), _rank(right))[0, 1])


def _configure(config: Config, stage: str) -> Config:
    source_ids = list(PARTITIONS[stage])
    config.mediq_source_ids = source_ids
    config.mediq_num_trials = len(source_ids)
    config.mediq_trial_batch_size = min(len(source_ids), 12)
    config.mediq_num_rounds = 1
    config.mediq_num_candidates = 2 if stage == "smoke" else 4
    config.run_id = f"icraft-profile-gate-{stage}"
    return config


def _models(config: Config) -> tuple[Any, Any]:
    pair = config.model_pairs[0]
    return (
        build_model_adapter(pair.questioner, config),
        build_model_adapter(pair.answerer, config),
    )


def _diagnosis_probability(
    env: MediQEnvironment, state: Any, task: Any
) -> tuple[np.ndarray, tuple[str, ...]]:
    return env._diagnosis_probabilities(state.hypotheses, state.probabilities, task)


def run_smoke(config: Config, questioner: Any, answerer: Any) -> dict[str, Any]:
    env = MediQEnvironment(config, answerer).configure_for_run(config)
    env.set_questioner(questioner)
    task = env.tasks[0]
    prior = env._prior_states_many([task], questioner)[0]
    candidates = env.generate_candidate_actions_many(
        [prior], [[]], questioner, config
    )[0]
    likelihoods = env.outcome_likelihoods_many(
        [(prior.hypotheses, action) for action in candidates]
    )
    return {
        "stage": "smoke",
        "source_ids": [task.source_id],
        "profiles": len(prior.hypotheses),
        "candidate_count": len(candidates),
        "likelihood_shapes": [list(matrix.shape) for matrix in likelihoods],
        "passed": len(prior.hypotheses) == 12 and len(candidates) == 2,
    }


def run_calibration(config: Config, questioner: Any, answerer: Any) -> dict[str, Any]:
    env = MediQEnvironment(config, answerer).configure_for_run(config)
    env.set_questioner(questioner)
    tasks = env.tasks
    priors = env._prior_states_many(tasks, questioner)
    candidates_many = env.generate_candidate_actions_many(
        priors, [[] for _task in tasks], questioner, config
    )
    flat_actions = [action for actions in candidates_many for action in actions]
    flat_tasks = [task for task, actions in zip(tasks, candidates_many) for _ in actions]
    flat_priors = [prior for prior, actions in zip(priors, candidates_many) for _ in actions]
    observations = env.observe_many(flat_actions, flat_tasks, np.random.default_rng(1304))
    scorer = CategoricalEIG()
    rows: list[dict[str, Any]] = []
    eigs: list[float] = []
    entropy_drops: list[float] = []
    truth_gains: list[float] = []
    available_truth_gains: list[float] = []
    available_favoring: list[bool] = []
    unavailable_moves: list[float] = []
    for task, prior, action, observation in zip(flat_tasks, flat_priors, flat_actions, observations):
        updated = env.update_belief_state(prior, [(action, observation)], questioner, config)
        prior_values, labels = _diagnosis_probability(env, prior, task)
        updated_values, _ = _diagnosis_probability(env, updated, task)
        true_index = labels.index(task.answer_idx)
        entropy_drop = _entropy(prior_values) - _entropy(updated_values)
        truth_gain = math.log(max(updated_values[true_index], 1e-300)) - math.log(max(prior_values[true_index], 1e-300))
        eig = scorer.select_action([action], prior, env, questioner, [], config).score
        branch_index = action.outcomes.index(observation.mapped_outcome)
        synthetic = env.branch_observation(action, branch_index)
        synthetic_updated = env.update_belief_state(prior, [(action, synthetic)], questioner, config)
        branch_error = float(np.max(np.abs(np.asarray(updated.probabilities) - np.asarray(synthetic_updated.probabilities))))
        unavailable = observation.cannot_answer
        if unavailable:
            unavailable_moves.append(float(np.max(np.abs(np.asarray(updated.probabilities) - np.asarray(prior.probabilities)))))
        else:
            available_truth_gains.append(truth_gain)
            available_favoring.append(bool(updated_values[true_index] > prior_values[true_index]))
        eigs.append(eig)
        entropy_drops.append(entropy_drop)
        truth_gains.append(truth_gain)
        rows.append({
            "source_id": task.source_id,
            "query": action.query,
            "outcome": observation.mapped_outcome,
            "available": not unavailable,
            "eig": eig,
            "entropy_drop": entropy_drop,
            "truth_log_gain": truth_gain,
            "branch_update_max_abs_error": branch_error,
        })
    prior_log_losses = []
    prior_briers = []
    for task, prior in zip(tasks, priors):
        values, labels = _diagnosis_probability(env, prior, task)
        true_index = labels.index(task.answer_idx)
        target = np.zeros(len(labels)); target[true_index] = 1.0
        prior_log_losses.append(-math.log(max(values[true_index], 1e-300)))
        prior_briers.append(float(np.sum((values - target) ** 2)))
    available = len(available_truth_gains)
    result = {
        "stage": "calibration",
        "source_ids": [task.source_id for task in tasks],
        "rows": rows,
        "diagnosis_prior_mean_log_loss": float(np.mean(prior_log_losses)),
        "diagnosis_prior_mean_brier": float(np.mean(prior_briers)),
        "available_count": available,
        "mean_available_truth_log_gain": float(np.mean(available_truth_gains)) if available else float("nan"),
        "available_true_label_favoring_fraction": float(np.mean(available_favoring)) if available else float("nan"),
        "max_unavailable_posterior_move": max(unavailable_moves, default=0.0),
        "max_branch_update_error": max(row["branch_update_max_abs_error"] for row in rows),
        "spearman_eig_entropy_drop": _spearman(eigs, entropy_drops),
        "spearman_eig_truth_log_gain": _spearman(eigs, truth_gains),
    }
    result["passed"] = (
        result["diagnosis_prior_mean_log_loss"] < math.log(4.0)
        and result["diagnosis_prior_mean_brier"] < 0.75
        and available >= 24
        and result["max_unavailable_posterior_move"] <= 1e-12
        and result["mean_available_truth_log_gain"] > 0.0
        and result["available_true_label_favoring_fraction"] >= 0.60
        and result["max_branch_update_error"] <= 1e-12
        and result["spearman_eig_entropy_drop"] >= 0.20
        and result["spearman_eig_truth_log_gain"] >= 0.20
    )
    return result


def run_structural(config: Config, questioner: Any, answerer: Any) -> dict[str, Any]:
    del answerer
    env = MediQEnvironment(config, None).configure_for_run(config)
    env.set_questioner(questioner)
    tasks = env.tasks
    priors = env._prior_states_many(tasks, questioner)
    candidates_many = env.generate_candidate_actions_many(
        priors, [[] for _task in tasks], questioner, config
    )
    one_step = CategoricalEIG()
    two_step = FullTwoStepCategoricalEIG()
    gaps: list[float] = []
    rows: list[dict[str, Any]] = []
    for task, prior, candidates in zip(tasks, priors, candidates_many):
        one = one_step.select_action(candidates, prior, env, questioner, [], config)
        two = two_step.select_action(candidates, prior, env, questioner, [], config)
        one_value = max(one.extras["candidate_scores"])
        two_value = max(two.extras["candidate_scores"])
        gaps.append(two_value - one_value)
        rows.append({
            "source_id": task.source_id,
            "one_step_best_value": one_value,
            "two_step_best_value": two_value,
            "gap": two_value - one_value,
            "one_step_query": one.action.query,
            "two_step_query": two.action.query,
        })
    rng = np.random.default_rng(1304)
    samples = np.asarray(gaps)[rng.integers(0, len(gaps), size=(10_000, len(gaps)))].mean(axis=1)
    lower = float(np.quantile(samples, 0.10))
    result = {
        "stage": "structural",
        "source_ids": [task.source_id for task in tasks],
        "rows": rows,
        "mean_gap": float(np.mean(gaps)),
        "positive_gaps": int(sum(gap >= 0.02 for gap in gaps)),
        "bootstrap_one_sided_90_lower": lower,
    }
    result["passed"] = result["mean_gap"] >= 0.02 and result["positive_gaps"] >= 8 and lower > 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_mediq_icraft_profile_gates_openrouter.yaml")
    parser.add_argument("--stage", choices=tuple(PARTITIONS), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = _configure(load_config(args.config), args.stage)
    questioner, answerer = _models(config)
    if args.stage == "smoke":
        result = run_smoke(config, questioner, answerer)
    elif args.stage == "calibration":
        result = run_calibration(config, questioner, answerer)
    else:
        result = run_structural(config, questioner, answerer)
    result["usage"] = {
        "questioner": questioner.usage_snapshot(),
        "answerer": answerer.usage_snapshot(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": args.stage, "passed": result["passed"], "output": str(output)}))


if __name__ == "__main__":
    main()
