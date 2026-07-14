#!/usr/bin/env python3
"""Replay frozen MediQ interactions under a new likelihood factorization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from environments.mediq import MediQAction
from environments.mediq.env import UNAVAILABLE_OUTCOME, MediQEnvironment
from helpers import build_models, load_config, resolve_run_id
from model_factory import build_model_adapter
from run_management import (
    add_item_artifact,
    create_run_context,
    item_base_metadata,
    set_item_metrics,
    write_json,
)


MIN_AVAILABLE_TRUE_FAVORED_RATE = 0.60


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


def _eig(prior: np.ndarray, likelihoods: np.ndarray) -> float:
    predictive = prior @ likelihoods
    marginal_entropy = _entropy(predictive)
    conditional_entropy = sum(
        prior[index] * _entropy(likelihoods[index])
        for index in range(len(prior))
    )
    return float(marginal_entropy - conditional_entropy)


def _selected_candidate(turn: dict[str, Any]) -> dict[str, Any]:
    matches = [
        candidate
        for candidate in turn["candidate_details"]
        if candidate["query"] == turn["query"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one selected candidate detail for {turn['query']!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def replay_records(
    records: Sequence[dict[str, Any]],
    environment: MediQEnvironment,
    *,
    expected_tasks: int = 5,
    expected_turns: int = 10,
    minimum_available_turns: int = 0,
    minimum_true_favored_rate: float = MIN_AVAILABLE_TRUE_FAVORED_RATE,
) -> dict[str, Any]:
    task_by_id = {task.task_id: task for task in environment.tasks}
    if {record["task_id"] for record in records} != set(task_by_id):
        raise ValueError("Source records do not exactly match configured MediQ tasks")

    states: list[dict[str, Any]] = []
    for record in records:
        task = task_by_id[record["task_id"]]
        first_candidate = _selected_candidate(record["turns"][0])
        prior_by_label = first_candidate["prior"]
        if prior_by_label is None:
            raise ValueError("Source run does not retain the initial selected-action prior")
        states.append(
            {
                "record": record,
                "task": task,
                "prior": np.asarray(
                    [prior_by_label[label] for label in task.option_labels],
                    dtype=float,
                ),
                "transcript": [],
            }
        )

    turn_records: list[dict[str, Any]] = []
    maximum_rounds = max(len(state["record"]["turns"]) for state in states)
    for round_index in range(maximum_rounds):
        active = [
            state
            for state in states
            if round_index < len(state["record"]["turns"])
        ]
        actions: list[MediQAction] = []
        for state in active:
            turn = state["record"]["turns"][round_index]
            actions.append(
                MediQAction(
                    query=turn["query"],
                    outcomes=tuple(turn["outcomes"]),
                    task=state["task"],
                    transcript=tuple(state["transcript"]),
                    prior_probabilities=tuple(float(value) for value in state["prior"]),
                )
            )

        matrices = environment.outcome_likelihoods_many(
            [
                (state["task"].option_labels, action)
                for state, action in zip(active, actions)
            ]
        )
        for state, action, likelihoods in zip(active, actions, matrices):
            turn = state["record"]["turns"][round_index]
            observed_outcome = turn["mapped_outcome"]
            if not turn["mapped_cleanly"] or observed_outcome is None:
                raise ValueError("Calibration replay requires clean frozen outcomes")
            prior = state["prior"]
            outcome_index = action.outcomes.index(observed_outcome)
            unavailable_index = action.outcomes.index(UNAVAILABLE_OUTCOME)
            predictive = prior @ likelihoods
            unnormalized = prior * likelihoods[:, outcome_index]
            posterior = unnormalized / float(np.sum(unnormalized))
            true_index = action.task.option_labels.index(action.task.answer_idx)
            true_likelihood = float(likelihoods[true_index, outcome_index])
            predictive_likelihood = float(predictive[outcome_index])
            unavailable = observed_outcome == UNAVAILABLE_OUTCOME
            turn_records.append(
                {
                    "task_id": action.task.task_id,
                    "source_id": action.task.source_id,
                    "round": round_index + 1,
                    "true_label": action.task.answer_idx,
                    "query": action.query,
                    "observed_outcome": observed_outcome,
                    "unavailable": unavailable,
                    "prior": dict(
                        zip(action.task.option_labels, prior.tolist(), strict=True)
                    ),
                    "posterior": dict(
                        zip(
                            action.task.option_labels,
                            posterior.tolist(),
                            strict=True,
                        )
                    ),
                    "likelihoods": {
                        label: likelihoods[index].tolist()
                        for index, label in enumerate(action.task.option_labels)
                    },
                    "eig": _eig(prior, likelihoods),
                    "realized_entropy_drop": _entropy(prior) - _entropy(posterior),
                    "realized_truth_log_probability_gain": math.log(
                        max(float(posterior[true_index]), 1e-300)
                    )
                    - math.log(max(float(prior[true_index]), 1e-300)),
                    "observed_outcome_predictive_probability": predictive_likelihood,
                    "observed_outcome_true_label_probability": true_likelihood,
                    "true_label_favored": true_likelihood > predictive_likelihood,
                    "unavailable_likelihood_span": float(
                        np.max(likelihoods[:, unavailable_index])
                        - np.min(likelihoods[:, unavailable_index])
                    ),
                    "posterior_linf_change": float(np.max(np.abs(posterior - prior))),
                    "record_answerability_probability": float(
                        1.0 - likelihoods[0, unavailable_index]
                    ),
                    "joint_projection_residual": float(
                        environment._data_estimation_projection_residuals.get(
                            action, 0.0
                        )
                    ),
                }
            )
            state["prior"] = posterior
            state["transcript"].append((action.query, turn["reply"]))

    available = [turn for turn in turn_records if not turn["unavailable"]]
    unavailable = [turn for turn in turn_records if turn["unavailable"]]

    def mean(key: str, rows: Sequence[dict[str, Any]]) -> float:
        return float(np.mean([row[key] for row in rows])) if rows else float("nan")

    favored_rate = (
        sum(turn["true_label_favored"] for turn in available) / len(available)
        if available
        else 0.0
    )
    summary = {
        "num_tasks": len(states),
        "num_turns": len(turn_records),
        "num_available_turns": len(available),
        "num_unavailable_turns": len(unavailable),
        "mean_eig": mean("eig", turn_records),
        "mean_realized_entropy_drop": mean("realized_entropy_drop", turn_records),
        "mean_truth_log_probability_gain": mean(
            "realized_truth_log_probability_gain", turn_records
        ),
        "available_mean_truth_log_probability_gain": mean(
            "realized_truth_log_probability_gain", available
        ),
        "available_true_label_favored_rate": favored_rate,
        "maximum_unavailable_likelihood_span": max(
            (turn["unavailable_likelihood_span"] for turn in turn_records),
            default=0.0,
        ),
        "maximum_unavailable_posterior_linf_change": max(
            (turn["posterior_linf_change"] for turn in unavailable),
            default=0.0,
        ),
        "maximum_joint_projection_residual": max(
            (turn["joint_projection_residual"] for turn in turn_records),
            default=0.0,
        ),
    }
    checks = {
        "frozen_replay_shape": summary["num_tasks"] == expected_tasks
        and summary["num_turns"] == expected_turns,
        "minimum_available_turn_count": summary["num_available_turns"]
        >= minimum_available_turns,
        "unavailable_is_label_independent": summary[
            "maximum_unavailable_likelihood_span"
        ]
        <= 1e-12,
        "unavailable_does_not_move_posterior": summary[
            "maximum_unavailable_posterior_linf_change"
        ]
        <= 1e-12,
        "available_true_label_favored_rate_passes_threshold": favored_rate
        >= minimum_true_favored_rate,
        "available_mean_truth_log_gain_positive": summary[
            "available_mean_truth_log_probability_gain"
        ]
        > 0.0,
        "overall_mean_truth_log_gain_nonnegative": summary[
            "mean_truth_log_probability_gain"
        ]
        >= 0.0,
        "coherent_joint_projection": summary[
            "maximum_joint_projection_residual"
        ]
        <= 1e-10,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "purpose": (
            "Likelihood-calibration diagnostic on frozen Step 0 actions/outcomes; "
            "not a policy-efficacy result"
        ),
        "gate_definition": {
            "available_true_label_favored_rate_minimum": (
                minimum_true_favored_rate
            ),
            "expected_tasks": expected_tasks,
            "expected_turns": expected_turns,
            "minimum_available_turns": minimum_available_turns,
            "available_mean_truth_log_gain_must_be_positive": True,
            "overall_mean_truth_log_gain_must_be_nonnegative": True,
            "unavailable_likelihood_and_posterior_tolerance": 1e-12,
        },
        "checks": checks,
        "summary": summary,
        "turns": turn_records,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# MediQ Likelihood Calibration Replay",
        "",
        f"Verdict: **{report['status'].upper()}**",
        "",
        "This is a frozen-action likelihood diagnostic, not policy-efficacy evidence.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        rendered = f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| `{key}` | {rendered} |")
    lines.extend(["", "## Checks", ""])
    for key, value in report["checks"].items():
        lines.append(f"- [{'x' if value else ' '}] `{key}`")
    lines.extend(
        [
            "",
            "## Turns",
            "",
            "| Task | Round | Outcome | EIG | Truth log gain | True favored | Query |",
            "|---|---:|---|---:|---:|---|---|",
        ]
    )
    for turn in report["turns"]:
        lines.append(
            f"| `{turn['task_id']}` | {turn['round']} | "
            f"{turn['observed_outcome']} | {turn['eig']:.4f} | "
            f"{turn['realized_truth_log_probability_gain']:+.4f} | "
            f"{'yes' if turn['true_label_favored'] else 'no'} | "
            f"{turn['query'].replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "backend_cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
        "backend_requests": int(snapshot.get("adapter_requests", 0)),
        "backend_prompt_tokens": int(snapshot.get("adapter_prompt_tokens", 0)),
        "backend_completion_tokens": int(
            snapshot.get("adapter_completion_tokens", 0)
        ),
        "backend_reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-item", default="000_EIG")
    parser.add_argument("--run-name", default="mediq-likelihood-calibration-replay")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--expected-tasks", type=int, default=5)
    parser.add_argument("--expected-turns", type=int, default=10)
    parser.add_argument("--minimum-available-turns", type=int, default=0)
    parser.add_argument(
        "--minimum-true-favored-rate",
        type=float,
        default=MIN_AVAILABLE_TRUE_FAVORED_RATE,
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(str(config_path))
    if config.task != "mediq":
        parser.error("calibration replay requires task: mediq")
    if config.mediq_likelihood_mode not in {
        "factored_record",
        "data_estimation",
    }:
        parser.error(
            "calibration replay requires factored_record or data_estimation likelihoods"
        )
    if len(config.model_pairs) != 1:
        parser.error("calibration replay requires exactly one model pair")

    source_path = (
        args.source_run
        / "items"
        / args.source_item
        / "mediq_interactions.json"
    )
    records = json.loads(source_path.read_text())
    config.run_id = resolve_run_id()
    context = create_run_context(
        args.output_root,
        config.run_id,
        args.run_name,
        config_path,
        task=config.task,
        cwd=Path.cwd(),
    )
    config.log_path = context.log_path
    context.write_config_snapshot(config)

    try:
        pair = config.model_pairs[0]
        models = build_models(
            config.model_pairs,
            lambda spec: build_model_adapter(spec, config=config),
            roles=("answerer",),
        )
        model = models[pair.answerer]
        environment = MediQEnvironment(config, model)
        environment.validate_config(config)
        environment.configure_for_run(config)
        environment.set_questioner(model)

        report = replay_records(
            records,
            environment,
            expected_tasks=args.expected_tasks,
            expected_turns=args.expected_turns,
            minimum_available_turns=args.minimum_available_turns,
            minimum_true_favored_rate=args.minimum_true_favored_rate,
        )
        report["source_run"] = str(args.source_run.resolve())
        report["source_artifact"] = str(source_path.resolve())
        report["likelihood_mode"] = config.mediq_likelihood_mode
        report["usage"] = _usage_snapshot(model)

        item = context.new_item(
            0,
            "likelihood_calibration",
            item_base_metadata(config, "likelihood_calibration", pair),
        )
        json_path = item.item_dir / "mediq_likelihood_calibration.json"
        markdown_path = item.item_dir / "REPORT.md"
        write_json(json_path, report)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")
        add_item_artifact(item, "calibration_report", json_path, context)
        add_item_artifact(item, "calibration_markdown", markdown_path, context)
        set_item_metrics(
            item,
            {
                "calibration_gate_pass": [float(report["status"] == "pass")],
                **{
                    key: [value]
                    for key, value in report["summary"].items()
                    if isinstance(value, (int, float))
                },
                **{key: [value] for key, value in report["usage"].items()},
            },
        )
        context.finish(status="completed")
        print(context.run_dir)
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        print(f"calibration_status={report['status']}")
    except Exception as exc:
        context.finish(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
