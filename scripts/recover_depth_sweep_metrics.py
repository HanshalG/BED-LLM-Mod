from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.defaults import register_defaults
from environments.location_finding.env import LocationBEDEnvironment
from environments.location_finding.types import LocationObservation
from helpers import Config, load_config
from scripts.llm_token_usage import summarize_llm_token_usage
from scripts.location_fixed_root_depth_sweep import (
    _DepthBranch,
    _entropy,
    _fidelity_summary,
    _jsonable,
    _metric_summary,
    _metric_summary_by_policy,
    _paired_delta_summary_by_policy,
    _parse_depth_list,
    _plot_depth_sweep,
    _plot_paired_trial_differences,
    _policy_specs,
    _run_metadata,
    _source_config_from_hidden_state,
    _truth_augmented_state,
    _truth_log_probability,
    _write_depth_sweep_report,
)


def load_decision_records(decisions_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with decisions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{decisions_path}:{line_number} is not valid JSON") from exc
    return records


def _record_observation(record: dict[str, Any]) -> LocationObservation:
    observation = record.get("observation")
    if not isinstance(observation, dict):
        raise ValueError("decision record is missing an observation object")
    query = observation.get("query", record.get("selected_root_query"))
    if query is None:
        raise ValueError("decision record observation is missing query")
    return LocationObservation(
        query=tuple(float(value) for value in query),
        value=float(observation["value"]),
    )


def _record_action(record: dict[str, Any]) -> tuple[float, float]:
    action = record.get("selected_root_query")
    if action is None:
        observation = record.get("observation") or {}
        action = observation.get("query")
    if action is None:
        raise ValueError("decision record is missing selected_root_query")
    return tuple(float(value) for value in action)


def _infer_max_depth(records: list[dict[str, Any]], fallback: int) -> int:
    depths = [
        int(depth)
        for record in records
        for depth in ([record.get("policy_depth")] + list((record.get("evaluations_by_depth") or {}).keys()))
        if depth is not None
    ]
    return max(depths, default=fallback)


def _summary_from_replayed_histories(
    *,
    config: Config,
    config_path: Path,
    decisions_path: Path,
    run_dir: Path,
    decisions: list[dict[str, Any]],
    questioner: Any,
    max_depth: int,
    include_myopic_controls: bool,
    strategy_depths: list[int] | None,
    eval_depths: list[int] | None,
    myopic_control_depths: list[int] | None,
    original_log_path: Path | None,
) -> dict[str, Any]:
    if config.task != "location_finding":
        raise ValueError("recover_depth_sweep_metrics only supports task=location_finding")
    if not decisions:
        raise ValueError("no decision records found")

    inferred_trials = max(int(record["trial_index"]) for record in decisions) + 1
    inferred_rounds = max(int(record["round_index"]) for record in decisions) + 1
    config.location_num_trials = inferred_trials
    config.location_num_rounds = inferred_rounds
    selected_eval_depths = eval_depths or list(range(1, max_depth + 1))
    policy_specs = _policy_specs(
        max_depth,
        include_myopic_controls=include_myopic_controls,
        strategy_depths=strategy_depths,
        myopic_control_depths=myopic_control_depths,
    )

    decisions_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for record in decisions:
        key = (int(record["trial_index"]), str(record["policy_label"]), int(record["round_index"]))
        if key in decisions_by_key:
            raise ValueError(f"duplicate decision record for trial/policy/round {key}")
        decisions_by_key[key] = record

    env = LocationBEDEnvironment(config=config)
    env.validate_config(config)
    env.rng = np.random.default_rng(config.location_seed)
    master_rng = np.random.default_rng(config.location_seed)
    # Match the normal runner's RNG advancement before hidden-state sampling.
    master_rng.normal(0.0, 1.0, size=(config.location_num_trials, config.location_num_rounds))

    branches: list[_DepthBranch] = []
    per_trial: list[dict[str, Any]] = []
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

    branch_records = {
        (record["trial_index"], record["policy_label"]): record
        for record in per_trial
    }
    recovered_decisions: list[dict[str, Any]] = []
    for round_index in range(config.location_num_rounds):
        updated_branches: list[_DepthBranch] = []
        selected_scores: list[float] = []
        for branch in branches:
            record = decisions_by_key.get((branch.trial_index, branch.policy_label, round_index))
            if record is None:
                continue
            action = _record_action(record)
            observation = _record_observation(record)
            branch.history.append((action, observation))
            updated_branches.append(branch)
            selected_scores.append(float(record.get("selected_eig", 0.0)))
            recovered_decisions.append(record)
        if not updated_branches:
            continue

        new_beliefs = env.update_belief_states(
            [branch.belief_state for branch in updated_branches],
            [branch.history for branch in updated_branches],
            questioner,
            config,
        )
        for branch, new_belief, selected_score in zip(updated_branches, new_beliefs, selected_scores):
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
            truth = _source_config_from_hidden_state(branch.hidden_state)
            metrics["truth_in_support"] = float(truth in branch.belief_state.hypotheses)
            metrics["selected_eig"] = float(selected_score)
            branch_record = branch_records[(branch.trial_index, branch.policy_label)]
            branch_record["round_metrics"].append({key: float(value) for key, value in metrics.items()})
            branch_record["history"] = [
                {
                    "action": list(action),
                    "observation": _jsonable(observation),
                }
                for action, observation in branch.history
            ]

    completed_decision_count = len(recovered_decisions)
    expected_decision_count = len(policy_specs) * config.location_num_trials * config.location_num_rounds
    recovery_warnings: list[str] = []
    if completed_decision_count < expected_decision_count:
        recovery_warnings.append(
            f"partial recovery: {completed_decision_count}/{expected_decision_count} expected decisions were present"
        )
    if original_log_path is None:
        original_log_path = run_dir / "run.log"

    return {
        "config_path": str(config_path),
        "decisions_path": str(decisions_path),
        "recovered": True,
        "recovery_warnings": recovery_warnings,
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
        "token_usage": summarize_llm_token_usage(original_log_path),
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
        "ranking_fidelity": _fidelity_summary(recovered_decisions, max_depth),
    }


def recover_depth_sweep_metrics(
    *,
    config: Config,
    config_path: Path,
    decisions_path: Path,
    run_dir: Path,
    output_path: Path,
    questioner: Any,
    max_depth: int | None = None,
    include_myopic_controls: bool = False,
    strategy_depths: list[int] | None = None,
    eval_depths: list[int] | None = None,
    myopic_control_depths: list[int] | None = None,
    original_log_path: Path | None = None,
    report_path: Path | None = None,
    plot_path: Path | None = None,
    paired_plot_path: Path | None = None,
) -> dict[str, Any]:
    decisions = load_decision_records(decisions_path)
    resolved_max_depth = _infer_max_depth(decisions, max_depth or 1)
    summary = _summary_from_replayed_histories(
        config=config,
        config_path=config_path,
        decisions_path=decisions_path,
        run_dir=run_dir,
        decisions=decisions,
        questioner=questioner,
        max_depth=resolved_max_depth,
        include_myopic_controls=include_myopic_controls,
        strategy_depths=strategy_depths,
        eval_depths=eval_depths,
        myopic_control_depths=myopic_control_depths,
        original_log_path=original_log_path,
    )
    if plot_path is not None:
        try:
            _plot_depth_sweep(summary, plot_path)
            summary["plot_path"] = str(plot_path)
        except Exception as exc:
            summary["plot_error"] = repr(exc)
    if paired_plot_path is not None:
        try:
            _plot_paired_trial_differences(summary, paired_plot_path)
            if paired_plot_path.exists():
                summary["paired_trial_delta_plot_path"] = str(paired_plot_path)
        except Exception as exc:
            summary["paired_trial_delta_plot_error"] = repr(exc)
    if report_path is not None:
        _write_depth_sweep_report(report_path, summary)
        summary["report_path"] = str(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover fixed-root depth sweep metrics from incremental decisions.")
    parser.add_argument("--config", "-c", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--paired-plot", type=Path)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--include-myopic-controls", action="store_true")
    parser.add_argument("--strategy-depths")
    parser.add_argument("--eval-depths")
    parser.add_argument("--myopic-control-depths")
    parser.add_argument("--json", action="store_true", help="Print the recovered summary path and warnings as JSON")
    args = parser.parse_args()

    register_defaults()
    config = load_config(str(args.config))
    decisions_path = args.decisions or args.run_dir / "fixed_root_depth_sweep_decisions.jsonl"
    output_path = args.output or args.run_dir / "fixed_root_depth_sweep_metrics_recovered.json"
    report_path = args.report or args.run_dir / "REPORT_recovered.md"
    plot_path = args.plot or args.run_dir / "fixed_root_depth_sweep_recovered.png"
    paired_plot_path = args.paired_plot or args.run_dir / "paired_trial_rmse_deltas_recovered.png"
    recovery_log_path = args.run_dir / "recover_depth_sweep_metrics.log"
    config.log_path = recovery_log_path
    if not recovery_log_path.exists():
        recovery_log_path.write_text("recover_depth_sweep_metrics\n", encoding="utf-8")

    if not config.model_pairs:
        raise ValueError("Config must include at least one model pair")
    from model import build_model_adapter

    questioner = build_model_adapter(config.model_pairs[0].questioner, config=config)
    summary = recover_depth_sweep_metrics(
        config=config,
        config_path=args.config,
        decisions_path=decisions_path,
        run_dir=args.run_dir,
        output_path=output_path,
        questioner=questioner,
        max_depth=args.max_depth,
        include_myopic_controls=args.include_myopic_controls,
        strategy_depths=_parse_depth_list(args.strategy_depths),
        eval_depths=_parse_depth_list(args.eval_depths),
        myopic_control_depths=_parse_depth_list(args.myopic_control_depths),
        original_log_path=args.run_dir / "run.log",
        report_path=report_path,
        plot_path=plot_path,
        paired_plot_path=paired_plot_path,
    )
    payload = {
        "output": str(output_path),
        "report": str(report_path),
        "plot": str(plot_path),
        "paired_plot": str(paired_plot_path),
        "warnings": summary.get("recovery_warnings", []),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
