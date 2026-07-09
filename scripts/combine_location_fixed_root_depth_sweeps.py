from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.location_fixed_root_depth_sweep import (
    _jsonable,
    _metric_summary,
    _metric_summary_by_policy,
    _paired_delta_summary_by_policy,
    _plot_depth_sweep,
    _plot_paired_trial_differences,
    _write_depth_sweep_report,
)


def _load_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "fixed_root_depth_sweep_metrics.json" if path.is_dir() else path
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if "per_trial" not in data or "aggregate" not in data:
        raise ValueError(f"{summary_path} does not look like a fixed-root depth sweep summary")
    data["_source_path"] = str(summary_path)
    return data


def _sum_nested_numbers(values: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        for key, item in value.items():
            if isinstance(item, dict):
                result[key] = _sum_nested_numbers(
                    [result.get(key, {}) if isinstance(result.get(key), dict) else {}, item]
                )
            elif isinstance(item, (int, float)):
                result[key] = result.get(key, 0) + item
            elif key not in result:
                result[key] = copy.deepcopy(item)
    return result


def _validate_compatible(summaries: list[dict[str, Any]]) -> None:
    fields = [
        "config_path",
        "max_depth",
        "strategy_depths",
        "eval_depths",
        "myopic_control_depths",
        "num_rounds",
        "location_seed",
        "location_source_prior",
        "location_source_radius",
        "location_signal_model",
        "location_signal_lengthscale",
        "location_signal_amplitude",
        "location_noise_sd",
        "location_max_step_radius",
        "location_strategy_num_rollouts",
        "location_strategy_num_candidates",
        "location_strategy_discount_factor",
        "location_strategy_rollout_score_mode",
        "location_strategy_rollout_scoring_support_mode",
        "location_strategy_rollout_refresh_hypotheses_each_step",
        "include_myopic_controls",
    ]
    first = summaries[0]
    for summary in summaries[1:]:
        for field in fields:
            if summary.get(field) != first.get(field):
                raise ValueError(
                    f"Incompatible split summaries for {field}: "
                    f"{first.get(field)!r} != {summary.get(field)!r}"
                )


def combine_depth_sweep_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("At least one summary is required")
    _validate_compatible(summaries)

    per_trial = [
        record
        for summary in summaries
        for record in summary.get("per_trial", [])
    ]
    seen_pairs: set[tuple[int, str]] = set()
    for record in per_trial:
        key = (int(record["trial_index"]), str(record["policy_label"]))
        if key in seen_pairs:
            raise ValueError(f"Duplicate trial/policy record in split summaries: {key}")
        seen_pairs.add(key)
    per_trial.sort(key=lambda record: (int(record["trial_index"]), str(record["policy_label"])))

    trial_indices = sorted({int(record["trial_index"]) for record in per_trial})
    max_depth = int(summaries[0]["max_depth"])
    combined = copy.deepcopy(summaries[0])
    combined.pop("_source_path", None)
    combined["combined_from"] = [summary.get("_source_path") for summary in summaries]
    combined["trial_offset"] = min(trial_indices) if trial_indices else 0
    combined["trial_indices"] = trial_indices
    combined["num_trials"] = len(trial_indices)
    combined["per_trial"] = per_trial
    combined["aggregate"] = _metric_summary_by_policy(per_trial)
    combined["aggregate_by_strategy_depth"] = _metric_summary(
        [record for record in per_trial if record["policy_kind"] == "StrategyEIG"],
        max_depth,
    )
    combined["paired_delta_vs_eig"] = _paired_delta_summary_by_policy(
        per_trial,
        baseline_label="EIG",
        rng=np.random.default_rng(int(combined.get("location_seed") or 0) + 99_991),
    )
    combined["ranking_fidelity_chunks"] = [
        summary.get("ranking_fidelity", {}) for summary in summaries
    ]
    combined["token_usage"] = _sum_nested_numbers(
        [
            summary.get("token_usage", {})
            for summary in summaries
            if isinstance(summary.get("token_usage"), dict)
        ]
    )
    combined["run_metadata"] = {
        **{
            key: value
            for key, value in summaries[0].get("run_metadata", {}).items()
            if key in {"questioner_model", "questioner_thinking", "questioner_thinking_max_new_tokens"}
        },
        "combined": True,
        "combined_from": [summary.get("run_metadata", {}) for summary in summaries],
    }
    return combined


def combine_depth_sweep_files(
    inputs: list[Path],
    *,
    output: Path,
    report: Path | None = None,
    plot: Path | None = None,
    paired_delta_plot: Path | None = None,
) -> dict[str, Any]:
    combined = combine_depth_sweep_summaries([_load_summary(path) for path in inputs])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(combined), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report is not None:
        _write_depth_sweep_report(report, combined)
        combined["public_report_path"] = str(report)
    if plot is not None:
        _plot_depth_sweep(combined, plot)
        combined["public_plot_path"] = str(plot)
    if paired_delta_plot is not None:
        _plot_paired_trial_differences(combined, paired_delta_plot)
        combined["public_paired_trial_delta_plot_path"] = str(paired_delta_plot)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine split fixed-root depth sweep metrics.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Metrics JSON files or run directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--paired-delta-plot", type=Path)
    args = parser.parse_args()

    combined = combine_depth_sweep_files(
        args.inputs,
        output=args.output,
        report=args.report,
        plot=args.plot,
        paired_delta_plot=args.paired_delta_plot,
    )
    print(f"combined_trials={combined['num_trials']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
