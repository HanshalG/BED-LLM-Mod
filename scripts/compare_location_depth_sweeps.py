from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.llm_token_usage import token_usage_report_lines

PAIRED_METRIC_NAMES = (
    "source_rmse",
    "expected_posterior_rmse",
    "posterior_entropy",
    "truth_log_probability",
)


def _load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "aggregate" not in data or "paired_delta_vs_eig" not in data:
        raise ValueError(f"{path} does not look like a fixed-root depth sweep summary")
    return data


def _policy_order(labels: list[str]) -> list[str]:
    def key(label: str) -> tuple[int, int | str]:
        if label == "naive":
            return (0, 0)
        if label == "naive+belief":
            return (1, 0)
        if label == "EIG":
            return (2, 0)
        if label.startswith("StrategyEIG-d"):
            try:
                return (3, int(label.rsplit("d", 1)[1]))
            except ValueError:
                return (3, label)
        if label.startswith("StrategyEIG-myopic-d"):
            try:
                return (4, int(label.rsplit("d", 1)[1]))
            except ValueError:
                return (4, label)
        return (5, label)

    return sorted(labels, key=key)


def compare_depth_sweeps(
    constrained_summary: dict[str, Any],
    unconstrained_summary: dict[str, Any],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "constrained": _comparison_side(constrained_summary),
        "unconstrained": _comparison_side(unconstrained_summary),
    }
    return comparison


def _comparison_side(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary.get("aggregate", {})
    paired = summary.get("paired_delta_vs_eig", {})
    policy_labels = _policy_order(list(aggregate))
    policies: dict[str, Any] = {}
    for policy_label in policy_labels:
        rmse = aggregate.get(policy_label, {}).get("source_rmse", {})
        policy_entry = {
            "final_rmse_mean": _maybe_float(rmse.get("final_mean")),
            "final_rmse_std": _maybe_float(rmse.get("final_std")),
            "rmse_trace": [
                _maybe_float(value)
                for value in rmse.get("mean_trace", [])
            ],
            "final_metrics": {
                metric_name: _maybe_float(aggregate.get(policy_label, {}).get(metric_name, {}).get("final_mean"))
                for metric_name in PAIRED_METRIC_NAMES
            },
        }
        if policy_label != "EIG":
            paired_metrics: dict[str, Any] = {}
            for metric_name in PAIRED_METRIC_NAMES:
                delta = paired.get(policy_label, {}).get(metric_name, {})
                paired_metrics[metric_name] = {
                    "mean": _maybe_float(delta.get("final_delta_mean")),
                    "ci95": delta.get("final_delta_ci95", [None, None]),
                    "p": _maybe_float(delta.get("wilcoxon_signed_rank_p")),
                }
            policy_entry["paired_delta_vs_eig"] = paired_metrics
            source_rmse_delta = paired_metrics["source_rmse"]
            policy_entry.update(
                {
                    "paired_delta_vs_eig_mean": source_rmse_delta["mean"],
                    "paired_delta_vs_eig_ci95": source_rmse_delta["ci95"],
                    "paired_delta_vs_eig_p": source_rmse_delta["p"],
                }
            )
        policies[policy_label] = policy_entry
    return {
        "config_path": summary.get("config_path"),
        "num_trials": summary.get("num_trials"),
        "num_rounds": summary.get("num_rounds"),
        "source_prior": summary.get("location_source_prior"),
        "signal_model": summary.get("location_signal_model"),
        "max_step_radius": summary.get("location_max_step_radius"),
        "run_metadata": summary.get("run_metadata", {}),
        "token_usage": summary.get("token_usage"),
        "policies": policies,
    }


def _maybe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def headline_policy_labels(
    comparison: dict[str, Any],
    *,
    side_name: str = "constrained",
    depths: tuple[int, ...] = (1, 3, 5),
) -> list[str]:
    policies = comparison.get(side_name, {}).get("policies", {})
    labels = ["EIG"] if "EIG" in policies else []
    labels.extend(
        label
        for depth in depths
        for label in (f"StrategyEIG-d{depth}",)
        if label in policies
    )
    return labels


def _headline_report_lines(comparison: dict[str, Any]) -> list[str]:
    side = comparison.get("constrained", {})
    labels = headline_policy_labels(comparison)
    if not labels:
        return []
    lines = [
        "## Headline Constrained Depths",
        "",
        "Paper-facing subset: greedy EIG and StrategyEIG depths 1, 3, and 5 on the constrained task.",
        "",
        "| policy | final RMSE | final RMSE std | paired final RMSE delta vs EIG | 95% CI | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        policy = side.get("policies", {}).get(label, {})
        if label == "EIG":
            lines.append(
                f"| `{label}` | {_format_optional(policy.get('final_rmse_mean'))} | "
                f"{_format_optional(policy.get('final_rmse_std'))} | n/a | n/a | n/a |"
            )
            continue
        delta = policy.get("paired_delta_vs_eig", {}).get("source_rmse", {})
        lines.append(
            f"| `{label}` | {_format_optional(policy.get('final_rmse_mean'))} | "
            f"{_format_optional(policy.get('final_rmse_std'))} | "
            f"{_format_optional(delta.get('mean'))} | {_format_ci(delta.get('ci95'))} | "
            f"{_format_optional(delta.get('p'), precision=4)} |"
        )
    lines.append("")
    return lines


def write_comparison_report(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# Location Depth Sweep Contrast Report",
        "",
        "This report compares the Phase 4 constrained and unconstrained fixed-root depth sweeps.",
        "",
        *_headline_report_lines(comparison),
    ]
    for side_name in ("constrained", "unconstrained"):
        side = comparison[side_name]
        lines.extend(
            [
                f"## {side_name.title()}",
                "",
                f"- Config: `{side.get('config_path')}`",
                f"- Trials: {side.get('num_trials')}",
                f"- Rounds: {side.get('num_rounds')}",
                f"- Source prior: `{side.get('source_prior')}`",
                f"- Signal model: `{side.get('signal_model')}`",
                f"- Max step radius: {side.get('max_step_radius')}",
                f"- Questioner model: `{side.get('run_metadata', {}).get('questioner_model')}`",
                f"- Host: `{side.get('run_metadata', {}).get('hostname')}`",
                f"- SLURM job: `{side.get('run_metadata', {}).get('slurm_job_id')}`",
                "",
                *token_usage_report_lines(side.get("token_usage")),
                "",
                "| policy | metric | final value | paired final delta vs EIG | 95% CI | Wilcoxon p |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for policy_label, policy in side["policies"].items():
            if policy_label == "EIG":
                for metric_name in PAIRED_METRIC_NAMES:
                    final_value = _format_optional(policy.get("final_metrics", {}).get(metric_name))
                    lines.append(f"| `{policy_label}` | `{metric_name}` | {final_value} | n/a | n/a | n/a |")
                continue
            for metric_name in PAIRED_METRIC_NAMES:
                metric_delta = policy.get("paired_delta_vs_eig", {}).get(metric_name, {})
                final_value = _format_optional(policy.get("final_metrics", {}).get(metric_name))
                delta = _format_optional(metric_delta.get("mean"))
                ci = metric_delta.get("ci95")
                ci_text = _format_ci(ci) if ci is not None else "n/a"
                p_text = _format_optional(metric_delta.get("p"), precision=4)
                lines.append(
                    f"| `{policy_label}` | `{metric_name}` | {final_value} | {delta} | {ci_text} | {p_text} |"
                )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_optional(value: Any, *, precision: int = 4) -> str:
    number = _maybe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{precision}f}"


def _format_ci(values: Any) -> str:
    if not isinstance(values, list | tuple) or len(values) != 2:
        return "n/a"
    low = _maybe_float(values[0])
    high = _maybe_float(values[1])
    if low is None or high is None:
        return "n/a"
    return f"[{low:.4f}, {high:.4f}]"


def plot_comparison(path: Path, comparison: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("source_rmse", "final RMSE delta vs EIG"),
        ("posterior_entropy", "final entropy delta vs EIG"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for col_idx, side_name in enumerate(("constrained", "unconstrained")):
        side = comparison[side_name]
        for row_idx, (metric_name, y_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            labels = []
            means = []
            lows = []
            highs = []
            for policy_label, policy in side["policies"].items():
                if policy_label == "EIG":
                    continue
                metric_delta = policy.get("paired_delta_vs_eig", {}).get(metric_name, {})
                mean = _maybe_float(metric_delta.get("mean"))
                ci = metric_delta.get("ci95")
                if mean is None or not isinstance(ci, list | tuple) or len(ci) != 2:
                    continue
                low = _maybe_float(ci[0])
                high = _maybe_float(ci[1])
                if low is None or high is None:
                    continue
                labels.append(policy_label)
                means.append(mean)
                lows.append(mean - low)
                highs.append(high - mean)
            x = np.arange(len(labels))
            ax.axhline(0.0, color="black", linewidth=0.8)
            if labels:
                ax.bar(x, means, yerr=np.asarray([lows, highs]), capsize=3)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{side_name.title()} {metric_name}")
            ax.set_ylabel(y_label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_headline_rmse(path: Path, comparison: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    side = comparison.get("constrained", {})
    labels = headline_policy_labels(comparison)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in labels:
        trace = side.get("policies", {}).get(label, {}).get("rmse_trace", [])
        values = [
            float(value)
            for value in trace
            if value is not None and math.isfinite(float(value))
        ]
        if not values:
            continue
        rounds = np.arange(1, len(values) + 1)
        ax.plot(rounds, values, marker="o", linewidth=1.8, label=label)
    ax.set_xlabel("round")
    ax.set_ylabel("mean RMSE")
    ax.set_title("Constrained location depth sweep")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare constrained and unconstrained location depth sweeps.")
    parser.add_argument("--constrained", type=Path, required=True, help="Constrained fixed_root_depth_sweep_metrics.json")
    parser.add_argument("--unconstrained", type=Path, required=True, help="Unconstrained fixed_root_depth_sweep_metrics.json")
    parser.add_argument("--output-dir", type=Path, default=Path("results/location_depth_sweep_contrast"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plots/location_depth_sweep_contrast"))
    parser.add_argument("--run-name", default="location_depth_sweep_contrast")
    args = parser.parse_args()

    comparison = compare_depth_sweeps(_load_summary(args.constrained), _load_summary(args.unconstrained))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.run_name}_summary.json"
    report_path = args.output_dir / f"{args.run_name}_REPORT.md"
    plot_path = args.plot_dir / f"{args.run_name}_depth_contrast.png"
    headline_plot_path = args.plot_dir / f"{args.run_name}_headline_rmse.png"

    summary_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_comparison_report(report_path, comparison)
    plot_comparison(plot_path, comparison)
    plot_headline_rmse(headline_plot_path, comparison)
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Plot: {plot_path}")
    print(f"Headline plot: {headline_plot_path}")


if __name__ == "__main__":
    main()
