from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "paired_delta_vs_eig" not in data or "location_strategy_num_rollouts" not in data:
        raise ValueError(f"{path} does not look like a fixed-root depth sweep summary")
    return data


def _maybe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _strategy_depth(policy_label: str) -> int | None:
    if not policy_label.startswith("StrategyEIG-d"):
        return None
    try:
        return int(policy_label.rsplit("d", 1)[1])
    except ValueError:
        return None


def compare_rollout_ablation(inputs: list[Path]) -> dict[str, Any]:
    if not inputs:
        raise ValueError("At least one fixed-root summary is required")

    runs: list[dict[str, Any]] = []
    for path in inputs:
        summary = _load_summary(path)
        rollouts = int(summary["location_strategy_num_rollouts"])
        policies: dict[str, Any] = {}
        for policy_label, metrics in sorted(summary.get("paired_delta_vs_eig", {}).items()):
            depth = _strategy_depth(str(policy_label))
            if depth is None:
                continue
            policy_entry: dict[str, Any] = {"depth": depth}
            for metric_name in ("source_rmse", "posterior_entropy"):
                metric = metrics.get(metric_name, {})
                policy_entry[metric_name] = {
                    "final_delta_mean": _maybe_float(metric.get("final_delta_mean")),
                    "final_delta_ci95": metric.get("final_delta_ci95", [None, None]),
                    "wilcoxon_signed_rank_p": _maybe_float(metric.get("wilcoxon_signed_rank_p")),
                }
            policies[str(policy_label)] = policy_entry
        runs.append(
            {
                "input_path": str(path),
                "rollouts": rollouts,
                "config_path": summary.get("config_path"),
                "num_trials": summary.get("num_trials"),
                "num_rounds": summary.get("num_rounds"),
                "max_step_radius": summary.get("location_max_step_radius"),
                "run_metadata": summary.get("run_metadata", {}),
                "policies": policies,
            }
        )

    runs.sort(key=lambda run: int(run["rollouts"]))
    return {"runs": runs}


def write_rollout_ablation_report(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# Location Rollout-Count Ablation",
        "",
        "This report compares fixed-root StrategyEIG depth sweeps at different rollout counts.",
        "",
        "| rollouts | policy | metric | final delta vs EIG | 95% CI | Wilcoxon p |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for run in comparison.get("runs", []):
        for policy_label, policy in run.get("policies", {}).items():
            for metric_name in ("source_rmse", "posterior_entropy"):
                metric = policy.get(metric_name, {})
                mean = _format_optional(metric.get("final_delta_mean"))
                ci = _format_ci(metric.get("final_delta_ci95"))
                p_value = _format_optional(metric.get("wilcoxon_signed_rank_p"), precision=4)
                lines.append(
                    f"| {int(run['rollouts'])} | `{policy_label}` | `{metric_name}` | {mean} | {ci} | {p_value} |"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def plot_rollout_ablation(path: Path, comparison: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("source_rmse", "final RMSE delta vs EIG"),
        ("posterior_entropy", "final entropy delta vs EIG"),
    ]
    policies = sorted(
        {
            policy_label
            for run in comparison.get("runs", [])
            for policy_label in run.get("policies", {})
        },
        key=lambda label: (_strategy_depth(label) or 10_000, label),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, (metric_name, y_label) in zip(axes, metrics):
        ax.axhline(0.0, color="black", linewidth=0.8)
        for policy_label in policies:
            xs = []
            ys = []
            for run in comparison.get("runs", []):
                metric = run.get("policies", {}).get(policy_label, {}).get(metric_name, {})
                mean = _maybe_float(metric.get("final_delta_mean"))
                if mean is None:
                    continue
                xs.append(int(run["rollouts"]))
                ys.append(mean)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=policy_label)
        ax.set_xlabel("StrategyEIG rollout count")
        ax.set_ylabel(y_label)
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed-root depth sweeps across StrategyEIG rollout counts.")
    parser.add_argument("inputs", nargs="+", type=Path, help="fixed_root_depth_sweep_metrics.json files")
    parser.add_argument("--output-dir", type=Path, default=Path("results/location_rollout_ablation"))
    parser.add_argument("--plot-dir", type=Path, default=Path("plots/location_rollout_ablation"))
    parser.add_argument("--run-name", default="location_rollout_ablation")
    args = parser.parse_args()

    comparison = compare_rollout_ablation([path.resolve() for path in args.inputs])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.run_name}_summary.json"
    report_path = args.output_dir / f"{args.run_name}_REPORT.md"
    plot_path = args.plot_dir / f"{args.run_name}.png"
    summary_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_rollout_ablation_report(report_path, comparison)
    plot_rollout_ablation(plot_path, comparison)
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
