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

from scripts.strategy_ranking_fidelity import _pearson, _spearman


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _clean_pair(xs: list[Any], ys: list[Any]) -> tuple[list[float], list[float]]:
    clean_x: list[float] = []
    clean_y: list[float] = []
    for x_value, y_value in zip(xs, ys):
        if x_value is None or y_value is None:
            continue
        x_float = float(x_value)
        y_float = float(y_value)
        if math.isfinite(x_float) and math.isfinite(y_float):
            clean_x.append(x_float)
            clean_y.append(y_float)
    return clean_x, clean_y


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return float(np.mean(clean))


def _std_error(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) <= 1:
        return 0.0 if clean else None
    return float(np.std(clean, ddof=1) / math.sqrt(len(clean)))


def _variance(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) <= 1:
        return 0.0 if clean else None
    return float(np.var(clean, ddof=1))


def _depth_record_metrics(record: dict[str, Any], depth: str) -> dict[str, Any] | None:
    realized = record.get("realized_by_depth", {}).get(str(depth))
    if not realized:
        return None
    entropy, rmse = _clean_pair(
        realized.get("entropy_drop_mean", []),
        realized.get("rmse_drop_mean", []),
    )
    truth, rmse_for_truth = _clean_pair(
        realized.get("truth_log_prob_mean", []),
        realized.get("rmse_drop_mean", []),
    )
    expected_rmse_drop, rmse_for_expected = _clean_pair(
        realized.get("expected_posterior_rmse_drop_mean", []),
        realized.get("rmse_drop_mean", []),
    )
    rmse_stds = [
        float(value)
        for value in realized.get("rmse_drop_std", [])
        if value is not None and math.isfinite(float(value))
    ]
    rmse_between = _variance(rmse)
    rmse_within = float(np.mean(np.square(rmse_stds))) if rmse_stds else None
    rmse_snr = None
    if rmse_between is not None and rmse_within is not None and rmse_within > 0.0:
        rmse_snr = float(rmse_between / rmse_within)
    execution = realized.get("strategy_execution_fidelity", {}) or {}
    return {
        "trial_index": int(record.get("trial_index", -1)),
        "round_index": int(record.get("round_index", -1)),
        "depth": int(depth),
        "n": len(entropy),
        "spearman_realized_entropy_vs_rmse_drop": _spearman(entropy, rmse),
        "pearson_realized_entropy_vs_rmse_drop": _pearson(entropy, rmse),
        "spearman_truth_log_prob_vs_rmse_drop": _spearman(truth, rmse_for_truth),
        "spearman_expected_posterior_rmse_drop_vs_rmse_drop": _spearman(expected_rmse_drop, rmse_for_expected),
        "rmse_var_between_strategies": rmse_between,
        "rmse_var_within_strategy": rmse_within,
        "rmse_snr_between_over_within": rmse_snr,
        "strategy_query_distance_ratio": execution.get("between_over_within_query_distance"),
    }


def analyze_rmse_repair(records: list[dict[str, Any]]) -> dict[str, Any]:
    depths = sorted(
        {
            int(depth)
            for record in records
            for depth in record.get("realized_by_depth", {}).keys()
        }
    )
    per_record: list[dict[str, Any]] = []
    by_depth: dict[str, Any] = {}
    for depth in depths:
        depth_items = [
            item
            for record in records
            if (item := _depth_record_metrics(record, str(depth))) is not None
        ]
        per_record.extend(depth_items)
        by_depth[str(depth)] = {}
        for key in (
            "spearman_realized_entropy_vs_rmse_drop",
            "pearson_realized_entropy_vs_rmse_drop",
            "spearman_truth_log_prob_vs_rmse_drop",
            "spearman_expected_posterior_rmse_drop_vs_rmse_drop",
            "rmse_snr_between_over_within",
            "strategy_query_distance_ratio",
        ):
            values = [
                float(item[key])
                for item in depth_items
                if item.get(key) is not None and math.isfinite(float(item[key]))
            ]
            by_depth[str(depth)][key] = {
                "mean": _mean(values),
                "se": _std_error(values),
                "n": len(values),
            }
    expected_values = [
        float(item["spearman_expected_posterior_rmse_drop_vs_rmse_drop"])
        for item in per_record
        if item.get("spearman_expected_posterior_rmse_drop_vs_rmse_drop") is not None
        and math.isfinite(float(item["spearman_expected_posterior_rmse_drop_vs_rmse_drop"]))
    ]
    expected_status = {
        "status": "available" if expected_values else "unavailable_from_current_records",
        "reason": (
            "Future ranking-fidelity records include candidate-level expected posterior "
            "RMSE drops and final posterior states, so expected posterior RMSE can be "
            "analyzed directly."
            if expected_values
            else "The aggregate ranking-fidelity JSONL stores candidate-level realized "
            "entropy drops, point-RMSE drops, truth-log-probability means, and "
            "query-distance diagnostics, but it does not store final posterior "
            "hypothesis supports/probabilities for each deployment. Expected "
            "posterior RMSE cannot be recomputed exactly without those posterior states."
        ),
    }
    return {
        "num_records": len(records),
        "depths": depths,
        "by_depth": by_depth,
        "per_record": per_record,
        "expected_posterior_rmse": expected_status,
    }


def write_report(path: Path, analysis: dict[str, Any]) -> None:
    lines = [
        "# RMSE Repair Analysis",
        "",
        f"- Records: {analysis['num_records']}",
        f"- Depths: {', '.join(str(depth) for depth in analysis['depths'])}",
        "",
        "## Realized-Realized Link",
        "",
        "| Depth | entropy-drop vs RMSE-drop Spearman | truth-log-prob vs RMSE-drop Spearman | expected-posterior-RMSE-drop vs RMSE-drop Spearman | RMSE SNR | query distance ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in analysis["depths"]:
        metrics = analysis["by_depth"][str(depth)]
        entropy = metrics["spearman_realized_entropy_vs_rmse_drop"]
        truth = metrics["spearman_truth_log_prob_vs_rmse_drop"]
        expected = metrics["spearman_expected_posterior_rmse_drop_vs_rmse_drop"]
        snr = metrics["rmse_snr_between_over_within"]
        qdist = metrics["strategy_query_distance_ratio"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(depth),
                    _format_mean_se(entropy),
                    _format_mean_se(truth),
                    _format_mean_se(expected),
                    _format_mean_se(snr),
                    _format_mean_se(qdist),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation: this table tests whether realized posterior-information gains are "
            "themselves rank-aligned with realized point-RMSE gains across candidate "
            "strategies. If these values are near zero, then point-RMSE is weakly rankable "
            "at the probe horizons even when the information metric is rankable.",
            "",
            "## Expected Posterior RMSE",
            "",
            f"Status: `{analysis['expected_posterior_rmse']['status']}`.",
            "",
            analysis["expected_posterior_rmse"]["reason"],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_mean_se(metric: dict[str, Any]) -> str:
    mean = metric.get("mean")
    se = metric.get("se")
    if mean is None:
        return "NA"
    if se is None:
        return f"{float(mean):.3f}"
    return f"{float(mean):.3f} +/- {float(se):.3f}"


def append_report_section(path: Path, analysis_report_path: Path, analysis: dict[str, Any]) -> None:
    marker = "## RMSE Repair Analysis"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    text = text.split(marker)[0].rstrip()
    lines = [
        text,
        "",
        marker,
        "",
        "The realized-realized repair analysis was computed from the aggregate records at",
        f"`{analysis_report_path}`.",
        "",
        "| Depth | entropy-drop vs RMSE-drop Spearman | truth-log-prob vs RMSE-drop Spearman | RMSE SNR |",
        "|---:|---:|---:|---:|",
    ]
    for depth in analysis["depths"]:
        metrics = analysis["by_depth"][str(depth)]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(depth),
                    _format_mean_se(metrics["spearman_realized_entropy_vs_rmse_drop"]),
                    _format_mean_se(metrics["spearman_truth_log_prob_vs_rmse_drop"]),
                    _format_mean_se(metrics["rmse_snr_between_over_within"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Expected posterior RMSE status: `{analysis['expected_posterior_rmse']['status']}`.",
            analysis["expected_posterior_rmse"]["reason"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RMSE repair diagnostics from ranking-fidelity records.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("results/ranking_fidelity/rmse_repair_analysis.json"))
    parser.add_argument("--output-report", type=Path, default=Path("results/ranking_fidelity/RMSE_REPAIR.md"))
    parser.add_argument("--append-to", type=Path)
    args = parser.parse_args()

    analysis = analyze_rmse_repair(_load_records(args.records))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.output_report, analysis)
    if args.append_to is not None:
        append_report_section(args.append_to, args.output_report, analysis)
    print(f"json: {args.output_json}")
    print(f"report: {args.output_report}")


if __name__ == "__main__":
    main()
