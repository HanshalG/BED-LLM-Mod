from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_summary_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "fixed_root_depth_sweep_metrics.json",
        path / "fixed_root_depth_sweep_metrics_recovered.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no fixed-root metrics JSON found under {path}")


def _load_decisions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _run_dir_for_summary(summary_path: Path) -> Path:
    return summary_path.parent


def _policy_records(summary: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {
        (int(record["trial_index"]), str(record["policy_label"])): record
        for record in summary.get("per_trial", [])
    }


def _decision_records(decisions: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in decisions:
        key = (int(record["trial_index"]), str(record["policy_label"]))
        grouped.setdefault(key, []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: int(item.get("round_index", 0)))
    return grouped


def _final_metric(record: dict[str, Any], metric_name: str) -> float | None:
    round_metrics = record.get("round_metrics") or []
    if not round_metrics or metric_name not in round_metrics[-1]:
        return None
    return float(round_metrics[-1][metric_name])


def _trajectory(record: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in record.get("history") or []:
        action = item.get("action")
        if action is not None and len(action) >= 2:
            points.append((float(action[0]), float(action[1])))
    return points


def _source_points(record: dict[str, Any]) -> list[tuple[float, float]]:
    hidden_state = record.get("hidden_state") or []
    points: list[tuple[float, float]] = []
    for source in hidden_state:
        if source is not None and len(source) >= 2:
            points.append((float(source[0]), float(source[1])))
    return points


def _strategy_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        strategy = record.get("selected_strategy")
        if not strategy:
            continue
        result.append(
            {
                "round_index": int(record.get("round_index", 0)),
                "selected_eig": record.get("selected_eig"),
                "root_query": record.get("selected_root_query"),
                "strategy": strategy,
            }
        )
    return result


def _candidate_rows(summary: dict[str, Any], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_policy = _policy_records(summary)
    decisions_by_policy = _decision_records(decisions)
    rows: list[dict[str, Any]] = []
    for (trial_index, policy_label), record in per_policy.items():
        if not policy_label.startswith("StrategyEIG-d"):
            continue
        baseline = per_policy.get((trial_index, "EIG"))
        final_rmse = _final_metric(record, "source_rmse")
        baseline_rmse = _final_metric(baseline or {}, "source_rmse") if baseline else None
        final_truth_logp = _final_metric(record, "truth_log_probability")
        baseline_truth_logp = _final_metric(baseline or {}, "truth_log_probability") if baseline else None
        rmse_delta = (
            None
            if final_rmse is None or baseline_rmse is None
            else float(final_rmse - baseline_rmse)
        )
        truth_logp_delta = (
            None
            if final_truth_logp is None or baseline_truth_logp is None
            else float(final_truth_logp - baseline_truth_logp)
        )
        rows.append(
            {
                "trial_index": trial_index,
                "policy_label": policy_label,
                "rmse_delta_vs_eig": rmse_delta,
                "truth_log_probability_delta_vs_eig": truth_logp_delta,
                "policy_final_source_rmse": final_rmse,
                "eig_final_source_rmse": baseline_rmse,
                "policy_final_truth_log_probability": final_truth_logp,
                "eig_final_truth_log_probability": baseline_truth_logp,
                "hidden_state": record.get("hidden_state"),
                "policy_trajectory": _trajectory(record),
                "eig_trajectory": _trajectory(baseline or {}),
                "source_points": _source_points(record),
                "selected_strategies": _strategy_records(decisions_by_policy.get((trial_index, policy_label), [])),
            }
        )
    return rows


def _sort_key(row: dict[str, Any]) -> tuple[int, float, float]:
    has_strategy = 0 if row.get("selected_strategies") else 1
    rmse_delta = row.get("rmse_delta_vs_eig")
    truth_delta = row.get("truth_log_probability_delta_vs_eig")
    return (
        has_strategy,
        float(rmse_delta) if rmse_delta is not None else 0.0,
        -float(truth_delta) if truth_delta is not None else 0.0,
    )


def _write_markdown(path: Path, examples: list[dict[str, Any]]) -> None:
    lines = ["# Qualitative Location Strategy Examples", ""]
    if not examples:
        lines.extend(["No StrategyEIG examples were available.", ""])
    for index, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"## Example {index}: trial {example['trial_index']} / {example['policy_label']}",
                "",
                f"- RMSE delta vs EIG: `{example.get('rmse_delta_vs_eig')}`",
                f"- Truth-log-prob delta vs EIG: `{example.get('truth_log_probability_delta_vs_eig')}`",
                f"- Plot: `{example.get('plot_path')}`",
                "",
            ]
        )
        for strategy in example.get("selected_strategies", []):
            wrapped = textwrap.fill(str(strategy["strategy"]), width=88)
            lines.extend(
                [
                    f"### Round {strategy['round_index'] + 1}",
                    "",
                    f"- Selected EIG: `{strategy.get('selected_eig')}`",
                    f"- Root query: `{strategy.get('root_query')}`",
                    "",
                    wrapped,
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_example(path: Path, example: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    for label, points, color in (
        ("EIG", example.get("eig_trajectory") or [], "tab:blue"),
        (example["policy_label"], example.get("policy_trajectory") or [], "tab:orange"),
    ):
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, marker="o", label=label, color=color)
        for round_index, (x_value, y_value) in enumerate(points, start=1):
            ax.text(x_value, y_value, str(round_index), fontsize=8, color=color)
    source_points = example.get("source_points") or []
    if source_points:
        ax.scatter(
            [point[0] for point in source_points],
            [point[1] for point in source_points],
            marker="*",
            s=140,
            color="black",
            label="truth",
        )
    ax.axhline(0.0, color="0.85", linewidth=0.8)
    ax.axvline(0.0, color="0.85", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"trial {example['trial_index']} {example['policy_label']}")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def extract_qualitative_examples(
    *,
    summary_path: Path,
    output_dir: Path,
    run_label: str | None = None,
    num_examples: int = 3,
) -> dict[str, Any]:
    resolved_summary_path = _resolve_summary_path(summary_path)
    summary = _load_json(resolved_summary_path)
    run_dir = _run_dir_for_summary(resolved_summary_path)
    decisions = _load_decisions(run_dir / "fixed_root_depth_sweep_decisions.jsonl")
    label = run_label or run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _candidate_rows(summary, decisions)
    rows.sort(key=_sort_key)
    examples = rows[: max(0, num_examples)]
    for index, example in enumerate(examples, start=1):
        example["run_label"] = label
        plot_path = output_dir / f"{label}_qualitative_example_{index}.png"
        try:
            _plot_example(plot_path, example)
            example["plot_path"] = str(plot_path)
        except Exception as exc:
            example["plot_error"] = repr(exc)

    json_path = output_dir / f"{label}_qualitative_examples.json"
    md_path = output_dir / f"{label}_qualitative_examples.md"
    json_path.write_text(json.dumps(examples, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, examples)
    return {
        "summary_path": str(resolved_summary_path),
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "num_examples": len(examples),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract qualitative StrategyEIG examples from location sweeps.")
    parser.add_argument("summary", type=Path, help="Run directory or fixed-root metrics JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("results/location_qualitative"))
    parser.add_argument("--run-label")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = extract_qualitative_examples(
        summary_path=args.summary,
        output_dir=args.output_dir,
        run_label=args.run_label,
        num_examples=args.num_examples,
    )
    if args.json:
        print(json.dumps({key: value for key, value in payload.items() if key != "examples"}, indent=2, sort_keys=True))
    else:
        print(f"json: {payload['json_path']}")
        print(f"markdown: {payload['markdown_path']}")
        print(f"num_examples: {payload['num_examples']}")


if __name__ == "__main__":
    main()

