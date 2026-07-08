from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.llm_token_usage import summarize_llm_token_usage


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _candidate_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [
        path / "fixed_root_depth_sweep_metrics.json",
        path / "strategy_ranking_fidelity_summary.json",
        path / "metrics.json",
        path / "run.log",
    ]


def _resolve_input(path: Path) -> Path:
    for candidate in _candidate_paths(path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No supported metrics/log file found for {path}")


def _empty_token_usage() -> dict[str, Any]:
    return {
        "total": {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "unknown_completion_token_records": 0,
        },
        "by_call_type": {},
        "by_model": {},
    }


def _token_usage_from_json(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    token_usage = data.get("token_usage")
    if isinstance(token_usage, dict):
        return token_usage
    log_path = path.parent / "run.log"
    if log_path.exists():
        return summarize_llm_token_usage(log_path)
    return _empty_token_usage()


def _infer_label(path: Path, data: dict[str, Any]) -> str:
    for key in ("run_name", "name"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    if path.name == "run.log":
        return path.parent.name
    return path.stem


def _infer_method(data: dict[str, Any]) -> str:
    if "aggregate_by_strategy_depth" in data:
        return "StrategyEIG fixed-root sweep"
    if "ranking_fidelity" in data or "score_variants" in data:
        return "StrategyEIG ranking fidelity"
    config = data.get("config")
    if isinstance(config, dict):
        method = config.get("method")
        if isinstance(method, str):
            return method
    return "unknown"


def _infer_depth(data: dict[str, Any]) -> str:
    max_depth = data.get("max_depth")
    if isinstance(max_depth, int):
        return f"1..{max_depth}"
    aggregate = data.get("aggregate")
    if isinstance(aggregate, dict):
        depths: set[int] = set()
        for variant in aggregate.values():
            if isinstance(variant, dict):
                for key in variant:
                    try:
                        depths.add(int(key))
                    except (TypeError, ValueError):
                        pass
        if depths:
            return ",".join(str(depth) for depth in sorted(depths))
    depth = data.get("depth") or data.get("location_strategy_planning_depth")
    if isinstance(depth, int):
        return str(depth)
    return ""


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _ratio(numerator: int, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return float(numerator) / denominator


def _geometric_tree_nodes(branching_factor: int, depth: int) -> int:
    if depth <= 0:
        return 0
    if branching_factor <= 1:
        return depth
    return sum(branching_factor**level for level in range(depth))


def _brute_force_cost_proxy(data: dict[str, Any]) -> list[dict[str, Any]]:
    if "aggregate_by_strategy_depth" not in data:
        return []
    max_depth = _int(data.get("max_depth"))
    num_trials = _int(data.get("num_trials"))
    num_rounds = _int(data.get("num_rounds"))
    branching_factor = _int(
        data.get("location_strategy_num_candidates")
        or data.get("location_target_num_candidates")
    )
    rollouts = _int(data.get("location_strategy_num_rollouts"))
    if max_depth <= 0 or num_trials <= 0 or num_rounds <= 0 or branching_factor <= 0:
        return []

    deployed_decisions = num_trials * num_rounds
    rows: list[dict[str, Any]] = []
    for depth in range(1, max_depth + 1):
        brute_force_nodes = _geometric_tree_nodes(branching_factor, depth)
        brute_force_leaf_sequences = branching_factor**depth
        strategy_rollout_paths = deployed_decisions * branching_factor * rollouts if rollouts > 0 else None
        strategy_simulated_steps = (
            strategy_rollout_paths * depth
            if strategy_rollout_paths is not None
            else None
        )
        rows.append(
            {
                "depth": depth,
                "branching_factor": branching_factor,
                "rollouts": rollouts,
                "deployed_decisions": deployed_decisions,
                "strategy_root_sets": deployed_decisions,
                "strategy_rollout_paths": strategy_rollout_paths,
                "strategy_simulated_steps": strategy_simulated_steps,
                "brute_force_candidate_sets": deployed_decisions * brute_force_nodes,
                "brute_force_leaf_sequences": deployed_decisions * brute_force_leaf_sequences,
                "brute_force_candidate_set_ratio_vs_strategy": float(brute_force_nodes),
            }
        )
    return rows


def row_from_path(input_path: Path) -> dict[str, Any]:
    resolved = _resolve_input(input_path)
    if resolved.suffix == ".log":
        data: dict[str, Any] = {}
        token_usage = summarize_llm_token_usage(resolved)
    else:
        data = _read_json(resolved)
        token_usage = _token_usage_from_json(resolved, data)

    total = token_usage.get("total", {})
    total_tokens = _int(total.get("total_tokens"))
    prompt_tokens = _int(total.get("prompt_tokens"))
    completion_tokens = _int(total.get("completion_tokens"))
    calls = _int(total.get("calls"))
    num_trials = _float_or_none(data.get("num_trials"))
    num_rounds = _float_or_none(data.get("num_rounds"))
    trial_rounds = None if num_trials is None or num_rounds is None else num_trials * num_rounds

    return {
        "label": _infer_label(resolved, data),
        "path": str(resolved),
        "method": _infer_method(data),
        "depth": _infer_depth(data),
        "num_trials": num_trials,
        "num_rounds": num_rounds,
        "calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_trial": _ratio(total_tokens, num_trials),
        "tokens_per_trial_round": _ratio(total_tokens, trial_rounds),
        "brute_force_cost_proxy": _brute_force_cost_proxy(data),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.1f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# LLM Cost vs Depth",
        "",
        "| label | method | depth | trials | rounds | calls | prompt tokens | completion tokens | total tokens | tokens/trial | tokens/trial-round |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["label"]),
                    _fmt(row["method"]),
                    _fmt(row["depth"]),
                    _fmt(row["num_trials"]),
                    _fmt(row["num_rounds"]),
                    _fmt(row["calls"]),
                    _fmt(row["prompt_tokens"]),
                    _fmt(row["completion_tokens"]),
                    _fmt(row["total_tokens"]),
                    _fmt(row["tokens_per_trial"]),
                    _fmt(row["tokens_per_trial_round"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Note: fixed-root depth sweeps report total run cost for the full depth set; "
        "use separate single-depth runs for strict per-depth wall-clock/token accounting."
    )
    proxy_rows = [
        (row, proxy)
        for row in rows
        for proxy in row.get("brute_force_cost_proxy", [])
        if isinstance(proxy, dict)
    ]
    if proxy_rows:
        lines.extend(
            [
                "",
                "## StrategyEIG vs brute-force n-step EIG proxy",
                "",
                (
                    "Candidate-tree proxy: brute-force depth-d EIG expands candidate sets at "
                    "`1 + B + ... + B^(d-1)` nodes per deployed decision; StrategyEIG uses "
                    "one generated strategy-root set per deployed decision plus rollout simulations."
                ),
                "",
                "| run | depth | B | rollouts | decisions | StrategyEIG root sets | StrategyEIG rollout paths | StrategyEIG simulated steps | brute-force candidate sets | brute-force leaf sequences | BF/Strategy root-set ratio |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row, proxy in proxy_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _fmt(row["label"]),
                        _fmt(proxy.get("depth")),
                        _fmt(proxy.get("branching_factor")),
                        _fmt(proxy.get("rollouts")),
                        _fmt(proxy.get("deployed_decisions")),
                        _fmt(proxy.get("strategy_root_sets")),
                        _fmt(proxy.get("strategy_rollout_paths")),
                        _fmt(proxy.get("strategy_simulated_steps")),
                        _fmt(proxy.get("brute_force_candidate_sets")),
                        _fmt(proxy.get("brute_force_leaf_sequences")),
                        _fmt(proxy.get("brute_force_candidate_set_ratio_vs_strategy")),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def write_cost_table(paths: list[Path], output_dir: Path, run_name: str) -> tuple[Path, Path]:
    rows = [row_from_path(path) for path in paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_name}_cost_vs_depth.json"
    md_path = output_dir / f"{run_name}_cost_vs_depth.md"
    json_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown_table(rows), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BED-LLM token-cost table from run logs/summaries.")
    parser.add_argument("paths", nargs="+", type=Path, help="Run dirs, summary JSON files, or run.log files")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cost_vs_depth"))
    parser.add_argument("--run-name", default="cost_vs_depth")
    args = parser.parse_args()
    json_path, md_path = write_cost_table(args.paths, args.output_dir, args.run_name)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
