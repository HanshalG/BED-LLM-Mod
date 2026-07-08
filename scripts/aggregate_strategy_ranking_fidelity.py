from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.strategy_ranking_fidelity import (  # noqa: E402
    _aggregate_variant_metrics,
    _gate_assessment,
    _jsonable,
    _write_plot,
    _write_report,
)
from scripts.llm_token_usage import merge_token_usage_summaries


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_records_path(summary_path: Path, records_path: str) -> Path:
    candidate = Path(records_path)
    if candidate.exists():
        return candidate
    relative = summary_path.parent / candidate.name
    if relative.exists():
        return relative
    return candidate


def _load_records_from_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _records_and_metadata_from_input(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path.suffix == ".jsonl":
        return _load_records_from_jsonl(path), {"records_path": str(path)}

    summary = _load_json(path)
    records_path = summary.get("records_path")
    if not records_path:
        raise ValueError(f"Summary does not include records_path: {path}")
    resolved_records_path = _resolve_records_path(path, str(records_path))
    records = _load_records_from_jsonl(resolved_records_path)
    return records, summary


def _infer_depths(records: list[dict[str, Any]], metadata: list[dict[str, Any]]) -> list[int]:
    depths: set[int] = set()
    for item in metadata:
        depths.update(int(depth) for depth in item.get("depths", []))
    for record in records:
        for variant_metrics in record.get("score_variant_metrics", {}).values():
            depths.update(int(depth) for depth in variant_metrics)
    return sorted(depths)


def _infer_score_variants(records: list[dict[str, Any]], metadata: list[dict[str, Any]]) -> list[str]:
    variants: set[str] = set()
    for item in metadata:
        variants.update(str(variant) for variant in item.get("score_variants", []))
    for record in records:
        variants.update(str(variant) for variant in record.get("score_variant_metrics", {}))
    return sorted(variants)


def aggregate_strategy_ranking_fidelity(
    inputs: list[Path],
    *,
    output_dir: Path,
    run_name: str,
    seed: int = 0,
) -> Path:
    if not inputs:
        raise ValueError("At least one input summary or records file is required")

    all_records: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for input_path in inputs:
        records, item_metadata = _records_and_metadata_from_input(input_path)
        all_records.extend(records)
        metadata.append(item_metadata)

    if not all_records:
        raise ValueError("No records found in inputs")

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"{run_name}_records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")

    depths = _infer_depths(all_records, metadata)
    score_variants = _infer_score_variants(all_records, metadata)
    state_rounds = sorted({int(record["round_index"]) for record in all_records})
    trial_indices = sorted({int(record["trial_index"]) for record in all_records})

    summary = {
        "generated": datetime.now().isoformat(),
        "config_path": "; ".join(str(item.get("config_path", "")) for item in metadata if item.get("config_path")),
        "questioner_model": "; ".join(
            sorted({str(item.get("questioner_model")) for item in metadata if item.get("questioner_model")})
        ),
        "hostname": "aggregate",
        "slurm_job_id": None,
        "num_trials": len(trial_indices),
        "num_probe_states": len(all_records),
        "state_rounds": state_rounds,
        "depths": depths,
        "deployments": sorted({item.get("deployments") for item in metadata if item.get("deployments") is not None}),
        "score_variants": score_variants,
        "location_seed": sorted({item.get("location_seed") for item in metadata if item.get("location_seed") is not None}),
        "location_num_rounds": sorted(
            {item.get("location_num_rounds") for item in metadata if item.get("location_num_rounds") is not None}
        ),
        "location_strategy_num_rollouts": sorted(
            {
                item.get("location_strategy_num_rollouts")
                for item in metadata
                if item.get("location_strategy_num_rollouts") is not None
            }
        ),
        "location_strategy_rollout_scoring_support_mode": sorted(
            {
                str(item.get("location_strategy_rollout_scoring_support_mode"))
                for item in metadata
                if item.get("location_strategy_rollout_scoring_support_mode")
            }
        ),
        "location_strategy_rollout_score_mode": sorted(
            {
                str(item.get("location_strategy_rollout_score_mode"))
                for item in metadata
                if item.get("location_strategy_rollout_score_mode")
            }
        ),
        "location_strategy_rollout_refresh_hypotheses_each_step": sorted(
            {
                item.get("location_strategy_rollout_refresh_hypotheses_each_step")
                for item in metadata
                if item.get("location_strategy_rollout_refresh_hypotheses_each_step") is not None
            }
        ),
        "location_strategy_rollout_final_refresh_enabled": sorted(
            {
                item.get("location_strategy_rollout_final_refresh_enabled")
                for item in metadata
                if item.get("location_strategy_rollout_final_refresh_enabled") is not None
            }
        ),
        "token_usage": merge_token_usage_summaries([item.get("token_usage") for item in metadata]),
        "input_paths": [str(path) for path in inputs],
        "records_path": str(records_path),
        "aggregate": _aggregate_variant_metrics(all_records, depths, score_variants, np.random.default_rng(seed)),
    }
    summary["gate_assessment"] = _gate_assessment(summary)

    summary_path = output_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(output_dir / f"{run_name}_REPORT.md", summary)
    _write_plot(output_dir / f"{run_name}_plot.png", summary)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate split StrategyEIG ranking-fidelity diagnostics.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Summary JSON or records JSONL files to aggregate")
    parser.add_argument("--output-dir", type=Path, default=Path("results/ranking_fidelity"))
    parser.add_argument("--run-name", default="aggregate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary_path = aggregate_strategy_ranking_fidelity(
        [path.resolve() for path in args.inputs],
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
    )
    print(f"[ranking-fidelity-aggregate] Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
