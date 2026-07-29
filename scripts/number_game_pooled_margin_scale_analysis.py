#!/usr/bin/env python3
"""Test one frozen scale normalization for pooled-support first-link margins."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_qwen_first_link_mechanism64 import _interval
from scripts.number_game_ranking_fidelity_audit import spearman_correlation


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-pooled-margin-scale-analysis-1"
SOURCE_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_pooled_first_link_confirmation32"
    / "number-game-qwen-pooled-first-link-confirmation32-20260729T094455Z"
    / "RESULT.json"
)
SOURCE_RESULT_SHA256 = (
    "cef6ded08050e6df07de62bbbf548635f229a8bf73fac149b4d0bd960f7b0253"
)
BOOTSTRAP_SEED = 63_700
BOOTSTRAP_SAMPLES = 20_000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_value(mapping: dict[str, Any], root: int) -> float:
    return float(mapping[str(root)])


def analysis_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for tree in source["trees"]:
        selection = tree["selection"]
        candidate = int(selection["crossfit_depth_three_root"])
        myopic = int(selection["myopic_root"])
        if candidate == myopic:
            continue
        risks = [
            float(value)
            for value in selection["crossfit_depth_three_brier"].values()
        ]
        scale = statistics.pstdev(risks)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"tree {tree['tree_seed']} has nonpositive risk scale"
            )
        raw_margin = (
            root_value(selection["crossfit_depth_three_brier"], myopic)
            - root_value(selection["crossfit_depth_three_brier"], candidate)
        )
        realized = (
            root_value(tree["per_root_endpoint_brier"], myopic)
            - root_value(tree["per_root_endpoint_brier"], candidate)
        )
        rows.append(
            {
                "tree_seed": int(tree["tree_seed"]),
                "risk_scale_population_sd": scale,
                "raw_predicted_advantage": raw_margin,
                "normalized_predicted_advantage": raw_margin / scale,
                "realized_advantage": realized,
            }
        )
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    raw = [row["raw_predicted_advantage"] for row in rows]
    normalized = [
        row["normalized_predicted_advantage"] for row in rows
    ]
    realized = [row["realized_advantage"] for row in rows]
    scales = [row["risk_scale_population_sd"] for row in rows]
    raw_rho = spearman_correlation(raw, realized)
    normalized_rho = spearman_correlation(normalized, realized)
    rng = random.Random(BOOTSTRAP_SEED)
    raw_bootstrap = []
    normalized_bootstrap = []
    delta_bootstrap = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [rng.choice(rows) for _ in rows]
        sample_realized = [
            row["realized_advantage"] for row in sample
        ]
        sample_raw = [
            row["raw_predicted_advantage"] for row in sample
        ]
        sample_normalized = [
            row["normalized_predicted_advantage"] for row in sample
        ]
        sample_raw_rho = spearman_correlation(
            sample_raw,
            sample_realized,
        )
        sample_normalized_rho = spearman_correlation(
            sample_normalized,
            sample_realized,
        )
        raw_bootstrap.append(sample_raw_rho)
        normalized_bootstrap.append(sample_normalized_rho)
        delta_bootstrap.append(sample_normalized_rho - sample_raw_rho)
    return {
        "tree_count": len(rows),
        "raw_margin_spearman": raw_rho,
        "raw_margin_spearman_95pct_bootstrap": _interval(raw_bootstrap),
        "normalized_margin_spearman": normalized_rho,
        "normalized_margin_spearman_95pct_bootstrap": _interval(
            normalized_bootstrap
        ),
        "normalized_minus_raw_spearman": normalized_rho - raw_rho,
        "normalized_minus_raw_spearman_95pct_bootstrap": _interval(
            delta_bootstrap
        ),
        "risk_scale": {
            "minimum": min(scales),
            "mean": sum(scales) / len(scales),
            "maximum": max(scales),
        },
        "risk_scale_to_realized_advantage_spearman": spearman_correlation(
            scales,
            realized,
        ),
        "rows": list(rows),
    }


def run_analysis(
    *,
    output_dir: Path,
    source_result_path: Path = SOURCE_RESULT,
) -> dict[str, Any]:
    if sha256_file(source_result_path) != SOURCE_RESULT_SHA256:
        raise ValueError("pooled confirmation source hash changed")
    source = json.loads(source_result_path.read_text(encoding="utf-8"))
    if source.get("status") != "gated_null":
        raise ValueError("pooled confirmation status changed")
    rows = analysis_rows(source)
    if len(rows) != 30:
        raise ValueError(f"expected 30 changed-root rows, got {len(rows)}")
    summary = summarize(rows)
    gates = {
        "normalized_spearman_at_least_point_two_five": (
            summary["normalized_margin_spearman"] >= 0.25
        ),
        "normalized_spearman_interval_above_zero": (
            summary["normalized_margin_spearman_95pct_bootstrap"][0] > 0.0
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "retrospective_scale_calibration_positive"
            if all(gates.values())
            else "retrospective_scale_calibration_null"
        ),
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "analysis_is_retrospective": True,
            "single_frozen_transformation": (
                "raw predicted advantage divided by population standard "
                "deviation of the eight within-tree candidate-root risks"
            ),
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "model_calls": 0,
            "cost_usd": 0.0,
            "cannot_rescue_source_status": True,
        },
        "gates": gates,
        "summary": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, default=SOURCE_RESULT)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    result = run_analysis(
        output_dir=args.output_dir,
        source_result_path=args.source_result,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "gates": result["gates"],
                "summary": {
                    key: value
                    for key, value in result["summary"].items()
                    if key != "rows"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
