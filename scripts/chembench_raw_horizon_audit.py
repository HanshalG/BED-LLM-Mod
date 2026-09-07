"""Bounded numerical mechanics audit; no source chemistry or LLM calls."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import monotonic

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.chembench_mopen.raw_belief import GaussianParticleModel
from environments.chembench_mopen.raw_horizon import plan_raw_horizon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = GaussianParticleModel(
        [[-0.3, -0.3, -0.3], [0.3, 0.3, 0.3]], 1, [[0], [1]], [0.5, 0.5]
    )
    rows = []
    for depth in (1, 2, 3):
        started = monotonic()
        try:
            plan = plan_raw_horizon(
                model,
                model.initial_state,
                depth,
                available=tuple(range(depth)),
                max_evaluations=1_000_000,
                max_seconds=30,
                tolerance=1e-5,
            )
            row = {"depth": depth, "status": "completed", "plan": asdict(plan)}
        except SearchLimitExceeded as exc:
            row = {"depth": depth, "status": "resource_limit", "message": str(exc)}
        except Exception as exc:
            row = {
                "depth": depth,
                "status": "execution_failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        row["wall_seconds"] = monotonic() - started
        rows.append(row)
        temp = args.output_dir / f"depth{depth}.tmp"
        temp.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output_dir / f"depth{depth}.json")
        if row["status"] != "completed":
            break
    root = Path(__file__).resolve().parents[1]
    result = {
        "status": "reference_completed"
        if len(rows) == 3 and all(r["status"] == "completed" for r in rows)
        else "reference_not_qualified",
        "cases": rows,
        "settings": {
            "max_evaluations_per_plan": 1_000_000,
            "max_seconds_per_plan": 30,
            "tolerance": 1e-5,
        },
        "source_hashes": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in [
                "scripts/chembench_raw_horizon_audit.py",
                "environments/chembench_mopen/raw_horizon.py",
                "environments/chembench_mopen/raw_integration.py",
                "environments/chembench_mopen/raw_belief.py",
            ]
        },
        "model_calls": 0,
        "cost_usd": 0,
        "chemistry_outcomes_opened": False,
        "authorizes_paid_calls": False,
        "positive_adaptivity_tested": False,
    }
    temp = args.output_dir / "RESULT.tmp"
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output_dir / "RESULT.json")
    print(result["status"])


if __name__ == "__main__":
    main()
