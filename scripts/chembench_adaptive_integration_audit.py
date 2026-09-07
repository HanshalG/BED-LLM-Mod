"""Replay banked synthetic reference values with a new numerical integrator."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.chembench_mopen.raw_belief import GaussianParticleModel
from environments.chembench_mopen.raw_integration import predictive_expectation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "results/nonmyopic/chembench_raw_integration/20260908-v1/RESULT.json"
    )
    data = source.read_bytes()
    assert (
        hashlib.sha256(data).hexdigest()
        == "3a3352b8087fb3d95be99fa03eda625e7684b7979df95a77549d7e87120dab23"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        rows = []
        for row in json.loads(data)["cases"]:
            separation = row["separation"]
            sigmas = [[0.3, 1], [2, 1]] if row["unequal_noise"] else 1
            model = GaussianParticleModel(
                [[-separation, -0.5], [separation, 0.5]], sigmas, [[0], [1]], [0.3, 0.7]
            )
            estimates = [
                predictive_expectation(
                    model,
                    model.initial_state,
                    a,
                    model.risk,
                    value_bound=0.25,
                    tolerance=1e-6,
                )
                for a in (0, 1)
            ]
            values = [e.value for e in estimates]
            references = row["reference_risks"]
            rows.append(
                {
                    "separation": separation,
                    "unequal_noise": row["unequal_noise"],
                    "estimates": [asdict(e) for e in estimates],
                    "absolute_error": max(
                        abs(v - r) for v, r in zip(values, references)
                    ),
                    "regret": references[int(np.argmin(values))] - min(references),
                }
            )
        passed = all(
            row["absolute_error"] <= 1e-6 and row["regret"] <= 1e-3 for row in rows
        )
        result = {
            "status": "one_step_reference_passed"
            if passed
            else "one_step_reference_failed",
            "cases": rows,
            "model_calls": 0,
            "cost_usd": 0,
            "chemistry_outcomes_opened": False,
            "depth_two_three_qualified": False,
            "authorizes_paid_calls": False,
            "source_hashes": {
                p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                for p in [
                    "scripts/chembench_adaptive_integration_audit.py",
                    "environments/chembench_mopen/raw_integration.py",
                    "environments/chembench_mopen/raw_belief.py",
                ]
            },
        }
    except Exception as exc:
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps({"status": "execution_failed", "error": str(exc)}) + "\n"
        )
        raise
    temporary = args.output_dir / "RESULT.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output_dir / "RESULT.json")
    print(result["status"])


if __name__ == "__main__":
    main()
