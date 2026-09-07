"""Qualify envelope knots against the unchanged crossing refinement protocol."""

import argparse
import hashlib
import json
from pathlib import Path
from environments.chembench_mopen.envelope_belief import EnvelopeGaussianModel
from scripts.chembench_batch_refinement import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    paths = [
        "scripts/chembench_envelope_refinement.py",
        "scripts/chembench_batch_refinement.py",
        "environments/chembench_mopen/envelope_belief.py",
        "environments/chembench_mopen/crossing_belief.py",
        "environments/chembench_mopen/batch_horizon.py",
        "environments/chembench_mopen/raw_belief.py",
        "environments/chembench_mopen/quantile_belief.py",
    ]
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
    try:
        result = execute(args.output_dir, model_type=EnvelopeGaussianModel)
    except Exception as exc:
        result = {
            "status": "execution_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    result.update(
        {
            "source_hashes": hashes,
            "settings": {
                "orders": [32, 64],
                "error_cap": 0.001,
                "regret_cap": 0.001,
                "adaptivity_floor": 0.001,
                "max_plan_states": 5000000,
                "max_plan_seconds": 60,
                "max_panel_seconds": 180,
                "workspace_bytes": 67108864,
                "rule": "equal_noise_upper_envelope_otherwise_all_crossings",
            },
            "model_calls": 0,
            "cost_usd": 0,
            "chemistry_outcomes_opened": False,
            "authorizes_paid_calls": False,
        }
    )
    temp = args.output_dir / "RESULT.tmp"
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output_dir / "RESULT.json")
    print(result["status"])


if __name__ == "__main__":
    main()
