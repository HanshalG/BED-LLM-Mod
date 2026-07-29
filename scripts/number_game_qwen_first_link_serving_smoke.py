#!/usr/bin/env python3
"""Run the exact-10 Qwen Number Game first-link serving gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_grok_planner_serving_smoke import run_smoke


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-first-link-serving-smoke-1"
MODEL_ID = "qwen/qwen3.7-plus"
MODEL_SEED = 60_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    try:
        result = run_smoke(
            output_dir=args.output_dir,
            run_id=args.run_id,
            model_id=MODEL_ID,
            model_seed=MODEL_SEED,
            interface_version=INTERFACE_VERSION,
        )
    except Exception as exc:
        checkpoint(
            args.output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
