#!/usr/bin/env python3
"""Run the fresh exact-10 serving gate for dynamic-vs-fixed confirmation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_qwen_pooled_support_serving_smoke as base


INTERFACE_VERSION = "number-game-qwen-dynamic-vs-fixed-serving-smoke-1"
MODEL_SEEDS = (65_900, 65_901)


@contextmanager
def configured_base() -> Iterator[None]:
    overrides = {
        "INTERFACE_VERSION": INTERFACE_VERSION,
        "MODEL_SEEDS": MODEL_SEEDS,
    }
    originals = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def run_smoke(
    *,
    output_dir: Path,
    run_id: str,
    adapter: Any | None = None,
) -> dict[str, Any]:
    with configured_base():
        return base.run_smoke(
            output_dir=output_dir,
            run_id=run_id,
            adapter=adapter,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    try:
        result = run_smoke(
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    except Exception as exc:
        checkpoint(
            args.output_dir / "FAILURE.json",
            {
                "schema_version": base.SCHEMA_VERSION,
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
