#!/usr/bin/env python3
"""Replay support-quality diagnostics on the fully fresh Qwen source."""

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

from scripts import number_game_dynamic_support_quality96 as quality


INTERFACE_VERSION = "number-game-qwen-fully-fresh-source-quality-1"
SOURCE_DIR = REPO_ROOT / (
    "results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/"
    "number-game-qwen-fully-fresh-daily-stages-20260806T000200Z/source"
)
SOURCE_RESULT = SOURCE_DIR / "RESULT.json"
SOURCE_TREES = SOURCE_DIR / "TREES.json"
SOURCE_TARGETS = SOURCE_DIR / "TARGETS.json"
SOURCE_RESULT_SHA256 = (
    "13fd3361a8ef8f525f68733182a9bdb30151e37a4a5e14dfb35dde78540ab523"
)
SOURCE_TREES_SHA256 = (
    "f812e4a356f5129a6f2f22d5f0f995b76b4ac624a0f312154064ff8c51f2c7b0"
)
SOURCE_TARGETS_SHA256 = (
    "b799a5d6609f5e8088e2f0115ec1eaa3c4520cebcd187283daf679714cdc5e2b"
)
TREE_COUNT = 32
BOOTSTRAP_SEED = 100_900
BOOTSTRAP_SAMPLES = 20_000


@contextmanager
def configured_quality() -> Iterator[None]:
    overrides = {
        "INTERFACE_VERSION": INTERFACE_VERSION,
        "SOURCE_DIR": SOURCE_DIR,
        "SOURCE_RESULT": SOURCE_RESULT,
        "SOURCE_TREES": SOURCE_TREES,
        "SOURCE_TARGETS": SOURCE_TARGETS,
        "SOURCE_RESULT_SHA256": SOURCE_RESULT_SHA256,
        "SOURCE_TREES_SHA256": SOURCE_TREES_SHA256,
        "SOURCE_TARGETS_SHA256": SOURCE_TARGETS_SHA256,
        "TREE_COUNT": TREE_COUNT,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "BOOTSTRAP_SAMPLES": BOOTSTRAP_SAMPLES,
    }
    originals = {name: getattr(quality, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(quality, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(quality, name, value)


def run_analysis(
    output_dir: Path,
    *,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    with configured_quality():
        return quality.run_analysis(
            output_dir,
            result_path=SOURCE_RESULT,
            trees_path=SOURCE_TREES,
            targets_path=SOURCE_TARGETS,
            bootstrap_samples=bootstrap_samples,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    result = run_analysis(args.output_dir)
    print(json.dumps(result["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
