#!/usr/bin/env python3
"""Run the corrected exact-10 Gemini validator fallback serving smoke."""

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
from scripts import number_game_validator_fallback_serving_smoke as base


INTERFACE_VERSION = "number-game-validator-fallback-serving-smoke-2"
SEEDS = tuple(range(85_000, 85_010))
V1_SMOKE_GATES = base.smoke_gates
EXPECTED_REJECTED_FIELDS = {
    "wrong_fields",
    "invalid_name",
    "invalid_expression",
    "inconsistent",
    "duplicate_extension",
}


def smoke_gates(
    *,
    supports: list[list[Any]],
    diagnostics: list[dict[str, Any]],
    usage: dict[str, Any],
    fallback_events: list[dict[str, Any]],
) -> dict[str, bool]:
    gates = V1_SMOKE_GATES(
        supports=supports,
        diagnostics=[
            {**item, "codec_mode": "strict_json"}
            for item in diagnostics
        ],
        usage=usage,
        fallback_events=fallback_events,
    )
    gates.pop("all_ten_draws_strict_json")
    gates["all_ten_base_strict_parser_contracts_valid"] = (
        len(diagnostics) == base.EXPECTED_REQUESTS
        and len(supports) == base.EXPECTED_REQUESTS
        and all(
            diagnostic.get("raw_count") == 24
            and diagnostic.get("valid_unique_count") == len(support)
            and set(diagnostic.get("rejected") or {})
            == EXPECTED_REJECTED_FIELDS
            for support, diagnostic in zip(
                supports,
                diagnostics,
                strict=True,
            )
        )
    )
    return gates


@contextmanager
def configured_base() -> Iterator[None]:
    overrides = {
        "INTERFACE_VERSION": INTERFACE_VERSION,
        "SEEDS": SEEDS,
        "smoke_gates": smoke_gates,
    }
    originals = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def run_smoke(*, output_dir: Path, run_id: str) -> dict[str, Any]:
    with configured_base():
        return base.run_smoke(output_dir=output_dir, run_id=run_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    try:
        result = run_smoke(output_dir=args.output_dir, run_id=args.run_id)
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
