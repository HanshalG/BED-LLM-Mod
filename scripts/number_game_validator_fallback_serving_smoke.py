#!/usr/bin/env python3
"""Run the exact-10 Gemini validator-path fallback serving smoke."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from threading import Lock
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_depth_three_development as depth
from scripts import number_game_qwen_planner_depth_three as qwen
from scripts.number_game_provider_seed_fallback import (
    DEFAULT_FALLBACK_OFFSETS,
    fallback_adapter_factory,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-validator-fallback-serving-smoke-1"
MODEL_ID = qwen.TARGET_MODEL_ID
SEEDS = tuple(range(79_000, 79_010))
EXPECTED_REQUESTS = 10
RUN_BUDGET_USD = 0.04
MIN_SUPPORT_SIZE = 16
MAX_FALLBACK_EVENTS = 10


@contextmanager
def fallback_adapters(
    events: list[dict[str, Any]],
) -> Iterator[None]:
    original = depth._adapter
    depth._adapter = fallback_adapter_factory(
        original,
        events=events,
        event_lock=Lock(),
    )
    try:
        yield
    finally:
        depth._adapter = original


def smoke_gates(
    *,
    supports: list[list[Any]],
    diagnostics: list[dict[str, Any]],
    usage: dict[str, Any],
    fallback_events: list[dict[str, Any]],
) -> dict[str, bool]:
    return {
        "exact_ten_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "provider_error_retries_accounted": (
            usage["provider_error_retries"] <= usage["retry_count"]
        ),
        "fallback_events_within_cap": (
            len(fallback_events) <= MAX_FALLBACK_EVENTS
        ),
        "all_ten_draws_strict_json": (
            len(diagnostics) == EXPECTED_REQUESTS
            and all(
                item["codec_mode"] == "strict_json"
                for item in diagnostics
            )
        ),
        "all_supports_have_at_least_sixteen_extensions": (
            len(supports) == EXPECTED_REQUESTS
            and all(len(support) >= MIN_SUPPORT_SIZE for support in supports)
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_smoke_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }


def run_smoke(
    *,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    with fallback_adapters(events):
        supports, diagnostics, responses, snapshots = (
            qwen._generate_supports(
                seeds=SEEDS,
                output_dir=output_dir,
                run_id=run_id,
                run_budget_usd=RUN_BUDGET_USD,
                target_model=MODEL_ID,
            )
        )
    usage = qwen._usage([], snapshots)
    gates = smoke_gates(
        supports=supports,
        diagnostics=diagnostics,
        usage=usage,
        fallback_events=events,
    )
    raw_path = private_dir / "RAW_RESPONSES.json"
    checkpoint(
        raw_path,
        {
            "seeds": list(SEEDS),
            "responses": responses,
        },
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all(gates.values()) else "failed_closed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "seeds": list(SEEDS),
            "expected_requests": EXPECTED_REQUESTS,
            "fallback_offsets": list(DEFAULT_FALLBACK_OFFSETS),
            "provider_error_fallback_only": True,
            "efficacy_used_for_authorization": False,
            "run_budget_usd": RUN_BUDGET_USD,
        },
        "usage": usage,
        "support_sizes": [len(support) for support in supports],
        "diagnostics": diagnostics,
        "fallback_events": events,
        "gates": gates,
        "raw_responses_sha256": hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest(),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


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
