#!/usr/bin/env python3
"""Run the default-routing structured DiscoverPhysics replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.discoverphysics_dark_matter_grounded_policy import (
    INITIAL_MAX_TOKENS,
    MODEL_ID,
    PROJECTED_COST_USD,
    RUN_BUDGET_USD,
)
from scripts.discoverphysics_dark_matter_semantic_smoke import (
    NonReasoningOpenRouterAdapter,
)
from scripts.discoverphysics_dark_matter_structured_replication import (
    SCHEMA_VERSION,
    run_replication,
)
from scripts.discoverphysics_oscillator_belief_smoke import (
    SmokeExecutionError,
    checkpoint,
)


INTERFACE_VERSION = "discoverphysics-dark-matter-structured-replication-2"


def use_default_structured_routing(
    payload: dict[str, Any],
) -> dict[str, Any]:
    routed = dict(payload)
    if "response_format" in routed:
        routed["provider"] = {"require_parameters": False}
    return routed


class DefaultRoutingStructuredAdapter(NonReasoningOpenRouterAdapter):
    """Keep strict output while allowing unrelated sampling knobs to be ignored."""

    def _payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        *,
        disable_reasoning: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages,
            temperature,
            n,
            max_tokens,
            disable_reasoning=disable_reasoning,
            response_format=response_format,
        )
        return use_default_structured_routing(payload)


def _adapter(
    *,
    run_id: str,
    output_dir: Path,
) -> DefaultRoutingStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=140.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=8,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=INITIAL_MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    spec = ModelSpec(
        model=MODEL_ID,
        backend="openrouter",
        max_model_len=65536,
    )
    return DefaultRoutingStructuredAdapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discoverphysics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "RESULT.json"
    failure_path = output_dir / "FAILURE.json"
    try:
        payload = run_replication(
            discoverphysics_root=args.discoverphysics_root.resolve(),
            output_dir=output_dir,
            run_id=args.run_id,
            adapter_factory=_adapter,
            interface_version=INTERFACE_VERSION,
        )
        checkpoint(result_path, payload)
    except SmokeExecutionError as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": str(exc),
            "endpoint_accessed": False,
            "usage": exc.usage,
        }
        checkpoint(failure_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
