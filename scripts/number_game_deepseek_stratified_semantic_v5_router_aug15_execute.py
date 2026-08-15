#!/usr/bin/env python3
"""Execute the transport-only V5 router successor once."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.number_game_bitmask_semantic_gate import SeededAdapter
from scripts import number_game_deepseek_stratified_semantic_v4 as gate
from scripts import number_game_deepseek_stratified_semantic_v4_verify as verifier
from scripts import number_game_deepseek_stratified_semantic_v4_aug15_execute as parent


INTERFACE_VERSION = "number-game-deepseek-stratified-semantic-v5-router-aug15-execute-1"
DATE = "2026-08-15"
OPENING_USAGE_USD = 220.339269126
DAILY_CAP_USD = 5.0
STAGE_CAP_USD = 0.20
MODEL_ID = gate.MODEL_ID
MAX_TOKENS = gate.MAX_TOKENS
CONCURRENCY = 64
ENDPOINTS_URL = f"https://openrouter.ai/api/v1/models/{MODEL_ID}/endpoints"
PRICE_CEILING = {"prompt": 0.20e-6, "completion": 0.50e-6}
ROOT = REPO_ROOT / "results/nonmyopic/number_game_deepseek_stratified_semantic_v5_router"
RUN = ROOT / "semantic-20260815"
BINDING = ROOT / "EXECUTION_BINDING.json"
RESULT = ROOT / "DAILY_RESULT_20260815.json"
FAILURE = ROOT / "DAILY_FAILURE_20260815.json"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-15-number-game-deepseek-stratified-semantic-v5-router.json"
RAW = RUN / "private/RAW_RESPONSES.json"
LABEL = RUN / "LABEL_FREE_RESULT.json"
VERIFY = RUN / "VERIFICATION.json"


def digest(path: Path) -> str:
    return parent.digest(path)


def load(path: Path) -> dict[str, Any]:
    return parent.load(path)


def read_catalog() -> dict[str, Any]:
    return parent.read_catalog()


def validate_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    data = catalog.get("data") or {}
    if data.get("id") != MODEL_ID:
        raise RuntimeError("exact DeepSeek route unavailable")
    eligible = []
    for row in data.get("endpoints", []):
        if row.get("status") != 0:
            continue
        supported = set(row.get("supported_parameters") or [])
        if (
            "seed" not in supported
            or "reasoning" not in supported
            or not ({"response_format", "structured_outputs"} & supported)
        ):
            continue
        try:
            prompt = float(row["pricing"]["prompt"])
            completion = float(row["pricing"]["completion"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("eligible DeepSeek price malformed") from error
        if not all(math.isfinite(value) and value >= 0 for value in (prompt, completion)):
            raise RuntimeError("eligible DeepSeek price malformed")
        eligible.append((str(row.get("provider_name")), prompt, completion))
    if not eligible:
        raise RuntimeError("no eligible DeepSeek endpoint")
    prompt = max(row[1] for row in eligible)
    completion = max(row[2] for row in eligible)
    if (
        prompt > PRICE_CEILING["prompt"] + 1e-15
        or completion > PRICE_CEILING["completion"] + 1e-15
    ):
        raise RuntimeError("DeepSeek eligible-provider price increased")
    return {
        "id": MODEL_ID,
        "provider": "parameter_constrained_router",
        "eligible_providers": sorted(row[0] for row in eligible),
        "prompt_price_usd_per_token": prompt,
        "completion_price_usd_per_token": completion,
    }


class RouterAdapter(SeededAdapter):
    def complete(self, messages, seeds, *, response_format, max_tokens):
        return parent.DeepSeekAdapter.complete(
            self,
            messages,
            seeds,
            response_format=response_format,
            max_tokens=max_tokens,
        )


def build_adapter(*, request_cap: float, authorize) -> RouterAdapter:
    config = Config(
        task="animals",
        run_id="number-game-deepseek-stratified-semantic-v5-router-20260815",
        log_path=RUN / "run.log",
        openrouter_budget_usd=245.0,
        openrouter_run_budget_usd=STAGE_CAP_USD,
        openrouter_projected_cost_usd=STAGE_CAP_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=request_cap,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return RouterAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65536),
        config,
        authorize=authorize,
    )


def validate_bindings() -> dict[str, str]:
    binding = load(BINDING)
    expected = {
        "v5_protocol": REPO_ROOT / "results/nonmyopic/NUMBER_GAME_DEEPSEEK_STRATIFIED_SEMANTIC_V5_ROUTER_PROTOCOL_20260815.md",
        "v4_protocol": gate.PROTOCOL,
        "v4_terminal": REPO_ROOT / "results/nonmyopic/NUMBER_GAME_DEEPSEEK_STRATIFIED_SEMANTIC_V4_TERMINAL_RESULT_20260815.md",
        "codec": REPO_ROOT / "scripts/number_game_stratified_atomic_particle_v3_codec.py",
        "producer": Path(gate.__file__).resolve(),
        "verifier": Path(verifier.__file__).resolve(),
        "transport_parent": Path(parent.__file__).resolve(),
        "wrapper": Path(__file__).resolve(),
        "tests": REPO_ROOT / "tests/test_number_game_deepseek_stratified_semantic_v5_router.py",
    }
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)) or row.get("sha256") != digest(path):
            raise RuntimeError(f"execution binding changed: {name}")
    required = {
        "date": DATE,
        "opening_total_usage_usd": OPENING_USAGE_USD,
        "daily_cap_usd": DAILY_CAP_USD,
        "stage_cap_usd": STAGE_CAP_USD,
        "model": MODEL_ID,
        "provider": "parameter_constrained_router",
        "requests": len(gate.MODEL_SEEDS),
        "maximum_http_attempts": len(gate.MODEL_SEEDS),
        "maximum_retries": 0,
        "block_size": gate.GROUP_SIZE,
        "concurrency": CONCURRENCY,
        "canonical_targets_authorized": False,
        "policy_endpoints_authorized": False,
    }
    if any(binding.get(key) != value for key, value in required.items()):
        raise RuntimeError("execution binding metadata changed")
    return {"execution_binding_sha256": digest(BINDING)}


@contextmanager
def configured_parent():
    replacements = {
        "INTERFACE_VERSION": INTERFACE_VERSION,
        "DATE": DATE,
        "OPENING_USAGE_USD": OPENING_USAGE_USD,
        "DAILY_CAP_USD": DAILY_CAP_USD,
        "STAGE_CAP_USD": STAGE_CAP_USD,
        "MODEL_ID": MODEL_ID,
        "MAX_TOKENS": MAX_TOKENS,
        "CONCURRENCY": CONCURRENCY,
        "ENDPOINTS_URL": ENDPOINTS_URL,
        "PRICE_CEILING": PRICE_CEILING,
        "ROOT": ROOT,
        "RUN": RUN,
        "BINDING": BINDING,
        "RESULT": RESULT,
        "FAILURE": FAILURE,
        "LEDGER": LEDGER,
        "RAW": RAW,
        "LABEL": LABEL,
        "VERIFY": VERIFY,
        "read_catalog": read_catalog,
        "validate_catalog": validate_catalog,
        "validate_bindings": validate_bindings,
        "build_adapter": build_adapter,
    }
    original = {name: getattr(parent, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(parent, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(parent, name, value)


def preflight(*, now=None, live_reader=parent.read_live_credits, catalog_reader=read_catalog):
    with configured_parent():
        return parent.preflight(
            now=now,
            live_reader=live_reader,
            catalog_reader=catalog_reader,
        )


def execute(*, now=None, live_reader=parent.read_live_credits, catalog_reader=read_catalog):
    with configured_parent():
        return parent.execute(
            now=now,
            live_reader=live_reader,
            catalog_reader=catalog_reader,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")
    value = preflight() if args.preflight else execute()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
