#!/usr/bin/env python3
"""Execute the frozen Aug 15 DeepSeek stratified semantic gate once."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_bitmask_semantic_gate import SeededAdapter
from scripts.openrouter_daily_budget import read_live_credits
from scripts import number_game_deepseek_stratified_semantic_v4 as gate
from scripts import number_game_deepseek_stratified_semantic_v4_verify as verifier


INTERFACE_VERSION = "number-game-deepseek-stratified-semantic-v4-aug15-execute-1"
DATE = "2026-08-15"
TIMEZONE = "Europe/London"
OPENING_USAGE_USD = 220.339269126
DAILY_CAP_USD = 5.0
STAGE_CAP_USD = gate.STAGE_CAP_USD
MODEL_ID = gate.MODEL_ID
PROVIDER = "GMICloud"
MAX_TOKENS = gate.MAX_TOKENS
CONCURRENCY = 64
ENDPOINTS_URL = f"https://openrouter.ai/api/v1/models/{MODEL_ID}/endpoints"
PRICE_CEILING = {"prompt": 0.07e-6, "completion": 0.14e-6}
ROOT = REPO_ROOT / "results/nonmyopic/number_game_deepseek_stratified_semantic_v4"
RUN = ROOT / "semantic-20260815"
BINDING = ROOT / "EXECUTION_BINDING.json"
RESULT = ROOT / "DAILY_RESULT_20260815.json"
FAILURE = ROOT / "DAILY_FAILURE_20260815.json"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-15-number-game-deepseek-stratified-semantic-v4.json"
RAW = RUN / "private/RAW_RESPONSES.json"
LABEL = RUN / "LABEL_FREE_RESULT.json"
VERIFY = RUN / "VERIFICATION.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("expected JSON object")
    return value


def read_catalog() -> dict[str, Any]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    request = Request(ENDPOINTS_URL, headers={"Authorization": f"Bearer {key}"})
    with urlopen(request, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("OpenRouter endpoint catalog malformed")
    return value


def validate_live(live: Mapping[str, Any]) -> dict[str, float]:
    try:
        credits, usage, balance = (
            float(live[key])
            for key in ("total_credits_usd", "total_usage_usd", "balance_usd")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("account values malformed") from error
    if (
        not all(math.isfinite(value) and value >= 0 for value in (credits, usage, balance))
        or abs(credits - usage - balance) > 1e-6
        or usage + 1e-12 < OPENING_USAGE_USD
    ):
        raise RuntimeError("account values invalid")
    return {
        "total_credits_usd": credits,
        "total_usage_usd": usage,
        "balance_usd": balance,
    }


def prior_spend(live: Mapping[str, Any]) -> float:
    return validate_live(live)["total_usage_usd"] - OPENING_USAGE_USD


def validate_catalog(catalog: Mapping[str, Any]) -> dict[str, float | str]:
    data = catalog.get("data") or {}
    if data.get("id") != MODEL_ID:
        raise RuntimeError("exact DeepSeek route unavailable")
    rows = [
        row
        for row in data.get("endpoints", [])
        if row.get("provider_name") == PROVIDER and row.get("status") == 0
    ]
    if len(rows) != 1:
        raise RuntimeError("pinned DeepSeek provider unavailable")
    row = rows[0]
    supported = set(row.get("supported_parameters") or [])
    try:
        prompt = float(row["pricing"]["prompt"])
        completion = float(row["pricing"]["completion"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("DeepSeek price malformed") from error
    if (
        "seed" not in supported
        or "reasoning" not in supported
        or not ({"response_format", "structured_outputs"} & supported)
        or not all(math.isfinite(value) and value >= 0 for value in (prompt, completion))
        or prompt > PRICE_CEILING["prompt"] + 1e-15
        or completion > PRICE_CEILING["completion"] + 1e-15
    ):
        raise RuntimeError("DeepSeek capability or price changed")
    return {
        "id": MODEL_ID,
        "provider": PROVIDER,
        "prompt_price_usd_per_token": prompt,
        "completion_price_usd_per_token": completion,
    }


def exposure(messages: Sequence[Sequence[Mapping[str, str]]], prices: Mapping[str, Any]) -> dict[str, Any]:
    prompt_price = float(prices["prompt_price_usd_per_token"])
    completion_price = float(prices["completion_price_usd_per_token"])
    values = [
        len(canonical_json(message).encode()) * prompt_price + MAX_TOKENS * completion_price
        for message in messages
    ]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError("semantic block exposure malformed")
    return {
        "request_count": len(values),
        "exact_block_exposure_usd": sum(values),
        "maximum_request_exposure_usd": max(values),
        "request_exposures_usd": values,
    }


def validate_bindings() -> dict[str, str]:
    binding = load(BINDING)
    expected = {
        "protocol": gate.PROTOCOL,
        "v3_terminal": REPO_ROOT / "results/nonmyopic/NUMBER_GAME_STRATIFIED_ATOMIC_PARTICLE_V3_TERMINAL_RESULT_20260815.md",
        "codec": REPO_ROOT / "scripts/number_game_stratified_atomic_particle_v3_codec.py",
        "producer": Path(gate.__file__).resolve(),
        "verifier": Path(verifier.__file__).resolve(),
        "wrapper": Path(__file__).resolve(),
        "tests": REPO_ROOT / "tests/test_number_game_deepseek_stratified_semantic_v4.py",
        "synthetic_test_helper": REPO_ROOT / "tests/test_number_game_atomic_particle_mechanics.py",
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
        "provider": PROVIDER,
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


def pristine(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def initial_block() -> list[list[dict[str, str]]]:
    return [row["messages"] for row in gate.group_requests(0)]


def maximum_request_exposure(prices: Mapping[str, Any]) -> float:
    prompts = [
        row["messages"]
        for group_index in range(len(gate.GROUPS))
        for row in gate.group_requests(group_index)
    ]
    return float(exposure(prompts, prices)["maximum_request_exposure_usd"])


def preflight(*, now: datetime | None = None, live_reader=read_live_credits, catalog_reader=read_catalog) -> dict[str, Any]:
    local = now.astimezone(ZoneInfo(TIMEZONE)) if now else datetime.now(ZoneInfo(TIMEZONE))
    if local.date().isoformat() != DATE:
        raise RuntimeError("wrong execution date")
    bindings = validate_bindings()
    if RESULT.exists() or FAILURE.exists() or not pristine(RUN) or not pristine(LEDGER):
        raise RuntimeError("DeepSeek semantic execution path is not pristine")
    prices = validate_catalog(catalog_reader())
    block = exposure(initial_block(), prices)
    live = validate_live(live_reader())
    if block["exact_block_exposure_usd"] > STAGE_CAP_USD + 1e-12:
        raise RuntimeError("initial semantic block exceeds stage cap")
    if (
        prior_spend(live) + block["exact_block_exposure_usd"] > DAILY_CAP_USD + 1e-12
        or live["balance_usd"] + 1e-12 < block["exact_block_exposure_usd"]
    ):
        raise RuntimeError("account allowance unavailable")
    return {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "bindings": bindings,
        "initial_block_authorization": {"model": prices, "exposure": block, "live": live},
        "budget": {
            "opening_total_usage_usd": OPENING_USAGE_USD,
            "prior_account_spend_usd": prior_spend(live),
            "daily_cap_usd": DAILY_CAP_USD,
            "stage_cap_usd": STAGE_CAP_USD,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


class DeepSeekAdapter(SeededAdapter):
    def _payload(self, messages, temperature, n, max_tokens=None, *, disable_reasoning=False, response_format=None):
        payload = super()._payload(
            messages,
            temperature,
            n,
            max_tokens,
            disable_reasoning=True,
            response_format=response_format,
        )
        payload["provider"] = {
            "require_parameters": True,
            "order": [PROVIDER],
            "allow_fallbacks": False,
        }
        return payload

    def complete(self, messages, seeds, *, response_format, max_tokens):
        if len(messages) != len(seeds):
            raise ValueError("message and seed counts differ")

        def one(item):
            prompt, seed = item
            self._seed.value = int(seed)
            try:
                return self._complete_request(
                    prompt,
                    gate.TEMPERATURE,
                    1,
                    max_tokens,
                    allow_forced_final=False,
                    disable_reasoning=True,
                    response_format=response_format,
                )[0]
            finally:
                del self._seed.value

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            return list(pool.map(one, zip(messages, seeds, strict=True)))


def build_adapter(*, request_cap: float, authorize) -> DeepSeekAdapter:
    config = Config(
        task="animals",
        run_id="number-game-deepseek-stratified-semantic-v4-20260815",
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
    return DeepSeekAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65536),
        config,
        authorize=authorize,
    )


def adapter_cost(adapter: DeepSeekAdapter | None) -> float:
    return 0.0 if adapter is None else float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))


def reconcile(ledger: Mapping[str, Any], status: str, local_cost: float, live: Mapping[str, Any] | None) -> dict[str, Any]:
    out = json.loads(json.dumps(ledger))
    execution_prior = float(out["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD
    recorded = max(
        float(out["recorded_actual_spend_usd"]),
        prior_spend(live) if live else 0.0,
        execution_prior + local_cost,
    )
    if recorded > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("daily cap exceeded")
    out["recorded_actual_spend_usd"] = recorded
    out["stage"].update({"status": status, "actual_cost_usd": local_cost})
    if live:
        out.update({
            "closing_total_credits_usd": live["total_credits_usd"],
            "closing_total_usage_usd": live["total_usage_usd"],
            "closing_balance_usd": live["balance_usd"],
        })
    return out


def execute(*, now: datetime | None = None, live_reader=read_live_credits, catalog_reader=read_catalog) -> dict[str, Any]:
    ready = preflight(now=now, live_reader=live_reader, catalog_reader=catalog_reader)
    live = validate_live(live_reader())
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "date": DATE,
        "timezone": TIMEZONE,
        "opening_total_usage_usd": OPENING_USAGE_USD,
        "execution_opening_total_credits_usd": live["total_credits_usd"],
        "execution_opening_total_usage_usd": live["total_usage_usd"],
        "execution_opening_balance_usd": live["balance_usd"],
        "recorded_actual_spend_usd": prior_spend(live),
        "daily_cap_usd": DAILY_CAP_USD,
        "execution_binding_sha256": ready["bindings"]["execution_binding_sha256"],
        "block_authorizations": [],
        "stage": {
            "status": "authorized_pending",
            "maximum_cost_usd": STAGE_CAP_USD,
            "maximum_http_attempts": len(gate.MODEL_SEEDS),
            "maximum_retries": 0,
        },
    }
    checkpoint(LEDGER, ledger)
    adapter: DeepSeekAdapter | None = None
    active_block = {"reserve": 0.0, "accepted_before": 0.0}
    active_lock = threading.Lock()
    try:
        prices = ready["initial_block_authorization"]["model"]
        request_cap = maximum_request_exposure(prices)

        def request_authorizer() -> None:
            current = validate_live(live_reader())
            with active_lock:
                reserve = float(active_block["reserve"])
                accepted_before = float(active_block["accepted_before"])
            accepted = adapter_cost(adapter)
            remaining = max(0.0, reserve - max(0.0, accepted - accepted_before))
            execution_prior = float(ledger["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD
            reconciled = max(prior_spend(current), execution_prior + accepted)
            if (
                accepted + remaining > STAGE_CAP_USD + 1e-12
                or reconciled + remaining > DAILY_CAP_USD + 1e-12
                or current["balance_usd"] + 1e-12 < remaining
            ):
                raise RuntimeError("per-request semantic allowance lost")

        adapter = build_adapter(request_cap=request_cap, authorize=request_authorizer)

        def block_authorizer(requests: Sequence[dict[str, Any]]) -> None:
            nonlocal ledger
            prices_now = validate_catalog(catalog_reader())
            block = exposure([row["messages"] for row in requests], prices_now)
            current = validate_live(live_reader())
            accepted = adapter_cost(adapter)
            execution_prior = float(ledger["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD
            reconciled = max(prior_spend(current), execution_prior + accepted)
            reserve = float(block["exact_block_exposure_usd"])
            if (
                accepted + reserve > STAGE_CAP_USD + 1e-12
                or reconciled + reserve > DAILY_CAP_USD + 1e-12
                or current["balance_usd"] + 1e-12 < reserve
            ):
                raise RuntimeError("semantic block allowance unavailable")
            ledger["block_authorizations"].append({
                "block_index": len(ledger["block_authorizations"]),
                "first_seed": requests[0]["seed"],
                "last_seed": requests[-1]["seed"],
                "model": prices_now,
                "exposure": block,
                "live": current,
                "accepted_cost_before_block_usd": accepted,
            })
            checkpoint(LEDGER, ledger)
            with active_lock:
                active_block["reserve"] = reserve
                active_block["accepted_before"] = accepted

        label = gate.produce(output_dir=RUN, adapter=adapter, block_authorizer=block_authorizer)
        verifier.verify(RUN, output=VERIFY)
        try:
            closing = validate_live(live_reader())
        except Exception:
            closing = None
        actual = adapter_cost(adapter)
        ledger = reconcile(ledger, label["status"], actual, closing)
        checkpoint(LEDGER, ledger)
        terminal = {
            "schema_version": 1,
            "interface_version": INTERFACE_VERSION,
            "status": label["status"],
            "authorizes": label["authorizes"],
            "label_free_result_sha256": digest(LABEL),
            "raw_sha256": digest(RAW),
            "verification_sha256": digest(VERIFY),
            "ledger_sha256": digest(LEDGER),
            "actual_cost_usd": actual,
            "canonical_targets_opened": False,
            "policy_endpoints_opened": False,
            "development_confirmation_opened": False,
        }
        checkpoint(RESULT, terminal)
        return terminal
    except Exception as error:
        try:
            closing = validate_live(live_reader())
        except Exception:
            closing = None
        actual = adapter_cost(adapter)
        ledger = reconcile(ledger, "failed_closed", actual, closing)
        checkpoint(LEDGER, ledger)
        failure = {
            "schema_version": 1,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "authorizes": "nothing",
            "error_type": type(error).__name__,
            "error": str(error),
            "actual_cost_usd": actual,
            "ledger_sha256": digest(LEDGER),
            "raw_exists": RAW.exists(),
            "label_exists": LABEL.exists(),
            "verification_exists": VERIFY.exists(),
            "canonical_targets_opened": False,
            "policy_endpoints_opened": False,
            "development_confirmation_opened": False,
        }
        checkpoint(FAILURE, failure)
        raise


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
