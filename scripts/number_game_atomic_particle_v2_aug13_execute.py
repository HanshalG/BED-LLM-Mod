#!/usr/bin/env python3
"""Execute the frozen Aug 13 atomic-particle Number Game mechanics once."""

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
from scripts.number_game_atomic_particle_v2_codec import DIVERSITY_CUES, MODEL_SEEDS, particle_messages
from scripts import number_game_atomic_particle_v2_mechanics as mechanics
from scripts import number_game_atomic_particle_v2_verify as verifier
from scripts.number_game_bitmask_semantic_gate import SeededAdapter
from scripts.openrouter_daily_budget import read_live_credits


INTERFACE_VERSION = "number-game-atomic-particle-v2-aug13-execute-1"
DATE = "2026-08-13"
TIMEZONE = "Europe/London"
OPENING_USAGE_USD = 220.134128880
DAILY_CAP_USD = 5.0
STAGE_CAP_USD = mechanics.STAGE_CAP_USD
MODEL_ID = mechanics.MODEL_ID
MAX_TOKENS = mechanics.MAX_TOKENS
TEMPERATURE = mechanics.TEMPERATURE
CONCURRENCY = 128
MODELS_URL = "https://openrouter.ai/api/v1/models"
PRICE_CEILING = {"prompt": .32e-6, "completion": 1.28e-6}
ROOT = REPO_ROOT / "results/nonmyopic/number_game_atomic_particle_v2_mechanics"
RUN = ROOT / "mechanics-20260813"
BINDING = ROOT / "EXECUTION_BINDING.json"
RESULT = ROOT / "DAILY_RESULT_20260813.json"
FAILURE = ROOT / "DAILY_FAILURE_20260813.json"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-number-game-atomic-particle-v2.json"
RAW = RUN / "private/RAW_RESPONSES.json"
TOPOLOGY = RUN / "private/TOPOLOGY.json"
VERIFY = RUN / "LABEL_FREE_VERIFICATION.json"
SCIENCE = RUN / "SCIENTIFIC_RESULT.json"


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
    request = Request(MODELS_URL, headers={"Authorization": f"Bearer {key}"})
    with urlopen(request, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("OpenRouter catalog malformed")
    return value


def validate_live(live: Mapping[str, Any]) -> dict[str, float]:
    try:
        credits, usage, balance = (float(live[key]) for key in ("total_credits_usd", "total_usage_usd", "balance_usd"))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("account values malformed") from error
    if (
        not all(math.isfinite(value) and value >= 0 for value in (credits, usage, balance))
        or abs(credits - usage - balance) > 1e-6
        or usage + 1e-12 < OPENING_USAGE_USD
    ):
        raise RuntimeError("account values invalid")
    return {"total_credits_usd": credits, "total_usage_usd": usage, "balance_usd": balance}


def prior_spend(live: Mapping[str, Any]) -> float:
    return validate_live(live)["total_usage_usd"] - OPENING_USAGE_USD


def validate_catalog(catalog: Mapping[str, Any]) -> dict[str, float | str]:
    rows = [row for row in catalog.get("data", []) if row.get("id") == MODEL_ID]
    if len(rows) != 1:
        raise RuntimeError("exact Qwen route unavailable")
    row = rows[0]
    architecture = row.get("architecture") or {}
    supported = set(row.get("supported_parameters") or [])
    try:
        prompt = float(row["pricing"]["prompt"])
        completion = float(row["pricing"]["completion"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("catalog price malformed") from error
    if (
        "text" not in set(architecture.get("input_modalities") or [])
        or "text" not in set(architecture.get("output_modalities") or [])
        or "seed" not in supported
        or not ({"response_format", "structured_outputs"} & supported)
        or not all(math.isfinite(value) and value >= 0 for value in (prompt, completion))
        or prompt > PRICE_CEILING["prompt"] + 1e-15
        or completion > PRICE_CEILING["completion"] + 1e-15
    ):
        raise RuntimeError("Qwen route capability or price changed")
    return {"id": MODEL_ID, "prompt_price_usd_per_token": prompt, "completion_price_usd_per_token": completion}


def exposure(messages: Sequence[Sequence[Mapping[str, str]]], prices: Mapping[str, Any]) -> dict[str, Any]:
    prompt_price = float(prices["prompt_price_usd_per_token"])
    completion_price = float(prices["completion_price_usd_per_token"])
    values = [len(canonical_json(message).encode()) * prompt_price + MAX_TOKENS * completion_price for message in messages]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError("block exposure malformed")
    return {
        "request_count": len(values),
        "exact_block_exposure_usd": sum(values),
        "maximum_request_exposure_usd": max(values),
        "request_exposures_usd": values,
    }


def validate_bindings() -> dict[str, str]:
    binding = load(BINDING)
    expected = {
        "protocol": mechanics.PROTOCOL,
        "parent_protocol": mechanics.PARENT_PROTOCOL,
        "half_bank_clarification": REPO_ROOT / "results/nonmyopic/NUMBER_GAME_ATOMIC_PARTICLE_HALF_BANK_CLARIFICATION_20260813.md",
        "collapse_clarification": REPO_ROOT / "results/nonmyopic/NUMBER_GAME_ATOMIC_PARTICLE_CANONICAL_COLLAPSE_CLARIFICATION_20260813.md",
        "codec": REPO_ROOT / "scripts/number_game_atomic_particle_v2_codec.py",
        "producer": Path(mechanics.__file__).resolve(),
        "verifier": Path(verifier.__file__).resolve(),
        "wrapper": Path(__file__).resolve(),
        "codec_tests": REPO_ROOT / "tests/test_number_game_atomic_particle_v2.py",
        "transaction_tests": REPO_ROOT / "tests/test_number_game_atomic_particle_v2_mechanics.py",
        "wrapper_tests": REPO_ROOT / "tests/test_number_game_atomic_particle_v2_execute.py",
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
        "requests": len(MODEL_SEEDS),
        "maximum_http_attempts": len(MODEL_SEEDS),
        "maximum_retries": 0,
        "block_size": mechanics.BLOCK_SIZE,
        "concurrency": CONCURRENCY,
        "canonical_targets_authorized_before_complete_bank": False,
        "classical_grammar_authorized_before_complete_bank": False,
    }
    if any(binding.get(key) != value for key, value in required.items()):
        raise RuntimeError("execution binding metadata changed")
    return {"execution_binding_sha256": digest(BINDING)}


def pristine(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def initial_block() -> list[list[dict[str, str]]]:
    return [particle_messages((), slot=index) for index in range(mechanics.BLOCK_SIZE)]


def maximum_request_exposure(prices: Mapping[str, Any]) -> float:
    prompts = [
        particle_messages(((100, False), (100, False)), slot=slot)
        for slot in range(len(DIVERSITY_CUES))
    ]
    return float(exposure(prompts, prices)["maximum_request_exposure_usd"])


def preflight(*, now: datetime | None = None, live_reader=read_live_credits, catalog_reader=read_catalog) -> dict[str, Any]:
    local = now.astimezone(ZoneInfo(TIMEZONE)) if now else datetime.now(ZoneInfo(TIMEZONE))
    if local.date().isoformat() != DATE:
        raise RuntimeError("wrong execution date")
    bindings = validate_bindings()
    if RESULT.exists() or FAILURE.exists() or not pristine(RUN) or not pristine(LEDGER):
        raise RuntimeError("atomic particle execution path is not pristine")
    prices = validate_catalog(catalog_reader())
    block = exposure(initial_block(), prices)
    live = validate_live(live_reader())
    if block["exact_block_exposure_usd"] > STAGE_CAP_USD + 1e-12:
        raise RuntimeError("initial block exceeds stage cap")
    if prior_spend(live) + block["exact_block_exposure_usd"] > DAILY_CAP_USD + 1e-12 or live["balance_usd"] + 1e-12 < block["exact_block_exposure_usd"]:
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


class AtomicAdapter(SeededAdapter):
    def complete(self, messages, seeds, *, response_format, max_tokens):
        if len(messages) != len(seeds):
            raise ValueError("message and seed counts differ")

        def one(item):
            prompt, seed = item
            self._seed.value = int(seed)
            try:
                return self._complete_request(
                    prompt,
                    TEMPERATURE,
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


def build_adapter(*, request_cap: float, authorize) -> AtomicAdapter:
    config = Config(
        task="animals",
        run_id="number-game-atomic-particle-v2-20260813",
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
    return AtomicAdapter(ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65536), config, authorize=authorize)


def adapter_cost(adapter: AtomicAdapter | None) -> float:
    return 0.0 if adapter is None else float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))


def reconcile(ledger: Mapping[str, Any], status: str, local_cost: float, live: Mapping[str, Any] | None) -> dict[str, Any]:
    out = json.loads(json.dumps(ledger))
    execution_prior = float(out["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD
    recorded = max(float(out["recorded_actual_spend_usd"]), prior_spend(live) if live else 0.0, execution_prior + local_cost)
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


def execute(*, live_reader=read_live_credits, catalog_reader=read_catalog) -> dict[str, Any]:
    ready = preflight(live_reader=live_reader, catalog_reader=catalog_reader)
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
        "stage": {"status": "authorized_pending", "maximum_cost_usd": STAGE_CAP_USD, "maximum_http_attempts": len(MODEL_SEEDS), "maximum_retries": 0},
    }
    checkpoint(LEDGER, ledger)
    adapter: AtomicAdapter | None = None
    active_block = {"reserve": 0.0, "accepted_before": 0.0}
    active_lock = threading.Lock()
    try:
        request_cap = maximum_request_exposure(ready["initial_block_authorization"]["model"])

        def request_authorizer() -> None:
            current = validate_live(live_reader())
            with active_lock:
                reserve = float(active_block["reserve"])
                accepted_before = float(active_block["accepted_before"])
            accepted = adapter_cost(adapter)
            remaining = max(0.0, reserve - max(0.0, accepted - accepted_before))
            execution_prior = float(ledger["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD
            reconciled = max(prior_spend(current), execution_prior + accepted)
            if accepted + remaining > STAGE_CAP_USD + 1e-12 or reconciled + remaining > DAILY_CAP_USD + 1e-12 or current["balance_usd"] + 1e-12 < remaining:
                raise RuntimeError("per-request account or stage allowance lost")

        adapter = build_adapter(request_cap=request_cap, authorize=request_authorizer)

        def block_authorizer(requests: Sequence[dict[str, Any]]) -> None:
            nonlocal ledger
            messages = [row["messages"] for row in requests]
            prices = validate_catalog(catalog_reader())
            block = exposure(messages, prices)
            current = validate_live(live_reader())
            accepted = adapter_cost(adapter)
            execution_prior = float(ledger["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD
            reconciled = max(prior_spend(current), execution_prior + accepted)
            reserve = float(block["exact_block_exposure_usd"])
            if accepted + reserve > STAGE_CAP_USD + 1e-12 or reconciled + reserve > DAILY_CAP_USD + 1e-12 or current["balance_usd"] + 1e-12 < reserve:
                raise RuntimeError("atomic block allowance unavailable")
            row = {
                "block_index": len(ledger["block_authorizations"]),
                "first_seed": requests[0]["seed"],
                "last_seed": requests[-1]["seed"],
                "model": prices,
                "exposure": block,
                "live": current,
                "accepted_cost_before_block_usd": accepted,
            }
            ledger["block_authorizations"].append(row)
            checkpoint(LEDGER, ledger)
            with active_lock:
                active_block["reserve"] = reserve
                active_block["accepted_before"] = accepted

        mechanics.produce_bank(output_dir=RUN, adapter=adapter, block_authorizer=block_authorizer)
        verifier.verify(RUN, output=VERIFY)
        replayed_banks = verifier.replay(load(RAW))
        science = mechanics.score_complete_bank({"banks": replayed_banks})
        science.update({
            "protocol_sha256": digest(mechanics.PROTOCOL),
            "half_bank_clarification_sha256": digest(REPO_ROOT / "results/nonmyopic/NUMBER_GAME_ATOMIC_PARTICLE_HALF_BANK_CLARIFICATION_20260813.md"),
            "collapse_clarification_sha256": digest(REPO_ROOT / "results/nonmyopic/NUMBER_GAME_ATOMIC_PARTICLE_CANONICAL_COLLAPSE_CLARIFICATION_20260813.md"),
            "raw_response_sha256": digest(RAW),
            "topology_sha256": digest(TOPOLOGY),
            "label_free_verification_sha256": digest(VERIFY),
        })
        checkpoint(SCIENCE, science)
        try:
            closing = validate_live(live_reader())
        except Exception:
            closing = None
        actual = adapter_cost(adapter)
        ledger = reconcile(ledger, science["status"], actual, closing)
        checkpoint(LEDGER, ledger)
        terminal = {
            "schema_version": 1,
            "interface_version": INTERFACE_VERSION,
            "status": science["status"],
            "authorizes": science["authorizes"],
            "scientific_result_sha256": digest(SCIENCE),
            "raw_sha256": digest(RAW),
            "topology_sha256": digest(TOPOLOGY),
            "verification_sha256": digest(VERIFY),
            "ledger_sha256": digest(LEDGER),
            "actual_cost_usd": actual,
            "canonical_targets_opened": True,
            "classical_grammar_opened": True,
            "development_opened": False,
            "confirmation_opened": False,
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
            "topology_exists": TOPOLOGY.exists(),
            "verification_exists": VERIFY.exists(),
            "scientific_result_exists": SCIENCE.exists(),
            "canonical_targets_opened": SCIENCE.exists(),
            "classical_grammar_opened": SCIENCE.exists(),
            "development_opened": False,
            "confirmation_opened": False,
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
