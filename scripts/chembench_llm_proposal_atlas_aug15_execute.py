#!/usr/bin/env python3
"""Execute the frozen Aug 15 ChemBench proposal-atlas semantic gate once."""

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
from scripts import chembench_llm_proposal_atlas as gate
from scripts import chembench_llm_proposal_atlas_verify as verifier


INTERFACE_VERSION = "chembench-llm-proposal-atlas-aug15-execute-1"
DATE = "2026-08-15"
TIMEZONE = "Europe/London"
OPENING_USAGE_USD = 220.339269126
DAILY_CAP_USD = 5.0
STAGE_CAP_USD = gate.STAGE_CAP_USD
MODEL_ID = gate.MODEL_ID
MAX_TOKENS = gate.MAX_COMPLETION_TOKENS
CONCURRENCY = gate.MAX_CONCURRENCY
ENDPOINTS_URL = f"https://openrouter.ai/api/v1/models/{MODEL_ID}/endpoints"
PRICE_CEILING = {"prompt": 0.20e-6, "completion": 0.50e-6}
ROOT = REPO_ROOT / "results/nonmyopic/chembench_llm_proposal_atlas"
RUN = ROOT / "proposal-atlas-v1-20260815"
MANIFEST = RUN / "PUBLIC_MANIFEST.json"
LABELS = RUN / "SEALED_LABELS.json"
SOURCE_SUMMARY = RUN / "SOURCE_SUMMARY.json"
SOURCE_VERIFICATION = RUN / "SOURCE_VERIFICATION.json"
BINDING = RUN / "EXECUTION_BINDING.json"
RAW = RUN / "private/RAW_RESPONSE_BANK.json"
EVALUATION = RUN / "SEMANTIC_EVALUATION.json"
SEMANTIC_VERIFICATION = RUN / "SEMANTIC_VERIFICATION.json"
RESULT = RUN / "DAILY_RESULT_20260815.json"
FAILURE = RUN / "DAILY_FAILURE_20260815.json"
LEDGER = (
    REPO_ROOT
    / "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-15-chembench-llm-proposal-atlas.json"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
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
        if (
            prompt > PRICE_CEILING["prompt"] + 1e-15
            or completion > PRICE_CEILING["completion"] + 1e-15
        ):
            raise RuntimeError("eligible DeepSeek provider exceeds the frozen price ceiling")
        eligible.append((str(row.get("provider_name")), prompt, completion))
    if not eligible:
        raise RuntimeError("no eligible DeepSeek endpoint")
    return {
        "id": MODEL_ID,
        "provider": "parameter_constrained_router",
        "eligible_providers": sorted(item[0] for item in eligible),
        "prompt_price_usd_per_token": max(item[1] for item in eligible),
        "completion_price_usd_per_token": max(item[2] for item in eligible),
    }


def request_messages(manifest: Mapping[str, Any], request: Mapping[str, Any]) -> list[dict[str, str]]:
    task = next(item for item in manifest["tasks"] if item["task_id"] == request["task_id"])
    prompt = task["prompts"][request["arm"]]
    if gate.payload_hash(prompt) != request["prompt_sha256"]:
        raise RuntimeError("request prompt no longer matches manifest")
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]


def request_exposure(
    manifest: Mapping[str, Any],
    prices: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_price = float(prices["prompt_price_usd_per_token"])
    completion_price = float(prices["completion_price_usd_per_token"])
    values = []
    for request in manifest["requests"]:
        messages = request_messages(manifest, request)
        values.append(
            len(canonical_json(messages).encode()) * prompt_price
            + MAX_TOKENS * completion_price
        )
    if len(values) != 126 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError("proposal-atlas exposure malformed")
    return {
        "request_count": len(values),
        "exact_stage_exposure_usd": sum(values),
        "maximum_request_exposure_usd": max(values),
        "request_exposures_usd": values,
    }


def validate_bindings() -> dict[str, str]:
    binding = load(BINDING)
    expected = {
        "protocol": REPO_ROOT
        / "results/nonmyopic/CHEMBENCH_LLM_PROPOSAL_ATLAS_PROTOCOL_20260815.md",
        "selection_clarification": REPO_ROOT
        / "results/nonmyopic/"
        "CHEMBENCH_LLM_PROPOSAL_ATLAS_SOURCE_SELECTION_CLARIFICATION_20260815.md",
        "factored_module": REPO_ROOT / "environments/chembench_mopen/factored.py",
        "atlas_module": REPO_ROOT / "environments/chembench_mopen/proposal_atlas.py",
        "producer": Path(gate.__file__).resolve(),
        "verifier": Path(verifier.__file__).resolve(),
        "wrapper": Path(__file__).resolve(),
        "tests": REPO_ROOT / "tests/test_chembench_proposal_atlas.py",
        "manifest": MANIFEST,
        "labels": LABELS,
        "source_summary": SOURCE_SUMMARY,
        "source_verification": SOURCE_VERIFICATION,
    }
    for name, path in expected.items():
        row = binding.get(name) or {}
        try:
            relative = str(path.relative_to(REPO_ROOT))
        except ValueError:
            relative = str(path)
        if row.get("path") != relative or row.get("sha256") != digest(path):
            raise RuntimeError(f"execution binding changed: {name}")
    required = {
        "date": DATE,
        "opening_total_usage_usd": OPENING_USAGE_USD,
        "daily_cap_usd": DAILY_CAP_USD,
        "stage_cap_usd": STAGE_CAP_USD,
        "model": MODEL_ID,
        "provider": "parameter_constrained_router",
        "requests": 126,
        "maximum_http_attempts": 126,
        "maximum_retries": 0,
        "concurrency": CONCURRENCY,
        "reasoning": "disabled",
        "policy_endpoints_authorized": False,
    }
    if any(binding.get(key) != value for key, value in required.items()):
        raise RuntimeError("execution binding metadata changed")
    implementation_commit = str(load(MANIFEST)["implementation_commit"])
    if binding.get("required_implementation_commit") != implementation_commit:
        raise RuntimeError("execution binding implementation commit mismatch")
    head = gate.git_value(("rev-parse", "HEAD"))
    subprocess_result = gate.git_value(("merge-base", implementation_commit, head))
    if subprocess_result != implementation_commit:
        raise RuntimeError("implementation commit is not an ancestor of HEAD")
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            head,
            "origin/codex/location-finding-llmstrategy",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"execution_binding_sha256": digest(BINDING), "pushed_head": head}


def paid_paths_pristine() -> bool:
    return all(
        not path.exists()
        for path in (RAW, EVALUATION, SEMANTIC_VERIFICATION, RESULT, FAILURE, LEDGER)
    )


def preflight(
    *,
    now: datetime | None = None,
    live_reader=read_live_credits,
    catalog_reader=read_catalog,
) -> dict[str, Any]:
    local = now.astimezone(ZoneInfo(TIMEZONE)) if now else datetime.now(ZoneInfo(TIMEZONE))
    if local.date().isoformat() != DATE:
        raise RuntimeError("wrong execution date")
    bindings = validate_bindings()
    if not paid_paths_pristine():
        raise RuntimeError("proposal-atlas paid path is not pristine")
    source_verification = load(SOURCE_VERIFICATION)
    if source_verification.get("status") != "passed":
        raise RuntimeError("source verification did not pass")
    manifest = load(MANIFEST)
    prices = validate_catalog(catalog_reader())
    exposure = request_exposure(manifest, prices)
    live = validate_live(live_reader())
    reserve = float(exposure["exact_stage_exposure_usd"])
    if reserve > STAGE_CAP_USD + 1e-12:
        raise RuntimeError("proposal-atlas exposure exceeds stage cap")
    if (
        prior_spend(live) + reserve > DAILY_CAP_USD + 1e-12
        or live["balance_usd"] + 1e-12 < reserve
    ):
        raise RuntimeError("account allowance unavailable")
    return {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "bindings": bindings,
        "authorization": {"model": prices, "exposure": exposure, "live": live},
        "budget": {
            "opening_total_usage_usd": OPENING_USAGE_USD,
            "prior_account_spend_usd": prior_spend(live),
            "daily_cap_usd": DAILY_CAP_USD,
            "stage_cap_usd": STAGE_CAP_USD,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


class ProposalAdapter(SeededAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reasoning: dict[tuple[int, str], bool] = {}
        self._reasoning_lock = threading.Lock()

    def _payload(
        self,
        messages: Any,
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        *,
        disable_reasoning: bool = False,
        response_format: Any = None,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages,
            temperature,
            n,
            max_tokens,
            disable_reasoning=True,
            response_format=response_format,
        )
        payload["provider"] = {"require_parameters": True}
        return payload

    def _post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        data = super()._post(payload)
        reasoning_present = False
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            if any(
                message.get(key)
                for key in ("reasoning", "reasoning_content", "reasoning_details")
            ):
                reasoning_present = True
        key = (
            int(payload["seed"]),
            hashlib.sha256(canonical_json(payload["messages"]).encode()).hexdigest(),
        )
        with self._reasoning_lock:
            self._reasoning[key] = reasoning_present
        return data

    def reasoning_for(self, seed: int, messages_sha256: str) -> bool:
        with self._reasoning_lock:
            return bool(self._reasoning.get((int(seed), messages_sha256), False))


def build_adapter(*, request_cap: float, authorize: Any) -> ProposalAdapter:
    config = Config(
        task="animals",
        run_id="chembench-llm-proposal-atlas-20260815",
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
    return ProposalAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65536),
        config,
        authorize=authorize,
    )


def adapter_cost(adapter: ProposalAdapter | None) -> float:
    if adapter is None:
        return 0.0
    return float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))


def adapter_usage(adapter: ProposalAdapter) -> dict[str, Any]:
    value = adapter.usage_snapshot()
    return {
        "accepted_requests": int(value.get("adapter_requests", 0)),
        "http_attempts": int(value.get("http_attempts", 0)),
        "retries": int(value.get("retry_count", 0)),
        "reasoning_tokens": int(value.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(value.get("forced_exits", 0)),
        "forced_final_requests": int(value.get("forced_final_requests", 0)),
        "cost_usd": float(value.get("adapter_cost_usd", 0.0)),
    }


def response_format(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "chembench_typed_mechanism_proposals",
            "strict": True,
            "schema": manifest["response_schema"],
        },
    }


def serve_requests(
    adapter: ProposalAdapter,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    requests = list(manifest["requests"])
    schema = response_format(manifest)

    def one(request_row: Mapping[str, Any]) -> dict[str, Any]:
        messages = request_messages(manifest, request_row)
        messages_sha = hashlib.sha256(canonical_json(messages).encode()).hexdigest()
        adapter._seed.value = int(request_row["seed"])
        content = None
        error_type = None
        error_message = None
        try:
            values = adapter._complete_request(
                messages,
                gate.TEMPERATURE,
                1,
                MAX_TOKENS,
                allow_forced_final=False,
                disable_reasoning=True,
                response_format=schema,
            )
            content = values[0]
        except Exception as error:
            error_type = type(error).__name__
            error_message = str(error)
        finally:
            del adapter._seed.value
        return {
            **request_row,
            "messages_sha256": messages_sha,
            "content": content,
            "error_type": error_type,
            "error": error_message,
        }

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        partial = list(pool.map(one, requests))
    metadata = {
        (int(item["seed"]), str(item["prompt_sha256"])): item
        for item in adapter.records()
    }
    records = []
    for item in partial:
        transport = metadata.get((int(item["seed"]), str(item["messages_sha256"])), {})
        reasoning_present = adapter.reasoning_for(
            int(item["seed"]), str(item["messages_sha256"])
        )
        clean = bool(
            transport.get("model_requested") == MODEL_ID
            and transport.get("model_returned") == MODEL_ID
            and transport.get("finish_reasons") == ["stop"]
            and isinstance(item.get("content"), str)
            and bool(item["content"].strip())
            and not reasoning_present
            and item.get("error_type") is None
        )
        records.append(
            {
                **item,
                "model_requested": transport.get("model_requested"),
                "model_returned": transport.get("model_returned"),
                "finish_reasons": transport.get("finish_reasons", []),
                "payload_sha256": transport.get("payload_sha256"),
                "reasoning_present": reasoning_present,
                "clean": clean,
            }
        )
    return sorted(records, key=lambda item: item["request_id"])


def reconcile(
    ledger: Mapping[str, Any],
    status: str,
    local_cost: float,
    live: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = json.loads(json.dumps(ledger))
    execution_prior = float(result["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD
    recorded = max(
        float(result["recorded_actual_spend_usd"]),
        prior_spend(live) if live else 0.0,
        execution_prior + local_cost,
    )
    if recorded > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("daily cap exceeded")
    result["recorded_actual_spend_usd"] = recorded
    result["stage"].update({"status": status, "actual_cost_usd": local_cost})
    if live:
        result.update(
            {
                "closing_total_credits_usd": live["total_credits_usd"],
                "closing_total_usage_usd": live["total_usage_usd"],
                "closing_balance_usd": live["balance_usd"],
            }
        )
    return result


def execute(
    *,
    now: datetime | None = None,
    live_reader=read_live_credits,
    catalog_reader=read_catalog,
) -> dict[str, Any]:
    ready = preflight(now=now, live_reader=live_reader, catalog_reader=catalog_reader)
    manifest = load(MANIFEST)
    labels = load(LABELS)
    prices = validate_catalog(catalog_reader())
    exposure = request_exposure(manifest, prices)
    live = validate_live(live_reader())
    reserve = float(exposure["exact_stage_exposure_usd"])
    if (
        reserve > STAGE_CAP_USD + 1e-12
        or prior_spend(live) + reserve > DAILY_CAP_USD + 1e-12
        or live["balance_usd"] + 1e-12 < reserve
    ):
        raise RuntimeError("allowance changed before ledger creation")
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
        "block_authorization": {
            "model": prices,
            "exposure": exposure,
            "live": live,
            "accepted_cost_before_block_usd": 0.0,
        },
        "stage": {
            "status": "authorized_pending",
            "maximum_cost_usd": STAGE_CAP_USD,
            "maximum_http_attempts": 126,
            "maximum_retries": 0,
        },
    }
    checkpoint(LEDGER, ledger)
    adapter: ProposalAdapter | None = None
    try:
        def request_authorizer() -> None:
            current = validate_live(live_reader())
            accepted = adapter_cost(adapter)
            remaining = max(0.0, reserve - accepted)
            execution_prior = (
                float(ledger["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD
            )
            reconciled = max(prior_spend(current), execution_prior + accepted)
            if (
                accepted + remaining > STAGE_CAP_USD + 1e-12
                or reconciled + remaining > DAILY_CAP_USD + 1e-12
                or current["balance_usd"] + 1e-12 < remaining
            ):
                raise RuntimeError("per-request proposal-atlas allowance lost")

        adapter = build_adapter(
            request_cap=float(exposure["maximum_request_exposure_usd"]),
            authorize=request_authorizer,
        )
        records = serve_requests(adapter, manifest)
        checkpoint(RAW, {"schema_version": 1, "records": records})
        evaluation = gate.evaluate_records(
            Path(manifest["source"]["root"]), manifest, labels, records
        )
        checkpoint(EVALUATION, evaluation)
        verification = verifier.verify(
            Path(manifest["source"]["root"]),
            MANIFEST,
            LABELS,
            SOURCE_SUMMARY,
            responses_path=RAW,
            evaluation_path=EVALUATION,
        )
        checkpoint(SEMANTIC_VERIFICATION, verification)
        usage = adapter_usage(adapter)
        transport_conditions = {
            "http_attempts_at_most_126": usage["http_attempts"] <= 126,
            "accepted_requests_at_most_126": usage["accepted_requests"] <= 126,
            "zero_retries": usage["retries"] == 0,
            "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
            "zero_forced_paths": usage["forced_exits"] == 0
            and usage["forced_final_requests"] == 0,
            "within_stage_cap": usage["cost_usd"] <= STAGE_CAP_USD + 1e-12,
            "independent_verification_passed": verification["status"] == "passed",
        }
        passed = evaluation["provisional_status"] == "passed_pending_independent_verification" and all(
            transport_conditions.values()
        )
        status = "passed" if passed else "failed_closed"
        try:
            closing = validate_live(live_reader())
        except Exception:
            closing = None
        ledger = reconcile(ledger, status, usage["cost_usd"], closing)
        checkpoint(LEDGER, ledger)
        terminal = {
            "schema_version": 1,
            "interface_version": INTERFACE_VERSION,
            "status": status,
            "authorizes": (
                "prospective_opened_v4_llm_policy_protocol_only" if passed else "nothing"
            ),
            "manifest_sha256": digest(MANIFEST),
            "labels_sha256": digest(LABELS),
            "raw_sha256": digest(RAW),
            "evaluation_sha256": digest(EVALUATION),
            "verification_sha256": digest(SEMANTIC_VERIFICATION),
            "ledger_sha256": digest(LEDGER),
            "usage": usage,
            "transport_conditions": transport_conditions,
            "semantic_conditions": evaluation["conditions"],
            "policy_endpoints_opened": False,
            "v5_opened": False,
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
            "evaluation_exists": EVALUATION.exists(),
            "verification_exists": SEMANTIC_VERIFICATION.exists(),
            "policy_endpoints_opened": False,
            "v5_opened": False,
        }
        checkpoint(FAILURE, failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")
    result = preflight() if args.preflight else execute()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
