#!/usr/bin/env python3
"""Execute or resume the frozen August 10 Bongard Luna gates."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import sys
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_image_integrity_audit as image_audit
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-aug10-execute-2"
EXPECTED_DATE = "2026-08-10"
TIMEZONE = "Europe/London"
OUTPUT_DIR = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_aug10_execution/"
    "bongard-openworld-luna-aug10-20260810"
)
SERVING_DIR = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_serving_smoke/"
    "bongard-openworld-luna-vlm-serving-smoke-20260810"
)
MECHANICS_DIR = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_mechanics_tree/"
    "bongard-openworld-luna-vlm-mechanics-tree-20260810"
)
DAILY_LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/2026-08-10.json"
)
SERVING_RUN_ID = "bongard-openworld-luna-vlm-serving-smoke-20260810"
MECHANICS_RUN_ID = "bongard-openworld-luna-vlm-mechanics-tree-20260810"
MODELS_URL = "https://openrouter.ai/api/v1/models"
IMAGE_INTEGRITY_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_image_integrity_audit/"
    "bongard-openworld-image-integrity-audit-20260806/MANIFEST.json"
)
IMAGE_INTEGRITY_MANIFEST_SHA256 = (
    "239943ae789ebdc2c0a03577a02b04890c6d00f50ce45639c5defc1624ccee96"
)
DEVELOPMENT_PROTOCOL_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_development32/"
    "PROTOCOL_MANIFEST.json"
)
DEVELOPMENT_PROTOCOL_MANIFEST_SHA256 = (
    "64f80983b3922556c279982fdbf966a861046345f698f2196cf823077b14ba46"
)
MINIMUM_STARTING_BALANCE_USD = 5.0
MINIMUM_RESERVED_PROMPT_TOKENS = 8_000
PRECHARGE_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/BONGARD_LUNA_PRECHARGE_AMENDMENT.md"
)
PRECHARGE_AMENDMENT_SHA256 = (
    "75acd7ae3b51e287a08e81e202d0cfcd95d83dfd3ed308f678b7fe7854bbbff4"
)


class PreExecutionGateError(RuntimeError):
    """Raised when the unopened paid sequence fails its read-only gate."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return mechanics.sha256_file(path)


def _artifact_path(output_dir: Path) -> Path | None:
    artifacts = [
        path
        for path in (output_dir / "RESULT.json", output_dir / "FAILURE.json")
        if path.exists()
    ]
    if len(artifacts) > 1:
        raise RuntimeError(f"ambiguous artifacts in {output_dir}")
    return artifacts[0] if artifacts else None


def _validate_date(now: datetime | None = None) -> None:
    timezone = ZoneInfo(TIMEZONE)
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    if local_now.date().isoformat() != EXPECTED_DATE:
        raise RuntimeError("the frozen Bongard sequence can run only on August 10")


def _validate_ledger(path: Path, *, require_mechanics: bool) -> dict[str, Any]:
    ledger = _load(path)
    if (
        ledger.get("date") != EXPECTED_DATE
        or ledger.get("timezone") != TIMEZONE
        or float(ledger.get("daily_cap_usd", 0.0)) != 5.0
        or float(ledger.get("recorded_actual_spend_usd", 0.0)) > 5.0 + 1e-12
    ):
        raise RuntimeError("August 10 ledger boundary is invalid")
    smoke = ledger.get("bongard_luna_vlm_serving_smoke") or {}
    if (
        smoke.get("interface_version") != serving.INTERFACE_VERSION
        or smoke.get("model") != serving.MODEL_ID
        or smoke.get("status") not in {"passed", "gated_null"}
    ):
        raise RuntimeError("ledger lacks the exact interface-v2 serving record")
    if require_mechanics:
        tree = ledger.get("bongard_luna_vlm_mechanics_tree") or {}
        if (
            tree.get("interface_version") != mechanics.INTERFACE_VERSION
            or tree.get("model") != mechanics.MODEL_ID
            or tree.get("status") not in {"mechanics_pass", "gated_null"}
        ):
            raise RuntimeError("ledger lacks the exact frozen mechanics record")
    return ledger


def read_openrouter_model_catalog() -> dict[str, Any]:
    headers = {}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(MODELS_URL, headers=headers)
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _validate_model_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        model
        for model in catalog.get("data", [])
        if model.get("id") == serving.MODEL_ID
    ]
    if len(matches) != 1:
        raise RuntimeError(f"OpenRouter does not expose exact model {serving.MODEL_ID}")
    model = matches[0]
    architecture = model.get("architecture") or {}
    modalities = set(architecture.get("input_modalities") or [])
    supported = set(model.get("supported_parameters") or [])
    top_provider = model.get("top_provider") or {}
    max_completion = int(top_provider.get("max_completion_tokens") or 0)
    if not {"text", "image"}.issubset(modalities):
        raise RuntimeError("frozen Luna endpoint no longer supports text and image input")
    if not ({"response_format", "structured_outputs"} & supported):
        raise RuntimeError("frozen Luna endpoint no longer supports structured output")
    if max_completion < serving.MAX_TOKENS:
        raise RuntimeError("frozen Luna endpoint completion limit is too small")
    pricing = model.get("pricing") or {}
    prompt_price = float(pricing.get("prompt", math.nan))
    completion_price = float(pricing.get("completion", math.nan))
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (prompt_price, completion_price)
    ):
        raise RuntimeError("frozen Luna endpoint pricing is missing or invalid")
    maximum_output_cost = completion_price * serving.MAX_TOKENS
    residual = serving.MAX_REQUEST_COST_USD - maximum_output_cost
    covered_prompt_tokens = (
        math.inf if prompt_price == 0.0 else residual / prompt_price
    )
    if (
        residual < 0.0
        or covered_prompt_tokens + 1e-9 < MINIMUM_RESERVED_PROMPT_TOKENS
    ):
        raise RuntimeError(
            "frozen Luna attempt-cost reservation no longer covers the "
            "output and prompt ceilings"
        )
    return {
        "id": model["id"],
        "context_length": int(model.get("context_length") or 0),
        "max_completion_tokens": max_completion,
        "input_modalities": sorted(modalities),
        "supports_structured_output": True,
        "prompt_usd_per_million_tokens": round(prompt_price * 1_000_000, 12),
        "completion_usd_per_million_tokens": round(
            completion_price * 1_000_000, 12
        ),
        "maximum_request_cost_usd": serving.MAX_REQUEST_COST_USD,
        "maximum_output_cost_usd": maximum_output_cost,
        "covered_prompt_tokens_at_live_price": covered_prompt_tokens,
    }


def _validate_precharge_amendment(
    path: Path = PRECHARGE_AMENDMENT,
) -> dict[str, Any]:
    digest = _sha256(path)
    if digest != PRECHARGE_AMENDMENT_SHA256:
        raise RuntimeError("Bongard precharge amendment hash changed")
    return {
        "path": str(path),
        "sha256": digest,
        "maximum_request_cost_usd": serving.MAX_REQUEST_COST_USD,
    }


def _validate_image_integrity_manifest(path: Path) -> dict[str, Any]:
    manifest_sha256 = _sha256(path)
    manifest = _load(path)
    source = manifest.get("source") or {}
    mechanics_record = manifest.get("mechanics") or {}
    summary = mechanics_record.get("summary") or {}
    if (
        manifest_sha256 != IMAGE_INTEGRITY_MANIFEST_SHA256
        or manifest.get("status") != "image_integrity_pass"
        or manifest.get("all_gates_pass") is not True
        or manifest.get("authorizes_paid_calls") is not False
        or not manifest.get("gates")
        or not all(manifest["gates"].values())
        or source.get("archive_sha256") != image_audit.ARCHIVE_SHA256
        or int(source.get("archive_size_bytes", 0)) != image_audit.ARCHIVE_SIZE
        or int(summary.get("tasks", 0)) != image_audit.EXPECTED_MECHANICS_TASKS
        or int(summary.get("images", 0)) != image_audit.EXPECTED_MECHANICS_IMAGES
    ):
        raise RuntimeError("image-integrity manifest is stale or invalid")
    return {
        "verified": True,
        "manifest_sha256": manifest_sha256,
        "archive_sha256": source["archive_sha256"],
        "archive_size_bytes": int(source["archive_size_bytes"]),
        "mechanics_tasks": int(summary["tasks"]),
        "mechanics_images": int(summary["images"]),
    }


def _verify_frozen_inputs() -> dict[str, Any]:
    precharge = _validate_precharge_amendment()
    source = image_audit.verify_bound_sources()
    image_integrity = _validate_image_integrity_manifest(
        IMAGE_INTEGRITY_MANIFEST
    )
    archive = image_audit.ARCHIVE_PATH
    if not archive.is_file() or archive.stat().st_size != image_audit.ARCHIVE_SIZE:
        raise RuntimeError("bound Bongard image archive is absent or changed in size")

    tasks = bed.load_mechanics_tasks()
    if (
        len(tasks) != image_audit.EXPECTED_MECHANICS_TASKS
        or sum(len(task.image_ids) for task in tasks)
        != image_audit.EXPECTED_MECHANICS_IMAGES
    ):
        raise RuntimeError("mechanics task or image count changed")
    cases = serving.build_smoke_cases(tasks)
    messages = [
        bed.build_belief_messages(case.task, case.history) for case in cases
    ]
    prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, message)
        for case, message in zip(cases, messages, strict=True)
    ]
    if len(cases) != serving.EXPECTED_REQUESTS or any(prompt_errors):
        raise RuntimeError("exact serving prompts are stale or expose hidden state")
    response_format = bed.belief_response_format()
    if (
        response_format.get("type") != "json_schema"
        or (response_format.get("json_schema") or {}).get("strict") is not True
    ):
        raise RuntimeError("strict belief response contract changed")

    protocol = development.verify_protocol_manifest(
        DEVELOPMENT_PROTOCOL_MANIFEST
    )
    if protocol["manifest_sha256"] != DEVELOPMENT_PROTOCOL_MANIFEST_SHA256:
        raise RuntimeError("development protocol manifest hash changed")
    return {
        "precharge_amendment": precharge,
        "source": source,
        "image_integrity": image_integrity,
        "archive_path": str(archive),
        "mechanics_tasks": len(tasks),
        "mechanics_images": sum(len(task.image_ids) for task in tasks),
        "serving_cases": len(cases),
        "serving_message_bytes": sum(
            len(bed.canonical_json(message).encode("utf-8"))
            for message in messages
        ),
        "belief_response_format_sha256": hashlib.sha256(
            bed.canonical_json(response_format).encode("utf-8")
        ).hexdigest(),
        "development_protocol_manifest_sha256": protocol["manifest_sha256"],
    }


def _verify_pristine_paths(
    *,
    output_dir: Path,
    serving_dir: Path,
    mechanics_dir: Path,
    daily_ledger: Path,
) -> dict[str, str]:
    if daily_ledger.exists():
        raise RuntimeError("August 10 daily ledger already exists")
    states = {}
    for name, path in (
        ("wrapper", output_dir),
        ("serving", serving_dir),
        ("mechanics", mechanics_dir),
    ):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise RuntimeError(f"{name} execution path is not pristine: {path}")
        states[name] = "absent" if not path.exists() else "empty"
    states["daily_ledger"] = "absent"
    return states


def preflight_aug10_sequence(
    *,
    output_dir: Path = OUTPUT_DIR,
    serving_dir: Path = SERVING_DIR,
    mechanics_dir: Path = MECHANICS_DIR,
    daily_ledger: Path = DAILY_LEDGER,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    model_catalog_reader: Callable[[], dict[str, Any]] = (
        read_openrouter_model_catalog
    ),
    frozen_inputs_validator: Callable[[], dict[str, Any]] = (
        _verify_frozen_inputs
    ),
) -> dict[str, Any]:
    """Validate the frozen sequence before its date without writes or model calls."""
    paths = _verify_pristine_paths(
        output_dir=output_dir,
        serving_dir=serving_dir,
        mechanics_dir=mechanics_dir,
        daily_ledger=daily_ledger,
    )
    frozen_inputs = frozen_inputs_validator()
    model = _validate_model_catalog(model_catalog_reader())
    live = live_reader()
    values = [
        float(live[field])
        for field in ("total_credits_usd", "total_usage_usd", "balance_usd")
    ]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("live OpenRouter credit values are non-finite")
    if float(live["balance_usd"]) + 1e-12 < MINIMUM_STARTING_BALANCE_USD:
        raise RuntimeError("live OpenRouter balance is below the $5 start gate")

    component_cap = serving.RUN_BUDGET_USD + mechanics.RUN_BUDGET_USD
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "execution_date": EXPECTED_DATE,
        "timezone": TIMEZONE,
        "execution_paths": paths,
        "frozen_inputs": frozen_inputs,
        "model": model,
        "live_credits": live,
        "budget": {
            "account_wide_daily_cap_usd": 5.0,
            "minimum_starting_balance_usd": MINIMUM_STARTING_BALANCE_USD,
            "serving_projected_cost_usd": serving.PROJECTED_COST_USD,
            "serving_maximum_cost_usd": serving.RUN_BUDGET_USD,
            "mechanics_maximum_cost_usd": mechanics.RUN_BUDGET_USD,
            "maximum_component_caps_usd": component_cap,
            "unallocated_daily_allowance_usd": 5.0 - component_cap,
            "mechanics_requires_observed_serving_projection": True,
            "unspent_allowance_does_not_roll_over": True,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


def validate_serving_artifact(
    path: Path, *, tasks: list[bed.VisualTask] | None = None
) -> dict[str, Any]:
    result = _load(path)
    protocol = result.get("protocol") or {}
    raw_path = path.parent / "private/RAW_RESPONSES.json"
    if (
        path.name != "RESULT.json"
        or result.get("status") not in {"passed", "gated_null"}
        or protocol.get("interface_version") != serving.INTERFACE_VERSION
        or protocol.get("model") != serving.MODEL_ID
        or protocol.get("actual_candidate_labels_accessed") is not False
        or protocol.get("endpoint_labels_accessed") is not False
        or not raw_path.is_file()
        or _sha256(raw_path) != result.get("raw_responses_sha256")
        or not result.get("gates")
        or result["gates"].get("all_pass") is not (
            result.get("status") == "passed"
        )
    ):
        raise RuntimeError("serving artifact is stale, partial, or inconsistent")
    if result["status"] == "passed":
        verification = mechanics.verify_serving_result(path, tasks=tasks)
    else:
        raw = _load(raw_path)
        cases = serving.build_smoke_cases(tasks or bed.load_mechanics_tasks())
        responses = raw.get("responses") or []
        if raw.get("case_ids") != [case.case_id for case in cases]:
            raise RuntimeError("gated-null serving case order changed")
        messages = [
            bed.build_belief_messages(case.task, case.history) for case in cases
        ]
        prompt_errors = [
            bed.prompt_hidden_state_errors(case.task, case.history, message)
            for case, message in zip(cases, messages, strict=True)
        ]
        beliefs = [
            bed.parse_belief_response(
                response,
                image_ids=case.task.image_ids,
                history=case.history,
            )
            for case, response in zip(cases, responses, strict=True)
        ]
        metrics = serving.serving_metrics(cases, beliefs)
        gates = serving.serving_gates(
            cases=cases,
            beliefs=beliefs,
            metrics=metrics,
            prompt_errors=prompt_errors,
            usage=result["usage"],
        )
        if (
            bed.canonical_json(metrics)
            != bed.canonical_json(result.get("metrics"))
            or gates != result["gates"]
            or gates["all_pass"]
        ):
            raise RuntimeError("gated-null serving artifact does not replay")
        verification = {
            "result_sha256": _sha256(path),
            "raw_responses_sha256": _sha256(raw_path),
            "verified": True,
        }
    return {
        **verification,
        "status": result["status"],
        "cost_usd": float(result["usage"]["run_cost_usd"]),
    }


class _MechanicsReplayAdapter:
    def __init__(self, *, raw: Mapping[str, Any], usage: Mapping[str, Any]) -> None:
        self.raw = raw
        self.usage = usage
        self.call_index = 0
        self.final_offset = 0
        pairing = raw.get("final_request_pairing") or {}
        self.final_batch_sizes = [
            int(batch["request_count"])
            for batch in pairing.get("dispatch_batches") or []
        ]

    def chat_complete_messages_batched_structured(self, batch_messages, **kwargs):
        del kwargs
        if self.call_index == 0:
            responses = self.raw.get("first_stage_responses") or []
        else:
            final_index = self.call_index - 1
            if final_index >= len(self.final_batch_sizes):
                raise RuntimeError("mechanics replay requested an extra model batch")
            expected_size = self.final_batch_sizes[final_index]
            if len(batch_messages) != expected_size:
                raise RuntimeError("mechanics replay terminal batch size changed")
            all_final = self.raw.get("final_responses") or []
            responses = all_final[
                self.final_offset : self.final_offset + expected_size
            ]
            self.final_offset += expected_size
        self.call_index += 1
        if len(responses) != len(batch_messages):
            raise RuntimeError("mechanics replay batch size changed")
        return list(responses)

    def chat_complete_seeded_messages_batched_structured(
        self, batch_messages, seeds, **kwargs
    ):
        if len(batch_messages) != len(seeds):
            raise RuntimeError("mechanics replay seed count changed")
        return self.chat_complete_messages_batched_structured(
            batch_messages, **kwargs
        )

    def usage_snapshot(self):
        return {
            "adapter_requests": self.usage["adapter_requests"],
            "http_attempts": self.usage["http_attempts"],
            "retry_count": self.usage["retry_count"],
            "provider_error_retries": self.usage["provider_error_retries"],
            "adapter_reasoning_tokens": self.usage["adapter_reasoning_tokens"],
            "forced_exits": self.usage["forced_exits"],
            "adapter_prompt_tokens": self.usage["prompt_tokens"],
            "adapter_completion_tokens": self.usage["completion_tokens"],
            "adapter_cost_usd": self.usage["run_cost_usd"],
        }


def validate_mechanics_artifact(
    path: Path,
    *,
    serving_result: Path,
    tasks: list[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    result = _load(path)
    protocol = result.get("protocol") or {}
    raw_path = path.parent / "private/RAW_RESPONSES.json"
    if (
        path.name != "RESULT.json"
        or result.get("status") not in {"mechanics_pass", "gated_null"}
        or protocol.get("interface_version") != mechanics.INTERFACE_VERSION
        or protocol.get("model") != mechanics.MODEL_ID
        or protocol.get("development_accessed") is not False
        or protocol.get("confirmation_accessed") is not False
        or protocol.get("sealed_test_accessed") is not False
        or not raw_path.is_file()
        or _sha256(raw_path) != result.get("raw_responses_sha256")
    ):
        raise RuntimeError("mechanics artifact is stale, partial, or inconsistent")
    raw = _load(raw_path)
    adapter = _MechanicsReplayAdapter(raw=raw, usage=result["usage"])
    with tempfile.TemporaryDirectory(prefix="bongard-mechanics-replay-") as tmp:
        replay = mechanics.run_mechanics(
            output_dir=Path(tmp),
            run_id="bongard-mechanics-independent-replay",
            serving_result=serving_result,
            tasks=tasks,
            adapter=adapter,
        )
    expected_calls = 1 + len(adapter.final_batch_sizes)
    if (
        adapter.call_index != expected_calls
        or adapter.final_offset != len(raw.get("final_responses") or [])
    ):
        raise RuntimeError("mechanics replay did not consume exact dispatch batches")
    if bed.canonical_json(replay) != bed.canonical_json(result):
        raise RuntimeError("mechanics result does not independently replay")
    return {
        "verified": True,
        "status": result["status"],
        "result_sha256": _sha256(path),
        "raw_responses_sha256": _sha256(raw_path),
        "cost_usd": float(result["usage"]["run_cost_usd"]),
    }


def _component_record(path: Path, verification: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact": str(path),
        "artifact_sha256": verification["result_sha256"],
        "raw_responses_sha256": verification["raw_responses_sha256"],
        "status": verification["status"],
        "cost_usd": verification["cost_usd"],
        "verified": verification["verified"],
    }


def _existing_or_partial(output_dir: Path) -> Path | None:
    artifact = _artifact_path(output_dir)
    if artifact is None and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"partial unbanked artifact exists in {output_dir}")
    return artifact


def execute_aug10_sequence(
    *,
    output_dir: Path = OUTPUT_DIR,
    serving_dir: Path = SERVING_DIR,
    mechanics_dir: Path = MECHANICS_DIR,
    daily_ledger: Path = DAILY_LEDGER,
    now: datetime | None = None,
    serving_runner: Callable[..., dict[str, Any]] = serving.execute_smoke,
    mechanics_runner: Callable[..., dict[str, Any]] = mechanics.execute_mechanics,
    serving_validator: Callable[[Path], dict[str, Any]] = validate_serving_artifact,
    mechanics_validator: Callable[..., dict[str, Any]] = validate_mechanics_artifact,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    fresh_preflight: Callable[..., dict[str, Any]] = preflight_aug10_sequence,
) -> dict[str, Any]:
    final_path = output_dir / "RESULT.json"
    if final_path.exists():
        final = _load(final_path)
        if final.get("interface_version") != INTERFACE_VERSION:
            raise RuntimeError("banked Aug10 wrapper interface changed")
        components = final.get("components") or {}
        serving_component = components.get("serving") or {}
        serving_artifact = Path(serving_component.get("artifact", ""))
        serving_verification = serving_validator(serving_artifact)
        if _component_record(
            serving_artifact, serving_verification
        ) != serving_component:
            raise RuntimeError("banked Aug10 serving component changed")
        mechanics_component = components.get("mechanics")
        if mechanics_component is not None:
            mechanics_artifact = Path(mechanics_component.get("artifact", ""))
            mechanics_verification = mechanics_validator(
                mechanics_artifact, serving_result=serving_artifact
            )
            if _component_record(
                mechanics_artifact, mechanics_verification
            ) != mechanics_component:
                raise RuntimeError("banked Aug10 mechanics component changed")
        _validate_ledger(
            daily_ledger, require_mechanics=mechanics_component is not None
        )
        return final

    _validate_date(now)
    if not daily_ledger.exists():
        try:
            preflight = fresh_preflight(
                output_dir=output_dir,
                serving_dir=serving_dir,
                mechanics_dir=mechanics_dir,
                daily_ledger=daily_ledger,
                live_reader=live_reader,
            )
        except Exception as exc:
            raise PreExecutionGateError(str(exc)) from exc
        if preflight.get("status") != "ready_without_paid_calls":
            raise PreExecutionGateError(
                "fresh August 10 preflight did not authorize execution"
            )
        live_opening = preflight.get("live_credits")
        if not isinstance(live_opening, dict):
            raise PreExecutionGateError(
                "fresh preflight omitted the live credit snapshot"
            )
        serving.initialize_daily_ledger(
            path=daily_ledger,
            live=live_opening,
            now=now,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "in_progress",
        "date": EXPECTED_DATE,
        "daily_ledger": str(daily_ledger),
        "components": {},
    }

    serving_artifact = _existing_or_partial(serving_dir)
    if serving_artifact is not None and serving_artifact.name == "FAILURE.json":
        raise RuntimeError("serving gate has a banked failed-closed artifact")
    if serving_artifact is None:
        serving_runner(
            output_dir=serving_dir,
            run_id=SERVING_RUN_ID,
            ledger_path=daily_ledger,
            now=now,
            live_reader=live_reader,
        )
        serving_artifact = _artifact_path(serving_dir)
    if serving_artifact is None or serving_artifact.name != "RESULT.json":
        raise RuntimeError("serving gate omitted its result artifact")
    serving_verification = serving_validator(serving_artifact)
    state["components"]["serving"] = _component_record(
        serving_artifact, serving_verification
    )
    checkpoint(output_dir / "EXECUTION_STATE.json", state)
    ledger = _validate_ledger(daily_ledger, require_mechanics=False)
    if serving_verification["status"] != "passed":
        state["status"] = "stopped_after_serving_gated_null"
        state["authorizes_development"] = False
        state["recorded_actual_spend_usd"] = ledger["recorded_actual_spend_usd"]
        checkpoint(final_path, state)
        return state

    mechanics_artifact = _existing_or_partial(mechanics_dir)
    if mechanics_artifact is not None and mechanics_artifact.name == "FAILURE.json":
        raise RuntimeError("mechanics gate has a banked failed-closed artifact")
    if mechanics_artifact is None:
        mechanics_runner(
            output_dir=mechanics_dir,
            run_id=MECHANICS_RUN_ID,
            serving_result=serving_artifact,
            ledger_path=daily_ledger,
            now=now,
        )
        mechanics_artifact = _artifact_path(mechanics_dir)
    if mechanics_artifact is None or mechanics_artifact.name != "RESULT.json":
        raise RuntimeError("mechanics gate omitted its result artifact")
    mechanics_verification = mechanics_validator(
        mechanics_artifact, serving_result=serving_artifact
    )
    state["components"]["mechanics"] = _component_record(
        mechanics_artifact, mechanics_verification
    )
    ledger = _validate_ledger(daily_ledger, require_mechanics=True)
    state["status"] = "complete"
    state["authorizes_development"] = (
        mechanics_verification["status"] == "mechanics_pass"
    )
    state["recorded_actual_spend_usd"] = ledger["recorded_actual_spend_usd"]
    state["remaining_daily_allowance_usd"] = max(
        0.0, 5.0 - float(ledger["recorded_actual_spend_usd"])
    )
    checkpoint(final_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--serving-dir", type=Path, default=SERVING_DIR)
    parser.add_argument("--mechanics-dir", type=Path, default=MECHANICS_DIR)
    parser.add_argument("--daily-ledger", type=Path, default=DAILY_LEDGER)
    args = parser.parse_args()
    function = preflight_aug10_sequence if args.preflight else execute_aug10_sequence
    result = function(
        output_dir=args.output_dir,
        serving_dir=args.serving_dir,
        mechanics_dir=args.mechanics_dir,
        daily_ledger=args.daily_ledger,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
