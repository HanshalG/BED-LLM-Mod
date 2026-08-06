#!/usr/bin/env python3
"""Execute or resume the frozen August 10 Bongard Luna gates."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import tempfile
import sys
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-aug10-execute-1"
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
            raise RuntimeError("ledger lacks the exact interface-v2 mechanics record")
    return ledger


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

    def chat_complete_messages_batched_structured(self, batch_messages, **kwargs):
        del kwargs
        if self.call_index == 0:
            responses = self.raw.get("first_stage_responses") or []
        elif self.call_index == 1:
            responses = self.raw.get("final_responses") or []
        else:
            raise RuntimeError("mechanics replay requested an extra model batch")
        self.call_index += 1
        if len(responses) != len(batch_messages):
            raise RuntimeError("mechanics replay batch size changed")
        return list(responses)

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
    if adapter.call_index != 2:
        raise RuntimeError("mechanics replay did not consume exactly two batches")
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
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--serving-dir", type=Path, default=SERVING_DIR)
    parser.add_argument("--mechanics-dir", type=Path, default=MECHANICS_DIR)
    parser.add_argument("--daily-ledger", type=Path, default=DAILY_LEDGER)
    args = parser.parse_args()
    result = execute_aug10_sequence(
        output_dir=args.output_dir,
        serving_dir=args.serving_dir,
        mechanics_dir=args.mechanics_dir,
        daily_ledger=args.daily_ledger,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
