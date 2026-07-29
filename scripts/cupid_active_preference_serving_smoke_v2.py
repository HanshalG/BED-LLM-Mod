#!/usr/bin/env python3
"""Run disjoint CUPID serving V2 with schema-enforced integer signatures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import cupid_active_preference_serving_smoke as v1
from scripts import cupid_active_preference_source_audit as source
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "cupid-active-preference-serving-smoke-2"
PLANNER_SEED = 37_510
TARGET_SEED = 37_610
V2_CASE_IDS = (
    "79+mathematics_professor:consistent",
    "66+horticulturist:consistent",
    "85+design_research_associate:contrastive",
    "212+maritime_preservationist:contrastive",
    "153+music_therapist:changing",
)


def integer_signature_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": v1.NUM_QUESTIONS,
        "maxItems": v1.NUM_QUESTIONS,
        "items": {
            "type": "integer",
            "enum": [0, 1],
        },
    }


def planner_response_format() -> dict[str, Any]:
    response_format = v1.planner_response_format()
    schema = response_format["json_schema"]["schema"]
    schema["properties"]["hypotheses"]["items"]["properties"][
        "answer_signature"
    ] = integer_signature_schema()
    response_format["json_schema"]["name"] = "cupid_preference_interview_v2"
    return response_format


def target_response_format() -> dict[str, Any]:
    response_format = v1.target_response_format()
    schema = response_format["json_schema"]["schema"]
    schema["properties"]["answer_signature"] = integer_signature_schema()
    response_format["json_schema"]["name"] = "cupid_preference_answers_v2"
    return response_format


def parse_integer_signature(value: Any, *, label: str) -> str:
    if not isinstance(value, list) or len(value) != v1.NUM_QUESTIONS:
        raise ValueError(f"{label} must contain exactly six integers")
    if any(type(item) is not int or item not in {0, 1} for item in value):
        raise ValueError(f"{label} must contain only integer 0 or 1")
    return "".join(str(item) for item in value)


def parse_planner_response(response: str) -> dict[str, Any]:
    value = v1.strict_json_object(response, label="planner V2 response")
    hypotheses = value.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise ValueError("planner V2 hypotheses are not an array")
    canonical = json.loads(json.dumps(value))
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, dict):
            raise ValueError(f"hypotheses[{index}] is not an object")
        canonical["hypotheses"][index]["answer_signature"] = (
            parse_integer_signature(
                hypothesis.get("answer_signature"),
                label=f"hypotheses[{index}].answer_signature",
            )
        )
    return v1.parse_planner_response(json.dumps(canonical))


def parse_target_response(response: str) -> str:
    value = v1.strict_json_object(response, label="target V2 response")
    if set(value) != {"answer_signature"}:
        raise ValueError("target V2 response has the wrong fields")
    return parse_integer_signature(
        value["answer_signature"],
        label="target answer_signature",
    )


def v2_serving_rows() -> list[dict[str, Any]]:
    manifest_path = (
        REPO_ROOT
        / "results/nonmyopic/cupid_active_preference_source_audit/"
        "cupid-active-preference-source-audit-20260729/MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    development_ids = {
        item["id"] for item in manifest["splits"]["development"]
    }
    if not set(V2_CASE_IDS) <= development_ids:
        raise ValueError("V2 case IDs are not all in the bound development split")
    v1_ids = {
        item["id"] for item in manifest["splits"]["serving_smoke"]
    }
    if set(V2_CASE_IDS) & v1_ids:
        raise ValueError("V2 cases overlap V1 serving cases")
    rows_by_id = {
        source.row_id(row): row for row in source.load_rows()
    }
    rows = [rows_by_id[row_id] for row_id in V2_CASE_IDS]
    if [source.row_id(row) for row in rows] != list(V2_CASE_IDS):
        raise ValueError("V2 case ordering changed")
    return rows


def _checkpoint_private(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_smoke(
    *,
    rows: Sequence[dict[str, Any]],
    planner_model: v1.StructuredModel,
    target_model: v1.StructuredModel,
    raw_path: Path,
) -> dict[str, Any]:
    if len(rows) != v1.NUM_CASES:
        raise ValueError(f"serving V2 requires exactly {v1.NUM_CASES} rows")
    raw: dict[str, Any] = {
        "source_row_ids": [source.row_id(row) for row in rows],
        "planner_responses": [],
        "target_responses": [],
        "hidden_target_accessed": False,
        "checklist_accessed": False,
    }
    try:
        planner_responses = planner_model.chat_complete_messages_batched_structured(
            [v1.planner_messages(row) for row in rows],
            temperature=v1.TEMPERATURE,
            block_size=v1.CONCURRENCY,
            response_format=planner_response_format(),
            max_new_tokens=v1.PLANNER_MAX_TOKENS,
        )
        raw["planner_responses"] = list(planner_responses)
        _checkpoint_private(raw_path, raw)
        if len(planner_responses) != v1.NUM_CASES:
            raise ValueError("planner V2 response count changed")
        planners = [
            parse_planner_response(response) for response in planner_responses
        ]

        raw["hidden_target_accessed"] = True
        target_responses = target_model.chat_complete_messages_batched_structured(
            [
                v1.target_messages(row, planner["questions"])
                for row, planner in zip(rows, planners, strict=True)
            ],
            temperature=0.0,
            block_size=v1.CONCURRENCY,
            response_format=target_response_format(),
            max_new_tokens=v1.TARGET_MAX_TOKENS,
        )
        raw["target_responses"] = list(target_responses)
        _checkpoint_private(raw_path, raw)
        if len(target_responses) != v1.NUM_CASES:
            raise ValueError("target V2 response count changed")
        target_signatures = [
            parse_target_response(response) for response in target_responses
        ]
        cases = [
            v1.case_metrics(
                row=row,
                planner=planner,
                target_signature=target_signature,
            )
            for row, planner, target_signature in zip(
                rows,
                planners,
                target_signatures,
                strict=True,
            )
        ]
        usage = v1._usage((planner_model, target_model))
    except Exception as exc:
        _checkpoint_private(raw_path, raw)
        raise v1.ServingExecutionError(
            f"{type(exc).__name__}: {exc}",
            v1._usage((planner_model, target_model)),
        ) from exc

    gates = v1.aggregate_gates(
        cases=cases,
        target_signatures=target_signatures,
        usage=usage,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_manifest_sha256": (
                "2f742ddbacd64eade99fa0148b3c5a11a8676f5cfceb5e666f0966fa6785f790"
            ),
            "v1_failure_sha256": (
                "a65e45184c26e1380e113fbc99088ad99741cbc1cb727e5c7cdfa2a4f5ed7e71"
            ),
            "planner_model": v1.PLANNER_MODEL_ID,
            "target_model": v1.TARGET_MODEL_ID,
            "planner_seed": PLANNER_SEED,
            "target_seed": TARGET_SEED,
            "temperature": v1.TEMPERATURE,
            "signature_representation": "array_of_six_integer_0_or_1",
            "expected_requests": v1.EXPECTED_REQUESTS,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "checklist_accessed": False,
            "policy_endpoint_accessed": False,
            "holdout_accessed": False,
        },
        "metrics": {
            "case_count": len(cases),
            "exact_target_coverage_cases": sum(
                case["target_signature_exactly_covered"] for case in cases
            ),
            "unique_target_signature_count": len(set(target_signatures)),
            "mean_nearest_target_hamming": sum(
                case["target_signature_nearest_hamming"] for case in cases
            )
            / len(cases),
            "mean_partition_entropy_nats": sum(
                case["mean_partition_entropy_nats"] for case in cases
            )
            / len(cases),
        },
        "cases": cases,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel(v1.DeterministicFixtureModel):
    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        responses = super().chat_complete_messages_batched_structured(
            batch_messages,
            temperature=temperature,
            block_size=block_size,
            response_format=response_format,
            max_new_tokens=max_new_tokens,
        )
        converted = []
        for response in responses:
            value = json.loads(response)
            if self.role == "planner":
                for hypothesis in value["hypotheses"]:
                    hypothesis["answer_signature"] = [
                        int(bit) for bit in hypothesis["answer_signature"]
                    ]
            else:
                value["answer_signature"] = [
                    int(bit) for bit in value["answer_signature"]
                ]
            converted.append(json.dumps(value))
        return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"

    if args.dry_run:
        planner: v1.StructuredModel = DeterministicFixtureModel("planner")
        target: v1.StructuredModel = DeterministicFixtureModel("target")
    else:
        planner = v1._adapter(
            model=v1.PLANNER_MODEL_ID,
            seed=PLANNER_SEED,
            run_id=f"{args.run_id}-planner",
            output_dir=args.output_dir,
        )
        target = v1._adapter(
            model=v1.TARGET_MODEL_ID,
            seed=TARGET_SEED,
            run_id=f"{args.run_id}-target",
            output_dir=args.output_dir,
        )
    try:
        payload = run_smoke(
            rows=v2_serving_rows(),
            planner_model=planner,
            target_model=target,
            raw_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
            "holdout_accessed": False,
            "policy_endpoint_accessed": False,
        }
        if isinstance(exc, v1.ServingExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        checkpoint(args.output_dir / "SERVING_FAILURE.json", failure)
        raise
    checkpoint(args.output_dir / "SERVING.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "metrics": payload["metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
