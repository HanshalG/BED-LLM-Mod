#!/usr/bin/env python3
"""Test an unambiguous Ambig-IaC particle schema without scientific endpoints."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.ambig_iac_first_link_smoke import (
    MODEL_ID,
    TASK_IDS,
    parse_generated_spec,
    spec_features,
    verify_source,
)


INTERFACE_VERSION = "ambig-iac-schema-v2-smoke-1"
EXPECTED_REQUESTS = len(TASK_IDS)
MAX_COST_USD = 0.15
V2_KEYS = frozenset({"resources", "dependencies", "attribute_keys"})


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


def parse_generated_spec_v2(text: str) -> dict[str, dict[str, Any]]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict) or set(payload) != V2_KEYS:
        raise ValueError("response must have exactly the three V2 keys")
    resource_rows = payload["resources"]
    dependency_rows = payload["dependencies"]
    attribute_rows = payload["attribute_keys"]
    if not isinstance(resource_rows, list) or not resource_rows:
        raise ValueError("resources must be a nonempty list")
    if not isinstance(dependency_rows, list) or not isinstance(attribute_rows, list):
        raise ValueError("dependencies and attribute_keys must be lists")

    resources: dict[str, str] = {}
    for row in resource_rows:
        if not isinstance(row, dict) or set(row) != {"label", "address"}:
            raise ValueError("each resource must have exactly label and address")
        label = row["label"]
        address = row["address"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError("resource label must be a nonempty string")
        if label in resources:
            raise ValueError("resource labels must be unique")
        resources[label] = address

    topology: dict[str, list[str]] = {}
    for row in dependency_rows:
        if not isinstance(row, dict) or set(row) != {"source", "depends_on"}:
            raise ValueError(
                "each dependency must have exactly source and depends_on"
            )
        source = row["source"]
        dependency = row["depends_on"]
        if source not in resources or dependency not in resources:
            raise ValueError("dependency references an unknown resource label")
        if source == dependency:
            raise ValueError("a resource cannot depend on itself")
        topology.setdefault(source, []).append(dependency)

    attributes: dict[str, dict[str, bool]] = {}
    for row in attribute_rows:
        if not isinstance(row, dict) or set(row) != {"label", "keys"}:
            raise ValueError("each attribute row must have exactly label and keys")
        label = row["label"]
        keys = row["keys"]
        if label not in resources:
            raise ValueError("attribute row references an unknown resource label")
        if (
            not isinstance(keys, list)
            or not keys
            or any(not isinstance(key, str) or not key.strip() for key in keys)
        ):
            raise ValueError("attribute keys must be a nonempty string list")
        attributes.setdefault(label, {}).update(
            {key: True for key in dict.fromkeys(keys)}
        )

    canonical = {
        "resources": resources,
        "topology": {
            source: list(dict.fromkeys(dependencies))
            for source, dependencies in topology.items()
        },
        "attributes": attributes,
    }
    return parse_generated_spec(json.dumps(canonical))


def particle_messages_v2(
    *,
    task_id: int,
    prompt: str,
    sample_index: int,
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    request = {
        "benchmark_task_id": task_id,
        "ambiguous_request": prompt,
        "clarification_history": history,
        "interpretation_sample": sample_index,
    }
    return [
        {
            "role": "system",
            "content": (
                "Generate one concrete plausible AWS Terraform specification for "
                "the underspecified request. Treat clarification answers as binding. "
                "Return exactly one JSON object and no prose. The object must have "
                "exactly these keys: resources, dependencies, attribute_keys. "
                "resources is a nonempty array of objects with exactly two string "
                "fields, label and address. label is a short local identifier; "
                "address is a real aws_* or awscc_* Terraform address such as "
                "aws_vpc.main. dependencies is an array of objects with exactly "
                "source and depends_on, both referring to resource labels. "
                "attribute_keys is an array of objects with exactly label and keys, "
                "where keys is a nonempty array of explicit Terraform attribute-name "
                "strings. Empty dependencies or attribute_keys arrays are allowed. "
                "Never nest an object inside label, address, source, or depends_on."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _usage_snapshot(model: ChatModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "adapter_cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
        "generator": snapshot,
    }


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_schema_smoke(
    config: Config,
    *,
    source_root: Path,
    raw_path: Path,
    model_adapter: ChatModel,
) -> dict[str, Any]:
    dataset_root = verify_source(source_root)
    messages = [
        particle_messages_v2(
            task_id=task_id,
            prompt=(dataset_root / str(task_id) / "prompt.txt").read_text(
                encoding="utf-8"
            ),
            sample_index=0,
            history=[],
        )
        for task_id in TASK_IDS
    ]
    responses = model_adapter.chat_complete_messages_batched(
        messages,
        temperature=0.7,
        block_size=min(config.openrouter_concurrency, EXPECTED_REQUESTS),
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    _checkpoint(raw_path, {"task_ids": list(TASK_IDS), "responses": responses})
    records = []
    for task_id, response in zip(TASK_IDS, responses, strict=True):
        try:
            spec = parse_generated_spec_v2(response)
            features = spec_features(spec)
            records.append(
                {
                    "task_id": task_id,
                    "valid": True,
                    "error": None,
                    "resource_count": len(spec["resources"]),
                    "feature_count": len(features),
                    "feature_dimension_count": len(
                        {feature.dimension for feature in features}
                    ),
                }
            )
        except Exception as exc:
            records.append(
                {
                    "task_id": task_id,
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "resource_count": None,
                    "feature_count": None,
                    "feature_dimension_count": None,
                }
            )
    usage = _usage_snapshot(model_adapter)
    generator = usage["generator"]
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": int(generator.get("http_attempts", -1))
        == EXPECTED_REQUESTS,
        "zero_transport_retries": int(generator.get("retry_count", -1)) == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": int(generator.get("forced_exits", -1)) == 0,
        "all_three_responses_valid": all(record["valid"] for record in records),
        "all_valid_specs_have_features": all(
            not record["valid"] or int(record["feature_count"]) > 0
            for record in records
        ),
        "cost_at_most_0_15": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "task_ids": list(TASK_IDS),
            "expected_requests": EXPECTED_REQUESTS,
            "temperature": 0.7,
            "reasoning_requested": False,
            "target_plan_files_opened": False,
            "scientific_scores_or_endpoints_computed": False,
            "repairs_or_reissues": 0,
        },
        "records": records,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        self.requests += len(batch_messages)
        return [
            json.dumps(
                {
                    "resources": [
                        {"label": "network", "address": "aws_vpc.main"},
                        {"label": "subnet", "address": "aws_subnet.public"},
                    ],
                    "dependencies": [
                        {"source": "subnet", "depends_on": "network"}
                    ],
                    "attribute_keys": [
                        {"label": "network", "keys": ["cidr_block"]}
                    ],
                }
            )
            for _ in batch_messages
        ]

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
            "http_attempts": self.requests,
            "retry_count": 0,
            "forced_exits": 0,
        }


def _build_model(config: Config) -> ChatModel:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("Ambig-IaC V2 schema config selects the wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.03
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = EXPECTED_REQUESTS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
        payload = run_schema_smoke(
            config,
            source_root=args.source_root,
            raw_path=raw_path,
            model_adapter=model,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "SMOKE_FAILURE.json", failure)
        raise
    output = args.output_dir / "SMOKE.json"
    _checkpoint(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "records": payload["records"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
