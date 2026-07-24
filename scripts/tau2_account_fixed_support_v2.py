#!/usr/bin/env python3
"""Rank the Tau2 account prerequisite with a fixed BED support."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.tau2_account_prerequisite_ranking_gate import (
    ROOT_ACTIONS,
    exact_action_values,
    exact_observations,
    official_signature_coverage,
    parse_rollout,
    rollout_messages,
    selected_prompts,
    summarize,
)


SCHEMA_VERSION = 2
SELECTION_SEED = 24319

FIXED_HYPOTHESES = (
    {
        "id": "h1",
        "description": (
            "Data allowance available; account roaming enabled; device roaming off"
        ),
        "data_allowance": "available",
        "account_roaming": "enabled",
        "device_roaming": "off",
    },
    {
        "id": "h2",
        "description": (
            "Data allowance available; account roaming disabled; device roaming on"
        ),
        "data_allowance": "available",
        "account_roaming": "disabled",
        "device_roaming": "on",
    },
    {
        "id": "h3",
        "description": (
            "Data allowance available; account roaming disabled; device roaming off"
        ),
        "data_allowance": "available",
        "account_roaming": "disabled",
        "device_roaming": "off",
    },
    {
        "id": "h4",
        "description": (
            "Data allowance exhausted; account roaming enabled; device roaming off"
        ),
        "data_allowance": "exhausted",
        "account_roaming": "enabled",
        "device_roaming": "off",
    },
    {
        "id": "h5",
        "description": (
            "Data allowance exhausted; account roaming disabled; device roaming on"
        ),
        "data_allowance": "exhausted",
        "account_roaming": "disabled",
        "device_roaming": "on",
    },
    {
        "id": "h6",
        "description": (
            "Data allowance exhausted; account roaming disabled; device roaming off"
        ),
        "data_allowance": "exhausted",
        "account_roaming": "disabled",
        "device_roaming": "off",
    },
)


def _usage(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot["adapter_requests"]),
        "reasoning_tokens": int(snapshot["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(snapshot["adapter_cost_usd"]),
        "model": snapshot,
    }


def _write_raw(path: Path | None, stage: str, raw: list[str]) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "stage": stage, "responses": raw},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_gate(
    config: Config,
    *,
    t3_dir: str | Path,
    stage: str,
    raw_checkpoint_path: Path | None = None,
    prompt_variants: Sequence[str] | None = None,
    fixed_hypotheses: Sequence[dict[str, str]] = FIXED_HYPOTHESES,
    observation_world_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    prompts = tuple(prompt_variants or selected_prompts(stage))
    if len(config.model_pairs) != 1:
        raise ValueError("Tau2 account V2 requires exactly one model pair")
    model = build_model_adapter(config.model_pairs[0].questioner, config)

    keys: list[tuple[int, str]] = []
    messages: list[list[dict[str, str]]] = []
    for prompt_index, prompt in enumerate(prompts):
        for action in ROOT_ACTIONS:
            keys.append((prompt_index, action))
            messages.append(
                rollout_messages(fixed_hypotheses, action, prompt)
            )
    raw = model.chat_complete_messages_batched(
        messages,
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    _write_raw(raw_checkpoint_path, stage, raw)
    if len(raw) != len(keys):
        raise ValueError("Tau2 account V2 response count changed")

    parsed: dict[tuple[int, str], dict[str, Any]] = {}
    for response, (prompt_index, action) in zip(raw, keys, strict=True):
        parsed[prompt_index, action] = parse_rollout(
            response,
            hypotheses=fixed_hypotheses,
            root_action=action,
        )

    observations = exact_observations(Path(t3_dir))
    if observation_world_ids is not None:
        observations = {
            world_id: observations[world_id]
            for world_id in observation_world_ids
        }
    exact_values = exact_action_values(observations)
    records: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        actions = []
        actions_by_id = {}
        for action in ROOT_ACTIONS:
            row = {
                "action_id": action,
                **parsed[prompt_index, action],
                **exact_values[action],
            }
            actions.append(row)
            actions_by_id[action] = row
        records.append(
            {
                "prompt_variant": prompt_index,
                "ticket": prompt,
                "hypotheses": list(fixed_hypotheses),
                "official_signature_coverage": official_signature_coverage(
                    fixed_hypotheses
                ),
                "actions": actions,
                "actions_by_id": actions_by_id,
            }
        )

    usage = _usage(model)
    summary = summarize(records, usage, stage=stage)
    expected_requests = len(prompts) * len(ROOT_ACTIONS)
    summary["gates"]["exact_physical_request_count"] = (
        int(usage["physical_requests"]) == expected_requests
    )
    summary["gates"]["all_pass"] = all(
        summary["gates"].values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "selection_seed": SELECTION_SEED,
            "fixed_support": True,
            "support_size": len(fixed_hypotheses),
            "standard_bed_support_visible_to_policy": True,
            "official_simulator_outputs_hidden_until_scoring": True,
            "llm_predicts_two_step_likelihood_partitions": True,
            "llm_receives_no_eig_or_entropy_values": True,
            "prompt_variants_share_one_physical_support_family": True,
            "root_actions": list(ROOT_ACTIONS),
            "no_reasoning": True,
            "expected_physical_requests": expected_requests,
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--t3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "formal"),
        required=True,
    )
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.05
        config.openrouter_run_budget_usd = 0.50
    else:
        config.openrouter_projected_cost_usd = 0.20
        config.openrouter_run_budget_usd = 2.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json"
        if args.stage == "serving_smoke"
        else "GATE.json"
    )
    failure_name = (
        "SERVING_SMOKE_FAILURE.json"
        if args.stage == "serving_smoke"
        else "GATE_FAILURE.json"
    )
    try:
        payload = run_gate(
            config,
            t3_dir=args.t3_dir,
            stage=args.stage,
            raw_checkpoint_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_responses_path": str(raw_path),
        }
        (args.output_dir / failure_name).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": payload["status"], **payload["summary"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
