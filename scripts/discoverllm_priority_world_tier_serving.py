#!/usr/bin/env python3
"""Run the DiscoverLLM coarse ordinal-tier realistic serving gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts import discoverllm_priority_world_mechanics as cardinal
from scripts import discoverllm_priority_world_ordinal_serving as ordinal


INTERFACE_VERSION = "discoverllm-priority-world-tier-serving-1"
MODEL_ID = "openai/gpt-5.4"
SERVING_TASK_ID = "creative_writing:artifact_385"
RESERVED_MECHANICS_TASK_IDS = (
    "technical_writing:artifact_333",
    "creative_writing:artifact_367",
    "technical_writing:artifact_249",
)
WORLD_PRESENTATION_SEED = 24_415
TIER_LABELS = ("H", "M", "L")
TIER_WEIGHTS = {"H": 4.0, "M": 2.0, "L": 1.0}
SENSITIVITY_TIER_WEIGHTS = (
    {"H": 3.0, "M": 2.0, "L": 1.0},
    {"H": 9.0, "M": 3.0, "L": 1.0},
)
EXPECTED_REQUESTS = 5
MAX_TRANSPORT_RETRIES = 2
PROJECTED_COST_USD = 0.10
MAX_COST_USD = 0.15


def _world_presentation(
    task: cardinal.MechanicsTask,
    stage: str,
) -> dict[str, dict[str, Any]]:
    labels = list(cardinal.WORLD_LABELS)
    digest = hashlib.sha256(
        f"{WORLD_PRESENTATION_SEED}:{task.key}:{stage}".encode("utf-8")
    ).digest()[:8]
    random.Random(int.from_bytes(digest, "big")).shuffle(labels)
    by_label = dict(zip(cardinal.WORLD_LABELS, task.worlds))
    return {label: by_label[label] for label in labels}


def _tier_messages(
    stage: str,
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"STAGE={stage}. Act as an independent semantic likelihood "
                "assessor. For every requested action-observation branch, "
                "assign each candidate priority world one coarse likelihood "
                "tier: H=strongly compatible, M=plausible but ambiguous, or "
                "L=weak or inconsistent. Judge only from the complete visible "
                "branch history. Worlds may share a tier. World labels are "
                "arbitrary and their presentation order is shuffled. Return "
                "exactly eight lines and no other text. Each line must be "
                "KEY|W1:X,W2:X,W3:X,W4:X in that exact world order, where "
                "every X is H, M, or L."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        },
    ]


def _root_tier_messages(
    task: cardinal.MechanicsTask,
    observations: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    return _tier_messages(
        "ROOT_TIERS",
        {
            "conversation": task.conversation,
            "candidate_actions": dict(
                zip(cardinal.ACTION_LABELS, task.actions)
            ),
            "observed_user_feedback": observations,
            "candidate_priority_worlds": _world_presentation(
                task,
                "ROOT_TIERS",
            ),
            "output_keys": sorted(cardinal._branch_keys()),
        },
    )


def _followup_tier_messages(
    task: cardinal.MechanicsTask,
    observations: dict[str, dict[str, str]],
    followups: dict[str, str],
    second_observations: dict[str, str],
) -> list[dict[str, str]]:
    branches: dict[str, dict[str, Any]] = {}
    for action_index, action in enumerate(cardinal.ACTION_LABELS):
        for observation in cardinal.OBSERVATION_LABELS:
            key = f"{action}_{observation}"
            branches[key] = {
                "conversation": task.conversation,
                "root_assistant_response": task.actions[action_index],
                "prior_user_feedback": observations[action][observation],
                "assistant_followup": followups[key],
                "new_user_feedback": second_observations[key],
            }
    return _tier_messages(
        "FOLLOWUP_TIERS",
        {
            "branches": branches,
            "candidate_priority_worlds": _world_presentation(
                task,
                "FOLLOWUP_TIERS",
            ),
            "output_keys": sorted(cardinal._branch_keys()),
        },
    )


def _parse_tiers(
    text: str,
    expected_keys: set[str],
) -> dict[str, dict[str, str]]:
    lines = text.strip().splitlines()
    if len(lines) != len(expected_keys):
        raise ValueError("tier response has the wrong line count")
    parsed: dict[str, dict[str, str]] = {}
    for line in lines:
        if line.count("|") != 1:
            raise ValueError("tier line has invalid separators")
        key, assignments_text = line.split("|")
        if key not in expected_keys or key in parsed:
            raise ValueError("tier line has an invalid or duplicate key")
        assignments = assignments_text.split(",")
        if len(assignments) != len(cardinal.WORLD_LABELS):
            raise ValueError("tier line has the wrong assignment count")
        world_tiers: dict[str, str] = {}
        for expected_world, assignment in zip(
            cardinal.WORLD_LABELS,
            assignments,
        ):
            if assignment.count(":") != 1:
                raise ValueError("tier assignment has invalid separators")
            world, tier = assignment.split(":")
            if world != expected_world or tier not in TIER_LABELS:
                raise ValueError("tier assignment is invalid")
            world_tiers[world] = tier
        parsed[key] = world_tiers
    if set(parsed) != expected_keys:
        raise ValueError("tier response is missing keys")
    return parsed


def _load_task(
    paths: dict[str, Path],
    manifest_path: Path,
) -> cardinal.MechanicsTask:
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    development_ids = tuple(frozen["splits"]["development"]["artifact_ids"])
    expected_prefix = (
        ordinal.SERVING_TASK_ID,
        SERVING_TASK_ID,
        *RESERVED_MECHANICS_TASK_IDS,
    )
    if development_ids[: len(expected_prefix)] != expected_prefix:
        raise ValueError("DiscoverLLM tier task reservation changed")
    return ordinal._load_task_by_id(paths, manifest_path, SERVING_TASK_ID)


def run_serving(
    config: Config,
    *,
    paths: dict[str, Path],
    manifest_path: Path,
    raw_path: Path,
    model: ordinal.ChatModel,
) -> dict[str, Any]:
    del config
    task = _load_task(paths, manifest_path)
    stages = ordinal._run_five_stages(
        task,
        raw_path=raw_path,
        model=model,
        root_likelihood_messages=_root_tier_messages,
        followup_likelihood_messages=_followup_tier_messages,
        likelihood_parser=_parse_tiers,
    )
    usage = stages["usage"]
    gates = ordinal._serving_gates(usage)
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "reasoning_requested": False,
            "temperature": 0.0,
            "task_id": task.key,
            "reserved_mechanics_task_ids": list(
                RESERVED_MECHANICS_TASK_IDS
            ),
            "manifest_sha256": cardinal.MANIFEST_SHA256,
            "world_presentation_seed": WORLD_PRESENTATION_SEED,
            "tier_weights": TIER_WEIGHTS,
            "sensitivity_tier_weights": list(
                SENSITIVITY_TIER_WEIGHTS
            ),
            "expected_logical_requests": EXPECTED_REQUESTS,
            "maximum_transport_retries": MAX_TRANSPORT_RETRIES,
            "semantic_reissues_allowed": False,
            "released_scores_read": False,
            "released_winner_labels_read": False,
            "truth_map_exposed_to_likelihood_scorer": False,
            "truth_map_exposed_to_policy": False,
            "semantic_content_emitted": False,
        },
        "parse_counts": {
            "root_observations": len(stages["root_observations"]),
            "root_tiers": len(stages["root_likelihoods"]),
            "followups": len(stages["followups"]),
            "followup_observations": len(
                stages["followup_observations"]
            ),
            "followup_tiers": len(stages["followup_likelihoods"]),
        },
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel(ordinal.DeterministicFixtureModel):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        stage = self._stage(batch_messages[0])
        if stage not in {"ROOT_TIERS", "FOLLOWUP_TIERS"}:
            return super().chat_complete_messages_batched(
                batch_messages,
                temperature,
                block_size,
                max_new_tokens,
            )
        del temperature, block_size, max_new_tokens
        responses = []
        for messages in batch_messages:
            payload = json.loads(messages[-1]["content"])
            responses.append(
                "\n".join(
                    f"{key}|W1:H,W2:M,W3:L,W4:L"
                    for key in payload["output_keys"]
                )
            )
        self.requests += len(responses)
        return responses


def _build_model(config: Config) -> ordinal.ChatModel:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("DiscoverLLM tier serving config selects wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--creative-writing", type=Path, required=True)
    parser.add_argument("--technical-writing", type=Path, required=True)
    parser.add_argument("--svg-drawing", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 1
    config.openrouter_max_retries = MAX_TRANSPORT_RETRIES
    config.openrouter_max_output_tokens = 2_200
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ordinal.ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
        result = run_serving(
            config,
            paths={
                "creative_writing": args.creative_writing,
                "technical_writing": args.technical_writing,
                "svg_drawing": args.svg_drawing,
            },
            manifest_path=args.manifest,
            raw_path=raw_path,
            model=model,
        )
        result["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, ordinal.ServingExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        ordinal._checkpoint(args.output_dir / "SERVING_FAILURE.json", failure)
        raise
    output_path = args.output_dir / "SERVING.json"
    ordinal._checkpoint(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output_path),
                "parse_counts": result["parse_counts"],
                "gates": result["gates"],
                "usage": result["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
