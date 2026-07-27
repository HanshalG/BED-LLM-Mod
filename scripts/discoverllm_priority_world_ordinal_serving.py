#!/usr/bin/env python3
"""Run the DiscoverLLM ordinal-likelihood realistic serving gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Callable, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts import discoverllm_priority_world_manifest as manifest_v1
from scripts import discoverllm_priority_world_manifest_v2 as manifest_v2
from scripts import discoverllm_priority_world_mechanics as cardinal


INTERFACE_VERSION = "discoverllm-priority-world-ordinal-serving-1"
MODEL_ID = "openai/gpt-5.4"
SERVING_TASK_ID = "svg_drawing:artifact_347"
RESERVED_MECHANICS_TASK_IDS = (
    "creative_writing:artifact_385",
    "technical_writing:artifact_333",
    "creative_writing:artifact_367",
)
WORLD_PRESENTATION_SEED = 24_414
EXPECTED_REQUESTS = 5
MAX_TRANSPORT_RETRIES = 2
PROJECTED_COST_USD = 0.10
MAX_COST_USD = 0.15


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class ServingExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _checkpoint(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_task(
    paths: dict[str, Path],
    manifest_path: Path,
) -> cardinal.MechanicsTask:
    if manifest_v1.sha256_file(manifest_path) != cardinal.MANIFEST_SHA256:
        raise ValueError("DiscoverLLM priority-world manifest hash changed")
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    development_ids = tuple(frozen["splits"]["development"]["artifact_ids"])
    expected_prefix = (SERVING_TASK_ID, *RESERVED_MECHANICS_TASK_IDS)
    if development_ids[: len(expected_prefix)] != expected_prefix:
        raise ValueError("DiscoverLLM ordinal task reservation changed")
    return _load_task_by_id(paths, manifest_path, SERVING_TASK_ID)


def _load_task_by_id(
    paths: dict[str, Path],
    manifest_path: Path,
    task_id: str,
) -> cardinal.MechanicsTask:
    if manifest_v1.sha256_file(manifest_path) != cardinal.MANIFEST_SHA256:
        raise ValueError("DiscoverLLM priority-world manifest hash changed")
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    development_ids = set(frozen["splits"]["development"]["artifact_ids"])
    if task_id not in development_ids:
        raise ValueError("DiscoverLLM task is outside the development split")
    for domain, expected_sha256 in manifest_v2.SOURCE_SHA256.items():
        if manifest_v1.sha256_file(paths[domain]) != expected_sha256:
            raise ValueError(f"DiscoverLLM {domain} Parquet hash changed")

    domain, artifact_id = task_id.split(":", 1)
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "DiscoverLLM ordinal serving requires pyarrow"
        ) from exc
    table = parquet.read_table(
        paths[domain],
        columns=[
            "artifact_id",
            "turn_id",
            "assistant_index",
            "prompt",
            "completion",
            "criteria_history",
        ],
        filters=[("artifact_id", "=", artifact_id)],
    )
    rows = table.to_pylist()
    if not rows:
        raise ValueError("ordinal serving artifact is absent")
    earliest_turn = min(int(row["turn_id"]) for row in rows)
    candidates = [
        {
            **row,
            "artifact_id": str(row["artifact_id"]),
            "assistant_index": int(row["assistant_index"]),
            "criteria_history": (
                json.loads(row["criteria_history"])
                if isinstance(row["criteria_history"], str)
                else row["criteria_history"]
            ),
        }
        for row in rows
        if int(row["turn_id"]) == earliest_turn
    ]
    candidates.sort(key=lambda row: row["assistant_index"])
    if len(candidates) != 2:
        raise ValueError("ordinal serving task no longer has two actions")
    histories = [candidate["criteria_history"] for candidate in candidates]
    if not histories[0] or histories[0] != histories[1]:
        raise ValueError("ordinal serving histories changed")
    roots = manifest_v1.eligible_world_roots(histories[0][-1])
    selected_ids = manifest_v2._world_ids(domain, artifact_id, roots)
    roots_by_id = {str(root["id"]): root for root in roots}
    worlds = tuple(
        cardinal._semantic_tree(roots_by_id[root_id])
        for root_id in selected_ids
    )
    conversation = tuple(
        {
            "role": str(message["role"]),
            "content": str(message["content"]),
        }
        for message in candidates[0]["prompt"]
    )
    return cardinal.MechanicsTask(
        key=task_id,
        conversation=conversation,
        actions=tuple(candidate["completion"] for candidate in candidates),
        worlds=worlds,
    )


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


def _ranking_messages(
    stage: str,
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"STAGE={stage}. Act as an independent semantic likelihood "
                "assessor. For each requested action-observation branch, rank "
                "all four candidate priority worlds from most to least likely. "
                "Judge only from the complete visible branch history. World "
                "labels are arbitrary and their presentation order is shuffled. "
                "Return exactly eight lines and no other text. Each line must "
                "be KEY|Wn>Wn>Wn>Wn, with every world appearing exactly once."
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


def _root_ranking_messages(
    task: cardinal.MechanicsTask,
    observations: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    return _ranking_messages(
        "ROOT_RANKINGS",
        {
            "conversation": task.conversation,
            "candidate_actions": dict(
                zip(cardinal.ACTION_LABELS, task.actions)
            ),
            "observed_user_feedback": observations,
            "candidate_priority_worlds": _world_presentation(
                task,
                "ROOT_RANKINGS",
            ),
            "output_keys": sorted(cardinal._branch_keys()),
        },
    )


def _followup_ranking_messages(
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
    return _ranking_messages(
        "FOLLOWUP_RANKINGS",
        {
            "branches": branches,
            "candidate_priority_worlds": _world_presentation(
                task,
                "FOLLOWUP_RANKINGS",
            ),
            "output_keys": sorted(cardinal._branch_keys()),
        },
    )


def _parse_rankings(
    text: str,
    expected_keys: set[str],
) -> dict[str, tuple[str, ...]]:
    lines = text.strip().splitlines()
    if len(lines) != len(expected_keys):
        raise ValueError("ranking response has the wrong line count")
    parsed: dict[str, tuple[str, ...]] = {}
    expected_worlds = set(cardinal.WORLD_LABELS)
    for line in lines:
        if line.count("|") != 1:
            raise ValueError("ranking line has invalid separators")
        key, ordered_text = line.split("|")
        if key not in expected_keys or key in parsed:
            raise ValueError("ranking line has an invalid or duplicate key")
        ordered = tuple(ordered_text.split(">"))
        if len(ordered) != len(cardinal.WORLD_LABELS):
            raise ValueError("ranking line has the wrong world count")
        if set(ordered) != expected_worlds:
            raise ValueError("ranking line is not a world permutation")
        parsed[key] = ordered
    if set(parsed) != expected_keys:
        raise ValueError("ranking response is missing keys")
    return parsed


def _serving_gates(usage: dict[str, Any]) -> dict[str, bool]:
    requests = int(usage.get("adapter_requests", -1))
    attempts = int(usage.get("http_attempts", -1))
    retries = int(usage.get("retry_count", -1))
    gates = {
        "exact_5_logical_requests": requests == EXPECTED_REQUESTS,
        "bounded_transport_retries": 0 <= retries <= MAX_TRANSPORT_RETRIES,
        "http_attempts_match_requests_plus_retries": (
            attempts == requests + retries
        ),
        "zero_reasoning_tokens": int(
            usage.get("adapter_reasoning_tokens", 0)
        )
        == 0,
        "zero_forced_exits": int(usage.get("forced_exits", 0)) == 0,
        "cost_at_most_0_15": float(
            usage.get("adapter_cost_usd", float("inf"))
        )
        <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def _run_five_stages(
    task: cardinal.MechanicsTask,
    *,
    raw_path: Path,
    model: ChatModel,
    root_likelihood_messages: Callable[
        [cardinal.MechanicsTask, dict[str, dict[str, str]]],
        list[dict[str, str]],
    ],
    followup_likelihood_messages: Callable[
        [
            cardinal.MechanicsTask,
            dict[str, dict[str, str]],
            dict[str, str],
            dict[str, str],
        ],
        list[dict[str, str]],
    ],
    likelihood_parser: Callable[
        [str, set[str]],
        dict[str, Any],
    ],
) -> dict[str, Any]:
    raw: dict[str, Any] = {"task_id": task.key}
    try:
        root_raw = model.chat_complete_messages_batched(
            [cardinal._root_observation_messages(task)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=2_200,
        )
        raw["root_observations"] = root_raw
        _checkpoint(raw_path, raw)
        root_output = cardinal._parse_text_object(
            root_raw[0],
            cardinal._root_observation_keys(),
            maximum_length=800,
        )
        observations, mapping = cardinal._shuffle_root_observations(
            task,
            root_output,
        )

        root_likelihood_raw = model.chat_complete_messages_batched(
            [root_likelihood_messages(task, observations)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=320,
        )
        raw["root_likelihoods"] = root_likelihood_raw
        _checkpoint(raw_path, raw)
        root_likelihoods = likelihood_parser(
            root_likelihood_raw[0],
            cardinal._branch_keys(),
        )

        followup_raw = model.chat_complete_messages_batched(
            [cardinal._followup_messages(task, observations)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=2_200,
        )
        raw["followups"] = followup_raw
        _checkpoint(raw_path, raw)
        followups = cardinal._parse_text_object(
            followup_raw[0],
            cardinal._branch_keys(),
            maximum_length=1_000,
        )

        second_raw = model.chat_complete_messages_batched(
            [
                cardinal._followup_observation_messages(
                    task,
                    observations,
                    mapping,
                    followups,
                )
            ],
            temperature=0.0,
            block_size=1,
            max_new_tokens=2_200,
        )
        raw["followup_observations"] = second_raw
        _checkpoint(raw_path, raw)
        second_observations = cardinal._parse_text_object(
            second_raw[0],
            cardinal._branch_keys(),
            maximum_length=800,
        )

        followup_likelihood_raw = model.chat_complete_messages_batched(
            [
                followup_likelihood_messages(
                    task,
                    observations,
                    followups,
                    second_observations,
                )
            ],
            temperature=0.0,
            block_size=1,
            max_new_tokens=320,
        )
        raw["followup_likelihoods"] = followup_likelihood_raw
        _checkpoint(raw_path, raw)
        followup_likelihoods = likelihood_parser(
            followup_likelihood_raw[0],
            cardinal._branch_keys(),
        )
        usage = model.usage_snapshot()
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}",
            model.usage_snapshot(),
        ) from exc
    return {
        "root_observations": root_output,
        "root_likelihoods": root_likelihoods,
        "followups": followups,
        "followup_observations": second_observations,
        "followup_likelihoods": followup_likelihoods,
        "usage": usage,
    }


def run_serving(
    config: Config,
    *,
    paths: dict[str, Path],
    manifest_path: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    del config
    task = _load_task(paths, manifest_path)
    stages = _run_five_stages(
        task,
        raw_path=raw_path,
        model=model,
        root_likelihood_messages=_root_ranking_messages,
        followup_likelihood_messages=_followup_ranking_messages,
        likelihood_parser=_parse_rankings,
    )

    usage = stages["usage"]
    gates = _serving_gates(usage)
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
            "root_rankings": len(stages["root_likelihoods"]),
            "followups": len(stages["followups"]),
            "followup_observations": len(
                stages["followup_observations"]
            ),
            "followup_rankings": len(stages["followup_likelihoods"]),
        },
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self) -> None:
        self.requests = 0

    @staticmethod
    def _stage(messages: list[dict[str, str]]) -> str:
        return messages[0]["content"].split("STAGE=", 1)[1].split(".", 1)[0]

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses: list[str] = []
        for messages in batch_messages:
            stage = self._stage(messages)
            payload = json.loads(messages[-1]["content"])
            if stage == "ROOT_OBSERVATIONS":
                value = {
                    key: f"Feedback grounded in {key.split('_')[1]}"
                    for key in payload["output_keys"]
                }
                response = json.dumps(value, separators=(",", ":"))
            elif stage in {"ROOT_RANKINGS", "FOLLOWUP_RANKINGS"}:
                response = "\n".join(
                    f"{key}|W1>W2>W3>W4"
                    for key in payload["output_keys"]
                )
            elif stage == "FOLLOWUP_POLICY":
                value = {
                    key: f"Diagnostic continuation for {key}?"
                    for key in payload["output_keys"]
                }
                response = json.dumps(value, separators=(",", ":"))
            elif stage == "FOLLOWUP_OBSERVATIONS":
                value = {
                    key: (
                        "Followup grounded in "
                        f"{payload['branches'][key]['private_priority_label']}"
                    )
                    for key in payload["output_keys"]
                }
                response = json.dumps(value, separators=(",", ":"))
            else:  # pragma: no cover
                raise AssertionError(stage)
            responses.append(response)
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
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
        raise ValueError("DiscoverLLM ordinal serving config selects wrong model")
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
    model: ChatModel = (
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
        if isinstance(exc, ServingExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "SERVING_FAILURE.json", failure)
        raise
    output_path = args.output_dir / "SERVING.json"
    _checkpoint(output_path, result)
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
