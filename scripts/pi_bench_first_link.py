#!/usr/bin/env python3
"""Run the preregistered Pi-Bench dynamic-support first-link experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Protocol, Sequence

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from openrouter_model import OpenRouterAdapter


SOURCE_COMMIT = "383910b1698758a198b86037c63a111c8edc32ad"
SOURCE_TREE = "a90d1c76b05c1c3651cb3156bc0f80b21490d751"
MANIFEST_SHA256 = (
    "ccdf9211016d6c77eefc6cb3aae4e0324640c252b9d3aa17551ad158fc61594e"
)
SCHEMA_VERSION = 1
INTERFACE_VERSION = "pi_bench_dynamic_support_v1"
POLICY_SEED = 24422

INITIAL_WORLD_COUNT = 8
INITIAL_QUESTION_COUNT = 6
ROLLOUT_WORLD_COUNT = 4
REFRESH_WORLD_COUNT = 4
REFRESH_QUESTION_COUNT = 4
ACTUAL_REFRESH_WORLD_COUNT = 8
ACTUAL_REFRESH_QUESTION_COUNT = 6
MIN_INITIAL_REQUIREMENTS = 3
MAX_REQUIREMENTS = 7

SERVING_TASK_IDS = (
    "Financier_task_001",
    "law_trainee_task_001",
    "marketer_task_001",
    "pharmacist_task_001",
    "researcher_task_001",
)
POLICIES = ("myopic", "depth2", "random", "naive_thinking")
BED_POLICIES = ("myopic", "depth2", "random")

GENERIC_QUESTION_PATTERNS = (
    re.compile(r"\banything else\b", re.IGNORECASE),
    re.compile(r"\bany other\b", re.IGNORECASE),
    re.compile(r"\beverything (?:you )?(?:need|want)\b", re.IGNORECASE),
    re.compile(r"\blist (?:all|every)\b", re.IGNORECASE),
    re.compile(r"\ball (?:your|the) requirements\b", re.IGNORECASE),
    re.compile(r"\btell me everything\b", re.IGNORECASE),
)


class StructuredModel(Protocol):
    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class PiBenchGPT54Adapter(OpenRouterAdapter):
    """Use only GPT-5.4 parameters advertised by OpenRouter routing."""

    def _payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        *,
        disable_reasoning: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages,
            temperature,
            n,
            max_tokens,
            disable_reasoning=disable_reasoning,
            response_format=response_format,
        )
        for unsupported in ("temperature", "top_p", "top_k", "n"):
            payload.pop(unsupported, None)
        if disable_reasoning or not self.reasoning_enabled:
            payload["reasoning"] = {"enabled": False, "exclude": True}
        return payload


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    persona: str
    persona_context: str
    initial_input: str


@dataclass(frozen=True)
class PrivateTask:
    task_id: str
    hidden_intents: tuple[str, ...]
    initial_statuses: tuple[str, ...]


@dataclass(frozen=True)
class RequirementWorld:
    requirements: tuple[str, ...]


@dataclass(frozen=True)
class BeliefAndQuestions:
    worlds: tuple[RequirementWorld, ...]
    questions: tuple[str, ...]


@dataclass(frozen=True)
class SemanticMap:
    """Question-major then world-major matched requirement indexes."""

    matches: tuple[tuple[int, ...], ...]
    question_count: int
    world_count: int

    def for_pair(self, question_index: int, world_index: int) -> tuple[int, ...]:
        offset = question_index * self.world_count + world_index
        return self.matches[offset]


@dataclass(frozen=True)
class RolloutBranch:
    branch_index: int
    root_question_index: int
    particle_world_index: int
    root_resolved_indexes: tuple[int, ...]
    simulated_response: str
    remaining_particle_requirements: tuple[str, ...]


@dataclass(frozen=True)
class BranchRefresh:
    worlds: tuple[RequirementWorld, ...]
    questions: tuple[str, ...]


@dataclass(frozen=True)
class BranchSemanticMap:
    support_map: SemanticMap
    particle_matches: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RootScores:
    immediate: tuple[float, ...]
    terminal: tuple[float, ...]
    rollout_world_indexes: tuple[int, ...]


@dataclass
class Trajectory:
    task_id: str
    policy: str
    statuses: list[str]
    questions: list[str]
    replies: list[str]
    selected_question_ids: list[str]
    provided_by_turn: list[list[int]]
    inferred_by_turn: list[list[int]]
    refreshed_belief: BeliefAndQuestions | None = None
    refreshed_truth_recall: float | None = None


@dataclass(frozen=True)
class TurnCase:
    case_id: str
    task_id: str
    hidden_intents: tuple[str, ...]
    statuses: tuple[str, ...]
    question: str


@dataclass(frozen=True)
class TurnResult:
    statuses: tuple[str, ...]
    reply: str
    provided_indexes: tuple[int, ...]
    inferred_indexes: tuple[int, ...]
    satisfaction_raw: str
    targeted_raw: str | None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_source(repo: Path, manifest_path: Path) -> dict[str, Any]:
    if _git_output(repo, "rev-parse", "HEAD") != SOURCE_COMMIT:
        raise ValueError("Pi-Bench source commit does not match preregistration")
    if _git_output(repo, "rev-parse", "HEAD^{tree}") != SOURCE_TREE:
        raise ValueError("Pi-Bench source tree does not match preregistration")
    if _git_output(repo, "status", "--short"):
        raise ValueError("Pi-Bench source checkout is not clean")
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Pi-Bench source manifest does not match preregistration")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("commit") != SOURCE_COMMIT:
        raise ValueError("manifest source commit is inconsistent")
    return payload


def _persona_context(profile: Mapping[str, Any]) -> str:
    role = profile.get("role") or ""
    if isinstance(role, Mapping):
        role = role.get("role_text", "")
    payload = {
        "role": str(role),
        "preferences": profile.get("preferences") or {},
        "long_term_goals": profile.get("long_term_goals") or [],
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def load_tasks(
    repo: Path,
    manifest: Mapping[str, Any],
    *,
    stage: str,
) -> tuple[list[PublicTask], dict[str, PrivateTask], dict[str, Any]]:
    if stage == "serving_smoke":
        partition = "mechanics"
        allowed_ids = set(SERVING_TASK_IDS)
    elif stage == "development":
        partition = "development"
        allowed_ids = None
    elif stage == "confirmation":
        partition = "confirmation"
        allowed_ids = None
    else:
        raise ValueError("stage must be serving_smoke, development, or confirmation")

    public_tasks: list[PublicTask] = []
    private_tasks: dict[str, PrivateTask] = {}
    excluded: list[dict[str, Any]] = []
    profiles: dict[str, Mapping[str, Any]] = {}
    for row in manifest["partitions"][partition]:
        task_id = str(row["task_id"])
        if allowed_ids is not None and task_id not in allowed_ids:
            continue
        task_path = repo / str(row["task_path"])
        task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        intents = task["intent"]["hidden_intent"]
        hidden_intents = tuple(str(item["content"]).strip() for item in intents)
        statuses = tuple(
            str(item.get("status", "not_provided")).strip() for item in intents
        )
        not_provided_count = sum(status == "not_provided" for status in statuses)
        if stage != "serving_smoke" and not_provided_count < 3:
            excluded.append(
                {
                    "task_id": task_id,
                    "reason": "fewer_than_three_initially_not_provided",
                    "count": not_provided_count,
                }
            )
            continue
        user_dir = task_path.parents[2]
        profile_key = str(user_dir)
        if profile_key not in profiles:
            profiles[profile_key] = yaml.safe_load(
                (user_dir / "profile.yaml").read_text(encoding="utf-8")
            )
        public_tasks.append(
            PublicTask(
                task_id=task_id,
                persona=str(row["persona"]),
                persona_context=_persona_context(profiles[profile_key]),
                initial_input=str(task["intent"]["initial_input"]),
            )
        )
        private_tasks[task_id] = PrivateTask(
            task_id=task_id,
            hidden_intents=hidden_intents,
            initial_statuses=statuses,
        )

    if stage == "serving_smoke":
        actual_ids = tuple(task.task_id for task in public_tasks)
        if actual_ids != SERVING_TASK_IDS:
            raise ValueError(
                f"serving task order mismatch: expected {SERVING_TASK_IDS}, "
                f"found {actual_ids}"
            )
    elif len(public_tasks) < 20:
        raise ValueError(
            f"{stage} opportunity gate requires at least 20 eligible tasks; "
            f"found {len(public_tasks)}"
        )
    return public_tasks, private_tasks, {
        "partition": partition,
        "included_task_ids": [task.task_id for task in public_tasks],
        "excluded": excluded,
    }


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _dedupe_key(value: str) -> str:
    return _normalize_text(value).casefold().rstrip("?.!")


def invalid_question_reason(question: str) -> str | None:
    normalized = _normalize_text(question)
    if not normalized:
        return "empty"
    if not normalized.endswith("?"):
        return "not_a_question"
    if len(normalized) > 320:
        return "too_long"
    for pattern in GENERIC_QUESTION_PATTERNS:
        if pattern.search(normalized):
            return "generic_or_omnibus"
    if normalized.count("?") > 2:
        return "too_many_questions"
    return None


def _strict_json_schema(
    name: str,
    properties: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": dict(properties),
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def _text_schema(*, maximum: int = 320) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "maxLength": maximum}


def _world_schema(*, min_requirements: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "requirements": {
                "type": "array",
                "items": _text_schema(),
                "minItems": min_requirements,
                "maxItems": MAX_REQUIREMENTS,
            }
        },
        "required": ["requirements"],
        "additionalProperties": False,
    }


def belief_response_format(
    *,
    name: str,
    world_count: int,
    question_count: int,
    min_requirements: int,
) -> dict[str, Any]:
    return _strict_json_schema(
        name,
        {
            "worlds": {
                "type": "array",
                "items": _world_schema(min_requirements=min_requirements),
                "minItems": world_count,
                "maxItems": world_count,
            },
            "questions": {
                "type": "array",
                "items": _text_schema(),
                "minItems": question_count,
                "maxItems": question_count,
            },
        },
    )


def semantic_map_response_format(
    *,
    name: str,
    pair_count: int,
) -> dict[str, Any]:
    return _strict_json_schema(
        name,
        {
            "matches": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "boolean"},
                    "minItems": 1,
                    "maxItems": MAX_REQUIREMENTS,
                },
                "minItems": pair_count,
                "maxItems": pair_count,
            }
        },
    )


def branch_refresh_response_format(
    *, name: str, branch_count: int
) -> dict[str, Any]:
    branch_schema = {
        "type": "object",
        "properties": {
            "worlds": {
                "type": "array",
                "items": _world_schema(min_requirements=1),
                "minItems": REFRESH_WORLD_COUNT,
                "maxItems": REFRESH_WORLD_COUNT,
            },
            "questions": {
                "type": "array",
                "items": _text_schema(),
                "minItems": REFRESH_QUESTION_COUNT,
                "maxItems": REFRESH_QUESTION_COUNT,
            },
        },
        "required": ["worlds", "questions"],
        "additionalProperties": False,
    }
    return _strict_json_schema(
        name,
        {
            "branches": {
                "type": "array",
                "items": branch_schema,
                "minItems": branch_count,
                "maxItems": branch_count,
            }
        },
    )


def branch_map_response_format(
    *, name: str, branch_count: int
) -> dict[str, Any]:
    branch_schema = {
        "type": "object",
        "properties": {
            "support_matches": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "boolean"},
                    "minItems": 1,
                    "maxItems": MAX_REQUIREMENTS,
                },
                "minItems": REFRESH_QUESTION_COUNT * REFRESH_WORLD_COUNT,
                "maxItems": REFRESH_QUESTION_COUNT * REFRESH_WORLD_COUNT,
            },
            "particle_matches": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "boolean"},
                    "minItems": 1,
                    "maxItems": MAX_REQUIREMENTS,
                },
                "minItems": REFRESH_QUESTION_COUNT,
                "maxItems": REFRESH_QUESTION_COUNT,
            },
        },
        "required": ["support_matches", "particle_matches"],
        "additionalProperties": False,
    }
    return _strict_json_schema(
        name,
        {
            "branches": {
                "type": "array",
                "items": branch_schema,
                "minItems": branch_count,
                "maxItems": branch_count,
            }
        },
    )


def boolean_decisions_response_format(name: str) -> dict[str, Any]:
    return _strict_json_schema(
        name,
        {
            "decisions": {
                "type": "array",
                "items": {"type": "boolean"},
                "minItems": 1,
                "maxItems": 32,
            }
        },
    )


def recall_response_format(name: str) -> dict[str, Any]:
    return _strict_json_schema(
        name,
        {
            "represented": {
                "type": "array",
                "items": {"type": "boolean"},
                "minItems": 1,
                "maxItems": 32,
            }
        },
    )


def naive_question_response_format() -> dict[str, Any]:
    return _strict_json_schema(
        "pi_bench_naive_question_v1",
        {"question": _text_schema()},
    )


def _messages(stage: str, instruction: str, payload: Any) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"STAGE={stage}. {instruction} Return only the requested strict "
                "JSON object. Do not mention hidden labels, answer keys, or this "
                "evaluation."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ),
        },
    ]


def initial_belief_messages(task: PublicTask) -> list[dict[str, str]]:
    return _messages(
        "initial_dynamic_support",
        (
            "Infer a diverse prior over coherent sets of requirements the person may "
            "have omitted. Generate exactly 8 worlds, each with 3-7 concise, "
            "non-overlapping requirements, and exactly 6 distinct clarification "
            "questions. Questions may target a coherent requirement group but must "
            "not ask for everything, use a generic anything-else prompt, or expose a "
            "checklist. Do not assert that any guessed requirement is true."
        ),
        {
            "persona_context": task.persona_context,
            "visible_initial_input": task.initial_input,
            "required_output": {
                "worlds": "8 ordered objects with requirements",
                "questions": "6 ordered question strings",
            },
        },
    )


def semantic_map_messages(
    task: PublicTask,
    belief: BeliefAndQuestions,
) -> list[dict[str, str]]:
    pairs = []
    for question_index, question in enumerate(belief.questions):
        for world_index, world in enumerate(belief.worlds):
            pairs.append(
                {
                    "question_index": question_index,
                    "world_index": world_index,
                    "question": question,
                    "requirements": list(world.requirements),
                }
            )
    return _messages(
        "initial_semantic_map",
        (
            "For every ordered pair, mark which requirements the question clearly "
            "and specifically asks about. Broad topical overlap is false. A close "
            "confirmation or short explicit options question is true. Return one "
            "boolean vector per pair in the supplied order; each vector must have "
            "exactly one entry per requirement."
        ),
        {
            "visible_initial_input": task.initial_input,
            "ordered_pairs": pairs,
            "required_output": "matches: ordered boolean vectors",
        },
    )


def branch_refresh_messages(
    task: PublicTask,
    branches: Sequence[RolloutBranch],
    initial_questions: Sequence[str],
) -> list[dict[str, str]]:
    payload_branches = [
        {
            "branch_index": branch.branch_index,
            "history": [
                {"role": "user", "content": task.initial_input},
                {
                    "role": "assistant",
                    "content": initial_questions[branch.root_question_index],
                },
                {"role": "user", "content": branch.simulated_response},
            ],
        }
        for branch in branches
    ]
    return _messages(
        "rollout_support_regeneration",
        (
            "Treat every branch independently. From only that branch's visible "
            "history and persona context, discard any earlier latent support and "
            "regenerate exactly 4 coherent worlds of still-unresolved requirements, "
            "with 1-7 requirements per world. Then generate exactly 4 distinct "
            "follow-up questions from the regenerated support. Do not repeat already "
            "answered requirements, ask for everything, or use a generic prompt. "
            "Return branches in input order."
        ),
        {
            "persona_context": task.persona_context,
            "branches": payload_branches,
            "required_output": "one worlds/questions object per branch",
        },
    )


def branch_map_messages(
    task: PublicTask,
    branches: Sequence[RolloutBranch],
    refreshes: Sequence[BranchRefresh],
) -> list[dict[str, str]]:
    payload_branches = []
    for branch, refresh in zip(branches, refreshes, strict=True):
        support_pairs = []
        for question_index, question in enumerate(refresh.questions):
            for world_index, world in enumerate(refresh.worlds):
                support_pairs.append(
                    {
                        "question_index": question_index,
                        "world_index": world_index,
                        "question": question,
                        "requirements": list(world.requirements),
                    }
                )
        particle_pairs = [
            {
                "question_index": question_index,
                "question": question,
                "requirements": list(branch.remaining_particle_requirements),
            }
            for question_index, question in enumerate(refresh.questions)
        ]
        payload_branches.append(
            {
                "branch_index": branch.branch_index,
                "support_pairs": support_pairs,
                "particle_pairs": particle_pairs,
            }
        )
    return _messages(
        "rollout_followup_semantic_map",
        (
            "For each branch independently, mark which requirements each follow-up "
            "question clearly and specifically asks about. Broad topical overlap is "
            "false. Return support_matches in question-major/world-major order and "
            "particle_matches in question order. Every boolean vector must exactly "
            "match its listed requirement count. Do not compare across branches."
        ),
        {
            "visible_initial_input": task.initial_input,
            "branches": payload_branches,
            "required_output": (
                "one support_matches and particle_matches object per branch"
            ),
        },
    )


def actual_refresh_messages(
    task: PublicTask,
    history: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    return _messages(
        "realized_support_regeneration",
        (
            "From only the visible dialogue and persona context, discard the prior "
            "latent support. Generate exactly 8 coherent worlds of still-unresolved "
            "requirements with 1-7 requirements per world, then exactly 6 distinct "
            "specific follow-up questions. Do not repeat requirements already "
            "answered, ask for everything, or use a generic anything-else prompt."
        ),
        {
            "persona_context": task.persona_context,
            "dialogue": [
                {"role": role, "content": content} for role, content in history
            ],
            "required_output": {
                "worlds": "8 ordered objects with requirements",
                "questions": "6 ordered question strings",
            },
        },
    )


def naive_question_messages(
    task: PublicTask,
    history: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    return _messages(
        "naive_thinking_clarification",
        (
            "Think carefully about the person's likely unstated needs, then return "
            "exactly one concise clarification question that will help you assist "
            "them well. The question must be specific; do not ask for everything or "
            "use a generic anything-else prompt."
        ),
        {
            "persona_context": task.persona_context,
            "dialogue": [
                {"role": role, "content": content} for role, content in history
            ],
            "required_output": "question",
        },
    )


def satisfaction_messages(case: TurnCase) -> list[dict[str, str]]:
    candidates = [
        {"idx": index + 1, "content": content}
        for index, (content, status) in enumerate(
            zip(case.hidden_intents, case.statuses, strict=True)
        )
        if status == "not_provided"
    ]
    return _messages(
        "official_intent_satisfaction",
        (
            "Apply the pinned Pi-Bench satisfaction policy. Judge only the listed "
            "not-provided intents from the latest assistant response, not earlier "
            "turns. True only when it precisely and explicitly provides specific "
            "details or concrete actions satisfying the intent. Vague, generic, "
            "topically related, or merely inquisitive text is false. Do not check "
            "facts outside the supplied context. Return decisions in candidate order."
        ),
        {
            "latest_assistant_response": case.question,
            "files_read_context": "<files />",
            "candidates": candidates,
            "required_output": "decisions: one boolean per candidate",
        },
    )


def targeted_messages(case: TurnCase) -> list[dict[str, str]]:
    candidates = [
        {"idx": index + 1, "content": content}
        for index, (content, status) in enumerate(
            zip(case.hidden_intents, case.statuses, strict=True)
        )
        if status == "not_provided"
    ]
    return _messages(
        "official_targeted_followup",
        (
            "Apply the pinned Pi-Bench targeted-followup policy. Judge only the "
            "listed not-provided intents from the latest assistant response. True "
            "only when it contains a clear follow-up question specifically about "
            "that intent, a close confirmation, or a short explicit options "
            "question. Generic prompts and broad topic overlap are false. Return "
            "decisions in candidate order."
        ),
        {
            "latest_assistant_response": case.question,
            "candidates": candidates,
            "required_output": "decisions: one boolean per candidate",
        },
    )


def truth_recall_messages(
    *,
    visible_initial_input: str,
    true_requirements: Sequence[str],
    belief: BeliefAndQuestions,
) -> list[dict[str, str]]:
    generated = [
        {
            "world_index": world_index,
            "requirement_index": requirement_index,
            "content": requirement,
        }
        for world_index, world in enumerate(belief.worlds)
        for requirement_index, requirement in enumerate(world.requirements)
    ]
    return _messages(
        "post_selection_truth_support_recall",
        (
            "This is an evaluator after policy selection. For each actual unresolved "
            "requirement, mark true when at least one generated requirement "
            "semantically represents the same concrete need closely enough that a "
            "specific question about the generated item would target the actual "
            "item. Broad topic overlap is false. Return decisions in actual-item "
            "order."
        ),
        {
            "visible_initial_input": visible_initial_input,
            "actual_requirements": list(true_requirements),
            "generated_support": generated,
            "required_output": "represented: one boolean per actual requirement",
        },
    )


def parse_belief(
    text: str,
    *,
    world_count: int,
    question_count: int,
    min_requirements: int,
) -> BeliefAndQuestions:
    payload = json.loads(text)
    if set(payload) != {"worlds", "questions"}:
        raise ValueError("belief response has unexpected keys")
    raw_worlds = payload["worlds"]
    raw_questions = payload["questions"]
    if len(raw_worlds) != world_count or len(raw_questions) != question_count:
        raise ValueError("belief response has wrong world or question count")

    worlds = []
    for raw_world in raw_worlds:
        if set(raw_world) != {"requirements"}:
            raise ValueError("world has unexpected keys")
        requirements = tuple(
            _normalize_text(value) for value in raw_world["requirements"]
        )
        if not min_requirements <= len(requirements) <= MAX_REQUIREMENTS:
            raise ValueError("world has wrong requirement count")
        if any(not value for value in requirements):
            raise ValueError("world has empty requirement")
        if len({_dedupe_key(value) for value in requirements}) != len(requirements):
            raise ValueError("world has duplicate requirements")
        worlds.append(RequirementWorld(requirements=requirements))

    questions = tuple(_normalize_text(value) for value in raw_questions)
    if len({_dedupe_key(value) for value in questions}) != len(questions):
        raise ValueError("belief has duplicate questions")
    invalid = [
        (index, reason)
        for index, question in enumerate(questions)
        if (reason := invalid_question_reason(question)) is not None
    ]
    if invalid:
        raise ValueError(f"belief has invalid questions: {invalid}")
    return BeliefAndQuestions(worlds=tuple(worlds), questions=questions)


def _indexes_from_flags(flags: Sequence[Any], expected: int) -> tuple[int, ...]:
    if len(flags) != expected or any(type(value) is not bool for value in flags):
        raise ValueError("semantic match vector has wrong shape or type")
    return tuple(index for index, value in enumerate(flags) if value)


def parse_semantic_map(
    text: str,
    belief: BeliefAndQuestions,
) -> SemanticMap:
    payload = json.loads(text)
    if set(payload) != {"matches"}:
        raise ValueError("semantic map has unexpected keys")
    expected_pairs = len(belief.questions) * len(belief.worlds)
    if len(payload["matches"]) != expected_pairs:
        raise ValueError("semantic map has wrong pair count")
    parsed = []
    for offset, flags in enumerate(payload["matches"]):
        world_index = offset % len(belief.worlds)
        parsed.append(
            _indexes_from_flags(
                flags, len(belief.worlds[world_index].requirements)
            )
        )
    return SemanticMap(
        matches=tuple(parsed),
        question_count=len(belief.questions),
        world_count=len(belief.worlds),
    )


def parse_branch_refreshes(
    text: str,
    *,
    branch_count: int,
) -> tuple[BranchRefresh, ...]:
    payload = json.loads(text)
    if set(payload) != {"branches"} or len(payload["branches"]) != branch_count:
        raise ValueError("branch refresh response has wrong shape")
    refreshes = []
    for raw_branch in payload["branches"]:
        belief = parse_belief(
            json.dumps(raw_branch),
            world_count=REFRESH_WORLD_COUNT,
            question_count=REFRESH_QUESTION_COUNT,
            min_requirements=1,
        )
        refreshes.append(
            BranchRefresh(worlds=belief.worlds, questions=belief.questions)
        )
    return tuple(refreshes)


def parse_branch_maps(
    text: str,
    *,
    branches: Sequence[RolloutBranch],
    refreshes: Sequence[BranchRefresh],
) -> tuple[BranchSemanticMap, ...]:
    payload = json.loads(text)
    if set(payload) != {"branches"}:
        raise ValueError("branch map has unexpected keys")
    if len(payload["branches"]) != len(branches):
        raise ValueError("branch map has wrong branch count")
    parsed = []
    for raw, branch, refresh in zip(
        payload["branches"], branches, refreshes, strict=True
    ):
        if set(raw) != {"support_matches", "particle_matches"}:
            raise ValueError("branch map item has unexpected keys")
        support_expected = REFRESH_QUESTION_COUNT * REFRESH_WORLD_COUNT
        if len(raw["support_matches"]) != support_expected:
            raise ValueError("branch support map has wrong pair count")
        support_matches = []
        for offset, flags in enumerate(raw["support_matches"]):
            world_index = offset % REFRESH_WORLD_COUNT
            support_matches.append(
                _indexes_from_flags(
                    flags, len(refresh.worlds[world_index].requirements)
                )
            )
        if len(raw["particle_matches"]) != REFRESH_QUESTION_COUNT:
            raise ValueError("branch particle map has wrong question count")
        particle_matches = tuple(
            _indexes_from_flags(
                flags, len(branch.remaining_particle_requirements)
            )
            for flags in raw["particle_matches"]
        )
        parsed.append(
            BranchSemanticMap(
                support_map=SemanticMap(
                    matches=tuple(support_matches),
                    question_count=REFRESH_QUESTION_COUNT,
                    world_count=REFRESH_WORLD_COUNT,
                ),
                particle_matches=particle_matches,
            )
        )
    return tuple(parsed)


def parse_boolean_decisions(text: str, *, expected: int) -> tuple[bool, ...]:
    payload = json.loads(text)
    if set(payload) != {"decisions"}:
        raise ValueError("decision response has unexpected keys")
    decisions = payload["decisions"]
    if len(decisions) != expected or any(type(value) is not bool for value in decisions):
        raise ValueError("decision response has wrong vector")
    return tuple(decisions)


def parse_recall(text: str, *, expected: int) -> tuple[bool, ...]:
    payload = json.loads(text)
    if set(payload) != {"represented"}:
        raise ValueError("recall response has unexpected keys")
    values = payload["represented"]
    if len(values) != expected or any(type(value) is not bool for value in values):
        raise ValueError("recall response has wrong vector")
    return tuple(values)


def parse_naive_question(text: str) -> str:
    payload = json.loads(text)
    if set(payload) != {"question"}:
        raise ValueError("naive response has unexpected keys")
    question = _normalize_text(payload["question"])
    reason = invalid_question_reason(question)
    if reason is not None:
        raise ValueError(f"naive question is invalid: {reason}")
    return question


def resolved_indexes(matches: Sequence[int], requirement_count: int) -> tuple[int, ...]:
    if requirement_count < 1:
        return ()
    if matches:
        result = tuple(sorted(set(int(index) for index in matches)))
        if result[0] < 0 or result[-1] >= requirement_count:
            raise ValueError("matched requirement index is out of range")
        return result
    return (0,)


def immediate_scores(
    belief: BeliefAndQuestions,
    semantic_map: SemanticMap,
) -> tuple[float, ...]:
    scores = []
    for question_index in range(len(belief.questions)):
        values = []
        for world_index, world in enumerate(belief.worlds):
            resolved = resolved_indexes(
                semantic_map.for_pair(question_index, world_index),
                len(world.requirements),
            )
            values.append(len(resolved) / len(world.requirements))
        scores.append(statistics.fmean(values))
    return tuple(scores)


def rollout_world_indexes(task_id: str) -> tuple[int, ...]:
    digest = hashlib.sha256(
        f"{POLICY_SEED}:{task_id}:rollout-worlds".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return tuple(sorted(rng.sample(range(INITIAL_WORLD_COUNT), ROLLOUT_WORLD_COUNT)))


def build_rollout_branches(
    task_id: str,
    belief: BeliefAndQuestions,
    semantic_map: SemanticMap,
) -> tuple[RolloutBranch, ...]:
    selected_worlds = rollout_world_indexes(task_id)
    branches = []
    for question_index in range(len(belief.questions)):
        for world_index in selected_worlds:
            world = belief.worlds[world_index]
            root_resolved = resolved_indexes(
                semantic_map.for_pair(question_index, world_index),
                len(world.requirements),
            )
            if len(root_resolved) == len(world.requirements):
                continue
            response = "\n".join(
                world.requirements[index] for index in root_resolved
            )
            remaining = tuple(
                requirement
                for index, requirement in enumerate(world.requirements)
                if index not in set(root_resolved)
            )
            branches.append(
                RolloutBranch(
                    branch_index=len(branches),
                    root_question_index=question_index,
                    particle_world_index=world_index,
                    root_resolved_indexes=root_resolved,
                    simulated_response=response,
                    remaining_particle_requirements=remaining,
                )
            )
    return tuple(branches)


def terminal_scores(
    belief: BeliefAndQuestions,
    semantic_map: SemanticMap,
    selected_worlds: Sequence[int],
    branches: Sequence[RolloutBranch],
    refreshes: Sequence[BranchRefresh],
    maps: Sequence[BranchSemanticMap],
) -> tuple[float, ...]:
    by_root: dict[int, list[float]] = defaultdict(list)
    for question_index in range(len(belief.questions)):
        for world_index in selected_worlds:
            world = belief.worlds[world_index]
            resolved = resolved_indexes(
                semantic_map.for_pair(question_index, world_index),
                len(world.requirements),
            )
            if len(resolved) == len(world.requirements):
                by_root[question_index].append(1.0)
    for branch, refresh, branch_map in zip(
        branches, refreshes, maps, strict=True
    ):
        followup_scores = immediate_scores(
            BeliefAndQuestions(
                worlds=refresh.worlds,
                questions=refresh.questions,
            ),
            branch_map.support_map,
        )
        followup_index = max(
            range(len(followup_scores)),
            key=lambda index: (followup_scores[index], -index),
        )
        particle_resolved = resolved_indexes(
            branch_map.particle_matches[followup_index],
            len(branch.remaining_particle_requirements),
        )
        original_count = len(
            belief.worlds[branch.particle_world_index].requirements
        )
        terminal = (
            len(branch.root_resolved_indexes) + len(particle_resolved)
        ) / original_count
        by_root[branch.root_question_index].append(terminal)
    if set(by_root) != set(range(len(belief.questions))):
        raise ValueError("terminal scores are missing root questions")
    return tuple(
        statistics.fmean(by_root[index])
        for index in range(len(belief.questions))
    )


def select_max(values: Sequence[float]) -> int:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("cannot select from empty or non-finite values")
    return max(range(len(values)), key=lambda index: (values[index], -index))


def random_root_index(task_id: str, question_count: int) -> int:
    digest = hashlib.sha256(
        f"{POLICY_SEED}:{task_id}:random-root".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).randrange(
        question_count
    )


def question_id(index: int, *, prefix: str = "Q") -> str:
    return f"{prefix}{index + 1}"


def support_fingerprint(belief: BeliefAndQuestions) -> str:
    return _sha256_json(
        [[_dedupe_key(item) for item in world.requirements] for world in belief.worlds]
    )


def public_payload_has_private_text(
    payload: Any,
    private_intents: Iterable[str],
) -> list[str]:
    encoded = _normalize_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True)
    ).casefold()
    leaked = []
    for intent in private_intents:
        normalized = _normalize_text(intent).casefold()
        if len(normalized) >= 12 and normalized in encoded:
            leaked.append(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
    return leaked


def assert_policy_messages_target_blind(
    messages: Sequence[Mapping[str, str]],
    private_task: PrivateTask,
    *,
    allowed_visible: Sequence[str],
) -> None:
    encoded = _normalize_text(
        json.dumps(list(messages), ensure_ascii=True, sort_keys=True)
    ).casefold()
    allowed = {
        _normalize_text(value).casefold()
        for value in allowed_visible
        if _normalize_text(value)
    }
    leaked = []
    for index, intent in enumerate(private_task.hidden_intents, start=1):
        normalized = _normalize_text(intent).casefold()
        if (
            len(normalized) >= 12
            and normalized in encoded
            and not any(normalized in visible for visible in allowed)
        ):
            leaked.append(index)
    if leaked:
        raise ValueError(
            f"private hidden intents leaked into policy payload: indexes={leaked}"
        )


@dataclass(frozen=True)
class PlanningResult:
    belief: BeliefAndQuestions
    semantic_map: SemanticMap
    branches: tuple[RolloutBranch, ...]
    refreshes: tuple[BranchRefresh, ...]
    branch_maps: tuple[BranchSemanticMap, ...]
    scores: RootScores


def _call_structured(
    model: StructuredModel,
    *,
    stage: str,
    messages: list[list[dict[str, str]]],
    response_format: dict[str, Any],
    block_size: int,
    max_new_tokens: int,
    raw_calls: list[dict[str, Any]],
) -> list[str]:
    if not messages:
        return []
    responses = model.chat_complete_messages_batched_structured(
        messages,
        temperature=0.0,
        block_size=block_size,
        response_format=response_format,
        max_new_tokens=max_new_tokens,
    )
    if len(responses) != len(messages):
        raise ValueError(f"{stage} returned the wrong response count")
    raw_calls.append(
        {
            "stage": stage,
            "messages": messages,
            "responses": responses,
        }
    )
    return responses


def run_initial_planning(
    tasks: Sequence[PublicTask],
    private_tasks: Mapping[str, PrivateTask],
    *,
    model: StructuredModel,
    block_size: int,
    raw_calls: list[dict[str, Any]],
) -> dict[str, PlanningResult]:
    initial_messages = [initial_belief_messages(task) for task in tasks]
    for task, messages in zip(tasks, initial_messages, strict=True):
        assert_policy_messages_target_blind(
            messages,
            private_tasks[task.task_id],
            allowed_visible=(task.initial_input, task.persona_context),
        )
    initial_responses = _call_structured(
        model,
        stage="initial_dynamic_support",
        messages=initial_messages,
        response_format=belief_response_format(
            name="pi_bench_initial_support_v1",
            world_count=INITIAL_WORLD_COUNT,
            question_count=INITIAL_QUESTION_COUNT,
            min_requirements=MIN_INITIAL_REQUIREMENTS,
        ),
        block_size=block_size,
        max_new_tokens=8192,
        raw_calls=raw_calls,
    )
    beliefs = {
        task.task_id: parse_belief(
            response,
            world_count=INITIAL_WORLD_COUNT,
            question_count=INITIAL_QUESTION_COUNT,
            min_requirements=MIN_INITIAL_REQUIREMENTS,
        )
        for task, response in zip(tasks, initial_responses, strict=True)
    }

    map_messages = [
        semantic_map_messages(task, beliefs[task.task_id]) for task in tasks
    ]
    for task, messages in zip(tasks, map_messages, strict=True):
        assert_policy_messages_target_blind(
            messages,
            private_tasks[task.task_id],
            allowed_visible=(task.initial_input, task.persona_context),
        )
    map_responses = _call_structured(
        model,
        stage="initial_semantic_map",
        messages=map_messages,
        response_format=semantic_map_response_format(
            name="pi_bench_initial_semantic_map_v1",
            pair_count=INITIAL_WORLD_COUNT * INITIAL_QUESTION_COUNT,
        ),
        block_size=block_size,
        max_new_tokens=8192,
        raw_calls=raw_calls,
    )
    semantic_maps = {
        task.task_id: parse_semantic_map(
            response, beliefs[task.task_id]
        )
        for task, response in zip(tasks, map_responses, strict=True)
    }
    branches = {
        task.task_id: build_rollout_branches(
            task.task_id,
            beliefs[task.task_id],
            semantic_maps[task.task_id],
        )
        for task in tasks
    }

    grouped_tasks: dict[int, list[PublicTask]] = defaultdict(list)
    for task in tasks:
        grouped_tasks[len(branches[task.task_id])].append(task)

    refreshes: dict[str, tuple[BranchRefresh, ...]] = {}
    for branch_count, task_group in sorted(grouped_tasks.items()):
        if branch_count == 0:
            for task in task_group:
                refreshes[task.task_id] = ()
            continue
        refresh_messages = [
            branch_refresh_messages(
                task,
                branches[task.task_id],
                beliefs[task.task_id].questions,
            )
            for task in task_group
        ]
        for task, messages in zip(task_group, refresh_messages, strict=True):
            assert_policy_messages_target_blind(
                messages,
                private_tasks[task.task_id],
                allowed_visible=(task.initial_input, task.persona_context),
            )
        refresh_responses = _call_structured(
            model,
            stage=f"rollout_support_regeneration_n{branch_count}",
            messages=refresh_messages,
            response_format=branch_refresh_response_format(
                name=f"pi_bench_rollout_refresh_n{branch_count}_v1",
                branch_count=branch_count,
            ),
            block_size=block_size,
            max_new_tokens=16384,
            raw_calls=raw_calls,
        )
        for task, response in zip(task_group, refresh_responses, strict=True):
            refreshes[task.task_id] = parse_branch_refreshes(
                response,
                branch_count=branch_count,
            )

    branch_maps: dict[str, tuple[BranchSemanticMap, ...]] = {}
    for branch_count, task_group in sorted(grouped_tasks.items()):
        if branch_count == 0:
            for task in task_group:
                branch_maps[task.task_id] = ()
            continue
        branch_map_messages_batch = [
            branch_map_messages(
                task,
                branches[task.task_id],
                refreshes[task.task_id],
            )
            for task in task_group
        ]
        for task, messages in zip(
            task_group, branch_map_messages_batch, strict=True
        ):
            assert_policy_messages_target_blind(
                messages,
                private_tasks[task.task_id],
                allowed_visible=(task.initial_input, task.persona_context),
            )
        branch_map_responses = _call_structured(
            model,
            stage=f"rollout_followup_semantic_map_n{branch_count}",
            messages=branch_map_messages_batch,
            response_format=branch_map_response_format(
                name=f"pi_bench_rollout_followup_map_n{branch_count}_v1",
                branch_count=branch_count,
            ),
            block_size=block_size,
            max_new_tokens=16384,
            raw_calls=raw_calls,
        )
        for task, response in zip(
            task_group, branch_map_responses, strict=True
        ):
            branch_maps[task.task_id] = parse_branch_maps(
                response,
                branches=branches[task.task_id],
                refreshes=refreshes[task.task_id],
            )

    results = {}
    for task in tasks:
        task_id = task.task_id
        immediate = immediate_scores(
            beliefs[task_id], semantic_maps[task_id]
        )
        terminal = terminal_scores(
            beliefs[task_id],
            semantic_maps[task_id],
            rollout_world_indexes(task_id),
            branches[task_id],
            refreshes[task_id],
            branch_maps[task_id],
        )
        results[task_id] = PlanningResult(
            belief=beliefs[task_id],
            semantic_map=semantic_maps[task_id],
            branches=branches[task_id],
            refreshes=refreshes[task_id],
            branch_maps=branch_maps[task_id],
            scores=RootScores(
                immediate=immediate,
                terminal=terminal,
                rollout_world_indexes=rollout_world_indexes(task_id),
            ),
        )
    return results


def _not_provided_positions(statuses: Sequence[str]) -> tuple[int, ...]:
    return tuple(
        index for index, status in enumerate(statuses) if status == "not_provided"
    )


def _dedupe_turn_cases(
    cases: Sequence[TurnCase],
) -> tuple[list[TurnCase], dict[str, str]]:
    unique: list[TurnCase] = []
    canonical_to_case_id: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for case in cases:
        key = _sha256_json(
            {
                "task_id": case.task_id,
                "hidden_intents": case.hidden_intents,
                "statuses": case.statuses,
                "question": case.question,
            }
        )
        if key not in canonical_to_case_id:
            canonical_to_case_id[key] = case.case_id
            unique.append(case)
        aliases[case.case_id] = canonical_to_case_id[key]
    return unique, aliases


def run_official_turns(
    cases: Sequence[TurnCase],
    *,
    model: StructuredModel,
    block_size: int,
    raw_calls: list[dict[str, Any]],
) -> dict[str, TurnResult]:
    unique_cases, aliases = _dedupe_turn_cases(cases)
    active_cases = [
        case for case in unique_cases if _not_provided_positions(case.statuses)
    ]
    results_by_canonical: dict[str, TurnResult] = {}
    for case in unique_cases:
        if case not in active_cases:
            results_by_canonical[case.case_id] = TurnResult(
                statuses=case.statuses,
                reply="",
                provided_indexes=(),
                inferred_indexes=(),
                satisfaction_raw="",
                targeted_raw=None,
            )

    satisfaction_responses = _call_structured(
        model,
        stage="official_intent_satisfaction",
        messages=[satisfaction_messages(case) for case in active_cases],
        response_format=boolean_decisions_response_format(
            "pi_bench_official_satisfaction_v1"
        ),
        block_size=block_size,
        max_new_tokens=1024,
        raw_calls=raw_calls,
    )
    post_satisfaction: dict[str, tuple[list[str], tuple[int, ...], str]] = {}
    followup_cases: list[TurnCase] = []
    for case, response in zip(
        active_cases, satisfaction_responses, strict=True
    ):
        candidates = _not_provided_positions(case.statuses)
        decisions = parse_boolean_decisions(response, expected=len(candidates))
        statuses = list(case.statuses)
        inferred = []
        for position, decision in zip(candidates, decisions, strict=True):
            if decision:
                statuses[position] = "inferred"
                inferred.append(position + 1)
        post_satisfaction[case.case_id] = (
            statuses,
            tuple(inferred),
            response,
        )
        if _not_provided_positions(statuses):
            followup_cases.append(
                TurnCase(
                    case_id=case.case_id,
                    task_id=case.task_id,
                    hidden_intents=case.hidden_intents,
                    statuses=tuple(statuses),
                    question=case.question,
                )
            )
        else:
            results_by_canonical[case.case_id] = TurnResult(
                statuses=tuple(statuses),
                reply="",
                provided_indexes=(),
                inferred_indexes=tuple(inferred),
                satisfaction_raw=response,
                targeted_raw=None,
            )

    targeted_responses = _call_structured(
        model,
        stage="official_targeted_followup",
        messages=[targeted_messages(case) for case in followup_cases],
        response_format=boolean_decisions_response_format(
            "pi_bench_official_targeted_v1"
        ),
        block_size=block_size,
        max_new_tokens=1024,
        raw_calls=raw_calls,
    )
    for case, response in zip(followup_cases, targeted_responses, strict=True):
        statuses, inferred, satisfaction_raw = post_satisfaction[case.case_id]
        candidates = _not_provided_positions(statuses)
        decisions = parse_boolean_decisions(response, expected=len(candidates))
        provided_positions = [
            position
            for position, decision in zip(candidates, decisions, strict=True)
            if decision
        ]
        if not provided_positions:
            provided_positions = [candidates[0]]
        for position in provided_positions:
            statuses[position] = "provided"
        reply = "\n".join(case.hidden_intents[position] for position in provided_positions)
        results_by_canonical[case.case_id] = TurnResult(
            statuses=tuple(statuses),
            reply=reply,
            provided_indexes=tuple(position + 1 for position in provided_positions),
            inferred_indexes=inferred,
            satisfaction_raw=satisfaction_raw,
            targeted_raw=response,
        )

    return {
        case.case_id: results_by_canonical[aliases[case.case_id]]
        for case in cases
    }


def _coverage(statuses: Sequence[str]) -> float:
    return sum(status != "not_provided" for status in statuses) / len(statuses)


def _trajectory_history(
    task: PublicTask,
    trajectory: Trajectory,
) -> list[tuple[str, str]]:
    history: list[tuple[str, str]] = [("user", task.initial_input)]
    for question, reply in zip(
        trajectory.questions, trajectory.replies, strict=True
    ):
        history.append(("assistant", question))
        if reply:
            history.append(("user", reply))
    return history


def _selected_root_indexes(
    task_id: str,
    planning: PlanningResult,
) -> dict[str, int]:
    return {
        "myopic": select_max(planning.scores.immediate),
        "depth2": select_max(planning.scores.terminal),
        "random": random_root_index(task_id, len(planning.belief.questions)),
    }


def _naive_questions(
    tasks: Sequence[PublicTask],
    histories: Sequence[Sequence[tuple[str, str]]],
    private_tasks: Mapping[str, PrivateTask],
    *,
    model: StructuredModel,
    block_size: int,
    raw_calls: list[dict[str, Any]],
    stage: str,
) -> dict[str, str]:
    messages = [
        naive_question_messages(task, history)
        for task, history in zip(tasks, histories, strict=True)
    ]
    for task, history, task_messages in zip(
        tasks, histories, messages, strict=True
    ):
        assert_policy_messages_target_blind(
            task_messages,
            private_tasks[task.task_id],
            allowed_visible=(
                task.initial_input,
                task.persona_context,
                *(content for _role, content in history),
            ),
        )
    responses = _call_structured(
        model,
        stage=stage,
        messages=messages,
        response_format=naive_question_response_format(),
        block_size=block_size,
        max_new_tokens=4096,
        raw_calls=raw_calls,
    )
    return {
        task.task_id: parse_naive_question(response)
        for task, response in zip(tasks, responses, strict=True)
    }


def run_actual_trajectories(
    tasks: Sequence[PublicTask],
    private_tasks: Mapping[str, PrivateTask],
    planning: Mapping[str, PlanningResult],
    *,
    bed_model: StructuredModel,
    naive_model: StructuredModel,
    block_size: int,
    raw_calls: list[dict[str, Any]],
) -> dict[tuple[str, str], Trajectory]:
    task_by_id = {task.task_id: task for task in tasks}
    trajectories: dict[tuple[str, str], Trajectory] = {}
    naive_first = _naive_questions(
        tasks,
        [[("user", task.initial_input)] for task in tasks],
        private_tasks,
        model=naive_model,
        block_size=block_size,
        raw_calls=raw_calls,
        stage="naive_thinking_turn1",
    )

    turn_one_cases = []
    for task in tasks:
        task_id = task.task_id
        private = private_tasks[task_id]
        root_indexes = _selected_root_indexes(task_id, planning[task_id])
        for policy in POLICIES:
            if policy in BED_POLICIES:
                root_index = root_indexes[policy]
                question = planning[task_id].belief.questions[root_index]
                selected_id = question_id(root_index)
            else:
                question = naive_first[task_id]
                selected_id = "NAIVE1"
            trajectory = Trajectory(
                task_id=task_id,
                policy=policy,
                statuses=list(private.initial_statuses),
                questions=[question],
                replies=[],
                selected_question_ids=[selected_id],
                provided_by_turn=[],
                inferred_by_turn=[],
            )
            trajectories[(task_id, policy)] = trajectory
            turn_one_cases.append(
                TurnCase(
                    case_id=f"{task_id}:{policy}:turn1",
                    task_id=task_id,
                    hidden_intents=private.hidden_intents,
                    statuses=private.initial_statuses,
                    question=question,
                )
            )

    turn_one_results = run_official_turns(
        turn_one_cases,
        model=bed_model,
        block_size=block_size,
        raw_calls=raw_calls,
    )
    for case in turn_one_cases:
        policy = case.case_id.split(":")[1]
        trajectory = trajectories[(case.task_id, policy)]
        result = turn_one_results[case.case_id]
        trajectory.statuses = list(result.statuses)
        trajectory.replies.append(result.reply)
        trajectory.provided_by_turn.append(list(result.provided_indexes))
        trajectory.inferred_by_turn.append(list(result.inferred_indexes))

    refresh_keys: list[tuple[str, str]] = []
    refresh_messages = []
    for task in tasks:
        for policy in BED_POLICIES:
            trajectory = trajectories[(task.task_id, policy)]
            if not _not_provided_positions(trajectory.statuses):
                continue
            messages = actual_refresh_messages(
                task, _trajectory_history(task, trajectory)
            )
            assert_policy_messages_target_blind(
                messages,
                private_tasks[task.task_id],
                allowed_visible=(
                    task.initial_input,
                    task.persona_context,
                    *trajectory.replies,
                ),
            )
            refresh_keys.append((task.task_id, policy))
            refresh_messages.append(messages)
    refresh_responses = _call_structured(
        bed_model,
        stage="realized_support_regeneration",
        messages=refresh_messages,
        response_format=belief_response_format(
            name="pi_bench_realized_refresh_v1",
            world_count=ACTUAL_REFRESH_WORLD_COUNT,
            question_count=ACTUAL_REFRESH_QUESTION_COUNT,
            min_requirements=1,
        ),
        block_size=block_size,
        max_new_tokens=8192,
        raw_calls=raw_calls,
    )
    for key, response in zip(refresh_keys, refresh_responses, strict=True):
        trajectories[key].refreshed_belief = parse_belief(
            response,
            world_count=ACTUAL_REFRESH_WORLD_COUNT,
            question_count=ACTUAL_REFRESH_QUESTION_COUNT,
            min_requirements=1,
        )

    actual_map_messages = []
    for task_id, policy in refresh_keys:
        actual_map_messages.append(
            semantic_map_messages(
                task_by_id[task_id],
                trajectories[(task_id, policy)].refreshed_belief,
            )
        )
    for (task_id, policy), messages in zip(
        refresh_keys, actual_map_messages, strict=True
    ):
        trajectory = trajectories[(task_id, policy)]
        task = task_by_id[task_id]
        assert_policy_messages_target_blind(
            messages,
            private_tasks[task_id],
            allowed_visible=(
                task.initial_input,
                task.persona_context,
                *trajectory.replies,
            ),
        )
    actual_map_responses = _call_structured(
        bed_model,
        stage="realized_followup_semantic_map",
        messages=actual_map_messages,
        response_format=semantic_map_response_format(
            name="pi_bench_realized_semantic_map_v1",
            pair_count=(
                ACTUAL_REFRESH_WORLD_COUNT * ACTUAL_REFRESH_QUESTION_COUNT
            ),
        ),
        block_size=block_size,
        max_new_tokens=8192,
        raw_calls=raw_calls,
    )
    for key, response in zip(refresh_keys, actual_map_responses, strict=True):
        trajectory = trajectories[key]
        assert trajectory.refreshed_belief is not None
        semantic_map = parse_semantic_map(
            response, trajectory.refreshed_belief
        )
        followup_index = select_max(
            immediate_scores(trajectory.refreshed_belief, semantic_map)
        )
        trajectory.questions.append(
            trajectory.refreshed_belief.questions[followup_index]
        )
        trajectory.selected_question_ids.append(
            question_id(followup_index, prefix="F")
        )

    naive_active_tasks = []
    naive_histories = []
    for task in tasks:
        trajectory = trajectories[(task.task_id, "naive_thinking")]
        if not _not_provided_positions(trajectory.statuses):
            continue
        naive_active_tasks.append(task)
        naive_histories.append(_trajectory_history(task, trajectory))
    naive_second = _naive_questions(
        naive_active_tasks,
        naive_histories,
        private_tasks,
        model=naive_model,
        block_size=block_size,
        raw_calls=raw_calls,
        stage="naive_thinking_turn2",
    )
    for task in naive_active_tasks:
        trajectory = trajectories[(task.task_id, "naive_thinking")]
        trajectory.questions.append(naive_second[task.task_id])
        trajectory.selected_question_ids.append("NAIVE2")

    turn_two_cases = []
    for task in tasks:
        private = private_tasks[task.task_id]
        for policy in POLICIES:
            trajectory = trajectories[(task.task_id, policy)]
            if len(trajectory.questions) < 2:
                continue
            turn_two_cases.append(
                TurnCase(
                    case_id=f"{task.task_id}:{policy}:turn2",
                    task_id=task.task_id,
                    hidden_intents=private.hidden_intents,
                    statuses=tuple(trajectory.statuses),
                    question=trajectory.questions[1],
                )
            )
    turn_two_results = run_official_turns(
        turn_two_cases,
        model=bed_model,
        block_size=block_size,
        raw_calls=raw_calls,
    )
    for case in turn_two_cases:
        policy = case.case_id.split(":")[1]
        trajectory = trajectories[(case.task_id, policy)]
        result = turn_two_results[case.case_id]
        trajectory.statuses = list(result.statuses)
        trajectory.replies.append(result.reply)
        trajectory.provided_by_turn.append(list(result.provided_indexes))
        trajectory.inferred_by_turn.append(list(result.inferred_indexes))
    return trajectories


def evaluate_truth_recall(
    tasks: Sequence[PublicTask],
    private_tasks: Mapping[str, PrivateTask],
    planning: Mapping[str, PlanningResult],
    trajectories: Mapping[tuple[str, str], Trajectory],
    *,
    model: StructuredModel,
    block_size: int,
    raw_calls: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    task_by_id = {task.task_id: task for task in tasks}
    initial_messages = []
    initial_expected = []
    for task in tasks:
        private = private_tasks[task.task_id]
        actual = [
            content
            for content, status in zip(
                private.hidden_intents, private.initial_statuses, strict=True
            )
            if status == "not_provided"
        ]
        initial_messages.append(
            truth_recall_messages(
                visible_initial_input=task.initial_input,
                true_requirements=actual,
                belief=planning[task.task_id].belief,
            )
        )
        initial_expected.append(len(actual))
    initial_responses = _call_structured(
        model,
        stage="initial_truth_support_recall",
        messages=initial_messages,
        response_format=recall_response_format(
            "pi_bench_initial_truth_recall_v1"
        ),
        block_size=block_size,
        max_new_tokens=2048,
        raw_calls=raw_calls,
    )
    initial_recall = {}
    for task, expected, response in zip(
        tasks, initial_expected, initial_responses, strict=True
    ):
        represented = parse_recall(response, expected=expected)
        initial_recall[task.task_id] = sum(represented) / expected

    refresh_keys = []
    refresh_messages = []
    refresh_expected = []
    refreshed_recall: dict[tuple[str, str], float] = {}
    for task in tasks:
        private = private_tasks[task.task_id]
        for policy in BED_POLICIES:
            trajectory = trajectories[(task.task_id, policy)]
            remaining = [
                content
                for content, status in zip(
                    private.hidden_intents, trajectory.statuses, strict=True
                )
                if status == "not_provided"
            ]
            if trajectory.refreshed_belief is None:
                if remaining:
                    raise ValueError(
                        "missing refreshed belief with unresolved true intents"
                    )
                refreshed_recall[(task.task_id, policy)] = 1.0
                continue
            # Recall is measured immediately after turn one, so reconstruct those
            # statuses by undoing any turn-two changes.
            turn_one_resolved = set(
                trajectory.provided_by_turn[0]
                + trajectory.inferred_by_turn[0]
            )
            remaining_after_one = [
                content
                for index, content in enumerate(
                    private.hidden_intents, start=1
                )
                if (
                    private.initial_statuses[index - 1] == "not_provided"
                    and index not in turn_one_resolved
                )
            ]
            if not remaining_after_one:
                refreshed_recall[(task.task_id, policy)] = 1.0
                continue
            refresh_keys.append((task.task_id, policy))
            refresh_expected.append(len(remaining_after_one))
            refresh_messages.append(
                truth_recall_messages(
                    visible_initial_input=task.initial_input,
                    true_requirements=remaining_after_one,
                    belief=trajectory.refreshed_belief,
                )
            )
    refresh_responses = _call_structured(
        model,
        stage="refreshed_truth_support_recall",
        messages=refresh_messages,
        response_format=recall_response_format(
            "pi_bench_refreshed_truth_recall_v1"
        ),
        block_size=block_size,
        max_new_tokens=2048,
        raw_calls=raw_calls,
    )
    for key, expected, response in zip(
        refresh_keys, refresh_expected, refresh_responses, strict=True
    ):
        represented = parse_recall(response, expected=expected)
        refreshed_recall[key] = sum(represented) / expected
    for key, value in refreshed_recall.items():
        trajectories[key].refreshed_truth_recall = value
    return initial_recall, refreshed_recall


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[cursor]]
        ):
            end += 1
        average_rank = (cursor + 1 + end) / 2
        for offset in range(cursor, end):
            ranks[order[offset]] = average_rank
        cursor = end
    return ranks


def spearman_correlation(
    left: Sequence[float], right: Sequence[float]
) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_ranks)
    )
    right_scale = math.sqrt(
        sum((value - right_mean) ** 2 for value in right_ranks)
    )
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence: float,
    seed: int,
    replicates: int = 10_000,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(replicates)
    )
    alpha = (1.0 - confidence) / 2.0
    lower_index = max(0, min(replicates - 1, int(alpha * replicates)))
    upper_index = max(
        0, min(replicates - 1, int((1.0 - alpha) * replicates) - 1)
    )
    return samples[lower_index], samples[upper_index]


def exact_one_sided_permutation_p(values: Sequence[float]) -> float:
    nonzero = [
        Fraction(value).limit_denominator(10_000)
        for value in values
        if abs(value) > 1e-12
    ]
    if not nonzero:
        return 1.0
    observed = sum(nonzero, start=Fraction(0))
    distribution: Counter[Fraction] = Counter({Fraction(0): 1})
    for value in nonzero:
        updated: Counter[Fraction] = Counter()
        for partial, count in distribution.items():
            updated[partial + value] += count
            updated[partial - value] += count
        distribution = updated
    favorable = sum(
        count for total, count in distribution.items() if total >= observed
    )
    return favorable / (2 ** len(nonzero))


def paired_summary(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    if not values:
        raise ValueError("paired summary requires values")
    ci90 = bootstrap_mean_interval(values, confidence=0.90, seed=seed)
    ci95 = bootstrap_mean_interval(values, confidence=0.95, seed=seed + 1)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "wins": sum(value > 1e-12 for value in values),
        "ties": sum(abs(value) <= 1e-12 for value in values),
        "losses": sum(value < -1e-12 for value in values),
        "bootstrap_ci90": list(ci90),
        "bootstrap_ci95": list(ci95),
        "one_sided_exact_permutation_p": exact_one_sided_permutation_p(
            values
        ),
    }


def _usage(model: StructuredModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "transport_retries": int(snapshot.get("retry_count", 0)),
        "prompt_tokens": int(snapshot.get("adapter_prompt_tokens", 0)),
        "completion_tokens": int(
            snapshot.get("adapter_completion_tokens", 0)
        ),
        "reasoning_tokens": int(
            snapshot.get("adapter_reasoning_tokens", 0)
        ),
        "cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "tracker": snapshot,
    }


def build_public_result(
    *,
    stage: str,
    tasks: Sequence[PublicTask],
    private_tasks: Mapping[str, PrivateTask],
    cohort: Mapping[str, Any],
    planning: Mapping[str, PlanningResult],
    trajectories: Mapping[tuple[str, str], Trajectory],
    initial_recall: Mapping[str, float],
    bed_usage: Mapping[str, Any],
    naive_usage: Mapping[str, Any],
    elapsed_seconds: float,
    private_raw_sha256: str,
) -> dict[str, Any]:
    rows = []
    for task in tasks:
        task_id = task.task_id
        private = private_tasks[task_id]
        plan = planning[task_id]
        selected = _selected_root_indexes(task_id, plan)
        policy_rows = {}
        for policy in POLICIES:
            trajectory = trajectories[(task_id, policy)]
            turn_one_resolved = set(
                trajectory.provided_by_turn[0]
                + trajectory.inferred_by_turn[0]
            )
            initial_resolved = {
                index + 1
                for index, status in enumerate(private.initial_statuses)
                if status != "not_provided"
            }
            turn_one_coverage = (
                len(initial_resolved | turn_one_resolved)
                / len(private.hidden_intents)
            )
            final_coverage = _coverage(trajectory.statuses)
            policy_rows[policy] = {
                "selected_question_ids": trajectory.selected_question_ids,
                "turn1_provided_indexes": trajectory.provided_by_turn[0],
                "turn1_inferred_indexes": trajectory.inferred_by_turn[0],
                "turn2_provided_indexes": (
                    trajectory.provided_by_turn[1]
                    if len(trajectory.provided_by_turn) > 1
                    else []
                ),
                "turn2_inferred_indexes": (
                    trajectory.inferred_by_turn[1]
                    if len(trajectory.inferred_by_turn) > 1
                    else []
                ),
                "covered_fraction_turn1": turn_one_coverage,
                "covered_fraction_turn2": final_coverage,
                "turn2_increment": final_coverage - turn_one_coverage,
                "refreshed_truth_recall": trajectory.refreshed_truth_recall,
                "refreshed_support_fingerprint": (
                    support_fingerprint(trajectory.refreshed_belief)
                    if trajectory.refreshed_belief is not None
                    else None
                ),
                "trajectory_complete_after_turn1": (
                    len(trajectory.questions) == 1
                ),
            }
        branch_fingerprints = [
            _sha256_json(
                [
                    list(world.requirements)
                    for world in refresh.worlds
                ]
            )
            for refresh in plan.refreshes
        ]
        rows.append(
            {
                "task_id": task_id,
                "persona": task.persona,
                "hidden_intent_count": len(private.hidden_intents),
                "initial_truth_support_recall": initial_recall[task_id],
                "root_question_count": len(plan.belief.questions),
                "initial_world_count": len(plan.belief.worlds),
                "rollout_world_indexes": list(
                    plan.scores.rollout_world_indexes
                ),
                "immediate_scores": list(plan.scores.immediate),
                "terminal_scores": list(plan.scores.terminal),
                "selected_root_ids": {
                    policy: question_id(index)
                    for policy, index in selected.items()
                },
                "myopic_depth2_root_changed": (
                    selected["myopic"] != selected["depth2"]
                ),
                "predicted_selected_terminal_advantage": (
                    plan.scores.terminal[selected["depth2"]]
                    - plan.scores.terminal[selected["myopic"]]
                ),
                "rollout_refresh_unique_fingerprint_count": len(
                    set(branch_fingerprints)
                ),
                "policies": policy_rows,
            }
        )

    coverage_differences = [
        row["policies"]["depth2"]["covered_fraction_turn2"]
        - row["policies"]["myopic"]["covered_fraction_turn2"]
        for row in rows
    ]
    changed_rows = [
        row for row in rows if row["myopic_depth2_root_changed"]
    ]
    recall_differences_changed = [
        row["policies"]["depth2"]["refreshed_truth_recall"]
        - row["policies"]["myopic"]["refreshed_truth_recall"]
        for row in changed_rows
    ]
    turn_one_differences = [
        row["policies"]["depth2"]["covered_fraction_turn1"]
        - row["policies"]["myopic"]["covered_fraction_turn1"]
        for row in rows
    ]
    increment_differences = [
        row["policies"]["depth2"]["turn2_increment"]
        - row["policies"]["myopic"]["turn2_increment"]
        for row in rows
    ]
    predicted_advantages = [
        row["predicted_selected_terminal_advantage"] for row in rows
    ]
    coverage_by_policy = {
        policy: {
            "mean_turn1": statistics.fmean(
                row["policies"][policy]["covered_fraction_turn1"]
                for row in rows
            ),
            "mean_turn2": statistics.fmean(
                row["policies"][policy]["covered_fraction_turn2"]
                for row in rows
            ),
            "sd_turn2": statistics.stdev(
                row["policies"][policy]["covered_fraction_turn2"]
                for row in rows
            )
            if len(rows) > 1
            else 0.0,
            "mean_turn2_increment": statistics.fmean(
                row["policies"][policy]["turn2_increment"] for row in rows
            ),
        }
        for policy in POLICIES
    }
    paired = paired_summary(coverage_differences, seed=POLICY_SEED)
    changed_recall_mean = (
        statistics.fmean(recall_differences_changed)
        if recall_differences_changed
        else 0.0
    )
    mean_increment_difference = statistics.fmean(increment_differences)
    mean_coverage_difference = statistics.fmean(coverage_differences)
    root_changed_count = len(changed_rows)
    path_sensitive_count = sum(
        row["rollout_refresh_unique_fingerprint_count"] >= 2 for row in rows
    )

    integrity_gates = {
        "all_tasks_complete": len(rows) == len(tasks),
        "exact_initial_world_count": all(
            row["initial_world_count"] == INITIAL_WORLD_COUNT for row in rows
        ),
        "exact_root_question_count": all(
            row["root_question_count"] == INITIAL_QUESTION_COUNT for row in rows
        ),
        "zero_bed_reasoning_tokens": int(bed_usage["reasoning_tokens"]) == 0,
        "positive_naive_reasoning_tokens": (
            int(naive_usage["reasoning_tokens"]) > 0
        ),
        "zero_forced_exits": (
            int(bed_usage["forced_exits"]) == 0
            and int(naive_usage["forced_exits"]) == 0
        ),
        "finite_endpoints": all(
            math.isfinite(
                float(row["policies"][policy]["covered_fraction_turn2"])
            )
            for row in rows
            for policy in POLICIES
        ),
    }
    if stage == "serving_smoke":
        decision_gates = {
            "exact_five_serving_tasks": len(rows) == 5,
            "path_sensitive_on_at_least_four": path_sensitive_count >= 4,
            "root_changes_on_at_least_one": root_changed_count >= 1,
        }
    elif stage == "development":
        mediation_threshold = 0.5 * mean_coverage_difference
        decision_gates = {
            "at_least_twenty_eligible_tasks": len(rows) >= 20,
            "root_changes_on_at_least_twenty_percent": (
                root_changed_count >= math.ceil(0.20 * len(rows))
            ),
            "changed_root_recall_gain_at_least_0_05": (
                changed_recall_mean >= 0.05
            ),
            "coverage_gain_at_least_0_03": mean_coverage_difference >= 0.03,
            "wins_exceed_losses_by_at_least_two": (
                paired["wins"] - paired["losses"] >= 2
            ),
            "at_least_half_gain_through_turn2": (
                mean_increment_difference >= mediation_threshold
            ),
        }
    else:
        random_difference = (
            coverage_by_policy["depth2"]["mean_turn2"]
            - coverage_by_policy["random"]["mean_turn2"]
        )
        mean_recall_all = statistics.fmean(
            row["policies"]["depth2"]["refreshed_truth_recall"]
            - row["policies"]["myopic"]["refreshed_truth_recall"]
            for row in rows
        )
        decision_gates = {
            "positive_mean_coverage_gain": mean_coverage_difference > 0.0,
            "one_sided_permutation_p_at_most_0_05": (
                paired["one_sided_exact_permutation_p"] <= 0.05
            ),
            "positive_mean_refreshed_recall_gain": mean_recall_all > 0.0,
            "wins_exceed_losses": paired["wins"] > paired["losses"],
            "depth2_not_below_random": random_difference >= 0.0,
        }
    gates = {**integrity_gates, **decision_gates}
    gates["all_pass"] = all(gates.values())
    summary = {
        "num_tasks": len(rows),
        "coverage_by_policy": coverage_by_policy,
        "myopic_depth2_root_changed_count": root_changed_count,
        "myopic_depth2_root_changed_fraction": root_changed_count / len(rows),
        "path_sensitive_task_count": path_sensitive_count,
        "changed_root_mean_refreshed_truth_recall_difference": (
            changed_recall_mean
        ),
        "mean_turn1_coverage_difference": statistics.fmean(
            turn_one_differences
        ),
        "mean_turn2_increment_difference": mean_increment_difference,
        "depth2_minus_myopic_coverage": paired,
        "predicted_vs_realized_advantage_spearman": spearman_correlation(
            predicted_advantages, coverage_differences
        ),
        "gates": gates,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": stage,
            "source_commit": SOURCE_COMMIT,
            "source_tree": SOURCE_TREE,
            "manifest_sha256": MANIFEST_SHA256,
            "policy_seed": POLICY_SEED,
            "initial_world_count": INITIAL_WORLD_COUNT,
            "initial_question_count": INITIAL_QUESTION_COUNT,
            "rollout_world_count": ROLLOUT_WORLD_COUNT,
            "refresh_world_count": REFRESH_WORLD_COUNT,
            "refresh_question_count": REFRESH_QUESTION_COUNT,
            "actual_refresh_world_count": ACTUAL_REFRESH_WORLD_COUNT,
            "actual_refresh_question_count": ACTUAL_REFRESH_QUESTION_COUNT,
            "bed_reasoning_disabled": True,
            "naive_reasoning_effort": "medium",
            "shared_compute_myopic_depth2": True,
            "private_raw_sha256": private_raw_sha256,
            "elapsed_seconds": elapsed_seconds,
            "cohort": cohort,
        },
        "summary": summary,
        "tasks": rows,
        "usage": {
            "bed_and_judge": dict(bed_usage),
            "naive_thinking": dict(naive_usage),
            "total_cost_usd": (
                float(bed_usage["cost_usd"])
                + float(naive_usage["cost_usd"])
            ),
            "total_physical_requests": (
                int(bed_usage["physical_requests"])
                + int(naive_usage["physical_requests"])
            ),
        },
    }
    leaked = public_payload_has_private_text(
        payload,
        (
            intent
            for private in private_tasks.values()
            for intent in private.hidden_intents
        ),
    )
    if leaked:
        raise ValueError(
            "public result contains private hidden-intent text hashes: "
            + ",".join(leaked)
        )
    return payload


def _checkpoint_private(path: Path, raw: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(raw, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_models(config: Config) -> tuple[StructuredModel, StructuredModel]:
    if len(config.model_pairs) != 1:
        raise ValueError("Pi-Bench first-link requires exactly one model pair")
    base = config.model_pairs[0].questioner
    if base.backend != "openrouter" or base.model != "openai/gpt-5.4":
        raise ValueError(
            "preregistered Pi-Bench protocol requires openai/gpt-5.4 via OpenRouter"
        )
    bed_spec = replace(
        base,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    naive_spec = replace(
        base,
        thinking=None,
        reasoning_effort="medium",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    return (
        PiBenchGPT54Adapter(bed_spec, config),
        PiBenchGPT54Adapter(naive_spec, config),
    )


def run_experiment(
    config: Config,
    *,
    stage: str,
    pi_bench_repo: Path,
    manifest_path: Path,
    private_raw_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    manifest = validate_source(pi_bench_repo, manifest_path)
    tasks, private_tasks, cohort = load_tasks(
        pi_bench_repo, manifest, stage=stage
    )
    bed_model, naive_model = build_models(config)
    raw: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "stage": stage,
        "source_commit": SOURCE_COMMIT,
        "cohort": cohort,
        "private_tasks": {
            task_id: {
                "hidden_intents": list(private.hidden_intents),
                "initial_statuses": list(private.initial_statuses),
            }
            for task_id, private in private_tasks.items()
        },
        "calls": [],
    }
    raw_calls = raw["calls"]
    try:
        planning = run_initial_planning(
            tasks,
            private_tasks,
            model=bed_model,
            block_size=config.batched_block_size,
            raw_calls=raw_calls,
        )
        _checkpoint_private(private_raw_path, raw)
        trajectories = run_actual_trajectories(
            tasks,
            private_tasks,
            planning,
            bed_model=bed_model,
            naive_model=naive_model,
            block_size=config.batched_block_size,
            raw_calls=raw_calls,
        )
        _checkpoint_private(private_raw_path, raw)
        initial_recall, _refreshed_recall = evaluate_truth_recall(
            tasks,
            private_tasks,
            planning,
            trajectories,
            model=bed_model,
            block_size=config.batched_block_size,
            raw_calls=raw_calls,
        )
        raw["selected"] = {
            f"{task_id}:{policy}": {
                "questions": trajectory.questions,
                "replies": trajectory.replies,
                "statuses": trajectory.statuses,
                "provided_by_turn": trajectory.provided_by_turn,
                "inferred_by_turn": trajectory.inferred_by_turn,
            }
            for (task_id, policy), trajectory in trajectories.items()
        }
        _checkpoint_private(private_raw_path, raw)
        private_hash = sha256_file(private_raw_path)
        result = build_public_result(
            stage=stage,
            tasks=tasks,
            private_tasks=private_tasks,
            cohort=cohort,
            planning=planning,
            trajectories=trajectories,
            initial_recall=initial_recall,
            bed_usage=_usage(bed_model),
            naive_usage=_usage(naive_model),
            elapsed_seconds=time.monotonic() - started,
            private_raw_sha256=private_hash,
        )
        return result
    except Exception:
        raw["failure_usage"] = {
            "bed_and_judge": _usage(bed_model),
            "naive_thinking": _usage(naive_model),
        }
        _checkpoint_private(private_raw_path, raw)
        raise


def validate_only(
    *,
    pi_bench_repo: Path,
    manifest_path: Path,
    stage: str,
) -> dict[str, Any]:
    manifest = validate_source(pi_bench_repo, manifest_path)
    tasks, private_tasks, cohort = load_tasks(
        pi_bench_repo, manifest, stage=stage
    )
    return {
        "status": "validated",
        "stage": stage,
        "task_count": len(tasks),
        "task_ids": [task.task_id for task in tasks],
        "private_task_count": len(private_tasks),
        "cohort": cohort,
        "source_commit": SOURCE_COMMIT,
        "manifest_sha256": MANIFEST_SHA256,
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
    }


def run_planning_preflight(
    config: Config,
    *,
    pi_bench_repo: Path,
    manifest_path: Path,
    private_raw_path: Path,
) -> dict[str, Any]:
    manifest = validate_source(pi_bench_repo, manifest_path)
    tasks, private_tasks, cohort = load_tasks(
        pi_bench_repo, manifest, stage="serving_smoke"
    )
    task = tasks[0]
    private = {task.task_id: private_tasks[task.task_id]}
    bed_model, _naive_model = build_models(config)
    raw: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "stage": "planning_preflight",
        "task_id": task.task_id,
        "calls": [],
    }
    try:
        planning = run_initial_planning(
            [task],
            private,
            model=bed_model,
            block_size=config.batched_block_size,
            raw_calls=raw["calls"],
        )[task.task_id]
        _checkpoint_private(private_raw_path, raw)
        usage = _usage(bed_model)
        selected = _selected_root_indexes(task.task_id, planning)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "stage": "planning_preflight",
                "task_id": task.task_id,
                "source_commit": SOURCE_COMMIT,
                "manifest_sha256": MANIFEST_SHA256,
                "formal_serving_cohort_unchanged": cohort[
                    "included_task_ids"
                ],
                "private_raw_sha256": sha256_file(private_raw_path),
            },
            "summary": {
                "initial_world_count": len(planning.belief.worlds),
                "root_question_count": len(planning.belief.questions),
                "incomplete_rollout_branch_count": len(planning.branches),
                "rollout_refresh_unique_fingerprint_count": len(
                    {
                        _sha256_json(
                            [
                                list(world.requirements)
                                for world in refresh.worlds
                            ]
                        )
                        for refresh in planning.refreshes
                    }
                ),
                "myopic_root_id": question_id(selected["myopic"]),
                "depth2_root_id": question_id(selected["depth2"]),
                "root_changed": (
                    selected["myopic"] != selected["depth2"]
                ),
                "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
                "zero_forced_exits": usage["forced_exits"] == 0,
            },
            "usage": usage,
        }
        leaked = public_payload_has_private_text(
            payload, private[task.task_id].hidden_intents
        )
        if leaked:
            raise ValueError("preflight public artifact contains private text")
        return payload
    except Exception:
        raw["failure_usage"] = _usage(bed_model)
        _checkpoint_private(private_raw_path, raw)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "development", "confirmation"),
        required=True,
    )
    parser.add_argument("--pi-bench-repo", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "results/nonmyopic/pi_bench_release_source_manifest.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--planning-preflight", action="store_true")
    args = parser.parse_args()
    if args.validate_only and args.planning_preflight:
        parser.error("--validate-only and --planning-preflight are exclusive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_run_dir = args.private_raw_dir / args.run_id
    private_run_dir.mkdir(parents=True, exist_ok=True)
    private_raw_path = private_run_dir / "RAW.json"
    output_name = (
        "VALIDATION.json"
        if args.validate_only
        else (
            "PREFLIGHT.json"
            if args.planning_preflight
            else f"{args.stage.upper()}.json"
        )
    )
    output_path = args.output_dir / output_name
    try:
        if args.validate_only:
            payload = validate_only(
                pi_bench_repo=args.pi_bench_repo.resolve(),
                manifest_path=args.manifest.resolve(),
                stage=args.stage,
            )
        else:
            if args.config is None:
                parser.error(
                    "--config is required unless --validate-only is used"
                )
            config = load_config(str(args.config))
            config.run_id = args.run_id
            config.log_path = args.output_dir / "run.log"
            config.openrouter_projected_cost_usd = 0.0
            config.openrouter_run_budget_usd = None
            if args.planning_preflight:
                if args.stage != "serving_smoke":
                    parser.error(
                        "--planning-preflight requires --stage serving_smoke"
                    )
                payload = run_planning_preflight(
                    config,
                    pi_bench_repo=args.pi_bench_repo.resolve(),
                    manifest_path=args.manifest.resolve(),
                    private_raw_path=private_raw_path,
                )
            else:
                payload = run_experiment(
                    config,
                    stage=args.stage,
                    pi_bench_repo=args.pi_bench_repo.resolve(),
                    manifest_path=args.manifest.resolve(),
                    private_raw_path=private_raw_path,
                )
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "interface_version": INTERFACE_VERSION,
            "error_type": type(exc).__name__,
            "private_raw_sha256": (
                sha256_file(private_raw_path)
                if private_raw_path.exists()
                else None
            ),
        }
        failure_path = args.output_dir / f"{args.stage.upper()}_FAILURE.json"
        failure_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
