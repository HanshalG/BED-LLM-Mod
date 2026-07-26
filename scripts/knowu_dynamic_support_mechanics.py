#!/usr/bin/env python3
"""Gate KnowU semantic support expansion on the frozen mechanics worlds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Protocol, Sequence

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts import knowu_bench_source_audit as source_audit
from scripts import knowu_dynamic_support_manifest_v2 as manifest_v2


INTERFACE_VERSION = "knowu-dynamic-support-mechanics-1"
MODEL_ID = "openai/gpt-5.4"
SUPPORT_SIZE = 4
QUESTION_COUNT = 4
TRUTH_MATCH_THRESHOLD = 70
RETRIEVED_LOG_COUNT = 8
EXPECTED_WORLDS = 6
EXPECTED_SERVING_REQUESTS = 10
EXPECTED_MECHANICS_REQUESTS = EXPECTED_WORLDS * (
    1 + 1 + QUESTION_COUNT + 1
)
SERVING_MAX_COST_USD = 0.10
MECHANICS_MAX_COST_USD = 0.75
EXPECTED_FIXTURE_SHA256: str | None = (
    "2cb9d0e34e3e13aee896b5d4f2c6bf0c470d1a92096a23128f6d8cc26905eff8"
)

TASK_SPECS = {
    "BuyComputerPreferenceAskUserTask": {
        "kind": "buy_computer",
        "query_terms": (
            "computer laptop macbook thinkpad windows linux ram gaming "
            "shopping platform taodian jingdian order purchase electronics "
            "price budget apple"
        ).split(),
        "category_terms": ("shopping", "computer"),
    },
    "MattermostLeaveNoticeTask": {
        "kind": "leave_notice",
        "query_terms": (
            "mattermost leave sick illness team group channel message "
            "communication tone style devops ai research work"
        ).split(),
        "category_terms": ("social", "message", "work", "mattermost"),
    },
}

PROFILE_LABELS = {
    "developer",
    "student",
    "professor",
    "persona",
    "profile",
    "identity",
    "kenji",
    "lucas",
    "aiden",
}


class StructuredChatModel(Protocol):
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


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class WorldFixture:
    world_id: str
    task_id: str
    task_kind: str
    goal_request: str
    profile_id: str
    retrieved_log_indices: tuple[int, ...]
    visible_logs: tuple[str, ...]
    truth_packet: dict[str, Any]


@dataclass(frozen=True)
class InitialPolicy:
    hypotheses: tuple[str, ...]
    dimensions: tuple[str, ...]
    questions: tuple[str, ...]


@dataclass(frozen=True)
class TruthJudgment:
    best_indices: tuple[int, ...]
    best_scores: tuple[int, ...]
    reasons: tuple[str, ...]

    @property
    def present(self) -> tuple[bool, ...]:
        return tuple(
            index > 0 and score >= TRUTH_MATCH_THRESHOLD
            for index, score in zip(self.best_indices, self.best_scores)
        )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _checkpoint(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _nested(value: dict[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _compact_dict(**values: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value not in (None, {}, [], "")
    }


def _truth_packet(profile: dict[str, Any], task_kind: str) -> dict[str, Any]:
    user = profile.get("user_profile", {})
    if task_kind == "buy_computer":
        return _compact_dict(
            identity=_compact_dict(
                full_name=_nested(user, "identity", "full_name"),
                nickname=_nested(user, "identity", "nickname"),
                contact_info=_nested(user, "identity", "contact_info"),
            ),
            locations=_nested(user, "locations"),
            devices=_nested(user, "digital_context", "devices"),
            shopping=_nested(user, "preferences", "shopping"),
            app_shopping=_nested(user, "preferences", "apps", "shopping"),
        )
    if task_kind == "leave_notice":
        return _compact_dict(
            occupation=_nested(user, "identity", "occupation"),
            social_preferences=_nested(user, "preferences", "social"),
            mattermost_preferences=_nested(
                user, "preferences", "apps", "mattermost"
            ),
            messaging_preferences=_nested(
                user, "preferences", "apps", "messaging"
            ),
            social_graph=_nested(user, "social_graph"),
            relevant_habits=_compact_dict(
                clock_out=_nested(user, "habits", "clock_out_routine"),
                on_call=_nested(user, "habits", "on_call_response"),
            ),
        )
    raise ValueError(f"unknown task kind: {task_kind}")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _retrieve_logs(
    logs: Sequence[dict[str, Any]],
    *,
    query_terms: Sequence[str],
    category_terms: Sequence[str],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    ranked: list[tuple[int, int, str]] = []
    for index, row in enumerate(logs):
        action = str(row.get("action", "")).strip()
        tokens = _tokens(action)
        score = sum(tokens.count(term) for term in query_terms)
        category = str(row.get("category", "")).casefold()
        if any(term in category for term in category_terms):
            score += 2
        ranked.append((score, index, action))
    selected = sorted(ranked, key=lambda row: (-row[0], row[1]))[
        :RETRIEVED_LOG_COUNT
    ]
    if len(selected) != RETRIEVED_LOG_COUNT or selected[-1][0] <= 0:
        raise ValueError("task-log retrieval no longer has eight positive rows")
    return (
        tuple(row[1] for row in selected),
        tuple(row[2] for row in selected),
    )


def build_world_fixtures(
    source_root: Path,
    manifest_path: Path,
    *,
    enforce_frozen: bool = True,
) -> tuple[list[WorldFixture], dict[str, Any]]:
    audit = source_audit.build_audit(source_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("passed"):
        raise ValueError("KnowU manifest did not pass")
    if tuple(manifest["splits"]["mechanics"]["family_ids"]) != (
        manifest_v2.EXPECTED_MECHANICS_IDS
    ):
        raise ValueError("KnowU mechanics task IDs changed")
    goals = manifest_v2.static_goals_by_class(source_root)
    families = {
        family["class_name"]: family
        for family in source_audit.preference_task_families(source_root)
    }
    profile_root = source_root / "src" / "knowu_bench" / "user_profile"
    log_root = source_root / "src" / "knowu_bench" / "user_logs"

    fixtures: list[WorldFixture] = []
    for task_number, task_id in enumerate(
        manifest_v2.EXPECTED_MECHANICS_IDS, start=1
    ):
        spec = TASK_SPECS[task_id]
        profiles = families[task_id]["supported_profiles"]
        for world_number, profile_id in enumerate(profiles, start=1):
            profile = yaml.safe_load(
                (profile_root / f"{profile_id}.yaml").read_text(
                    encoding="utf-8"
                )
            )
            logs = json.loads(
                (log_root / f"{profile_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            indices, visible_logs = _retrieve_logs(
                logs,
                query_terms=spec["query_terms"],
                category_terms=spec["category_terms"],
            )
            fixtures.append(
                WorldFixture(
                    world_id=f"T{task_number}W{world_number}",
                    task_id=task_id,
                    task_kind=spec["kind"],
                    goal_request=goals[task_id],
                    profile_id=profile_id,
                    retrieved_log_indices=indices,
                    visible_logs=visible_logs,
                    truth_packet=_truth_packet(profile, spec["kind"]),
                )
            )
    if len(fixtures) != EXPECTED_WORLDS:
        raise ValueError("KnowU mechanics world count changed")

    private_fixture = {
        "source": audit["source"],
        "worlds": [
            {
                "world_id": fixture.world_id,
                "task_id": fixture.task_id,
                "task_kind": fixture.task_kind,
                "goal_request": fixture.goal_request,
                "profile_id": fixture.profile_id,
                "retrieved_log_indices": fixture.retrieved_log_indices,
                "visible_logs": fixture.visible_logs,
                "truth_packet": fixture.truth_packet,
            }
            for fixture in fixtures
        ],
    }
    fixture_hash = _sha256_value(private_fixture)
    if enforce_frozen:
        if EXPECTED_FIXTURE_SHA256 is None:
            raise ValueError("KnowU mechanics fixture hash is not frozen")
        if fixture_hash != EXPECTED_FIXTURE_SHA256:
            raise ValueError("KnowU mechanics fixture changed")
    public_fixture = {
        "interface_version": INTERFACE_VERSION,
        "source": audit["source"],
        "private_fixture_sha256": fixture_hash,
        "task_ids": list(manifest_v2.EXPECTED_MECHANICS_IDS),
        "world_count": len(fixtures),
        "profile_prior": "uniform_within_each_supported-profile task family",
        "profile_labels_visible_to_policy": False,
        "retrieval": {
            "method": (
                "frozen lowercase alphanumeric term frequency plus "
                "task-category bonus"
            ),
            "top_k": RETRIEVED_LOG_COUNT,
        },
        "worlds": [
            {
                "world_id": fixture.world_id,
                "task_id": fixture.task_id,
                "retrieved_log_indices": fixture.retrieved_log_indices,
                "visible_logs_sha256": _sha256_value(fixture.visible_logs),
                "truth_packet_sha256": _sha256_value(fixture.truth_packet),
            }
            for fixture in fixtures
        ],
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }
    return fixtures, public_fixture


def _json_schema_format(
    name: str, properties: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def initial_response_format() -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {}
    for index in range(1, SUPPORT_SIZE + 1):
        properties[f"h{index}"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 260,
        }
    for index in range(1, QUESTION_COUNT + 1):
        properties[f"d{index}"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 60,
        }
        properties[f"q{index}"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 160,
        }
    return _json_schema_format("knowu_initial_support", properties)


def answer_response_format() -> dict[str, Any]:
    return _json_schema_format(
        "knowu_profile_answers",
        {
            f"a{index}": {
                "type": "string",
                "minLength": 1,
                "maxLength": 220,
            }
            for index in range(1, QUESTION_COUNT + 1)
        },
    )


def refresh_response_format() -> dict[str, Any]:
    return _json_schema_format(
        "knowu_refreshed_support",
        {
            f"h{index}": {
                "type": "string",
                "minLength": 1,
                "maxLength": 260,
            }
            for index in range(1, SUPPORT_SIZE + 1)
        },
    )


def judgment_response_format() -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = {}
    for label in ("initial", "q1", "q2", "q3", "q4"):
        properties[f"{label}_best_index"] = {
            "type": "integer",
            "minimum": 0,
            "maximum": SUPPORT_SIZE,
        }
        properties[f"{label}_best_score"] = {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        }
        properties[f"{label}_reason"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 240,
        }
    return _json_schema_format("knowu_truth_support_judgment", properties)


def _messages(stage: str, instruction: str, payload: dict[str, Any]):
    return [
        {
            "role": "system",
            "content": (
                f"STAGE={stage}. {instruction} Return only the requested "
                "strict JSON object. Do not reveal or invent profile labels."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                payload, ensure_ascii=True, separators=(",", ":")
            ),
        },
    ]


def initial_messages(
    goal_request: str,
    task_kind: str,
    visible_logs: Sequence[str],
) -> list[dict[str, str]]:
    task_requirement = (
        "Each hypothesis must jointly specify the likely desired computer "
        "(type/brand or ecosystem, use, OS or performance constraints, and "
        "budget when supported) and likely shopping platform."
        if task_kind == "buy_computer"
        else
        "Each hypothesis must jointly specify the likely Mattermost "
        "destination/channel and the likely leave-message tone, style, and "
        "format."
    )
    return _messages(
        "INITIAL",
        (
            "Infer four distinct, coherent task-relevant preference states "
            "from behavior logs. State preferences directly, without naming "
            "a demographic, occupation, persona, or known user type. "
            f"{task_requirement} Also propose four distinct clarification "
            "questions. Every question must target exactly one unresolved "
            "dimension, contain exactly one question mark, and must not ask "
            "for a complete product/order/message or a user/profile label."
        ),
        {
            "task_kind": task_kind,
            "user_request": goal_request,
            "behavior_logs": list(visible_logs),
            "output_contract": {
                "h1_to_h4": "four complete preference-state hypotheses",
                "d1_to_d4": "four unique atomic dimension names",
                "q1_to_q4": "one atomic question per dimension",
            },
        },
    )


def answer_messages(
    goal_request: str,
    questions: Sequence[str],
    truth_packet: dict[str, Any],
) -> list[dict[str, str]]:
    return _messages(
        "ANSWERS",
        (
            "Act as the hidden user described only by PRIVATE_USER_STATE. "
            "Answer each clarification independently and naturally in first "
            "person. Use the private state and current request; do not name "
            "a profile, persona, benchmark, or hidden state. Do not volunteer "
            "answers to other questions."
        ),
        {
            "current_request": goal_request,
            "questions": {
                f"q{index}": question
                for index, question in enumerate(questions, start=1)
            },
            "PRIVATE_USER_STATE": truth_packet,
        },
    )


def refresh_messages(
    goal_request: str,
    task_kind: str,
    visible_logs: Sequence[str],
    initial: InitialPolicy,
    question: str,
    answer: str,
) -> list[dict[str, str]]:
    requirement = (
        "Each state must jointly cover computer preference and shopping "
        "platform."
        if task_kind == "buy_computer"
        else
        "Each state must jointly cover Mattermost destination/channel and "
        "message tone/style/format."
    )
    return _messages(
        "REFRESH",
        (
            "Regenerate four distinct, coherent task-relevant preference "
            "states after exactly one clarification. This branch is "
            "independent: use only the listed behavior logs, old support, "
            "and the single question-answer pair. State preferences directly "
            "without demographic, occupation, persona, or profile labels. "
            f"{requirement}"
        ),
        {
            "task_kind": task_kind,
            "user_request": goal_request,
            "behavior_logs": list(visible_logs),
            "old_support": list(initial.hypotheses),
            "single_clarification": {
                "question": question,
                "answer": answer,
            },
        },
    )


def judgment_messages(
    goal_request: str,
    task_kind: str,
    truth_packet: dict[str, Any],
    supports: Sequence[Sequence[str]],
) -> list[dict[str, str]]:
    return _messages(
        "JUDGE",
        (
            "Act as a strict semantic endpoint. For each candidate support, "
            "find the single hypothesis that best represents the true "
            "task-relevant preference state. Score 0-100 for joint coverage "
            "of decision-critical preferences, not wording. Partial states "
            "that omit a required task dimension must score below 70. "
            "Use best_index 1-4 only when best_score is at least 70; otherwise "
            "use best_index 0. Judge supports independently."
        ),
        {
            "task_kind": task_kind,
            "user_request": goal_request,
            "TRUE_USER_STATE": truth_packet,
            "required_match_threshold": TRUTH_MATCH_THRESHOLD,
            "candidate_supports": {
                label: list(support)
                for label, support in zip(
                    ("initial", "q1", "q2", "q3", "q4"), supports
                )
            },
        },
    )


def _strict_object(text: str, expected_keys: set[str]) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("response is not the exact expected JSON object")
    return value


def _clean_text(value: Any, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError("response text field is not a string")
    text = " ".join(value.split())
    if not text or len(text) > maximum:
        raise ValueError("response text field has invalid length")
    return text


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))


def _contains_profile_label(text: str) -> bool:
    return bool(PROFILE_LABELS.intersection(_tokens(text)))


def _parse_hypotheses(
    value: dict[str, Any],
) -> tuple[str, ...]:
    hypotheses = tuple(
        _clean_text(value[f"h{index}"], maximum=260)
        for index in range(1, SUPPORT_SIZE + 1)
    )
    if len({_normalize(item) for item in hypotheses}) != SUPPORT_SIZE:
        raise ValueError("support hypotheses are not distinct")
    if any(_contains_profile_label(item) for item in hypotheses):
        raise ValueError("support hypothesis contains a forbidden profile label")
    return hypotheses


def _validate_question(question: str) -> None:
    if question.count("?") != 1 or not question.endswith("?"):
        raise ValueError("question must contain exactly one terminal question mark")
    normalized = f" {_normalize(question)} "
    if _contains_profile_label(question):
        raise ValueError("question contains a forbidden profile label")
    if any(token in normalized for token in (" and ", " or ")):
        raise ValueError("question combines dimensions")
    if "/" in question or "&" in question:
        raise ValueError("question combines dimensions")
    terminal_patterns = (
        r"\bwhat exact (computer|laptop|product|message|text)\b",
        r"\bwhich exact (computer|laptop|product|message|text)\b",
        r"\btell me (exactly )?what to (buy|order|send|write)\b",
        r"\bprovide the (complete|full) (order|message|text)\b",
    )
    if any(re.search(pattern, normalized) for pattern in terminal_patterns):
        raise ValueError("question requests a complete terminal action")


def parse_initial(text: str) -> InitialPolicy:
    expected = {
        *(f"h{index}" for index in range(1, SUPPORT_SIZE + 1)),
        *(f"d{index}" for index in range(1, QUESTION_COUNT + 1)),
        *(f"q{index}" for index in range(1, QUESTION_COUNT + 1)),
    }
    value = _strict_object(text, expected)
    hypotheses = _parse_hypotheses(value)
    dimensions = tuple(
        _clean_text(value[f"d{index}"], maximum=60)
        for index in range(1, QUESTION_COUNT + 1)
    )
    questions = tuple(
        _clean_text(value[f"q{index}"], maximum=160)
        for index in range(1, QUESTION_COUNT + 1)
    )
    if len({_normalize(item) for item in dimensions}) != QUESTION_COUNT:
        raise ValueError("question dimensions are not distinct")
    if len({_normalize(item) for item in questions}) != QUESTION_COUNT:
        raise ValueError("questions are not distinct")
    for question in questions:
        _validate_question(question)
    return InitialPolicy(hypotheses, dimensions, questions)


def parse_answers(text: str) -> tuple[str, ...]:
    expected = {f"a{index}" for index in range(1, QUESTION_COUNT + 1)}
    value = _strict_object(text, expected)
    answers = tuple(
        _clean_text(value[f"a{index}"], maximum=220)
        for index in range(1, QUESTION_COUNT + 1)
    )
    if any(_contains_profile_label(answer) for answer in answers):
        raise ValueError("answer leaks a forbidden profile label")
    return answers


def parse_refresh(text: str) -> tuple[str, ...]:
    expected = {f"h{index}" for index in range(1, SUPPORT_SIZE + 1)}
    return _parse_hypotheses(_strict_object(text, expected))


def parse_judgment(text: str) -> TruthJudgment:
    labels = ("initial", "q1", "q2", "q3", "q4")
    expected = {
        f"{label}_{suffix}"
        for label in labels
        for suffix in ("best_index", "best_score", "reason")
    }
    value = _strict_object(text, expected)
    indices: list[int] = []
    scores: list[int] = []
    reasons: list[str] = []
    for label in labels:
        index = value[f"{label}_best_index"]
        score = value[f"{label}_best_score"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= SUPPORT_SIZE
        ):
            raise ValueError("judge best index is invalid")
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= 100
        ):
            raise ValueError("judge best score is invalid")
        if (index == 0) != (score < TRUTH_MATCH_THRESHOLD):
            raise ValueError("judge index and threshold decision disagree")
        indices.append(index)
        scores.append(score)
        reasons.append(
            _clean_text(value[f"{label}_reason"], maximum=240)
        )
    return TruthJudgment(tuple(indices), tuple(scores), tuple(reasons))


def _usage(model: StructuredChatModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retry_count": int(snapshot.get("retry_count", 0)),
        "reasoning_tokens": int(
            snapshot.get("adapter_reasoning_tokens", 0)
        ),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "adapter_cost_usd": float(
            snapshot.get("adapter_cost_usd", 0.0)
        ),
        "model": snapshot,
    }


def _complete(
    model: StructuredChatModel,
    messages: list[list[dict[str, str]]],
    *,
    response_format: dict[str, Any],
    max_new_tokens: int,
) -> list[str]:
    return model.chat_complete_messages_batched_structured(
        messages,
        temperature=0.0,
        block_size=len(messages),
        response_format=response_format,
        max_new_tokens=max_new_tokens,
    )


def _synthetic_initial(index: int) -> list[dict[str, str]]:
    return initial_messages(
        f"Choose an item for synthetic request {index}.",
        "buy_computer",
        (
            "Usually compares practical devices before buying.",
            "Uses one shopping service for routine purchases.",
        ),
    )


def run_serving_gate(
    model: StructuredChatModel,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    raw: dict[str, Any] = {"interface_version": INTERFACE_VERSION}
    try:
        initial_responses = _complete(
            model,
            [_synthetic_initial(index) for index in range(4)],
            response_format=initial_response_format(),
            max_new_tokens=900,
        )
        initial = [parse_initial(response) for response in initial_responses]
        raw["initial_responses"] = initial_responses
        _checkpoint(raw_path, raw)

        answer_responses = _complete(
            model,
            [
                answer_messages(
                    "Choose a synthetic computer.",
                    initial[index].questions,
                    {
                        "shopping": {
                            "computer": "portable Linux laptop",
                            "platform": "ExampleShop",
                        }
                    },
                )
                for index in range(2)
            ],
            response_format=answer_response_format(),
            max_new_tokens=600,
        )
        answers = [parse_answers(response) for response in answer_responses]
        raw["answer_responses"] = answer_responses
        _checkpoint(raw_path, raw)

        refresh_responses = _complete(
            model,
            [
                refresh_messages(
                    "Choose a synthetic computer.",
                    "buy_computer",
                    ("Usually compares practical devices before buying.",),
                    initial[index],
                    initial[index].questions[0],
                    answers[index][0],
                )
                for index in range(2)
            ],
            response_format=refresh_response_format(),
            max_new_tokens=650,
        )
        refreshed = [
            parse_refresh(response) for response in refresh_responses
        ]
        raw["refresh_responses"] = refresh_responses
        _checkpoint(raw_path, raw)

        judge_responses = _complete(
            model,
            [
                judgment_messages(
                    "Choose a synthetic computer.",
                    "buy_computer",
                    {
                        "shopping": {
                            "computer": "portable Linux laptop",
                            "platform": "ExampleShop",
                        }
                    },
                    (
                        initial[index].hypotheses,
                        refreshed[index],
                        refreshed[index],
                        refreshed[index],
                        refreshed[index],
                    ),
                )
                for index in range(2)
            ],
            response_format=judgment_response_format(),
            max_new_tokens=900,
        )
        judgments = [
            parse_judgment(response) for response in judge_responses
        ]
        raw["judge_responses"] = judge_responses
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}", _usage(model)
        ) from exc

    gates = {
        "exact_10_physical_requests": (
            usage["physical_requests"] == EXPECTED_SERVING_REQUESTS
        ),
        "exact_10_http_attempts": (
            usage["http_attempts"] == EXPECTED_SERVING_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_strict_schema_objects_parse": (
            len(initial) == 4
            and len(answers) == 2
            and len(refreshed) == 2
            and len(judgments) == 2
        ),
        "cost_at_most_0_10": (
            usage["adapter_cost_usd"] <= SERVING_MAX_COST_USD
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": "serving",
            "model": MODEL_ID,
            "response_format": "chat_strict_json_schema",
            "expected_requests": EXPECTED_SERVING_REQUESTS,
            "reasoning_requested": False,
            "scientific_endpoint_evaluated": False,
            "repairs_or_reissues": 0,
        },
        "gates": gates,
        "usage": usage,
    }


def run_mechanics_gate(
    fixtures: Sequence[WorldFixture],
    model: StructuredChatModel,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "private_fixtures": [
            {
                "world_id": fixture.world_id,
                "task_id": fixture.task_id,
                "profile_id": fixture.profile_id,
                "visible_logs": fixture.visible_logs,
                "truth_packet": fixture.truth_packet,
            }
            for fixture in fixtures
        ],
    }
    try:
        initial_requests = [
            initial_messages(
                fixture.goal_request,
                fixture.task_kind,
                fixture.visible_logs,
            )
            for fixture in fixtures
        ]
        for request in initial_requests:
            payload = json.loads(request[-1]["content"])
            if set(payload) != {
                "task_kind",
                "user_request",
                "behavior_logs",
                "output_contract",
            }:
                raise ValueError("initial policy payload contains private fields")
        initial_responses = _complete(
            model,
            initial_requests,
            response_format=initial_response_format(),
            max_new_tokens=1000,
        )
        initial = [
            parse_initial(response) for response in initial_responses
        ]
        raw["initial_responses"] = initial_responses
        _checkpoint(raw_path, raw)

        answer_responses = _complete(
            model,
            [
                answer_messages(
                    fixture.goal_request,
                    policy.questions,
                    fixture.truth_packet,
                )
                for fixture, policy in zip(fixtures, initial)
            ],
            response_format=answer_response_format(),
            max_new_tokens=700,
        )
        answers = [
            parse_answers(response) for response in answer_responses
        ]
        raw["answer_responses"] = answer_responses
        _checkpoint(raw_path, raw)

        refresh_requests: list[list[dict[str, str]]] = []
        refresh_layout: list[tuple[int, int]] = []
        for world_index, (fixture, policy, world_answers) in enumerate(
            zip(fixtures, initial, answers)
        ):
            for question_index, (question, answer) in enumerate(
                zip(policy.questions, world_answers)
            ):
                request = refresh_messages(
                    fixture.goal_request,
                    fixture.task_kind,
                    fixture.visible_logs,
                    policy,
                    question,
                    answer,
                )
                payload = json.loads(request[-1]["content"])
                clarification = payload["single_clarification"]
                if set(clarification) != {"question", "answer"}:
                    raise ValueError("refresh branch is not isolated")
                refresh_requests.append(request)
                refresh_layout.append((world_index, question_index))
        refresh_responses = _complete(
            model,
            refresh_requests,
            response_format=refresh_response_format(),
            max_new_tokens=650,
        )
        parsed_refreshes = [
            parse_refresh(response) for response in refresh_responses
        ]
        refreshed: list[list[tuple[str, ...] | None]] = [
            [None] * QUESTION_COUNT for _ in fixtures
        ]
        for (world_index, question_index), support in zip(
            refresh_layout, parsed_refreshes
        ):
            refreshed[world_index][question_index] = support
        if any(
            support is None
            for world_supports in refreshed
            for support in world_supports
        ):
            raise ValueError("refresh branch is missing")
        raw["refresh_responses"] = refresh_responses
        _checkpoint(raw_path, raw)

        judge_responses = _complete(
            model,
            [
                judgment_messages(
                    fixture.goal_request,
                    fixture.task_kind,
                    fixture.truth_packet,
                    (
                        policy.hypotheses,
                        *[
                            support
                            for support in world_supports
                            if support is not None
                        ],
                    ),
                )
                for fixture, policy, world_supports in zip(
                    fixtures, initial, refreshed
                )
            ],
            response_format=judgment_response_format(),
            max_new_tokens=1100,
        )
        judgments = [
            parse_judgment(response) for response in judge_responses
        ]
        raw["judge_responses"] = judge_responses
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}", _usage(model)
        ) from exc

    world_metrics = []
    for fixture, policy, judgment in zip(fixtures, initial, judgments):
        initial_present = judgment.present[0]
        branch_present = judgment.present[1:]
        score_gains = tuple(
            score - judgment.best_scores[0]
            for score in judgment.best_scores[1:]
        )
        world_metrics.append(
            {
                "world_id": fixture.world_id,
                "task_id": fixture.task_id,
                "question_dimensions": policy.dimensions,
                "questions": policy.questions,
                "initial_truth_present": initial_present,
                "branch_truth_present": branch_present,
                "truth_support_scores": judgment.best_scores,
                "truth_support_score_gains": score_gains,
                "support_gain_range": max(score_gains) - min(score_gains),
                "truth_entry_branches": [
                    index + 1
                    for index, present in enumerate(branch_present)
                    if not initial_present and present
                ],
            }
        )
    missing_worlds = [
        row for row in world_metrics if not row["initial_truth_present"]
    ]
    entry_worlds = [
        row for row in missing_worlds if row["truth_entry_branches"]
    ]
    expected_requests = len(fixtures) * (2 + QUESTION_COUNT + 1)
    gates = {
        "exact_42_physical_requests": (
            expected_requests == EXPECTED_MECHANICS_REQUESTS
            and usage["physical_requests"] == expected_requests
        ),
        "exact_42_http_attempts": (
            usage["http_attempts"] == expected_requests
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_policy_answer_refresh_and_judge_objects_parse": True,
        "profile_labels_hidden_from_support_policy": True,
        "each_refresh_contains_one_question_answer_pair": True,
        "initial_truth_missing_in_at_least_one_world": bool(missing_worlds),
        "missing_truth_enters_after_at_least_one_atomic_question": bool(
            entry_worlds
        ),
        "positive_truth_support_score_gain_exists": any(
            max(row["truth_support_score_gains"]) > 0
            for row in world_metrics
        ),
        "cost_at_most_0_75": (
            usage["adapter_cost_usd"] <= MECHANICS_MAX_COST_USD
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": "mechanics",
            "model": MODEL_ID,
            "response_format": "chat_strict_json_schema",
            "world_count": len(fixtures),
            "support_size": SUPPORT_SIZE,
            "question_count": QUESTION_COUNT,
            "truth_match_threshold": TRUTH_MATCH_THRESHOLD,
            "expected_requests": EXPECTED_MECHANICS_REQUESTS,
            "policy_requests": len(fixtures) * (1 + QUESTION_COUNT),
            "profile_conditioned_user_requests": len(fixtures),
            "post_policy_judge_requests": len(fixtures),
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "branch_isolation": (
                "one physical refresh request per world/question branch"
            ),
        },
        "summary": {
            "worlds": len(world_metrics),
            "initial_truth_missing_worlds": len(missing_worlds),
            "worlds_with_truth_entry": len(entry_worlds),
            "truth_entry_events": sum(
                len(row["truth_entry_branches"]) for row in entry_worlds
            ),
            "mean_support_gain_range": (
                sum(row["support_gain_range"] for row in world_metrics)
                / len(world_metrics)
            ),
        },
        "world_metrics": world_metrics,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self) -> None:
        self.requests = 0

    @staticmethod
    def _stage(messages: list[dict[str, str]]) -> str:
        return messages[0]["content"].split("STAGE=", 1)[1].split(".", 1)[0]

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, response_format, max_new_tokens
        responses = []
        for messages in batch_messages:
            stage = self._stage(messages)
            if stage == "INITIAL":
                value = {
                    **{
                        f"h{index}": (
                            f"Computer state {index} uses Shop{index} with "
                            f"operating system {index} within budget {index}."
                        )
                        for index in range(1, 5)
                    },
                    **{
                        f"d{index}": f"dimension {index}"
                        for index in range(1, 5)
                    },
                    **{
                        f"q{index}": f"What is preference dimension {index}?"
                        for index in range(1, 5)
                    },
                }
            elif stage == "ANSWERS":
                value = {
                    f"a{index}": f"My preference for dimension {index} is value {index}."
                    for index in range(1, 5)
                }
            elif stage == "REFRESH":
                value = {
                    f"h{index}": (
                        f"Updated computer state {index} uses Shop{index} "
                        f"with operating system {index} within budget {index}."
                    )
                    for index in range(1, 5)
                }
            elif stage == "JUDGE":
                value = {}
                for label in ("initial", "q1", "q2", "q3", "q4"):
                    present = label == "q1"
                    value[f"{label}_best_index"] = 1 if present else 0
                    value[f"{label}_best_score"] = 90 if present else 50
                    value[f"{label}_reason"] = (
                        "Joint task preferences match."
                        if present
                        else "A required task preference is missing."
                    )
            else:  # pragma: no cover
                raise AssertionError(stage)
            responses.append(json.dumps(value, separators=(",", ":")))
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


def _build_model(config: Config) -> StructuredChatModel:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID or spec.backend != "openrouter":
        raise ValueError("KnowU mechanics config selects the wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("fixture", "serving", "mechanics"), required=True
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-fixture-hash", action="store_true")
    args = parser.parse_args()

    fixtures, public_fixture = build_world_fixtures(
        args.source_root.resolve(),
        args.manifest.resolve(),
        enforce_frozen=not args.show_fixture_hash,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "fixture":
        _checkpoint(args.output_dir / "FIXTURE.json", public_fixture)
        print(json.dumps(public_fixture, indent=2, sort_keys=True))
        return
    if args.config is None or args.private_raw_dir is None or not args.run_id:
        parser.error(
            "--config, --private-raw-dir, and --run-id are required "
            "for serving/mechanics"
        )

    config = load_config(args.config)
    config.run_id = args.run_id
    config.openrouter_concurrency = (
        EXPECTED_SERVING_REQUESTS
        if args.stage == "serving"
        else 24
    )
    config.openrouter_projected_cost_usd = (
        0.04 if args.stage == "serving" else 0.50
    )
    config.openrouter_run_budget_usd = (
        SERVING_MAX_COST_USD
        if args.stage == "serving"
        else MECHANICS_MAX_COST_USD
    )
    config.openrouter_max_output_tokens = 1_100
    config.openrouter_max_retries = 0
    config.log_path = args.output_dir / "run.log"
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: StructuredChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )

    try:
        result = (
            run_serving_gate(model, raw_path=raw_path)
            if args.stage == "serving"
            else run_mechanics_gate(fixtures, model, raw_path=raw_path)
        )
        result["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
        result["protocol"]["fixture_sha256"] = public_fixture[
            "private_fixture_sha256"
        ]
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "GATE_FAILURE.json", failure)
        raise
    output_path = args.output_dir / f"{args.stage.upper()}.json"
    _checkpoint(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output_path),
                "gates": result["gates"],
                "usage": result["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
