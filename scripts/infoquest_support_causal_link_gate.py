#!/usr/bin/env python3
"""Test whether InfoQuest support refresh improves next-turn discovery."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Protocol, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts import infoquest_cached_trajectory_opportunity as trajectory_audit
from scripts import infoquest_llm_bed_manifest as source_manifest


INTERFACE_VERSION = "infoquest-support-causal-link-1"
MECHANICS_IDS = (0, 1, 4)
SUPPORT_SIZE = 8
ROOT_COUNT = 5
TRUTH_THRESHOLD = 70
GENERATOR_MODEL_ID = "openai/gpt-5.4"
SIMULATOR_MODEL_ID = "google/gemini-2.5-flash"
SUPPORT_JUDGE_MODEL_ID = "google/gemma-4-26b-a4b-it"
CHECKLIST_JUDGE_MODEL_ID = "openai/gpt-5.4-mini"
EXPECTED_SERVING_REQUESTS = 10
EXPECTED_MECHANICS_REQUESTS = 165
SERVING_MAX_COST_USD = 0.15
MECHANICS_MAX_COST_USD = 1.0
SHUFFLE_PERMUTATION = (1, 3, 4, 2, 0)
BASELINE_SHA256 = trajectory_audit.BASELINE_SHA256[0]
EXPECTED_PRIVATE_FIXTURE_SHA256 = (
    "63248345976e884b9fac127033415fe68bacb3a0e32a32e498285851719251cc"
)
EXPECTED_FIXTURE_HASHES = {
    "I0W1": "d9f287f43ed84c5f6bfeb27c12c19c86ae8897171aa8f3105ee6f1241514b456",
    "I0W2": "84a3f1ceea996bf4a4e62fc3974ae96c11e7baee22d219cd58e37dd08b5597aa",
    "I1W1": "3f1ed6f0dfbfecaa1203d97cbb75c623c1ffac1df30d0b76d127ea1109cadadb",
    "I1W2": "3c087a81846557e5aa3c8388d708b390184dab69e4a3f45a75a1c7bcfe465d84",
    "I4W1": "316ec07b6f019310ae688bb19b678525ecfab2f8f4f0f7212ec7cc3d95b421c1",
    "I4W2": "7c0fac177a238c883a7db2db2b6c8c7caa98e76adb4bd557c272a009930f6d72",
}


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class WorldFixture:
    fixture_id: str
    record_id: int
    world: int
    seed_message: str
    simulator_system: str
    truth_packet: dict[str, Any]
    checklist: tuple[str, ...]


@dataclass(frozen=True)
class InitialPolicy:
    hypotheses: tuple[str, ...]
    roots: tuple[str, ...]


@dataclass(frozen=True)
class RefreshPolicy:
    hypotheses: tuple[str, ...]
    followup: str


@dataclass(frozen=True)
class SupportJudgment:
    indices: tuple[int, ...]
    scores: tuple[int, ...]


@dataclass(frozen=True)
class ChecklistJudgment:
    immediate: tuple[tuple[int, ...], ...]
    dynamic: tuple[tuple[int, ...], ...]
    fixed: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ModelBundle:
    generator: ChatModel
    simulator: ChatModel
    support_judge: ChatModel
    checklist_judge: ChatModel


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _checkpoint(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_jsonl_by_id(path: Path, expected_sha256: str) -> dict[int, dict]:
    return trajectory_audit._rows_by_id(
        path,
        expected_sha256=expected_sha256,
    )


def build_fixtures(
    *,
    settings_path: Path,
    baseline_path: Path,
) -> tuple[list[WorldFixture], dict[str, Any]]:
    settings = _load_jsonl_by_id(
        settings_path,
        source_manifest.SOURCE_SHA256["settings"],
    )
    baseline = _load_jsonl_by_id(baseline_path, BASELINE_SHA256)
    fixtures: list[WorldFixture] = []
    for record_id in MECHANICS_IDS:
        setting_row = settings[record_id]
        for world in (1, 2):
            setting = setting_row[f"setting{world}"]
            checklist = setting["checklist"]
            if (
                not isinstance(checklist, list)
                or len(checklist) != 5
                or not all(
                    isinstance(item, str) and item.strip()
                    for item in checklist
                )
            ):
                raise ValueError("InfoQuest mechanics checklist changed")
            history = baseline[record_id][f"user_history{world}"]
            if (
                not isinstance(history, list)
                or not history
                or history[0].get("role") != "system"
                or not isinstance(history[0].get("content"), str)
                or not history[0]["content"].strip()
            ):
                raise ValueError("InfoQuest simulator system prompt changed")
            truth_packet = {
                key: setting[key]
                for key in (
                    "description",
                    "goal",
                    "obstacle",
                    "constraints",
                    "solution",
                    "persona",
                )
            }
            fixtures.append(
                WorldFixture(
                    fixture_id=f"I{record_id}W{world}",
                    record_id=record_id,
                    world=world,
                    seed_message=setting_row["seed_message"],
                    simulator_system=history[0]["content"],
                    truth_packet=truth_packet,
                    checklist=tuple(checklist),
                )
            )
    private_fixture = [
        {
            "fixture_id": fixture.fixture_id,
            "seed_message": fixture.seed_message,
            "simulator_system": fixture.simulator_system,
            "truth_packet": fixture.truth_packet,
            "checklist": fixture.checklist,
        }
        for fixture in fixtures
    ]
    private_fixture_sha256 = _sha256_value(private_fixture)
    fixture_hashes = {
        fixture.fixture_id: _sha256_value(row)
        for fixture, row in zip(fixtures, private_fixture)
    }
    if private_fixture_sha256 != EXPECTED_PRIVATE_FIXTURE_SHA256:
        raise ValueError("InfoQuest private mechanics fixture changed")
    if fixture_hashes != EXPECTED_FIXTURE_HASHES:
        raise ValueError("InfoQuest per-world mechanics fixtures changed")
    public_fixture = {
        "interface_version": INTERFACE_VERSION,
        "mechanics_ids": list(MECHANICS_IDS),
        "worlds": len(fixtures),
        "settings_sha256": source_manifest.SOURCE_SHA256["settings"],
        "baseline_sha256": BASELINE_SHA256,
        "private_fixture_sha256": private_fixture_sha256,
        "fixture_hashes": fixture_hashes,
        "semantic_content_emitted": False,
        "opportunity_v1_read": False,
        "opportunity_v2_read": False,
        "development_read": False,
        "holdout_read": False,
        "openrouter_calls": 0,
        "oatml_jobs": 0,
    }
    return fixtures, public_fixture


def _strip_single_fence(response: str) -> str:
    text = response.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        raise ValueError("response has an incomplete Markdown fence")
    if lines[0].strip() not in {"```", "```json", "```JSON"}:
        raise ValueError("response has an unsupported Markdown fence")
    if any(line.strip().startswith("```") for line in lines[1:-1]):
        raise ValueError("response contains nested Markdown fences")
    return "\n".join(lines[1:-1]).strip()


def _parse_exact_object(response: str, expected_keys: set[str]) -> dict:
    try:
        value = json.loads(_strip_single_fence(response))
    except json.JSONDecodeError as exc:
        raise ValueError("response is not one JSON object") from exc
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("response JSON fields do not match the schema")
    return value


def _clean_text(value: Any, *, maximum: int = 600) -> str:
    if not isinstance(value, str):
        raise ValueError("response field is not a string")
    text = " ".join(value.split())
    if not text or len(text) > maximum:
        raise ValueError("response field has invalid length")
    return text


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _parse_atomic_question(value: Any) -> str:
    question = _clean_text(value, maximum=300)
    lowered = f" {question.casefold()} "
    if (
        question.count("?") != 1
        or not question.endswith("?")
        or " and " in lowered
        or " or " in lowered
        or "/" in question
        or ";" in question
    ):
        raise ValueError("question is not one atomic question")
    return question


def parse_initial(response: str) -> InitialPolicy:
    expected = {
        *(f"h{index}" for index in range(1, SUPPORT_SIZE + 1)),
        *(f"q{index}" for index in range(1, ROOT_COUNT + 1)),
    }
    value = _parse_exact_object(response, expected)
    hypotheses = tuple(
        _clean_text(value[f"h{index}"])
        for index in range(1, SUPPORT_SIZE + 1)
    )
    roots = tuple(
        _parse_atomic_question(value[f"q{index}"])
        for index in range(1, ROOT_COUNT + 1)
    )
    if len({_normalize(item) for item in hypotheses}) != SUPPORT_SIZE:
        raise ValueError("initial hypotheses are not distinct")
    if len({_normalize(item) for item in roots}) != ROOT_COUNT:
        raise ValueError("initial roots are not distinct")
    return InitialPolicy(hypotheses=hypotheses, roots=roots)


def parse_refresh(response: str) -> RefreshPolicy:
    expected = {
        *(f"h{index}" for index in range(1, SUPPORT_SIZE + 1)),
        "followup",
    }
    value = _parse_exact_object(response, expected)
    hypotheses = tuple(
        _clean_text(value[f"h{index}"])
        for index in range(1, SUPPORT_SIZE + 1)
    )
    if len({_normalize(item) for item in hypotheses}) != SUPPORT_SIZE:
        raise ValueError("refreshed hypotheses are not distinct")
    return RefreshPolicy(
        hypotheses=hypotheses,
        followup=_parse_atomic_question(value["followup"]),
    )


def parse_fixed_followup(response: str) -> str:
    value = _parse_exact_object(response, {"followup"})
    return _parse_atomic_question(value["followup"])


def parse_support_judgment(response: str) -> SupportJudgment:
    text = _strip_single_fence(response)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    labels = ["I", *(f"Q{index}" for index in range(1, ROOT_COUNT + 1))]
    if len(lines) != len(labels):
        raise ValueError("support judgment does not contain six lines")
    indices: list[int] = []
    scores: list[int] = []
    for line, label in zip(lines, labels):
        match = re.fullmatch(rf"{label}\|(\d{{2}})\|(\d{{1,3}})", line)
        if match is None:
            raise ValueError("support judgment line is malformed")
        index = int(match.group(1))
        score = int(match.group(2))
        if not 0 <= index <= SUPPORT_SIZE or not 0 <= score <= 100:
            raise ValueError("support judgment value is out of range")
        if (index == 0) != (score < TRUTH_THRESHOLD):
            raise ValueError("support judgment threshold and index disagree")
        indices.append(index)
        scores.append(score)
    return SupportJudgment(tuple(indices), tuple(scores))


def _parse_bits(value: str) -> tuple[int, ...]:
    if re.fullmatch(r"[01]{5}", value) is None:
        raise ValueError("checklist bit string is malformed")
    return tuple(int(bit) for bit in value)


def parse_checklist_judgment(response: str) -> ChecklistJudgment:
    text = _strip_single_fence(response)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != ROOT_COUNT:
        raise ValueError("checklist judgment does not contain five lines")
    immediate: list[tuple[int, ...]] = []
    dynamic: list[tuple[int, ...]] = []
    fixed: list[tuple[int, ...]] = []
    for root_index, line in enumerate(lines, start=1):
        match = re.fullmatch(
            rf"Q{root_index}\|([01]{{5}})\|([01]{{5}})\|([01]{{5}})",
            line,
        )
        if match is None:
            raise ValueError("checklist judgment line is malformed")
        immediate_bits = _parse_bits(match.group(1))
        dynamic_bits = _parse_bits(match.group(2))
        fixed_bits = _parse_bits(match.group(3))
        if any(
            immediate_bit > total_bit
            for immediate_bit, total_bit in zip(
                immediate_bits,
                dynamic_bits,
            )
        ) or any(
            immediate_bit > total_bit
            for immediate_bit, total_bit in zip(
                immediate_bits,
                fixed_bits,
            )
        ):
            raise ValueError("two-turn checklist result loses immediate credit")
        immediate.append(immediate_bits)
        dynamic.append(dynamic_bits)
        fixed.append(fixed_bits)
    return ChecklistJudgment(
        tuple(immediate),
        tuple(dynamic),
        tuple(fixed),
    )


def initial_messages(seed_message: str) -> list[dict[str, str]]:
    request = {"ambiguous_seed_message": seed_message}
    return [
        {
            "role": "system",
            "content": (
                "STAGE=INITIAL. The user sent an ambiguous request. Generate "
                "eight distinct, concrete latent contexts that could explain "
                "their hidden goal, obstacle, and constraints. Do not use "
                "catch-all hypotheses. Then generate five distinct atomic "
                "clarification questions that could reveal different hidden "
                "information. Questions must each contain one question mark, "
                "end with it, and contain neither and nor or. Output only one "
                "JSON object with string fields h1..h8 and q1..q5."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def simulator_root_messages(
    simulator_system: str,
    root: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": simulator_system},
        {"role": "user", "content": root},
    ]


def refresh_messages(
    seed_message: str,
    initial: InitialPolicy,
    root: str,
    answer: str,
) -> list[dict[str, str]]:
    request = {
        "ambiguous_seed_message": seed_message,
        "initial_hypotheses": list(initial.hypotheses),
        "clarification_question": root,
        "user_answer": answer,
    }
    return [
        {
            "role": "system",
            "content": (
                "STAGE=REFRESH. Regenerate eight distinct concrete latent "
                "contexts using the complete question and answer as binding "
                "evidence. New contexts may enter and old contexts may leave. "
                "Then choose one atomic follow-up question using the refreshed "
                "support. Output only one JSON object with string fields "
                "h1..h8 and followup. The followup must contain one question "
                "mark, end with it, and contain neither and nor or."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def fixed_followup_messages(
    seed_message: str,
    initial: InitialPolicy,
    root: str,
    answer: str,
) -> list[dict[str, str]]:
    request = {
        "ambiguous_seed_message": seed_message,
        "fixed_initial_hypotheses": list(initial.hypotheses),
        "clarification_question": root,
        "user_answer": answer,
    }
    return [
        {
            "role": "system",
            "content": (
                "STAGE=FIXED. Keep the supplied initial hypothesis support "
                "fixed: do not add, remove, rewrite, or replace hypotheses. "
                "Using only that fixed support and the observed answer, choose "
                "one atomic follow-up question. Output only one JSON object "
                "with the string field followup. It must contain one question "
                "mark, end with it, and contain neither and nor or."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def simulator_followup_messages(
    simulator_system: str,
    root: str,
    root_answer: str,
    followup: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": simulator_system},
        {"role": "user", "content": root},
        {"role": "assistant", "content": root_answer},
        {"role": "user", "content": followup},
    ]


def support_judge_messages(
    fixture: WorldFixture,
    initial: InitialPolicy,
    refreshed: Sequence[RefreshPolicy],
) -> list[dict[str, str]]:
    request = {
        "hidden_context": fixture.truth_packet,
        "supports": {
            "I": list(initial.hypotheses),
            **{
                f"Q{index}": list(policy.hypotheses)
                for index, policy in enumerate(refreshed, start=1)
            },
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "STAGE=SUPPORT_JUDGE. For each support, find the one hypothesis "
                "that best jointly covers the hidden goal, obstacle, and "
                "important constraints. Score semantic joint coverage 0..100. "
                f"A score below {TRUTH_THRESHOLD} means no hypothesis covers "
                "the hidden context and must use index 00; otherwise use the "
                "1-based best index 01..08. Output exactly six lines and no "
                "other text: I|index|score, then Q1..Q5 in order."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def checklist_judge_messages(
    fixture: WorldFixture,
    initial: InitialPolicy,
    root_answers: Sequence[str],
    dynamic: Sequence[RefreshPolicy],
    dynamic_answers: Sequence[str],
    fixed_followups: Sequence[str],
    fixed_answers: Sequence[str],
) -> list[dict[str, str]]:
    transcripts = []
    for index in range(ROOT_COUNT):
        transcripts.append(
            {
                "label": f"Q{index + 1}",
                "root": initial.roots[index],
                "root_answer": root_answers[index],
                "dynamic_followup": dynamic[index].followup,
                "dynamic_answer": dynamic_answers[index],
                "fixed_followup": fixed_followups[index],
                "fixed_answer": fixed_answers[index],
            }
        )
    request = {
        "checklist_items": list(fixture.checklist),
        "transcripts": transcripts,
    }
    return [
        {
            "role": "system",
            "content": (
                "STAGE=CHECKLIST_JUDGE. For each Q1..Q5 transcript, mark each "
                "of the five checklist information needs as discovered only "
                "when the policy question and hidden-user answer explicitly "
                "address it. Produce three five-bit strings: after root only, "
                "after root plus dynamic followup, and after root plus fixed "
                "followup. Later strings cannot lose earlier credit. Output "
                "exactly five lines and no other text: "
                "Qn|immediate_bits|dynamic_bits|fixed_bits."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _complete(
    model: ChatModel,
    messages: list[list[dict[str, str]]],
    *,
    max_new_tokens: int,
) -> list[str]:
    return model.chat_complete_messages_batched(
        messages,
        temperature=0.0,
        block_size=len(messages),
        max_new_tokens=max_new_tokens,
    )


def _model_usage(model: ChatModel) -> dict[str, Any]:
    return model.usage_snapshot()


def aggregate_usage(models: ModelBundle) -> dict[str, Any]:
    snapshots = {
        "generator": _model_usage(models.generator),
        "simulator": _model_usage(models.simulator),
        "support_judge": _model_usage(models.support_judge),
        "checklist_judge": _model_usage(models.checklist_judge),
    }
    return {
        "physical_requests": sum(
            int(value.get("adapter_requests", 0))
            for value in snapshots.values()
        ),
        "http_attempts": sum(
            int(value.get("http_attempts", 0))
            for value in snapshots.values()
        ),
        "retry_count": sum(
            int(value.get("retry_count", 0))
            for value in snapshots.values()
        ),
        "reasoning_tokens": sum(
            int(value.get("adapter_reasoning_tokens", 0))
            for value in snapshots.values()
        ),
        "forced_exits": sum(
            int(value.get("forced_exits", 0))
            for value in snapshots.values()
        ),
        "adapter_cost_usd": sum(
            float(value.get("adapter_cost_usd", 0.0))
            for value in snapshots.values()
        ),
        "models": snapshots,
    }


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
        rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("correlation vectors have invalid lengths")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left)
    right_scale = sum((value - right_mean) ** 2 for value in right)
    if left_scale == 0 or right_scale == 0:
        return 0.0
    return numerator / (left_scale * right_scale) ** 0.5


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_rankdata(left), _rankdata(right))


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _count_bits(bits: Sequence[int]) -> int:
    return sum(bits)


def _public_metrics(
    fixtures: Sequence[WorldFixture],
    initials: dict[int, InitialPolicy],
    root_answers: Sequence[Sequence[str]],
    refreshed: Sequence[Sequence[RefreshPolicy]],
    fixed_followups: Sequence[Sequence[str]],
    dynamic_answers: Sequence[Sequence[str]],
    fixed_answers: Sequence[Sequence[str]],
    support_judgments: Sequence[SupportJudgment],
    checklist_judgments: Sequence[ChecklistJudgment],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    fixture_metrics: list[dict[str, Any]] = []
    all_support_scores: list[float] = []
    all_dynamic_total: list[float] = []
    all_immediate: list[float] = []
    all_shuffled_scores: list[float] = []
    all_dynamic_minus_fixed: list[float] = []
    dynamic_selected_values: list[float] = []
    myopic_selected_values: list[float] = []
    random_expected_values: list[float] = []
    selected_dynamic_minus_fixed: list[float] = []
    selection_wins = selection_ties = selection_losses = 0

    for fixture_index, fixture in enumerate(fixtures):
        initial = initials[fixture.record_id]
        support = support_judgments[fixture_index]
        checklist = checklist_judgments[fixture_index]
        scores = list(support.scores[1:])
        immediate = [
            _count_bits(bits) for bits in checklist.immediate
        ]
        dynamic_total = [
            _count_bits(bits) for bits in checklist.dynamic
        ]
        fixed_total = [_count_bits(bits) for bits in checklist.fixed]
        dynamic_incremental = [
            total - first for total, first in zip(dynamic_total, immediate)
        ]
        shuffled_scores = [scores[index] for index in SHUFFLE_PERMUTATION]
        dynamic_index = _argmax(scores)
        myopic_index = _argmax(immediate)
        dynamic_value = dynamic_total[dynamic_index]
        myopic_value = dynamic_total[myopic_index]
        random_value = sum(dynamic_total) / ROOT_COUNT
        if dynamic_value > myopic_value:
            selection_wins += 1
        elif dynamic_value < myopic_value:
            selection_losses += 1
        else:
            selection_ties += 1
        continuation_differences = [
            dynamic_value_ - fixed_value_
            for dynamic_value_, fixed_value_ in zip(
                dynamic_total,
                fixed_total,
            )
        ]
        row = {
            "fixture_id": fixture.fixture_id,
            "initial_truth_index": support.indices[0],
            "initial_truth_score": support.scores[0],
            "refreshed_truth_scores": scores,
            "refreshed_truth_indices": list(support.indices[1:]),
            "immediate_checklist_counts": immediate,
            "dynamic_checklist_counts": dynamic_total,
            "fixed_checklist_counts": fixed_total,
            "dynamic_incremental_counts": dynamic_incremental,
            "support_score_range": max(scores) - min(scores),
            "support_vs_dynamic_total_spearman": _spearman(
                scores,
                dynamic_total,
            ),
            "support_vs_dynamic_incremental_spearman": _spearman(
                scores,
                dynamic_incremental,
            ),
            "immediate_vs_dynamic_total_spearman": _spearman(
                immediate,
                dynamic_total,
            ),
            "shuffled_support_vs_dynamic_total_spearman": _spearman(
                shuffled_scores,
                dynamic_total,
            ),
            "dynamic_selected_root": dynamic_index + 1,
            "myopic_selected_root": myopic_index + 1,
            "dynamic_selected_total": dynamic_value,
            "myopic_selected_total": myopic_value,
            "random_expected_total": random_value,
            "dynamic_selected_fixed_total": fixed_total[dynamic_index],
            "root_hashes": [
                _sha256_value(root) for root in initial.roots
            ],
            "root_answer_hashes": [
                _sha256_value(value)
                for value in root_answers[fixture_index]
            ],
            "refreshed_support_hashes": [
                _sha256_value(policy.hypotheses)
                for policy in refreshed[fixture_index]
            ],
            "dynamic_followup_hashes": [
                _sha256_value(policy.followup)
                for policy in refreshed[fixture_index]
            ],
            "fixed_followup_hashes": [
                _sha256_value(value)
                for value in fixed_followups[fixture_index]
            ],
            "dynamic_answer_hashes": [
                _sha256_value(value)
                for value in dynamic_answers[fixture_index]
            ],
            "fixed_answer_hashes": [
                _sha256_value(value)
                for value in fixed_answers[fixture_index]
            ],
        }
        fixture_metrics.append(row)
        all_support_scores.extend(scores)
        all_dynamic_total.extend(dynamic_total)
        all_immediate.extend(immediate)
        all_shuffled_scores.extend(shuffled_scores)
        all_dynamic_minus_fixed.extend(continuation_differences)
        dynamic_selected_values.append(dynamic_value)
        myopic_selected_values.append(myopic_value)
        random_expected_values.append(random_value)
        selected_dynamic_minus_fixed.append(
            dynamic_value - fixed_total[dynamic_index]
        )

    mean_fixture_support_rho = sum(
        row["support_vs_dynamic_total_spearman"]
        for row in fixture_metrics
    ) / len(fixture_metrics)
    mean_fixture_immediate_rho = sum(
        row["immediate_vs_dynamic_total_spearman"]
        for row in fixture_metrics
    ) / len(fixture_metrics)
    mean_fixture_shuffled_rho = sum(
        row["shuffled_support_vs_dynamic_total_spearman"]
        for row in fixture_metrics
    ) / len(fixture_metrics)
    metrics = {
        "fixtures": len(fixtures),
        "root_world_cells": len(fixtures) * ROOT_COUNT,
        "initial_truth_omissions": sum(
            judgment.scores[0] < TRUTH_THRESHOLD
            for judgment in support_judgments
        ),
        "truth_entry_events": sum(
            judgment.scores[0] < TRUTH_THRESHOLD
            and score >= TRUTH_THRESHOLD
            for judgment in support_judgments
            for score in judgment.scores[1:]
        ),
        "support_gain_at_least_10_events": sum(
            score - judgment.scores[0] >= 10
            for judgment in support_judgments
            for score in judgment.scores[1:]
        ),
        "fixtures_with_support_range_at_least_10": sum(
            row["support_score_range"] >= 10 for row in fixture_metrics
        ),
        "pooled_support_vs_dynamic_total_spearman": _spearman(
            all_support_scores,
            all_dynamic_total,
        ),
        "pooled_support_vs_dynamic_incremental_spearman": _spearman(
            all_support_scores,
            [
                total - first
                for total, first in zip(
                    all_dynamic_total,
                    all_immediate,
                )
            ],
        ),
        "pooled_immediate_vs_dynamic_total_spearman": _spearman(
            all_immediate,
            all_dynamic_total,
        ),
        "pooled_shuffled_support_vs_dynamic_total_spearman": _spearman(
            all_shuffled_scores,
            all_dynamic_total,
        ),
        "mean_fixture_support_vs_dynamic_total_spearman": (
            mean_fixture_support_rho
        ),
        "mean_fixture_immediate_vs_dynamic_total_spearman": (
            mean_fixture_immediate_rho
        ),
        "mean_fixture_shuffled_vs_dynamic_total_spearman": (
            mean_fixture_shuffled_rho
        ),
        "dynamic_selected_mean_total": (
            sum(dynamic_selected_values) / len(dynamic_selected_values)
        ),
        "myopic_selected_mean_total": (
            sum(myopic_selected_values) / len(myopic_selected_values)
        ),
        "random_expected_mean_total": (
            sum(random_expected_values) / len(random_expected_values)
        ),
        "dynamic_minus_myopic_selected_mean": (
            sum(
                dynamic - myopic
                for dynamic, myopic in zip(
                    dynamic_selected_values,
                    myopic_selected_values,
                )
            )
            / len(fixtures)
        ),
        "dynamic_minus_random_selected_mean": (
            sum(
                dynamic - random_value
                for dynamic, random_value in zip(
                    dynamic_selected_values,
                    random_expected_values,
                )
            )
            / len(fixtures)
        ),
        "dynamic_selection_wins_ties_losses_vs_myopic": [
            selection_wins,
            selection_ties,
            selection_losses,
        ],
        "dynamic_minus_fixed_continuation_mean_all_roots": (
            sum(all_dynamic_minus_fixed) / len(all_dynamic_minus_fixed)
        ),
        "dynamic_minus_fixed_continuation_mean_selected_roots": (
            sum(selected_dynamic_minus_fixed)
            / len(selected_dynamic_minus_fixed)
        ),
        "dynamic_continuation_wins_ties_losses_all_roots": [
            sum(value > 0 for value in all_dynamic_minus_fixed),
            sum(value == 0 for value in all_dynamic_minus_fixed),
            sum(value < 0 for value in all_dynamic_minus_fixed),
        ],
    }
    gates = {
        "at_least_1_initial_truth_omission": (
            metrics["initial_truth_omissions"] >= 1
        ),
        "at_least_2_truth_entry_events": (
            metrics["truth_entry_events"] >= 2
        ),
        "at_least_4_fixtures_have_support_range_10": (
            metrics["fixtures_with_support_range_at_least_10"] >= 4
        ),
        "mean_fixture_support_rho_at_least_0_20": (
            metrics["mean_fixture_support_vs_dynamic_total_spearman"] >= 0.20
        ),
        "pooled_support_rho_at_least_0_25": (
            metrics["pooled_support_vs_dynamic_total_spearman"] >= 0.25
        ),
        "support_rho_beats_immediate_by_0_10": (
            metrics["mean_fixture_support_vs_dynamic_total_spearman"]
            >= metrics["mean_fixture_immediate_vs_dynamic_total_spearman"]
            + 0.10
        ),
        "support_rho_beats_shuffled_by_0_15": (
            metrics["mean_fixture_support_vs_dynamic_total_spearman"]
            >= metrics["mean_fixture_shuffled_vs_dynamic_total_spearman"]
            + 0.15
        ),
        "dynamic_selection_beats_myopic_by_0_15": (
            metrics["dynamic_minus_myopic_selected_mean"] >= 0.15
        ),
        "dynamic_selection_beats_random_by_0_15": (
            metrics["dynamic_minus_random_selected_mean"] >= 0.15
        ),
        "dynamic_selection_more_wins_than_losses": (
            selection_wins > selection_losses
        ),
        "dynamic_continuation_beats_fixed_by_0_10": (
            metrics["dynamic_minus_fixed_continuation_mean_all_roots"] >= 0.10
        ),
        "dynamic_continuation_more_wins_than_losses": (
            metrics["dynamic_continuation_wins_ties_losses_all_roots"][0]
            > metrics["dynamic_continuation_wins_ties_losses_all_roots"][2]
        ),
    }
    return metrics, fixture_metrics, gates


def run_serving_gate(
    models: ModelBundle,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    raw: dict[str, Any] = {"interface_version": INTERFACE_VERSION}
    try:
        initial_raw = _complete(
            models.generator,
            [initial_messages(f"Synthetic ambiguous request {index}.") for index in range(2)],
            max_new_tokens=1_200,
        )
        raw["initial"] = initial_raw
        _checkpoint(raw_path, raw)
        initial = [parse_initial(value) for value in initial_raw]

        synthetic_system = (
            "You are a hidden user. Answer the latest specific question in "
            "one concise sentence and reveal at most one detail."
        )
        root_answer_raw = _complete(
            models.simulator,
            [
                simulator_root_messages(
                    synthetic_system,
                    initial[index].roots[0],
                )
                for index in range(2)
            ],
            max_new_tokens=160,
        )
        root_answers = [
            _clean_text(value, maximum=1_000) for value in root_answer_raw
        ]
        raw["root_answers"] = root_answer_raw
        _checkpoint(raw_path, raw)

        refresh_raw = _complete(
            models.generator,
            [
                refresh_messages(
                    f"Synthetic ambiguous request {index}.",
                    initial[index],
                    initial[index].roots[0],
                    root_answers[index],
                )
                for index in range(2)
            ],
            max_new_tokens=1_100,
        )
        refreshed = [parse_refresh(value) for value in refresh_raw]
        raw["refresh"] = refresh_raw
        _checkpoint(raw_path, raw)

        fixed_raw = _complete(
            models.generator,
            [
                fixed_followup_messages(
                    "Synthetic ambiguous request.",
                    initial[0],
                    initial[0].roots[0],
                    root_answers[0],
                )
            ],
            max_new_tokens=240,
        )
        fixed = [parse_fixed_followup(value) for value in fixed_raw]
        raw["fixed"] = fixed_raw

        followup_raw = _complete(
            models.simulator,
            [
                simulator_followup_messages(
                    synthetic_system,
                    initial[0].roots[0],
                    root_answers[0],
                    refreshed[0].followup,
                )
            ],
            max_new_tokens=160,
        )
        followup_answer = [_clean_text(value) for value in followup_raw]
        raw["followup_answers"] = followup_raw

        fixture = WorldFixture(
            fixture_id="SYNTHETIC",
            record_id=-1,
            world=1,
            seed_message="Synthetic ambiguous request.",
            simulator_system=synthetic_system,
            truth_packet={
                "description": "Synthetic hidden context",
                "goal": "Choose a suitable option",
                "obstacle": "Missing one detail",
                "constraints": ["Limited budget"],
                "solution": "Ask a specific question",
                "persona": "Synthetic user",
            },
            checklist=tuple(f"Checklist item {index}" for index in range(5)),
        )
        support_raw = _complete(
            models.support_judge,
            [
                support_judge_messages(
                    fixture,
                    initial[0],
                    [refreshed[0]] * ROOT_COUNT,
                )
            ],
            max_new_tokens=220,
        )
        support = [parse_support_judgment(value) for value in support_raw]
        raw["support_judgment"] = support_raw

        checklist_raw = _complete(
            models.checklist_judge,
            [
                checklist_judge_messages(
                    fixture,
                    initial[0],
                    [root_answers[0]] * ROOT_COUNT,
                    [refreshed[0]] * ROOT_COUNT,
                    [followup_answer[0]] * ROOT_COUNT,
                    [fixed[0]] * ROOT_COUNT,
                    [followup_answer[0]] * ROOT_COUNT,
                )
            ],
            max_new_tokens=260,
        )
        checklist = [
            parse_checklist_judgment(value) for value in checklist_raw
        ]
        raw["checklist_judgment"] = checklist_raw
        _checkpoint(raw_path, raw)
        usage = aggregate_usage(models)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            aggregate_usage(models),
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
        "all_stage_parsers_pass": (
            len(initial) == 2
            and len(root_answers) == 2
            and len(refreshed) == 2
            and len(fixed) == 1
            and len(followup_answer) == 1
            and len(support) == 1
            and len(checklist) == 1
        ),
        "cost_at_most_0_15": (
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
            "expected_requests": EXPECTED_SERVING_REQUESTS,
            "models": {
                "generator": GENERATOR_MODEL_ID,
                "simulator": SIMULATOR_MODEL_ID,
                "support_judge": SUPPORT_JUDGE_MODEL_ID,
                "checklist_judge": CHECKLIST_JUDGE_MODEL_ID,
            },
            "reasoning_requested": False,
            "scientific_endpoint_evaluated": False,
            "repairs_or_reissues": 0,
        },
        "gates": gates,
        "usage": usage,
    }


def run_mechanics_gate(
    fixtures: Sequence[WorldFixture],
    models: ModelBundle,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "private_fixtures": [
            {
                "fixture_id": fixture.fixture_id,
                "seed_message": fixture.seed_message,
                "simulator_system": fixture.simulator_system,
                "truth_packet": fixture.truth_packet,
                "checklist": fixture.checklist,
            }
            for fixture in fixtures
        ],
    }
    try:
        fixtures_by_record = {
            fixture.record_id: fixture for fixture in fixtures
        }
        initial_raw = _complete(
            models.generator,
            [
                initial_messages(fixtures_by_record[record_id].seed_message)
                for record_id in MECHANICS_IDS
            ],
            max_new_tokens=1_300,
        )
        initials = {
            record_id: parse_initial(response)
            for record_id, response in zip(MECHANICS_IDS, initial_raw)
        }
        raw["initial"] = initial_raw
        _checkpoint(raw_path, raw)

        root_requests = []
        for fixture in fixtures:
            initial = initials[fixture.record_id]
            root_requests.extend(
                simulator_root_messages(
                    fixture.simulator_system,
                    root,
                )
                for root in initial.roots
            )
        root_answer_raw = _complete(
            models.simulator,
            root_requests,
            max_new_tokens=220,
        )
        root_answers_flat = [
            _clean_text(value, maximum=1_200) for value in root_answer_raw
        ]
        root_answers = [
            root_answers_flat[index : index + ROOT_COUNT]
            for index in range(0, len(root_answers_flat), ROOT_COUNT)
        ]
        raw["root_answers"] = root_answer_raw
        _checkpoint(raw_path, raw)

        refresh_requests = []
        fixed_requests = []
        for fixture_index, fixture in enumerate(fixtures):
            initial = initials[fixture.record_id]
            for root_index, root in enumerate(initial.roots):
                answer = root_answers[fixture_index][root_index]
                refresh_requests.append(
                    refresh_messages(
                        fixture.seed_message,
                        initial,
                        root,
                        answer,
                    )
                )
                fixed_requests.append(
                    fixed_followup_messages(
                        fixture.seed_message,
                        initial,
                        root,
                        answer,
                    )
                )
        refresh_raw = _complete(
            models.generator,
            refresh_requests,
            max_new_tokens=1_200,
        )
        refreshed_flat = [parse_refresh(value) for value in refresh_raw]
        refreshed = [
            refreshed_flat[index : index + ROOT_COUNT]
            for index in range(0, len(refreshed_flat), ROOT_COUNT)
        ]
        raw["refresh"] = refresh_raw
        _checkpoint(raw_path, raw)

        fixed_raw = _complete(
            models.generator,
            fixed_requests,
            max_new_tokens=260,
        )
        fixed_flat = [parse_fixed_followup(value) for value in fixed_raw]
        fixed_followups = [
            fixed_flat[index : index + ROOT_COUNT]
            for index in range(0, len(fixed_flat), ROOT_COUNT)
        ]
        raw["fixed_followups"] = fixed_raw
        _checkpoint(raw_path, raw)

        followup_requests = []
        for fixture_index, fixture in enumerate(fixtures):
            initial = initials[fixture.record_id]
            for root_index, root in enumerate(initial.roots):
                common = (
                    fixture.simulator_system,
                    root,
                    root_answers[fixture_index][root_index],
                )
                followup_requests.append(
                    simulator_followup_messages(
                        *common,
                        refreshed[fixture_index][root_index].followup,
                    )
                )
                followup_requests.append(
                    simulator_followup_messages(
                        *common,
                        fixed_followups[fixture_index][root_index],
                    )
                )
        followup_raw = _complete(
            models.simulator,
            followup_requests,
            max_new_tokens=220,
        )
        followup_clean = [
            _clean_text(value, maximum=1_200) for value in followup_raw
        ]
        dynamic_answers: list[list[str]] = []
        fixed_answers: list[list[str]] = []
        cursor = 0
        for _fixture in fixtures:
            dynamic_row = []
            fixed_row = []
            for _root in range(ROOT_COUNT):
                dynamic_row.append(followup_clean[cursor])
                fixed_row.append(followup_clean[cursor + 1])
                cursor += 2
            dynamic_answers.append(dynamic_row)
            fixed_answers.append(fixed_row)
        raw["followup_answers"] = followup_raw
        _checkpoint(raw_path, raw)

        support_raw = _complete(
            models.support_judge,
            [
                support_judge_messages(
                    fixture,
                    initials[fixture.record_id],
                    refreshed[fixture_index],
                )
                for fixture_index, fixture in enumerate(fixtures)
            ],
            max_new_tokens=260,
        )
        support_judgments = [
            parse_support_judgment(value) for value in support_raw
        ]
        raw["support_judgments"] = support_raw
        _checkpoint(raw_path, raw)

        checklist_raw = _complete(
            models.checklist_judge,
            [
                checklist_judge_messages(
                    fixture,
                    initials[fixture.record_id],
                    root_answers[fixture_index],
                    refreshed[fixture_index],
                    dynamic_answers[fixture_index],
                    fixed_followups[fixture_index],
                    fixed_answers[fixture_index],
                )
                for fixture_index, fixture in enumerate(fixtures)
            ],
            max_new_tokens=320,
        )
        checklist_judgments = [
            parse_checklist_judgment(value) for value in checklist_raw
        ]
        raw["checklist_judgments"] = checklist_raw
        _checkpoint(raw_path, raw)
        usage = aggregate_usage(models)
        metrics, fixture_metrics, scientific_gates = _public_metrics(
            fixtures,
            initials,
            root_answers,
            refreshed,
            fixed_followups,
            dynamic_answers,
            fixed_answers,
            support_judgments,
            checklist_judgments,
        )
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            aggregate_usage(models),
        ) from exc

    mechanics_gates = {
        "exact_165_physical_requests": (
            usage["physical_requests"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "exact_165_http_attempts": (
            usage["http_attempts"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_1_00": (
            usage["adapter_cost_usd"] <= MECHANICS_MAX_COST_USD
        ),
        "exact_6_fixtures_30_root_cells": (
            metrics["fixtures"] == 6
            and metrics["root_world_cells"] == 30
        ),
    }
    gates = {**mechanics_gates, **scientific_gates}
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": "mechanics",
            "mechanics_ids": list(MECHANICS_IDS),
            "support_size": SUPPORT_SIZE,
            "root_count": ROOT_COUNT,
            "truth_threshold": TRUTH_THRESHOLD,
            "expected_requests": EXPECTED_MECHANICS_REQUESTS,
            "shuffle_permutation": list(SHUFFLE_PERMUTATION),
            "models": {
                "generator_refresh_fixed": GENERATOR_MODEL_ID,
                "simulator": SIMULATOR_MODEL_ID,
                "support_judge": SUPPORT_JUDGE_MODEL_ID,
                "checklist_judge": CHECKLIST_JUDGE_MODEL_ID,
            },
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "opportunity_or_later_split_read": False,
            "causal_policy_efficacy_claimed": False,
        },
        "metrics": metrics,
        "fixture_metrics": fixture_metrics,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self, role: str) -> None:
        self.role = role
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses = []
        for messages in batch_messages:
            if self.role == "simulator":
                responses.append("The relevant hidden detail is value one.")
                continue
            stage = messages[0]["content"].split("STAGE=", 1)[1].split(".", 1)[0]
            if stage == "INITIAL":
                value = {
                    **{
                        f"h{index}": (
                            f"Concrete hidden context {index} has goal {index}, "
                            f"obstacle {index}, and constraint {index}."
                        )
                        for index in range(1, SUPPORT_SIZE + 1)
                    },
                    **{
                        f"q{index}": f"What is hidden detail {index}?"
                        for index in range(1, ROOT_COUNT + 1)
                    },
                }
                responses.append(json.dumps(value, separators=(",", ":")))
            elif stage == "REFRESH":
                value = {
                    **{
                        f"h{index}": (
                            f"Refreshed context {index} has goal {index}, "
                            f"obstacle {index}, and constraint {index}."
                        )
                        for index in range(1, SUPPORT_SIZE + 1)
                    },
                    "followup": "What is the next hidden detail?",
                }
                responses.append(json.dumps(value, separators=(",", ":")))
            elif stage == "FIXED":
                responses.append(
                    json.dumps(
                        {"followup": "What is the fixed hidden detail?"},
                        separators=(",", ":"),
                    )
                )
            elif stage == "SUPPORT_JUDGE":
                responses.append(
                    "\n".join(
                        (
                            "I|00|50",
                            "Q1|00|60",
                            "Q2|01|80",
                            "Q3|01|90",
                            "Q4|01|70",
                            "Q5|00|40",
                        )
                    )
                )
            elif stage == "CHECKLIST_JUDGE":
                responses.append(
                    "\n".join(
                        (
                            "Q1|00000|11000|10000",
                            "Q2|00000|11100|10000",
                            "Q3|00000|11110|11000",
                            "Q4|00000|11000|10000",
                            "Q5|00000|10000|00000",
                        )
                    )
                )
            else:  # pragma: no cover
                raise AssertionError(stage)
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


def _nonthinking_spec(spec: Any, model_id: str) -> Any:
    return replace(
        spec,
        model=model_id,
        backend="openrouter",
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )


def _build_models(config: Config) -> ModelBundle:
    base = config.model_pairs[0].questioner
    return ModelBundle(
        generator=build_model_adapter(
            _nonthinking_spec(base, GENERATOR_MODEL_ID),
            config,
        ),
        simulator=build_model_adapter(
            _nonthinking_spec(base, SIMULATOR_MODEL_ID),
            config,
        ),
        support_judge=build_model_adapter(
            _nonthinking_spec(base, SUPPORT_JUDGE_MODEL_ID),
            config,
        ),
        checklist_judge=build_model_adapter(
            _nonthinking_spec(base, CHECKLIST_JUDGE_MODEL_ID),
            config,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("fixture", "serving", "mechanics"),
        required=True,
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fixtures, public_fixture = build_fixtures(
        settings_path=args.settings,
        baseline_path=args.baseline,
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
    config.openrouter_projected_cost_usd = (
        0.08 if args.stage == "serving" else 0.65
    )
    config.openrouter_run_budget_usd = (
        SERVING_MAX_COST_USD
        if args.stage == "serving"
        else MECHANICS_MAX_COST_USD
    )
    config.openrouter_concurrency = (
        EXPECTED_SERVING_REQUESTS
        if args.stage == "serving"
        else 30
    )
    config.openrouter_max_retries = 0
    config.openrouter_max_output_tokens = 1_400
    config.log_path = args.output_dir / "run.log"
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    models = (
        ModelBundle(
            generator=DeterministicFixtureModel("generator"),
            simulator=DeterministicFixtureModel("simulator"),
            support_judge=DeterministicFixtureModel("support_judge"),
            checklist_judge=DeterministicFixtureModel("checklist_judge"),
        )
        if args.dry_run
        else _build_models(config)
    )

    try:
        result = (
            run_serving_gate(models, raw_path=raw_path)
            if args.stage == "serving"
            else run_mechanics_gate(fixtures, models, raw_path=raw_path)
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

    output_name = (
        "SERVING.json" if args.stage == "serving" else "MECHANICS.json"
    )
    _checkpoint(args.output_dir / output_name, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gates"]["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
