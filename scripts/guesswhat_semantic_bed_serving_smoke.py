#!/usr/bin/env python3
"""Qualify visual question, likelihood, and oracle serving for GuessWhat?!."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Protocol, Sequence
import urllib.request

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import guesswhat_semantic_bed_source_audit as source
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_predictive_risk_replication import (
    SeededStructuredAdapter,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "guesswhat-semantic-bed-serving-smoke-1"
SOURCE_AUDIT_SHA256 = (
    "027a7f49fc599001eca6fbac0b3fe3e6be021d8cb3512eacad3ee167d0f5fff2"
)
SOURCE_MANIFEST_PATH = (
    REPO_ROOT
    / "results/nonmyopic/guesswhat_semantic_bed_source_audit/"
    "guesswhat-semantic-bed-source-audit-20260729/MANIFEST.json"
)
PLANNER_MODEL_ID = "openai/gpt-5.4-mini"
LIKELIHOOD_MODEL_ID = "google/gemini-2.5-flash"
ORACLE_MODEL_ID = "qwen/qwen3-vl-32b-instruct"
PLANNER_SEED = 39_100
LIKELIHOOD_SEED = 39_200
ORACLE_SEED = 39_300
PLANNER_TEMPERATURE = 0.7
LIKELIHOOD_TEMPERATURE = 0.0
ORACLE_TEMPERATURE = 0.0
NUM_CASES = 2
NUM_QUESTIONS = 4
ORACLE_QUESTIONS_PER_CASE = 3
EXPECTED_REQUESTS = (
    NUM_CASES + NUM_CASES + NUM_CASES * ORACLE_QUESTIONS_PER_CASE
)
CONCURRENCY = 6
PLANNER_MAX_TOKENS = 1_200
LIKELIHOOD_MAX_TOKENS = 1_800
ORACLE_MAX_TOKENS = 120
PROJECTED_COST_USD = 0.08
RUN_BUDGET_USD = 0.30
MIN_DISCRIMINATORY_RANGE = 40
MIN_DISCRIMINATORY_QUESTIONS_PER_CASE = 2
MIN_BALANCED_QUESTIONS_PER_CASE = 2
MIN_CROSS_MODEL_CONSISTENT = 5
LOW_PROBABILITY = 40
HIGH_PROBABILITY = 60


class StructuredModel(Protocol):
    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class ServingExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class ServingCase:
    dialogue_id: str
    picture_id: str
    row_sha256: str
    image_url: str
    image_width: int
    image_height: int
    objects: tuple[dict[str, Any], ...]
    target_index: int

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(f"C{index}" for index in range(1, len(self.objects) + 1))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_json_object(response: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not exact JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys are {sorted(value)}, expected {sorted(expected)}"
        )


def planner_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "guesswhat_visual_questions",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["questions"],
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": NUM_QUESTIONS,
                        "maxItems": NUM_QUESTIONS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "text"],
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "minLength": 2,
                                    "maxLength": 2,
                                },
                                "text": {
                                    "type": "string",
                                    "minLength": 8,
                                    "maxLength": 180,
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def likelihood_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "guesswhat_visual_likelihoods",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rows"],
                "properties": {
                    "rows": {
                        "type": "array",
                        "minItems": NUM_QUESTIONS,
                        "maxItems": NUM_QUESTIONS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["question_id", "candidates"],
                            "properties": {
                                "question_id": {
                                    "type": "string",
                                    "minLength": 2,
                                    "maxLength": 2,
                                },
                                "candidates": {
                                    "type": "array",
                                    "minItems": source.MIN_OBJECTS,
                                    "maxItems": source.MAX_OBJECTS,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": [
                                            "candidate_id",
                                            "yes_probability",
                                        ],
                                        "properties": {
                                            "candidate_id": {
                                                "type": "string",
                                                "minLength": 2,
                                                "maxLength": 3,
                                            },
                                            "yes_probability": {
                                                "type": "integer",
                                                "minimum": 0,
                                                "maximum": 100,
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def oracle_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "guesswhat_visual_oracle",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["answer"],
                "properties": {
                    "answer": {
                        "type": "string",
                        "enum": ["Yes", "No"],
                    }
                },
            },
        },
    }


QUESTION_START_RE = re.compile(
    r"^(is|are|does|do|can|has|have|was|were)\b",
    re.IGNORECASE,
)
LABEL_REFERENCE_RE = re.compile(
    r"\b(?:c[0-9]+|box(?:es)?|bounding|label(?:ed)?)\b",
    re.IGNORECASE,
)


def parse_planner_response(response: str) -> list[dict[str, str]]:
    value = strict_json_object(response, label="planner response")
    _exact_keys(value, {"questions"}, "planner response")
    questions = value["questions"]
    if not isinstance(questions, list) or len(questions) != NUM_QUESTIONS:
        raise ValueError("planner must return exactly four questions")
    parsed = []
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError("planner question is not an object")
        _exact_keys(item, {"id", "text"}, f"planner question {index}")
        expected_id = f"Q{index}"
        text = item["text"]
        if item["id"] != expected_id:
            raise ValueError(f"planner question {index} id is not {expected_id}")
        if not isinstance(text, str):
            raise ValueError("planner question text is not a string")
        text = " ".join(text.split())
        if not 8 <= len(text) <= 180 or not text.endswith("?"):
            raise ValueError("planner question has invalid shape")
        if QUESTION_START_RE.match(text) is None:
            raise ValueError("planner question is not a yes/no question")
        if LABEL_REFERENCE_RE.search(text):
            raise ValueError("planner question refers to overlay labels")
        parsed.append({"id": expected_id, "text": text})
    normalized = {item["text"].casefold() for item in parsed}
    if len(normalized) != NUM_QUESTIONS:
        raise ValueError("planner questions are not unique")
    return parsed


def parse_likelihood_response(
    response: str,
    *,
    candidate_ids: Sequence[str],
) -> list[dict[str, Any]]:
    value = strict_json_object(response, label="likelihood response")
    _exact_keys(value, {"rows"}, "likelihood response")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != NUM_QUESTIONS:
        raise ValueError("likelihood response must have four rows")
    parsed = []
    for question_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError("likelihood row is not an object")
        _exact_keys(
            row,
            {"question_id", "candidates"},
            f"likelihood row {question_index}",
        )
        expected_question = f"Q{question_index}"
        if row["question_id"] != expected_question:
            raise ValueError("likelihood question ids are not ordered")
        candidates = row["candidates"]
        if not isinstance(candidates, list) or len(candidates) != len(candidate_ids):
            raise ValueError("likelihood candidate count changed")
        probabilities = []
        for candidate_index, (item, expected_id) in enumerate(
            zip(candidates, candidate_ids, strict=True),
            start=1,
        ):
            if not isinstance(item, dict):
                raise ValueError("likelihood candidate is not an object")
            _exact_keys(
                item,
                {"candidate_id", "yes_probability"},
                f"likelihood candidate {candidate_index}",
            )
            probability = item["yes_probability"]
            if item["candidate_id"] != expected_id:
                raise ValueError("likelihood candidate ids are not ordered")
            if (
                isinstance(probability, bool)
                or not isinstance(probability, int)
                or not 0 <= probability <= 100
            ):
                raise ValueError("likelihood probability is invalid")
            probabilities.append(probability)
        parsed.append(
            {
                "question_id": expected_question,
                "probabilities": probabilities,
            }
        )
    return parsed


def parse_oracle_response(response: str) -> str:
    value = strict_json_object(response, label="oracle response")
    _exact_keys(value, {"answer"}, "oracle response")
    if value["answer"] not in {"Yes", "No"}:
        raise ValueError("oracle answer is not Yes or No")
    return str(value["answer"])


def _source_rows_by_id() -> dict[str, dict[str, Any]]:
    return {source.row_id(row): row for row in source.load_rows()}


def load_cases() -> list[ServingCase]:
    if sha256_file(SOURCE_MANIFEST_PATH) != SOURCE_AUDIT_SHA256:
        raise ValueError("GuessWhat source manifest changed")
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
    specs = manifest["splits"]["serving_smoke"]
    if len(specs) != NUM_CASES:
        raise ValueError("serving split no longer has two cases")
    rows = _source_rows_by_id()
    cases = []
    for spec in specs:
        row = rows[spec["dialogue_id"]]
        if source.row_sha256(row) != spec["row_sha256"]:
            raise ValueError(f"source row {spec['dialogue_id']} changed")
        if source.eligibility_errors(row):
            raise ValueError(f"source row {spec['dialogue_id']} is ineligible")
        objects = tuple(row["objects"].values())
        target_object_id = str(row["object_id"])
        target_indexes = [
            index
            for index, obj in enumerate(objects, start=1)
            if str(obj["object_id"]) == target_object_id
        ]
        if len(target_indexes) != 1:
            raise ValueError("target does not map to exactly one candidate")
        cases.append(
            ServingCase(
                dialogue_id=spec["dialogue_id"],
                picture_id=spec["picture_id"],
                row_sha256=spec["row_sha256"],
                image_url=spec["image_url"],
                image_width=int(row["picture"]["width"]),
                image_height=int(row["picture"]["height"]),
                objects=objects,
                target_index=target_indexes[0],
            )
        )
    return cases


def download_image(case: ServingCase) -> bytes:
    request = urllib.request.Request(
        case.image_url,
        headers={"User-Agent": "BED-LLM-Mod/GuessWhat-source-audit"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def decode_image(image_bytes: bytes, case: ServingCase) -> Image.Image:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    if image.size != (case.image_width, case.image_height):
        raise ValueError(
            f"image {case.picture_id} size {image.size} changed from "
            f"{(case.image_width, case.image_height)}"
        )
    return image


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x, y, width, height = bbox
    left = int(round(x))
    top = int(round(y))
    right = int(round(x + width))
    bottom = int(round(y + height))
    draw.rectangle((left, top, right, bottom), outline=color, width=4)
    text_box = draw.textbbox((left, top), label, font=_font())
    draw.rectangle(text_box, fill=color)
    draw.text((left, top), label, fill=(255, 255, 255), font=_font())


def candidate_overlay(image: Image.Image, case: ServingCase) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    colors = (
        (220, 20, 60),
        (0, 110, 220),
        (0, 150, 80),
        (180, 90, 0),
        (150, 40, 180),
        (20, 150, 170),
        (210, 70, 120),
        (80, 80, 220),
        (80, 150, 20),
        (210, 120, 0),
        (120, 50, 50),
        (30, 120, 120),
    )
    for index, obj in enumerate(case.objects, start=1):
        _draw_box(
            draw,
            obj["bbox"],
            color=colors[index - 1],
            label=f"C{index}",
        )
    return overlay


def target_overlay(image: Image.Image, case: ServingCase) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    _draw_box(
        draw,
        case.objects[case.target_index - 1]["bbox"],
        color=(220, 20, 60),
        label="TARGET",
    )
    return overlay


def image_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def image_data_url(image: Image.Image) -> str:
    payload = base64.b64encode(image_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{payload}"


def multimodal_message(text: str, image: Image.Image) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(image)},
                },
            ],
        }
    ]


def planner_messages(
    case: ServingCase,
    overlay: Image.Image,
) -> list[dict[str, Any]]:
    request = {
        "task": (
            "The image contains numbered candidate objects. A hidden target is "
            "one candidate. Propose four distinct visual yes/no questions that "
            "would efficiently identify it."
        ),
        "candidate_ids": list(case.candidate_ids),
        "requirements": [
            "Questions must be answerable from the image for any candidate.",
            "Use visible category, attribute, relation, or position.",
            "Do not refer to candidate IDs, boxes, labels, or annotations.",
            "Avoid four paraphrases of the same distinction.",
            "Return Q1 through Q4 in order.",
        ],
    }
    return multimodal_message(source.canonical_json(request), overlay)


def likelihood_messages(
    case: ServingCase,
    overlay: Image.Image,
    questions: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    request = {
        "task": (
            "For each question and each numbered candidate, estimate the "
            "probability from 0 to 100 that a truthful visual oracle would "
            "answer Yes if that candidate were the hidden target."
        ),
        "candidate_ids": list(case.candidate_ids),
        "questions": list(questions),
        "requirements": [
            "Condition on each candidate separately.",
            "Use the visible object inside that candidate box.",
            "Return Q1 through Q4 and candidates in the supplied order.",
            "Use calibrated uncertainty, not explanations.",
        ],
    }
    return multimodal_message(source.canonical_json(request), overlay)


def oracle_messages(
    target_image: Image.Image,
    question: dict[str, str],
) -> list[dict[str, Any]]:
    request = {
        "task": (
            "The red TARGET box marks the hidden object. Answer the supplied "
            "visual yes/no question about that object."
        ),
        "question_id": question["id"],
        "question": question["text"],
        "requirements": [
            "Answer only Yes or No in the schema.",
            "Base the answer on the object in the TARGET box.",
            "If the image is ambiguous, choose the better supported answer.",
        ],
    }
    return multimodal_message(source.canonical_json(request), target_image)


def _usage(models: Sequence[StructuredModel]) -> dict[str, Any]:
    snapshots = [model.usage_snapshot() for model in models]
    totals: dict[str, Any] = {}
    for key in (
        "adapter_requests",
        "http_attempts",
        "retry_count",
        "provider_error_retries",
        "adapter_reasoning_tokens",
        "forced_exits",
        "adapter_prompt_tokens",
        "adapter_completion_tokens",
        "adapter_cost_usd",
    ):
        totals[key] = sum(float(item.get(key, 0) or 0) for item in snapshots)
    for key in (
        "adapter_requests",
        "http_attempts",
        "retry_count",
        "provider_error_retries",
        "adapter_reasoning_tokens",
        "forced_exits",
        "adapter_prompt_tokens",
        "adapter_completion_tokens",
    ):
        totals[key] = int(totals[key])
    totals["run_cost_usd"] = totals.pop("adapter_cost_usd")
    totals["models"] = snapshots
    return totals


def _checkpoint_private(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def case_metrics(
    *,
    case: ServingCase,
    image_bytes: bytes,
    candidate_image: Image.Image,
    questions: Sequence[dict[str, str]],
    likelihoods: Sequence[dict[str, Any]],
    oracle_answers: Sequence[str],
) -> dict[str, Any]:
    question_metrics = []
    consistent = 0
    for index, (question, likelihood) in enumerate(
        zip(questions, likelihoods, strict=True)
    ):
        probabilities = likelihood["probabilities"]
        probability_range = max(probabilities) - min(probabilities)
        mean_probability = sum(probabilities) / (100.0 * len(probabilities))
        oracle_answer = (
            oracle_answers[index]
            if index < ORACLE_QUESTIONS_PER_CASE
            else None
        )
        target_probability = probabilities[case.target_index - 1]
        cross_model_consistent = None
        if oracle_answer is not None:
            cross_model_consistent = (
                target_probability >= HIGH_PROBABILITY
                if oracle_answer == "Yes"
                else target_probability <= LOW_PROBABILITY
            )
            consistent += int(cross_model_consistent)
        question_metrics.append(
            {
                "question_id": question["id"],
                "question": question["text"],
                "likelihood_range": probability_range,
                "unique_probability_count": len(set(probabilities)),
                "uniform_prior_yes_mass": mean_probability,
                "oracle_answer": oracle_answer,
                "cross_model_consistent": cross_model_consistent,
            }
        )
    return {
        "dialogue_id": case.dialogue_id,
        "picture_id": case.picture_id,
        "candidate_count": len(case.objects),
        "source_image_sha256": sha256_bytes(image_bytes),
        "candidate_overlay_sha256": sha256_bytes(
            image_png_bytes(candidate_image)
        ),
        "question_count": len(questions),
        "unique_question_count": len(
            {question["text"].casefold() for question in questions}
        ),
        "discriminatory_question_count": sum(
            item["likelihood_range"] >= MIN_DISCRIMINATORY_RANGE
            and item["unique_probability_count"] >= 3
            for item in question_metrics
        ),
        "balanced_question_count": sum(
            0.2 <= item["uniform_prior_yes_mass"] <= 0.8
            for item in question_metrics
        ),
        "oracle_yes_count": sum(answer == "Yes" for answer in oracle_answers),
        "oracle_no_count": sum(answer == "No" for answer in oracle_answers),
        "cross_model_consistent_count": consistent,
        "questions": question_metrics,
    }


def aggregate_gates(
    *,
    cases: Sequence[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, bool]:
    answer_set = {
        question["oracle_answer"]
        for case in cases
        for question in case["questions"]
        if question["oracle_answer"] is not None
    }
    consistent = sum(
        case["cross_model_consistent_count"] for case in cases
    )
    gates = {
        "exact_10_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "exact_10_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_ten_schemas_parse": len(cases) == NUM_CASES,
        "both_cases_have_four_unique_semantic_questions": all(
            case["unique_question_count"] == NUM_QUESTIONS for case in cases
        ),
        "both_cases_have_at_least_two_discriminatory_questions": all(
            case["discriminatory_question_count"]
            >= MIN_DISCRIMINATORY_QUESTIONS_PER_CASE
            for case in cases
        ),
        "both_cases_have_at_least_two_balanced_questions": all(
            case["balanced_question_count"]
            >= MIN_BALANCED_QUESTIONS_PER_CASE
            for case in cases
        ),
        "oracle_answers_include_yes_and_no": answer_set == {"Yes", "No"},
        "at_least_five_of_six_cross_model_answers_are_consistent": (
            consistent >= MIN_CROSS_MODEL_CONSISTENT
        ),
        "each_case_has_at_least_two_cross_model_consistent_answers": all(
            case["cross_model_consistent_count"] >= 2 for case in cases
        ),
        "cost_at_most_0_30": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def run_smoke(
    *,
    cases: Sequence[ServingCase],
    planner_model: StructuredModel,
    likelihood_model: StructuredModel,
    oracle_model: StructuredModel,
    raw_path: Path,
    image_loader: Callable[[ServingCase], bytes] = download_image,
) -> dict[str, Any]:
    if len(cases) != NUM_CASES:
        raise ValueError("serving smoke requires exactly two cases")
    raw: dict[str, Any] = {
        "case_ids": [case.dialogue_id for case in cases],
        "planner_responses": [],
        "likelihood_responses": [],
        "oracle_responses": [],
        "human_dialogue_prompted_or_scored": False,
        "game_outcome_prompted_or_scored": False,
        "policy_endpoint_accessed": False,
        "development_accessed": False,
        "holdout_accessed": False,
    }
    try:
        image_bytes = [image_loader(case) for case in cases]
        images = [
            decode_image(payload, case)
            for payload, case in zip(image_bytes, cases, strict=True)
        ]
        candidate_images = [
            candidate_overlay(image, case)
            for image, case in zip(images, cases, strict=True)
        ]
        target_images = [
            target_overlay(image, case)
            for image, case in zip(images, cases, strict=True)
        ]

        planner_responses = (
            planner_model.chat_complete_messages_batched_structured(
                [
                    planner_messages(case, overlay)
                    for case, overlay in zip(
                        cases, candidate_images, strict=True
                    )
                ],
                temperature=PLANNER_TEMPERATURE,
                block_size=NUM_CASES,
                response_format=planner_response_format(),
                max_new_tokens=PLANNER_MAX_TOKENS,
            )
        )
        raw["planner_responses"] = list(planner_responses)
        _checkpoint_private(raw_path, raw)
        if len(planner_responses) != NUM_CASES:
            raise ValueError("planner response count changed")
        planners = [
            parse_planner_response(response)
            for response in planner_responses
        ]

        likelihood_responses = (
            likelihood_model.chat_complete_messages_batched_structured(
                [
                    likelihood_messages(case, overlay, questions)
                    for case, overlay, questions in zip(
                        cases, candidate_images, planners, strict=True
                    )
                ],
                temperature=LIKELIHOOD_TEMPERATURE,
                block_size=NUM_CASES,
                response_format=likelihood_response_format(),
                max_new_tokens=LIKELIHOOD_MAX_TOKENS,
            )
        )
        raw["likelihood_responses"] = list(likelihood_responses)
        _checkpoint_private(raw_path, raw)
        if len(likelihood_responses) != NUM_CASES:
            raise ValueError("likelihood response count changed")
        likelihoods = [
            parse_likelihood_response(
                response,
                candidate_ids=case.candidate_ids,
            )
            for response, case in zip(
                likelihood_responses, cases, strict=True
            )
        ]

        oracle_requests = [
            oracle_messages(target_image, question)
            for target_image, questions in zip(
                target_images, planners, strict=True
            )
            for question in questions[:ORACLE_QUESTIONS_PER_CASE]
        ]
        oracle_responses = (
            oracle_model.chat_complete_messages_batched_structured(
                oracle_requests,
                temperature=ORACLE_TEMPERATURE,
                block_size=len(oracle_requests),
                response_format=oracle_response_format(),
                max_new_tokens=ORACLE_MAX_TOKENS,
            )
        )
        raw["oracle_responses"] = list(oracle_responses)
        _checkpoint_private(raw_path, raw)
        if len(oracle_responses) != len(oracle_requests):
            raise ValueError("oracle response count changed")
        parsed_oracle = [
            parse_oracle_response(response) for response in oracle_responses
        ]
        oracle_by_case = [
            parsed_oracle[
                index * ORACLE_QUESTIONS_PER_CASE:
                (index + 1) * ORACLE_QUESTIONS_PER_CASE
            ]
            for index in range(NUM_CASES)
        ]

        public_cases = [
            case_metrics(
                case=case,
                image_bytes=payload,
                candidate_image=candidate_image,
                questions=questions,
                likelihoods=case_likelihoods,
                oracle_answers=answers,
            )
            for (
                case,
                payload,
                candidate_image,
                target_image,
                questions,
                case_likelihoods,
                answers,
            ) in zip(
                cases,
                image_bytes,
                candidate_images,
                target_images,
                planners,
                likelihoods,
                oracle_by_case,
                strict=True,
            )
        ]
        usage = _usage(
            (planner_model, likelihood_model, oracle_model)
        )
    except Exception as exc:
        _checkpoint_private(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage((planner_model, likelihood_model, oracle_model)),
        ) from exc

    gates = aggregate_gates(cases=public_cases, usage=usage)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_audit_sha256": SOURCE_AUDIT_SHA256,
            "planner_model": PLANNER_MODEL_ID,
            "likelihood_model": LIKELIHOOD_MODEL_ID,
            "oracle_model": ORACLE_MODEL_ID,
            "planner_seed": PLANNER_SEED,
            "likelihood_seed": LIKELIHOOD_SEED,
            "oracle_seed": ORACLE_SEED,
            "expected_requests": EXPECTED_REQUESTS,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "human_dialogue_prompted_or_scored": False,
            "game_outcome_prompted_or_scored": False,
            "policy_endpoint_accessed": False,
            "development_accessed": False,
            "holdout_accessed": False,
        },
        "metrics": {
            "case_count": len(public_cases),
            "question_count": sum(
                case["question_count"] for case in public_cases
            ),
            "discriminatory_question_count": sum(
                case["discriminatory_question_count"]
                for case in public_cases
            ),
            "balanced_question_count": sum(
                case["balanced_question_count"] for case in public_cases
            ),
            "cross_model_consistent_count": sum(
                case["cross_model_consistent_count"]
                for case in public_cases
            ),
        },
        "cases": public_cases,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(
        self,
        role: str,
        *,
        oracle_answers: Sequence[str] = (),
    ) -> None:
        self.role = role
        self.oracle_answers = list(oracle_answers)
        self.requests = 0

    @staticmethod
    def _text(messages: list[dict[str, Any]]) -> dict[str, Any]:
        content = messages[-1]["content"]
        text = next(
            item["text"]
            for item in content
            if item.get("type") == "text"
        )
        return json.loads(text)

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, response_format, max_new_tokens
        responses = []
        for messages in batch_messages:
            request = self._text(messages)
            if self.role == "planner":
                responses.append(
                    json.dumps(
                        {
                            "questions": [
                                {"id": "Q1", "text": "Is it on the left side?"},
                                {"id": "Q2", "text": "Is it on the right side?"},
                                {"id": "Q3", "text": "Is it a person?"},
                                {"id": "Q4", "text": "Does it touch the ground?"},
                            ]
                        }
                    )
                )
            elif self.role == "likelihood":
                candidate_ids = request["candidate_ids"]
                rows = []
                for question_index in range(1, NUM_QUESTIONS + 1):
                    rows.append(
                        {
                            "question_id": f"Q{question_index}",
                            "candidates": [
                                {
                                    "candidate_id": candidate_id,
                                    "yes_probability": (10, 30, 90)[
                                        (
                                            candidate_index
                                            + question_index
                                        )
                                        % 3
                                    ],
                                }
                                for candidate_index, candidate_id in enumerate(
                                    candidate_ids,
                                    start=1,
                                )
                            ],
                        }
                    )
                responses.append(json.dumps({"rows": rows}))
            else:
                answer_index = self.requests + len(responses)
                responses.append(
                    json.dumps(
                        {"answer": self.oracle_answers[answer_index]}
                    )
                )
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def fixture_models(
    cases: Sequence[ServingCase],
) -> tuple[
    DeterministicFixtureModel,
    DeterministicFixtureModel,
    DeterministicFixtureModel,
]:
    answers = []
    for case in cases:
        for question_index in range(1, ORACLE_QUESTIONS_PER_CASE + 1):
            probability = (10, 30, 90)[
                (case.target_index + question_index) % 3
            ]
            answers.append(
                "Yes" if probability >= HIGH_PROBABILITY else "No"
            )
    return (
        DeterministicFixtureModel("planner"),
        DeterministicFixtureModel("likelihood"),
        DeterministicFixtureModel("oracle", oracle_answers=answers),
    )


def _adapter(
    *,
    model: str,
    seed: int,
    run_id: str,
    output_dir: Path,
    max_tokens: int,
) -> SeededStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=500.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=max_tokens,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return SeededStructuredAdapter(
        ModelSpec(model=model, backend="openrouter", max_model_len=262_144),
        config,
        request_seed=seed,
    )


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
    cases = load_cases()

    if args.dry_run:
        planner, likelihood, oracle = fixture_models(cases)
    else:
        planner = _adapter(
            model=PLANNER_MODEL_ID,
            seed=PLANNER_SEED,
            run_id=f"{args.run_id}-planner",
            output_dir=args.output_dir,
            max_tokens=PLANNER_MAX_TOKENS,
        )
        likelihood = _adapter(
            model=LIKELIHOOD_MODEL_ID,
            seed=LIKELIHOOD_SEED,
            run_id=f"{args.run_id}-likelihood",
            output_dir=args.output_dir,
            max_tokens=LIKELIHOOD_MAX_TOKENS,
        )
        oracle = _adapter(
            model=ORACLE_MODEL_ID,
            seed=ORACLE_SEED,
            run_id=f"{args.run_id}-oracle",
            output_dir=args.output_dir,
            max_tokens=ORACLE_MAX_TOKENS,
        )
    try:
        payload = run_smoke(
            cases=cases,
            planner_model=planner,
            likelihood_model=likelihood,
            oracle_model=oracle,
            raw_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
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
            )
        )
    except ServingExecutionError as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "source_audit_sha256": SOURCE_AUDIT_SHA256,
                "expected_requests": EXPECTED_REQUESTS,
                "reasoning_requested": False,
                "repairs_or_reissues": 0,
            },
            "error": str(exc),
            "usage": exc.usage,
            "private_raw_sha256": sha256_file(raw_path),
        }
        checkpoint(args.output_dir / "FAILURE.json", failure)
        raise


if __name__ == "__main__":
    main()
