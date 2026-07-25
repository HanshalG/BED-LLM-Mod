#!/usr/bin/env python3
"""Test EIG ranking over regenerated InteractComp answer supports."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import unicodedata
from typing import Any, Iterable, Protocol, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.atd_code_first_link_audit import spearman


INTERFACE_VERSION = "interactcomp-first-link-opportunity-1"
SOURCE_REPOSITORY = "https://github.com/FoundationAgents/InteractComp"
SOURCE_COMMIT = "9cdf7f804f527ad32a405efaa6c86aae03692556"
ENCRYPTED_DATA_SHA256 = (
    "0bd0ccc4b69c228c04c15b1147211adc6c6483b852d74e5ac7e5a34c9db80496"
)
ELIGIBLE_MANIFEST_SHA256 = (
    "9375f5cb3e52590c5eafca25c96f6378c6f44da674162ffdf63c5fe949eeb6c6"
)
GENERATOR_MODEL_ID = "openai/gpt-5.4-mini"
RESPONDER_MODEL_ID = "openai/gpt-5.4"
SEED = 24_377
TASK_INDICES = (75, 37)
EXPECTED_TASK_IDS = (76, 38)
PARTICLE_COUNT = 8
QUESTION_COUNT = 4
MIN_REFRESH_PARTICLES = 6
GENERATOR_REQUESTS = (
    len(TASK_INDICES) * PARTICLE_COUNT
    + len(TASK_INDICES) * QUESTION_COUNT
    + len(TASK_INDICES) * PARTICLE_COUNT
    + len(TASK_INDICES) * QUESTION_COUNT * PARTICLE_COUNT
)
RESPONDER_REQUESTS = len(TASK_INDICES) * QUESTION_COUNT
EXPECTED_REQUESTS = GENERATOR_REQUESTS + RESPONDER_REQUESTS
MAX_COST_USD = 1.50


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class FirstLinkExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class Hypothesis:
    entity: str
    profile: str

    def as_dict(self) -> dict[str, str]:
        return {"entity": self.entity, "profile": self.profile}


def _derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def _decrypt_field(ciphertext: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext)
    key = _derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode("utf-8")


def _read_encrypted_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _decrypted_fields(
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    materialized = []
    for row in rows:
        token = row["canary"]
        record = {"id": row["id"]}
        for field in fields:
            record[field] = _decrypt_field(row[field], token)
        materialized.append(record)
    return materialized


def verify_source(source_root: Path) -> tuple[Path, list[dict[str, Any]]]:
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(
            f"InteractComp commit is {commit}, expected {SOURCE_COMMIT}"
        )
    path = source_root / "data" / "dataset" / "InteractComp210.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ENCRYPTED_DATA_SHA256:
        raise ValueError(
            f"InteractComp data hash is {digest}, expected {ENCRYPTED_DATA_SHA256}"
        )
    rows = _read_encrypted_rows(path)
    if len(rows) != 210 or tuple(rows[index]["id"] for index in TASK_INDICES) != (
        EXPECTED_TASK_IDS
    ):
        raise ValueError("InteractComp task manifest does not match")
    return path, rows


def parse_hypothesis(text: str) -> Hypothesis:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError("hypothesis must contain exactly two nonempty lines")
    if not lines[0].startswith("ENTITY: ") or not lines[1].startswith("PROFILE: "):
        raise ValueError("hypothesis lines must use ENTITY and PROFILE prefixes")
    entity = lines[0][len("ENTITY: ") :].strip()
    profile = lines[1][len("PROFILE: ") :].strip()
    if not entity or not profile or len(entity) > 160 or len(profile) > 800:
        raise ValueError("hypothesis entity or profile is empty or too long")
    return Hypothesis(entity=entity, profile=profile)


def parse_question(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("question must contain exactly one nonempty line")
    question = lines[0]
    if not question.endswith("?") or not (8 <= len(question) <= 300):
        raise ValueError("question must be a bounded single question")
    return question


def parse_classification(text: str) -> str:
    value = text.strip().upper()
    if re.fullmatch(r"[YNU]{4}", value) is None:
        raise ValueError("classification must be exactly four Y/N/U characters")
    return value


def parse_responder_answer(text: str) -> str:
    value = text.strip().lower().splitlines()[0].strip()
    if value in {"yes", "y", "true", "correct"}:
        return "Y"
    if value in {"no", "n", "false", "incorrect"}:
        return "N"
    if value in {"i don't know", "i dont know", "idk", "unknown"}:
        return "U"
    raise ValueError("responder answer is not yes, no, or i don't know")


def normalize_entity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def entropy_from_labels(labels: Iterable[str]) -> float:
    values = list(labels)
    if not values:
        return 0.0
    counts = {value: values.count(value) for value in set(values)}
    total = len(values)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _initial_messages(
    *,
    task_id: int,
    question: str,
    sample_index: int,
) -> list[dict[str, str]]:
    request = {
        "benchmark_task_id": task_id,
        "ambiguous_question": question,
        "interpretation_sample": sample_index,
    }
    return [
        {
            "role": "system",
            "content": (
                "Generate one concrete candidate entity that could answer the "
                "ambiguous question, plus a short profile of attributes that would "
                "distinguish it. Different interpretation_sample values should "
                "explore materially different plausible candidates. Do not ask a "
                "question and do not give multiple entities. Output exactly two "
                "lines:\nENTITY: <candidate name>\nPROFILE: <candidate attributes>"
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _question_messages(
    *,
    task_id: int,
    question: str,
    hypotheses: Sequence[Hypothesis],
    question_index: int,
) -> list[dict[str, str]]:
    request = {
        "benchmark_task_id": task_id,
        "ambiguous_question": question,
        "candidate_support": [
            hypothesis.as_dict() for hypothesis in hypotheses
        ],
        "question_index": question_index,
    }
    return [
        {
            "role": "system",
            "content": (
                "Write one neutral yes/no clarification question that best "
                "distinguishes the candidate support. Ask about exactly one "
                "attribute, relationship, time, place, or property. Do not mention "
                "or ask the user to confirm a candidate entity name. Different "
                "question_index values should target different distinctions. "
                "Output exactly the question on one line and nothing else."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _classification_messages(
    *,
    hypothesis: Hypothesis,
    questions: Sequence[str],
) -> list[dict[str, str]]:
    request = {
        "candidate": hypothesis.as_dict(),
        "questions": list(questions),
    }
    return [
        {
            "role": "system",
            "content": (
                "Classify how the candidate profile would answer each of the four "
                "yes/no questions. Use Y when the profile clearly supports it, N "
                "when the profile clearly contradicts it, and U when unknown. "
                "Output exactly four characters from Y/N/U in question order, with "
                "no spaces or prose."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _responder_messages(*, context: str, question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the InteractComp closed-mode responder. Reply with exactly "
                "one of: yes, no, or i don't know. Treat the supplied context as the "
                "entire truth and use only that context. Answer yes only if it "
                "clearly states the proposition, no only if it contradicts the "
                "proposition, and i don't know otherwise. Do not use outside "
                "knowledge and do not explain."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT\n{context}\n\nQUESTION\n{question}",
        },
    ]


def _refresh_messages(
    *,
    task_id: int,
    original_question: str,
    clarification_question: str,
    clarification_answer: str,
    sample_index: int,
) -> list[dict[str, str]]:
    request = {
        "benchmark_task_id": task_id,
        "ambiguous_question": original_question,
        "clarification": {
            "question": clarification_question,
            "answer": {"Y": "yes", "N": "no", "U": "i don't know"}[
                clarification_answer
            ],
        },
        "interpretation_sample": sample_index,
    }
    return [
        {
            "role": "system",
            "content": (
                "Regenerate one concrete candidate entity after the clarification. "
                "Treat the answer as binding evidence. Different "
                "interpretation_sample values should explore materially different "
                "remaining candidates. Output exactly two lines:\n"
                "ENTITY: <candidate name>\nPROFILE: <candidate attributes>"
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _snapshot(model: ChatModel) -> dict[str, Any]:
    return model.usage_snapshot()


def aggregate_usage(
    generator: ChatModel,
    responder: ChatModel,
) -> dict[str, Any]:
    generator_snapshot = _snapshot(generator)
    responder_snapshot = _snapshot(responder)
    return {
        "physical_requests": int(
            generator_snapshot.get("adapter_requests", 0)
        )
        + int(responder_snapshot.get("adapter_requests", 0)),
        "reasoning_tokens": int(
            generator_snapshot.get("adapter_reasoning_tokens", 0)
        )
        + int(responder_snapshot.get("adapter_reasoning_tokens", 0)),
        "adapter_cost_usd": float(generator_snapshot.get("adapter_cost_usd", 0.0))
        + float(responder_snapshot.get("adapter_cost_usd", 0.0)),
        "http_attempts": int(generator_snapshot.get("http_attempts", 0))
        + int(responder_snapshot.get("http_attempts", 0)),
        "retry_count": int(generator_snapshot.get("retry_count", 0))
        + int(responder_snapshot.get("retry_count", 0)),
        "forced_exits": int(generator_snapshot.get("forced_exits", 0))
        + int(responder_snapshot.get("forced_exits", 0)),
        "generator": generator_snapshot,
        "responder": responder_snapshot,
    }


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _truth_mass(
    hypotheses: Sequence[Hypothesis],
    target: str,
) -> float:
    normalized_target = normalize_entity(target)
    return sum(
        normalize_entity(hypothesis.entity) == normalized_target
        for hypothesis in hypotheses
    ) / len(hypotheses)


def score_task(
    *,
    task_id: int,
    initial: Sequence[Hypothesis],
    classifications: Sequence[str],
    questions: Sequence[str],
    true_responses: Sequence[str],
    refreshed: Sequence[Sequence[Hypothesis]],
    target: str,
) -> dict[str, Any]:
    eigs = [
        entropy_from_labels(
            classification[index] for classification in classifications
        )
        for index in range(len(questions))
    ]
    endpoints = [
        _truth_mass(root_hypotheses, target)
        for root_hypotheses in refreshed
    ]
    initial_mass = _truth_mass(initial, target)
    selected_index = _argmax(eigs)
    oracle_index = _argmax(endpoints)
    selected_endpoint = endpoints[selected_index]
    return {
        "task_id": task_id,
        "initial_particle_count": len(initial),
        "unique_question_count": len(set(questions)),
        "informative_question_count": sum(value >= 0.30 for value in eigs),
        "non_unknown_true_response_count": sum(
            response != "U" for response in true_responses
        ),
        "refresh_particle_counts": [
            len(root_hypotheses) for root_hypotheses in refreshed
        ],
        "initial_truth_mass": initial_mass,
        "endpoint_range": max(endpoints) - min(endpoints),
        "selected_root_index": selected_index,
        "oracle_root_index": oracle_index,
        "selected_endpoint": selected_endpoint,
        "oracle_endpoint": endpoints[oracle_index],
        "candidate_mean_endpoint": statistics.fmean(endpoints),
        "selected_gain_over_initial": selected_endpoint - initial_mass,
        "selected_gain_over_candidate_mean": selected_endpoint
        - statistics.fmean(endpoints),
        "score_endpoint_spearman": spearman(eigs, endpoints),
        "roots": [
            {
                "root_index": index,
                "estimated_eig": eigs[index],
                "predicted_response_counts": {
                    label: sum(
                        classification[index] == label
                        for classification in classifications
                    )
                    for label in ("Y", "N", "U")
                },
                "true_response": true_responses[index],
                "truth_mass": endpoints[index],
                "refresh_particle_count": len(refreshed[index]),
            }
            for index in range(len(questions))
        ],
    }


def run_opportunity(
    config: Config,
    *,
    source_root: Path,
    raw_path: Path,
    generator: ChatModel,
    responder: ChatModel,
) -> dict[str, Any]:
    _path, encrypted_rows = verify_source(source_root)
    visible = _decrypted_fields(encrypted_rows, ("question",))
    tasks = {
        index: {
            "task_id": int(visible[index]["id"]),
            "question": visible[index]["question"],
        }
        for index in TASK_INDICES
    }
    raw: dict[str, Any] = {
        "task_indices": list(TASK_INDICES),
        "task_ids": list(EXPECTED_TASK_IDS),
    }
    try:
        initial_messages = [
            _initial_messages(
                task_id=tasks[index]["task_id"],
                question=tasks[index]["question"],
                sample_index=sample_index,
            )
            for index in TASK_INDICES
            for sample_index in range(PARTICLE_COUNT)
        ]
        initial_raw = generator.chat_complete_messages_batched(
            initial_messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=400,
        )
        raw["initial"] = initial_raw
        initial: dict[int, list[Hypothesis]] = {}
        for offset, index in enumerate(TASK_INDICES):
            hypotheses = [
                parse_hypothesis(response)
                for response in initial_raw[
                    offset * PARTICLE_COUNT : (offset + 1) * PARTICLE_COUNT
                ]
            ]
            if len(hypotheses) != PARTICLE_COUNT:
                raise ValueError("initial population width changed")
            initial[index] = hypotheses

        question_messages = [
            _question_messages(
                task_id=tasks[index]["task_id"],
                question=tasks[index]["question"],
                hypotheses=initial[index],
                question_index=question_index,
            )
            for index in TASK_INDICES
            for question_index in range(QUESTION_COUNT)
        ]
        question_raw = generator.chat_complete_messages_batched(
            question_messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=160,
        )
        raw["questions"] = question_raw
        questions: dict[int, list[str]] = {}
        for offset, index in enumerate(TASK_INDICES):
            task_questions = [
                parse_question(response)
                for response in question_raw[
                    offset * QUESTION_COUNT : (offset + 1) * QUESTION_COUNT
                ]
            ]
            for question in task_questions:
                normalized_question = normalize_entity(question)
                if any(
                    normalize_entity(hypothesis.entity) in normalized_question
                    for hypothesis in initial[index]
                    if len(normalize_entity(hypothesis.entity)) >= 4
                ):
                    raise ValueError("clarification question names a candidate entity")
            questions[index] = task_questions

        classification_messages = [
            _classification_messages(
                hypothesis=hypothesis,
                questions=questions[index],
            )
            for index in TASK_INDICES
            for hypothesis in initial[index]
        ]
        classification_raw = generator.chat_complete_messages_batched(
            classification_messages,
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=16,
        )
        raw["classifications"] = classification_raw
        classifications: dict[int, list[str]] = {}
        for offset, index in enumerate(TASK_INDICES):
            classifications[index] = [
                parse_classification(response)
                for response in classification_raw[
                    offset * PARTICLE_COUNT : (offset + 1) * PARTICLE_COUNT
                ]
            ]

        # Freeze all EIG scores before hidden contexts are decrypted.
        frozen_eigs = {
            index: [
                entropy_from_labels(
                    classification[root_index]
                    for classification in classifications[index]
                )
                for root_index in range(QUESTION_COUNT)
            ]
            for index in TASK_INDICES
        }

        contexts = _decrypted_fields(encrypted_rows, ("context",))
        responder_messages = [
            _responder_messages(
                context=contexts[index]["context"],
                question=question,
            )
            for index in TASK_INDICES
            for question in questions[index]
        ]
        responder_raw = responder.chat_complete_messages_batched(
            responder_messages,
            temperature=0.0,
            block_size=RESPONDER_REQUESTS,
            max_new_tokens=16,
        )
        raw["true_responses"] = responder_raw
        true_responses: dict[int, list[str]] = {}
        for offset, index in enumerate(TASK_INDICES):
            true_responses[index] = [
                parse_responder_answer(response)
                for response in responder_raw[
                    offset * QUESTION_COUNT : (offset + 1) * QUESTION_COUNT
                ]
            ]

        refresh_messages = [
            _refresh_messages(
                task_id=tasks[index]["task_id"],
                original_question=tasks[index]["question"],
                clarification_question=questions[index][root_index],
                clarification_answer=true_responses[index][root_index],
                sample_index=sample_index,
            )
            for index in TASK_INDICES
            for root_index in range(QUESTION_COUNT)
            for sample_index in range(PARTICLE_COUNT)
        ]
        refresh_raw = generator.chat_complete_messages_batched(
            refresh_messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=400,
        )
        raw["refresh"] = refresh_raw
        refreshed: dict[int, list[list[Hypothesis]]] = {}
        cursor = 0
        for index in TASK_INDICES:
            task_roots = []
            for _root_index in range(QUESTION_COUNT):
                root_hypotheses = []
                for _sample_index in range(PARTICLE_COUNT):
                    response = refresh_raw[cursor]
                    cursor += 1
                    try:
                        root_hypotheses.append(parse_hypothesis(response))
                    except Exception:
                        continue
                if len(root_hypotheses) < MIN_REFRESH_PARTICLES:
                    raise ValueError(
                        f"task {index} refresh has {len(root_hypotheses)} valid "
                        f"particles; {MIN_REFRESH_PARTICLES} required"
                    )
                task_roots.append(root_hypotheses)
            refreshed[index] = task_roots
        raw["all_target_blind_calls_and_scores_complete"] = True
        _checkpoint(raw_path, raw)
        usage = aggregate_usage(generator, responder)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise FirstLinkExecutionError(
            f"{type(exc).__name__}: {exc}",
            aggregate_usage(generator, responder),
        ) from exc

    # Exact target answers load only after every model call and EIG score freezes.
    targets = _decrypted_fields(encrypted_rows, ("answer",))
    task_records = [
        score_task(
            task_id=tasks[index]["task_id"],
            initial=initial[index],
            classifications=classifications[index],
            questions=questions[index],
            true_responses=true_responses[index],
            refreshed=refreshed[index],
            target=targets[index]["answer"],
        )
        for index in TASK_INDICES
    ]
    for index, record in zip(TASK_INDICES, task_records, strict=True):
        if [root["estimated_eig"] for root in record["roots"]] != frozen_eigs[index]:
            raise AssertionError("target loading changed a frozen EIG score")

    finite_rhos = [
        record["score_endpoint_spearman"]
        for record in task_records
        if record["score_endpoint_spearman"] is not None
    ]
    mean_rho = statistics.fmean(finite_rhos) if finite_rhos else None
    mean_selected_gain_initial = statistics.fmean(
        record["selected_gain_over_initial"] for record in task_records
    )
    mean_selected_gain_mean = statistics.fmean(
        record["selected_gain_over_candidate_mean"]
        for record in task_records
    )
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_transport_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_initial_populations_have_eight_particles": all(
            record["initial_particle_count"] == PARTICLE_COUNT
            for record in task_records
        ),
        "at_least_three_unique_questions_each": all(
            record["unique_question_count"] >= 3 for record in task_records
        ),
        "at_least_two_informative_questions_each": all(
            record["informative_question_count"] >= 2
            for record in task_records
        ),
        "at_least_two_non_unknown_true_responses_each": all(
            record["non_unknown_true_response_count"] >= 2
            for record in task_records
        ),
        "all_refreshes_have_at_least_six_particles": all(
            min(record["refresh_particle_counts"]) >= MIN_REFRESH_PARTICLES
            for record in task_records
        ),
        "target_appears_after_at_least_one_root_each": all(
            record["oracle_endpoint"] >= 1 / PARTICLE_COUNT
            for record in task_records
        ),
        "endpoint_range_at_least_one_particle_each": all(
            record["endpoint_range"] >= 1 / PARTICLE_COUNT
            for record in task_records
        ),
        "finite_positive_rho_each": all(
            record["score_endpoint_spearman"] is not None
            and record["score_endpoint_spearman"] > 0.0
            for record in task_records
        ),
        "mean_rho_at_least_0_20": mean_rho is not None and mean_rho >= 0.20,
        "mean_selected_gain_over_initial_at_least_0_125": (
            mean_selected_gain_initial >= 0.125
        ),
        "mean_selected_gain_over_candidate_mean_at_least_0_05": (
            mean_selected_gain_mean >= 0.05
        ),
        "cost_at_most_1_50": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "encrypted_data_sha256": ENCRYPTED_DATA_SHA256,
            "eligible_manifest_sha256": ELIGIBLE_MANIFEST_SHA256,
            "generator_model": GENERATOR_MODEL_ID,
            "responder_model": RESPONDER_MODEL_ID,
            "seed": SEED,
            "task_indices": list(TASK_INDICES),
            "task_ids": list(EXPECTED_TASK_IDS),
            "particle_count": PARTICLE_COUNT,
            "question_count": QUESTION_COUNT,
            "expected_requests": EXPECTED_REQUESTS,
            "generator_expected_requests": GENERATOR_REQUESTS,
            "responder_expected_requests": RESPONDER_REQUESTS,
            "reasoning_requested": False,
            "hidden_context_loaded_after_eig_scores": True,
            "target_answer_loaded_after_all_model_calls": True,
            "repairs_or_reissues": 0,
        },
        "metrics": {
            "task_count": len(task_records),
            "mean_score_endpoint_spearman": mean_rho,
            "mean_initial_truth_mass": statistics.fmean(
                record["initial_truth_mass"] for record in task_records
            ),
            "mean_selected_endpoint": statistics.fmean(
                record["selected_endpoint"] for record in task_records
            ),
            "mean_candidate_endpoint": statistics.fmean(
                record["candidate_mean_endpoint"] for record in task_records
            ),
            "mean_selected_gain_over_initial": mean_selected_gain_initial,
            "mean_selected_gain_over_candidate_mean": mean_selected_gain_mean,
        },
        "tasks": task_records,
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
            user = messages[-1]["content"]
            if self.role == "responder":
                responses.append("yes")
                continue
            request = json.loads(user)
            if "questions" in request:
                sample = int(
                    normalize_entity(request["candidate"]["entity"])[-1:] or "0"
                )
                responses.append(("YNUU", "NYUU", "UNYU")[sample % 3])
            elif "candidate_support" in request:
                responses.append(
                    f"Does the target have fixture attribute {request['question_index']}?"
                )
            else:
                sample = int(request["interpretation_sample"])
                responses.append(
                    f"ENTITY: Fixture Candidate {sample}\n"
                    f"PROFILE: Fixture profile {sample}."
                )
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
            "http_attempts": self.requests,
            "retry_count": 0,
            "forced_exits": 0,
        }


def _nonthinking_spec(spec: Any) -> Any:
    return replace(
        spec,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )


def _build_models(config: Config) -> tuple[ChatModel, ChatModel]:
    generator_spec = _nonthinking_spec(config.model_pairs[0].questioner)
    responder_spec = _nonthinking_spec(config.model_pairs[0].answerer)
    if generator_spec.model != GENERATOR_MODEL_ID:
        raise ValueError("InteractComp config selects the wrong generator")
    if responder_spec.model != RESPONDER_MODEL_ID:
        raise ValueError("InteractComp config selects the wrong responder")
    return (
        build_model_adapter(generator_spec, config),
        build_model_adapter(responder_spec, config),
    )


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
    config.openrouter_projected_cost_usd = 0.40
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 64
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    if args.dry_run:
        generator: ChatModel = DeterministicFixtureModel("generator")
        responder: ChatModel = DeterministicFixtureModel("responder")
    else:
        generator, responder = _build_models(config)
    try:
        payload = run_opportunity(
            config,
            source_root=args.source_root,
            raw_path=raw_path,
            generator=generator,
            responder=responder,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, FirstLinkExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "OPPORTUNITY_FAILURE.json", failure)
        raise
    output = args.output_dir / "OPPORTUNITY.json"
    _checkpoint(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
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
