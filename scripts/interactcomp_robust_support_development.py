#!/usr/bin/env python3
"""Develop a misspecification-aware InteractComp root score on open tasks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.atd_code_first_link_audit import spearman
from scripts.interactcomp_first_link_opportunity import (
    ChatModel,
    GENERATOR_MODEL_ID,
    Hypothesis,
    PARTICLE_COUNT,
    QUESTION_COUNT,
    TASK_INDICES,
    _argmax,
    _checkpoint,
    _classification_messages,
    _decrypted_fields,
    entropy_from_labels,
    normalize_entity,
    parse_classification,
    parse_hypothesis,
    verify_source,
)


INTERFACE_VERSION = "interactcomp-robust-support-development-1"
PRIOR_RUN_ID = "interactcomp-first-link-opportunity-20260725T102000Z"
PRIOR_PUBLIC_SHA256 = (
    "86c741d9cf38f83b282b8f5048d50d10b54a563ca05bdec897ae7ec72b9a0e91"
)
PRIOR_PRIVATE_SHA256 = (
    "fb90c37a65d004a79b1b0aff8dce986aaf0de3ff0abbd633b68aff01b83d3f95"
)
EXPECTED_REQUESTS = len(TASK_INDICES) * PARTICLE_COUNT * 2
MIN_UNIQUE_AUXILIARY = 4
MAX_COST_USD = 0.25


class DevelopmentExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(labels: Sequence[str]) -> dict[str, float]:
    if not labels:
        raise ValueError("response label population is empty")
    return {
        label: sum(value == label for value in labels) / len(labels)
        for label in ("Y", "N", "U")
    }


def _entropy_distribution(distribution: dict[str, float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in distribution.values()
        if probability > 0.0
    )


def balanced_model_information(
    current_labels: Sequence[str],
    auxiliary_labels: Sequence[str],
) -> float:
    """Return I(Z; Y) for balanced current/auxiliary model identity Z."""
    current = _distribution(current_labels)
    auxiliary = _distribution(auxiliary_labels)
    mixture = {
        label: 0.5 * current[label] + 0.5 * auxiliary[label]
        for label in ("Y", "N", "U")
    }
    return (
        _entropy_distribution(mixture)
        - 0.5 * _entropy_distribution(current)
        - 0.5 * _entropy_distribution(auxiliary)
    )


def robust_root_scores(
    current_classifications: Sequence[str],
    auxiliary_classifications: Sequence[str],
) -> tuple[list[float], list[float], list[float]]:
    if len(current_classifications) != PARTICLE_COUNT:
        raise ValueError("current population width changed")
    if len(auxiliary_classifications) != PARTICLE_COUNT:
        raise ValueError("auxiliary population width changed")
    current_eigs = []
    model_information = []
    robust_scores = []
    for root_index in range(QUESTION_COUNT):
        current_labels = [
            classification[root_index]
            for classification in current_classifications
        ]
        auxiliary_labels = [
            classification[root_index]
            for classification in auxiliary_classifications
        ]
        eig = entropy_from_labels(current_labels)
        misspecification_information = balanced_model_information(
            current_labels,
            auxiliary_labels,
        )
        current_eigs.append(eig)
        model_information.append(misspecification_information)
        robust_scores.append(eig + misspecification_information)
    return current_eigs, model_information, robust_scores


def _auxiliary_messages(
    *,
    task_id: int,
    question: str,
    initial: Sequence[Hypothesis],
    sample_index: int,
) -> list[dict[str, str]]:
    request = {
        "benchmark_task_id": task_id,
        "ambiguous_question": question,
        "excluded_current_support": [
            hypothesis.as_dict() for hypothesis in initial
        ],
        "outside_sample": sample_index,
    }
    return [
        {
            "role": "system",
            "content": (
                "Generate one plausible answer candidate outside the supplied "
                "current support. Look for a materially different interpretation "
                "or a shared assumption the current candidates may have missed. "
                "The entity must not be equivalent to any excluded candidate. "
                "Different outside_sample values should explore different "
                "possibilities. Do not ask a question and do not give multiple "
                "entities. Output exactly two lines:\n"
                "ENTITY: <candidate name>\nPROFILE: <candidate attributes>"
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _usage(model: ChatModel) -> dict[str, Any]:
    return model.usage_snapshot()


def _load_prior_inputs(
    source_root: Path,
    prior_private_path: Path,
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, list[Hypothesis]],
    dict[int, list[str]],
    dict[int, list[str]],
]:
    _path, encrypted_rows = verify_source(source_root)
    if _sha256(prior_private_path) != PRIOR_PRIVATE_SHA256:
        raise ValueError("prior private artifact hash changed")
    visible = _decrypted_fields(encrypted_rows, ("question",))
    tasks = {
        index: {
            "task_id": int(visible[index]["id"]),
            "question": visible[index]["question"],
        }
        for index in TASK_INDICES
    }
    raw = json.loads(prior_private_path.read_text(encoding="utf-8"))
    initial = {}
    questions = {}
    classifications = {}
    for offset, index in enumerate(TASK_INDICES):
        initial[index] = [
            parse_hypothesis(response)
            for response in raw["initial"][
                offset * PARTICLE_COUNT : (offset + 1) * PARTICLE_COUNT
            ]
        ]
        questions[index] = raw["questions"][
            offset * QUESTION_COUNT : (offset + 1) * QUESTION_COUNT
        ]
        classifications[index] = [
            parse_classification(response)
            for response in raw["classifications"][
                offset * PARTICLE_COUNT : (offset + 1) * PARTICLE_COUNT
            ]
        ]
    return tasks, initial, questions, classifications


def _score_against_prior_endpoints(
    *,
    prior_public_path: Path,
    frozen_scores: dict[int, dict[str, list[float]]],
    unique_auxiliary_counts: dict[int, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _sha256(prior_public_path) != PRIOR_PUBLIC_SHA256:
        raise ValueError("prior public artifact hash changed")
    prior = json.loads(prior_public_path.read_text(encoding="utf-8"))
    prior_by_id = {int(record["task_id"]): record for record in prior["tasks"]}
    records = []
    for index in TASK_INDICES:
        task_id = int(prior["protocol"]["task_ids"][TASK_INDICES.index(index)])
        prior_record = prior_by_id[task_id]
        endpoints = [float(root["truth_mass"]) for root in prior_record["roots"]]
        expected_eigs = [
            float(root["estimated_eig"]) for root in prior_record["roots"]
        ]
        scores = frozen_scores[index]
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(
                scores["current_eig"],
                expected_eigs,
                strict=True,
            )
        ):
            raise AssertionError("cached current EIG does not match prior artifact")
        eig_index = _argmax(scores["current_eig"])
        robust_index = _argmax(scores["robust"])
        oracle_index = _argmax(endpoints)
        records.append(
            {
                "task_id": task_id,
                "unique_auxiliary_count": unique_auxiliary_counts[index],
                "eig_selected_root_index": eig_index,
                "robust_selected_root_index": robust_index,
                "oracle_root_index": oracle_index,
                "eig_selected_endpoint": endpoints[eig_index],
                "robust_selected_endpoint": endpoints[robust_index],
                "oracle_endpoint": endpoints[oracle_index],
                "robust_endpoint_gain_over_eig": (
                    endpoints[robust_index] - endpoints[eig_index]
                ),
                "eig_endpoint_spearman": spearman(
                    scores["current_eig"],
                    endpoints,
                ),
                "robust_endpoint_spearman": spearman(
                    scores["robust"],
                    endpoints,
                ),
                "roots": [
                    {
                        "root_index": root_index,
                        "current_eig": scores["current_eig"][root_index],
                        "model_misspecification_information": scores[
                            "model_information"
                        ][root_index],
                        "robust_score": scores["robust"][root_index],
                        "truth_mass": endpoints[root_index],
                    }
                    for root_index in range(QUESTION_COUNT)
                ],
            }
        )
    finite_rhos = [
        record["robust_endpoint_spearman"]
        for record in records
        if record["robust_endpoint_spearman"] is not None
    ]
    summary = {
        "mean_robust_endpoint_spearman": (
            statistics.fmean(finite_rhos) if finite_rhos else None
        ),
        "mean_robust_endpoint_gain_over_eig": statistics.fmean(
            record["robust_endpoint_gain_over_eig"] for record in records
        ),
        "robust_selects_oracle_count": sum(
            record["robust_selected_root_index"]
            == record["oracle_root_index"]
            for record in records
        ),
    }
    return records, summary


def run_development(
    config: Config,
    *,
    source_root: Path,
    prior_public_path: Path,
    prior_private_path: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    tasks, initial, questions, current_classifications = _load_prior_inputs(
        source_root,
        prior_private_path,
    )
    raw: dict[str, Any] = {
        "prior_run_id": PRIOR_RUN_ID,
        "task_indices": list(TASK_INDICES),
    }
    try:
        auxiliary_messages = [
            _auxiliary_messages(
                task_id=tasks[index]["task_id"],
                question=tasks[index]["question"],
                initial=initial[index],
                sample_index=sample_index,
            )
            for index in TASK_INDICES
            for sample_index in range(PARTICLE_COUNT)
        ]
        auxiliary_raw = model.chat_complete_messages_batched(
            auxiliary_messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=400,
        )
        raw["auxiliary"] = auxiliary_raw
        auxiliary: dict[int, list[Hypothesis]] = {}
        unique_counts = {}
        for offset, index in enumerate(TASK_INDICES):
            population = [
                parse_hypothesis(response)
                for response in auxiliary_raw[
                    offset * PARTICLE_COUNT : (offset + 1) * PARTICLE_COUNT
                ]
            ]
            excluded = {
                normalize_entity(hypothesis.entity)
                for hypothesis in initial[index]
            }
            if any(
                normalize_entity(hypothesis.entity) in excluded
                for hypothesis in population
            ):
                raise ValueError("auxiliary population repeats current support")
            auxiliary[index] = population
            unique_counts[index] = len(
                {
                    normalize_entity(hypothesis.entity)
                    for hypothesis in population
                }
            )

        classification_messages = [
            _classification_messages(
                hypothesis=hypothesis,
                questions=questions[index],
            )
            for index in TASK_INDICES
            for hypothesis in auxiliary[index]
        ]
        auxiliary_classification_raw = model.chat_complete_messages_batched(
            classification_messages,
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=16,
        )
        raw["auxiliary_classifications"] = auxiliary_classification_raw
        auxiliary_classifications = {}
        frozen_scores = {}
        for offset, index in enumerate(TASK_INDICES):
            auxiliary_classifications[index] = [
                parse_classification(response)
                for response in auxiliary_classification_raw[
                    offset * PARTICLE_COUNT : (offset + 1) * PARTICLE_COUNT
                ]
            ]
            current_eig, model_information, robust = robust_root_scores(
                current_classifications[index],
                auxiliary_classifications[index],
            )
            frozen_scores[index] = {
                "current_eig": current_eig,
                "model_information": model_information,
                "robust": robust,
            }
        raw["all_scores_frozen_before_endpoint_load"] = True
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise DevelopmentExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    records, summary = _score_against_prior_endpoints(
        prior_public_path=prior_public_path,
        frozen_scores=frozen_scores,
        unique_auxiliary_counts=unique_counts,
    )
    gates = {
        "exact_request_count": int(usage.get("adapter_requests", 0))
        == EXPECTED_REQUESTS,
        "exact_http_attempt_count": int(usage.get("http_attempts", 0))
        == EXPECTED_REQUESTS,
        "zero_transport_retries": int(usage.get("retry_count", 0)) == 0,
        "zero_reasoning_tokens": int(
            usage.get("adapter_reasoning_tokens", 0)
        )
        == 0,
        "zero_forced_exits": int(usage.get("forced_exits", 0)) == 0,
        "at_least_four_unique_auxiliary_each": all(
            record["unique_auxiliary_count"] >= MIN_UNIQUE_AUXILIARY
            for record in records
        ),
        "robust_selects_oracle_each": all(
            record["robust_selected_root_index"]
            == record["oracle_root_index"]
            for record in records
        ),
        "positive_robust_rho_each": all(
            record["robust_endpoint_spearman"] is not None
            and record["robust_endpoint_spearman"] > 0.0
            for record in records
        ),
        "mean_robust_rho_at_least_0_50": (
            summary["mean_robust_endpoint_spearman"] is not None
            and summary["mean_robust_endpoint_spearman"] >= 0.50
        ),
        "mean_endpoint_gain_over_eig_at_least_0_0625": (
            summary["mean_robust_endpoint_gain_over_eig"] >= 0.0625
        ),
        "cost_at_most_0_25": float(usage.get("adapter_cost_usd", 0.0))
        <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "prior_run_id": PRIOR_RUN_ID,
            "prior_public_sha256": PRIOR_PUBLIC_SHA256,
            "prior_private_sha256": PRIOR_PRIVATE_SHA256,
            "model": GENERATOR_MODEL_ID,
            "reasoning_requested": False,
            "task_indices": list(TASK_INDICES),
            "particle_count": PARTICLE_COUNT,
            "question_count": QUESTION_COUNT,
            "expected_requests": EXPECTED_REQUESTS,
            "balanced_model_prior": 0.5,
            "robust_score": "current_eig_plus_model_identity_mutual_information",
            "endpoints_loaded_after_scores_frozen": True,
            "repairs_or_reissues": 0,
        },
        "metrics": summary,
        "tasks": records,
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
        responses = []
        patterns = ("YYYY", "NNNN", "UUUU", "YNUY")
        for messages in batch_messages:
            request = json.loads(messages[-1]["content"])
            if "questions" in request:
                entity = normalize_entity(request["candidate"]["entity"])
                sample = int(entity[-1])
                responses.append(patterns[sample % len(patterns)])
            else:
                sample = int(request["outside_sample"])
                responses.append(
                    f"ENTITY: Fixture Alternative {sample}\n"
                    f"PROFILE: Alternative fixture profile {sample}."
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


def _build_model(config: Config) -> ChatModel:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != GENERATOR_MODEL_ID:
        raise ValueError("InteractComp robust-support config selects wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prior-public-path", type=Path, required=True)
    parser.add_argument("--prior-private-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.05
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 32
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
        payload = run_development(
            config,
            source_root=args.source_root,
            prior_public_path=args.prior_public_path,
            prior_private_path=args.prior_private_path,
            raw_path=raw_path,
            model=model,
        )
        payload["protocol"]["private_raw_sha256"] = _sha256(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, DevelopmentExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = _sha256(raw_path)
        _checkpoint(args.output_dir / "DEVELOPMENT_FAILURE.json", failure)
        raise
    output = args.output_dir / "DEVELOPMENT.json"
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
