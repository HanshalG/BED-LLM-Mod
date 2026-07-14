#!/usr/bin/env python3
"""Audit the official-data MediQ integration smoke before claims runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


ERROR_RE = re.compile(
    r"Traceback|RuntimeError|ValueError|CUDA out of memory|Killed", re.I
)
TARGET_LEAK_RE = re.compile(r"\b(?:answer|option|choice)\s*[A-Z]\b", re.I)
COMPOUND_QUERY_RE = re.compile(r"\b(?:and|or)\b", re.I)
UNAVAILABLE_MARKERS = (
    "unavailable",
    "not in record",
    "cannot answer",
    "unknown",
    "not provided",
    "not recorded",
)
UNAVAILABLE_OUTCOME = "Information unavailable / not in record"
CANONICAL_OUTCOMES = ["Yes", "No", UNAVAILABLE_OUTCOME]
PATIENT_CANNOT_ANSWER = (
    "The patient cannot answer this question from the supplied record."
)
MEDIQ_REPOSITORY = "https://github.com/stellali7/MediQ.git"
MEDIQ_COMMIT = "faa2ce62fef0423e35af4c31d7537aad973173eb"
MEDIQ_IMEDQA_SHA256 = (
    "3bfc7090d060dd8d11e4237344ed78846707faab433a84d078191627ad3c9526"
)
MEDIQ_IMEDQA_RAW_ROWS = 1272
MEDIQ_IMEDQA_EXCLUDED_IDS = ["224", "298", "779"]

QUERY_STOPWORDS = {
    "a",
    "an",
    "any",
    "are",
    "at",
    "can",
    "child",
    "could",
    "current",
    "currently",
    "did",
    "do",
    "does",
    "experience",
    "experienced",
    "experiences",
    "for",
    "from",
    "had",
    "has",
    "have",
    "history",
    "in",
    "is",
    "of",
    "on",
    "patient",
    "recent",
    "report",
    "reported",
    "reports",
    "the",
    "there",
    "to",
    "was",
    "were",
    "will",
    "with",
    "would",
}


def _query_content_tokens(query: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", query.casefold())
        if token not in QUERY_STOPWORDS
    }


def _queries_semantically_equivalent(left: str, right: str) -> bool:
    left_tokens = _query_content_tokens(left)
    right_tokens = _query_content_tokens(right)
    if not left_tokens or not right_tokens:
        return left.strip().casefold() == right.strip().casefold()
    if left_tokens == right_tokens:
        return True
    overlap = len(left_tokens & right_tokens)
    return overlap >= 2 and overlap / len(left_tokens | right_tokens) >= 0.8


def _candidate_contract_error(trial: dict[str, Any], query: str) -> str | None:
    if COMPOUND_QUERY_RE.search(query):
        return "compound query"
    if not re.match(
        r"^(?:is|are|was|were|has|have|had|do|does|did|can|could|would|will)\b",
        query.strip(),
        flags=re.IGNORECASE,
    ):
        return "not a binary predicate"
    normalized = " ".join(re.findall(r"[a-z0-9]+", query.casefold()))
    options = trial.get("options", {})
    if isinstance(options, dict):
        for option in options.values():
            option_normalized = " ".join(
                re.findall(r"[a-z0-9]+", str(option).casefold())
            )
            if len(option_normalized) >= 4 and option_normalized in normalized:
                return f"directly asks about answer option {option!r}"
    if re.search(
        r"\b(?:treated|treatment|given|administered|prescribed|prescription|"
        r"received|therapy|medication|drug|antibiotic|managed|management|"
        r"ordered|performed|obtained|diagnosed|diagnosis)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "asks about diagnosis or management rather than patient evidence"
    if re.search(
        r"\b(?:hemodynamically stable|clinically stable|critically ill|toxic appearing)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return "asks for a derived clinical judgment"
    return None


def _load_one(run_dir: Path, name: str) -> tuple[Path, Any]:
    matches = sorted(run_dir.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} under {run_dir}, found {len(matches)}")
    return matches[0], json.loads(matches[0].read_text())


def _metric_max(metrics: dict[str, Any], name: str) -> float:
    values = metrics.get(name, [])
    return max((float(value) for value in values), default=0.0)


def _categorical_eig(prior: np.ndarray, likelihoods: np.ndarray) -> float:
    predictive = prior @ likelihoods
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(
            likelihoods,
            predictive[None, :],
            out=np.zeros_like(likelihoods),
            where=predictive[None, :] > 0.0,
        )
        terms = prior[:, None] * likelihoods * np.log(np.maximum(ratios, 1e-300))
    return float(np.sum(np.where(np.isfinite(terms), terms, 0.0)))


def _candidate_diagnostics(
    trial: dict[str, Any],
    turn: dict[str, Any],
    prior_queries: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    option_labels = list(trial.get("options", {}))
    details = turn.get("candidate_details")
    if not isinstance(details, list) or not details:
        return ["missing candidate likelihood diagnostics"], summaries
    set_validation = turn.get("candidate_set_semantic_validation")
    if (
        not isinstance(set_validation, dict)
        or set_validation.get("valid") is not True
        or not isinstance(set_validation.get("reason"), str)
        or not set_validation["reason"].strip()
    ):
        errors.append("missing successful candidate-set semantic validation")

    scores: list[float] = []
    current_queries: list[str] = []
    for candidate_index, candidate in enumerate(details):
        prefix = f"candidate {candidate_index}"
        outcomes = candidate.get("outcomes")
        prior_raw = candidate.get("prior")
        likelihood_raw = candidate.get("likelihoods")
        predictive_raw = candidate.get("predictive_outcome_probabilities")
        score = candidate.get("score")
        try:
            query = str(candidate.get("query", ""))
            contract_error = _candidate_contract_error(trial, query)
            if contract_error is not None:
                raise ValueError(contract_error)
            if any(
                _queries_semantically_equivalent(query, previous)
                for previous in prior_queries
            ):
                raise ValueError("semantically repeats an earlier query")
            if any(
                _queries_semantically_equivalent(query, previous)
                for previous in current_queries
            ):
                raise ValueError("semantically duplicates another candidate")
            current_queries.append(query)
            if not isinstance(outcomes, list) or not 3 <= len(outcomes) <= 5:
                raise ValueError("requires 3-5 outcomes")
            if len({str(value).casefold() for value in outcomes}) != len(outcomes):
                raise ValueError("outcomes are duplicated")
            unavailable_count = sum(
                any(marker in str(outcome).casefold() for marker in UNAVAILABLE_MARKERS)
                for outcome in outcomes
            )
            if unavailable_count != 1 or outcomes.count(UNAVAILABLE_OUTCOME) != 1:
                raise ValueError("requires exactly one canonical unavailable outcome")
            if outcomes != CANONICAL_OUTCOMES:
                raise ValueError("requires canonical Yes/No/unavailable outcomes")
            validation = candidate.get("semantic_validation")
            if (
                not isinstance(validation, dict)
                or validation.get("valid") is not True
                or not isinstance(validation.get("reason"), str)
                or not validation["reason"].strip()
            ):
                raise ValueError("missing successful semantic candidate validation")
            if not isinstance(prior_raw, dict) or list(prior_raw) != option_labels:
                raise ValueError("prior labels do not match task options")
            if not isinstance(likelihood_raw, dict) or list(likelihood_raw) != option_labels:
                raise ValueError("likelihood labels do not match task options")
            prior = np.asarray([prior_raw[label] for label in option_labels], dtype=float)
            likelihoods = np.asarray(
                [likelihood_raw[label] for label in option_labels], dtype=float
            )
            if likelihoods.shape != (len(option_labels), len(outcomes)):
                raise ValueError("likelihood matrix has the wrong shape")
            if (
                not np.all(np.isfinite(prior))
                or not np.all(np.isfinite(likelihoods))
                or np.any(prior < 0.0)
                or np.any(likelihoods < 0.0)
            ):
                raise ValueError("prior or likelihood contains invalid values")
            if not math.isclose(float(np.sum(prior)), 1.0, abs_tol=1e-6):
                raise ValueError("prior does not sum to one")
            if not np.allclose(np.sum(likelihoods, axis=1), 1.0, atol=1e-6):
                raise ValueError("likelihood rows do not sum to one")
            predictive = prior @ likelihoods
            if not isinstance(predictive_raw, dict) or list(predictive_raw) != outcomes:
                raise ValueError("predictive outcome labels do not match")
            logged_predictive = np.asarray(
                [predictive_raw[outcome] for outcome in outcomes], dtype=float
            )
            if not np.allclose(predictive, logged_predictive, atol=1e-8):
                raise ValueError("logged predictive probabilities are inconsistent")
            recomputed_score = _categorical_eig(prior, likelihoods)
            score_value = float(score)
            if not math.isfinite(score_value) or not math.isclose(
                score_value, recomputed_score, rel_tol=1e-7, abs_tol=1e-9
            ):
                raise ValueError("logged EIG score is inconsistent")
            scores.append(score_value)
            row_span = max(
                (
                    float(np.sum(np.abs(left - right)))
                    for left_index, left in enumerate(likelihoods)
                    for right in likelihoods[left_index + 1 :]
                ),
                default=0.0,
            )
            summaries.append(
                {
                    "query": query,
                    "outcomes": outcomes,
                    "eig": score_value,
                    "max_label_likelihood_l1_span": row_span,
                }
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"{prefix}: {exc}")

    selected_score = turn.get("selected_score")
    if scores and (
        not isinstance(selected_score, (int, float))
        or not math.isclose(float(selected_score), max(scores), abs_tol=1e-8)
    ):
        errors.append("selected score is not the maximum candidate EIG")
    return errors, summaries


def analyze(
    run_dir: Path,
    *,
    expected_tasks: int = 5,
    expected_rounds: int = 2,
    coverage_threshold: float = 0.85,
    relevance_threshold: float = 0.85,
) -> dict[str, Any]:
    artifact_path, trials = _load_one(run_dir, "mediq_interactions.json")
    manifest_path, manifest = _load_one(run_dir, "mediq_data_manifest.json")
    if not isinstance(trials, list):
        raise ValueError("MediQ interaction artifact must be a list")
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise ValueError(f"Missing {metrics_path}")
    payload = json.loads(metrics_path.read_text())
    items = payload.get("items", [])
    if len(items) != 1 or items[0].get("method") != "EIG":
        raise ValueError("MediQ integration smoke requires exactly one EIG metrics item")
    metrics = items[0].get("metrics", {})

    turns = [turn for trial in trials for turn in trial.get("turns", [])]
    clean_turns = sum(turn.get("mapped_cleanly") is True for turn in turns)
    grounded_turns = sum(turn.get("grounded") is True for turn in turns)
    relevant_turns = sum(turn.get("relevant") is True for turn in turns)
    denominator = len(turns)
    coverage = clean_turns / denominator if denominator else 0.0
    grounding_rate = grounded_turns / denominator if denominator else 0.0
    relevance_rate = relevant_turns / denominator if denominator else 0.0

    grounding_errors: list[str] = []
    mapping_errors: list[str] = []
    relevance_errors: list[str] = []
    unmapped_turns: list[str] = []
    candidate_errors: list[str] = []
    target_leaking_queries: list[str] = []
    compound_queries: list[str] = []
    task_summaries: list[dict[str, Any]] = []
    all_candidate_summaries: list[dict[str, Any]] = []
    for trial in trials:
        task_id = str(trial.get("task_id"))
        facts = trial.get("facts", [])
        turn_summaries: list[dict[str, Any]] = []
        prior_queries: list[str] = []
        for turn_index, turn in enumerate(trial.get("turns", [])):
            prefix = f"{task_id} turn {turn_index + 1}"
            query = str(turn.get("query", ""))
            if TARGET_LEAK_RE.search(query):
                target_leaking_queries.append(f"{prefix}: {query}")
            if COMPOUND_QUERY_RE.search(query):
                compound_queries.append(f"{prefix}: {query}")
            indices = turn.get("selected_fact_indices")
            reply = turn.get("reply")
            cannot_answer = turn.get("cannot_answer") is True
            try:
                if not isinstance(indices, list) or any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in indices
                ):
                    raise ValueError("invalid selected fact indices")
                if cannot_answer:
                    expected_reply = PATIENT_CANNOT_ANSWER
                    if indices:
                        raise ValueError("cannot-answer reply selected facts")
                else:
                    if not indices or any(not 0 <= index < len(facts) for index in indices):
                        raise ValueError("selected facts are empty or out of range")
                    expected_reply = "\n".join(facts[index] for index in indices)
                if reply != expected_reply or turn.get("grounded") is not True:
                    raise ValueError("reply is not a verbatim assembly of selected facts")
            except (TypeError, ValueError) as exc:
                grounding_errors.append(f"{prefix}: {exc}")

            outcomes = turn.get("outcomes", [])
            mapped_cleanly = turn.get("mapped_cleanly") is True
            mapped_outcome = turn.get("mapped_outcome")
            if turn.get("relevant") is not True:
                relevance_errors.append(f"{prefix}: patient reply was judged irrelevant")
            if mapped_cleanly and mapped_outcome not in outcomes:
                mapping_errors.append(f"{prefix}: clean mapping is not a supplied outcome")
            elif not mapped_cleanly and mapped_outcome is not None:
                mapping_errors.append(f"{prefix}: unclean mapping retained an outcome")
            elif not mapped_cleanly:
                unmapped_turns.append(prefix)
            errors, candidate_summaries = _candidate_diagnostics(
                trial, turn, prior_queries
            )
            candidate_errors.extend(f"{prefix}: {error}" for error in errors)
            all_candidate_summaries.extend(candidate_summaries)
            turn_summaries.append(
                {
                    "query": query,
                    "outcomes": outcomes,
                    "reply": reply,
                    "selected_fact_indices": indices,
                    "mapped_outcome": turn.get("mapped_outcome"),
                    "selected_eig": turn.get("selected_score"),
                    "realized_entropy_drop": turn.get("metrics", {}).get(
                        "realized_entropy_drop"
                    ),
                    "realized_truth_log_probability_gain": turn.get("metrics", {}).get(
                        "realized_truth_log_probability_gain"
                    ),
                    "observed_outcome_predictive_probability": turn.get(
                        "metrics", {}
                    ).get("observed_outcome_predictive_probability"),
                    "candidate_summaries": candidate_summaries,
                }
            )
            prior_queries.append(query)
        task_summaries.append(
            {
                "task_id": task_id,
                "source_id": trial.get("source_id"),
                "initial_info": trial.get("initial_info"),
                "question": trial.get("question"),
                "options": trial.get("options"),
                "answer_idx": trial.get("answer_idx"),
                "turns": turn_summaries,
            }
        )

    task_ids = [trial.get("task_id") for trial in trials]
    source_ids = [str(trial.get("source_id")) for trial in trials]
    official_ids = all(
        trial.get("dataset") == "imedqa"
        and str(trial.get("task_id", "")).startswith("mediq:imedqa:")
        for trial in trials
    )
    exact_rounds = all(
        len(trial.get("turns", [])) == expected_rounds for trial in trials
    )
    structured_failures = _metric_max(metrics, "structured_parse_failures")
    candidate_validation_failures = _metric_max(
        metrics, "candidate_validation_failures"
    )
    candidate_set_validation_checks = _metric_max(
        metrics, "candidate_set_validation_checks"
    )
    relevance_failures = _metric_max(metrics, "patient_relevance_failures")

    log_path = run_dir / "run.log"
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    error_lines = sorted(
        {line.strip() for line in log_text.splitlines() if ERROR_RE.search(line)}
    )
    forced_exits = max(
        int(_metric_max(metrics, "backend_forced_exits")),
        log_text.count("Forced thinking exit"),
    )
    requests = int(_metric_max(metrics, "backend_requests"))
    valid_manifest = (
        manifest.get("repository") == MEDIQ_REPOSITORY
        and manifest.get("commit") == MEDIQ_COMMIT
        and manifest.get("dataset") == "imedqa"
        and manifest.get("expected_sha256") == MEDIQ_IMEDQA_SHA256
        and manifest.get("raw_row_count") == MEDIQ_IMEDQA_RAW_ROWS
        and manifest.get("usable_row_count")
        == MEDIQ_IMEDQA_RAW_ROWS - len(MEDIQ_IMEDQA_EXCLUDED_IDS)
        and manifest.get("skip_unusable_tasks") is True
        and manifest.get("excluded_source_ids") == MEDIQ_IMEDQA_EXCLUDED_IDS
        and [str(value) for value in manifest.get("selected_source_ids", [])]
        == source_ids
    )
    checks = {
        "expected_task_count": len(trials) == expected_tasks,
        "pinned_official_data_manifest": valid_manifest,
        "unique_official_task_ids": official_ids and len(set(task_ids)) == len(task_ids),
        "exact_round_count": exact_rounds,
        "answer_set_coverage": coverage >= coverage_threshold,
        "verbatim_patient_grounding": grounding_rate == 1.0 and not grounding_errors,
        "patient_relevance": relevance_rate >= relevance_threshold
        and not relevance_errors,
        "valid_mapping_contract": not mapping_errors,
        "zero_structured_parse_failures": structured_failures == 0.0,
        "zero_candidate_validation_failures": candidate_validation_failures == 0.0,
        "candidate_sets_semantically_validated": (
            candidate_set_validation_checks >= denominator
        ),
        "zero_patient_relevance_failures": relevance_failures == 0.0,
        "valid_finite_target_eig_tables": not candidate_errors,
        "no_target_leaking_queries": not target_leaking_queries,
        "no_compound_queries": not compound_queries,
        "no_runtime_errors": not error_lines,
    }
    automated_pass = all(checks.values())
    positive_eig = sum(float(item["eig"]) > 1e-9 for item in all_candidate_summaries)
    return {
        "status": (
            "automated_pass_manual_review_pending"
            if automated_pass
            else "automated_fail"
        ),
        "automated_pass": automated_pass,
        "manual_transcript_review_required": True,
        "manual_review_focus": [
            "Are generated yes/no predicates atomic, clinically sensible, and non-redundant?",
            "Can the selected patient facts explicitly establish Yes or No for each predicate?",
            "Do selected facts directly answer the question rather than merely concern the case?",
        ],
        "checks": checks,
        "run_dir": str(run_dir.resolve()),
        "interaction_artifact": str(artifact_path.resolve()),
        "data_manifest": str(manifest_path.resolve()),
        "num_tasks": len(trials),
        "num_turns": denominator,
        "expected_tasks": expected_tasks,
        "expected_rounds": expected_rounds,
        "answer_set_coverage": coverage,
        "coverage_threshold": coverage_threshold,
        "patient_grounding_rate": grounding_rate,
        "patient_relevance_rate": relevance_rate,
        "relevance_threshold": relevance_threshold,
        "structured_parse_failures": structured_failures,
        "structured_parse_retries": _metric_max(metrics, "structured_parse_retries"),
        "candidate_validation_checks": _metric_max(
            metrics, "candidate_validation_checks"
        ),
        "candidate_validation_retries": _metric_max(
            metrics, "candidate_validation_retries"
        ),
        "candidate_validation_failures": candidate_validation_failures,
        "candidate_set_validation_checks": candidate_set_validation_checks,
        "candidate_set_validation_rejections": _metric_max(
            metrics, "candidate_set_validation_rejections"
        ),
        "patient_relevance_failures": relevance_failures,
        "grounding_errors": grounding_errors,
        "mapping_errors": mapping_errors,
        "relevance_errors": relevance_errors,
        "unmapped_turns": unmapped_turns,
        "candidate_diagnostic_errors": candidate_errors,
        "target_leaking_queries": target_leaking_queries,
        "compound_queries": compound_queries,
        "num_candidate_scores": len(all_candidate_summaries),
        "positive_candidate_eig_rate": (
            positive_eig / len(all_candidate_summaries)
            if all_candidate_summaries
            else 0.0
        ),
        "backend_requests": requests,
        "backend_prompt_tokens": int(_metric_max(metrics, "backend_prompt_tokens")),
        "backend_completion_tokens": int(
            _metric_max(metrics, "backend_completion_tokens")
        ),
        "backend_reasoning_tokens": int(
            _metric_max(metrics, "backend_reasoning_tokens")
        ),
        "backend_cost_usd": _metric_max(metrics, "backend_cost_usd"),
        "forced_thinking_exits": forced_exits,
        "forced_exit_rate": forced_exits / requests if requests else None,
        "runtime_error_lines": error_lines,
        "endpoint_accuracy": (
            metrics.get("accuracy", [None])[-1] if metrics.get("accuracy") else None
        ),
        "endpoint_accuracy_is_diagnostic_only": True,
        "task_summaries": task_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-tasks", type=int, default=5)
    parser.add_argument("--expected-rounds", type=int, default=2)
    parser.add_argument("--coverage-threshold", type=float, default=0.85)
    parser.add_argument("--relevance-threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.run_dir,
        expected_tasks=args.expected_tasks,
        expected_rounds=args.expected_rounds,
        coverage_threshold=args.coverage_threshold,
        relevance_threshold=args.relevance_threshold,
    )
    output = args.output or args.run_dir / "mediq_step0_analysis.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
