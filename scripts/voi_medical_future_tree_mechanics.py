#!/usr/bin/env python3
"""Build and score a coherent two-step semantic medical question tree."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter


INTERFACE_VERSION = "voi-medical-future-tree-mechanics-1"
MODEL_ID = "openai/gpt-5.4"
SOURCE_SHA256 = "e851864a9cb53c36304245bc3213a8a894cf7b86f8945d60978923e1f1ef0169"
SOURCE_ROWS = 499
ROOT_COUNT = 4
FOLLOWUPS_PER_BRANCH = 2
OUTCOMES = ("Yes", "No", "Maybe")
EXPECTED_REQUESTS = 18
PROJECTED_COST_USD = 0.18
MAX_COST_USD = 0.35
MIN_DEPTH_TWO_GAIN_NATS = 0.03
MIN_SCORE_RANGE_NATS = 0.05

DIAGNOSES = (
    "Enteritis",
    "Gastritis",
    "Gastroenteritis",
    "Esophagitis",
    "Cholecystitis",
    "Appendicitis",
    "Pancreatitis",
    "Gastric ulcer",
    "Constipation",
    "Cold",
    "Irritable bowel syndrome",
    "Diarrhea",
    "Allergic rhinitis",
    "Upper respiratory tract infection",
    "Pneumonia",
)


class MechanicsExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_empirical_prior(path: Path) -> dict[str, float]:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("MedDG source hash mismatch")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != SOURCE_ROWS:
        raise ValueError("MedDG source row count mismatch")
    counts = {diagnosis: 0 for diagnosis in DIAGNOSES}
    for row in rows:
        if not isinstance(row, dict) or row.get("target") not in counts:
            raise ValueError("MedDG contains an unexpected diagnosis")
        counts[str(row["target"])] += 1
    if any(value <= 0 for value in counts.values()):
        raise ValueError("every frozen diagnosis must have positive prior mass")
    total = sum(counts.values())
    return {diagnosis: counts[diagnosis] / total for diagnosis in DIAGNOSES}


def parse_questions(text: str, expected: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != expected:
        raise ValueError(f"expected exactly {expected} question lines")
    questions = []
    for index, line in enumerate(lines, start=1):
        prefix = f"Q{index}|"
        if not line.startswith(prefix):
            raise ValueError("question line has the wrong prefix")
        question = " ".join(line[len(prefix) :].split())
        if not question or not question.endswith("?"):
            raise ValueError("question must be nonempty and end with '?'")
        if "|" in question:
            raise ValueError("question contains the field delimiter")
        questions.append(question)
    if len({normalize_question(value) for value in questions}) != expected:
        raise ValueError("questions must be distinct after normalization")
    return questions


def parse_answer_matrix(
    text: str,
    questions: Sequence[str],
) -> dict[str, dict[str, str]]:
    expected = len(questions) * len(DIAGNOSES)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != expected:
        raise ValueError(f"expected exactly {expected} answer-map lines")
    matrix = {question: {} for question in questions}
    for line in lines:
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 3:
            raise ValueError("answer-map line must have three fields")
        qid, diagnosis, answer = fields
        match = re.fullmatch(r"Q([1-9][0-9]*)", qid)
        if match is None:
            raise ValueError("answer-map question ID is invalid")
        qindex = int(match.group(1)) - 1
        if not 0 <= qindex < len(questions):
            raise ValueError("answer-map question ID is out of range")
        if diagnosis not in DIAGNOSES:
            raise ValueError("answer-map diagnosis is unexpected")
        if answer not in OUTCOMES:
            raise ValueError("answer-map outcome is invalid")
        question = questions[qindex]
        if diagnosis in matrix[question]:
            raise ValueError("answer-map cell is duplicated")
        matrix[question][diagnosis] = answer
    if any(set(values) != set(DIAGNOSES) for values in matrix.values()):
        raise ValueError("answer-map matrix is incomplete")
    return matrix


def entropy(belief: Mapping[str, float]) -> float:
    return -sum(value * math.log(value) for value in belief.values() if value > 0.0)


def branch_beliefs(
    belief: Mapping[str, float],
    answer_map: Mapping[str, str],
) -> dict[str, tuple[float, dict[str, float]]]:
    result = {}
    for outcome in OUTCOMES:
        mass = sum(
            probability
            for diagnosis, probability in belief.items()
            if answer_map[diagnosis] == outcome
        )
        posterior = (
            {
                diagnosis: probability / mass
                for diagnosis, probability in belief.items()
                if answer_map[diagnosis] == outcome
            }
            if mass > 0.0
            else {}
        )
        result[outcome] = (mass, posterior)
    return result


def expected_entropy_after_question(
    belief: Mapping[str, float],
    answer_map: Mapping[str, str],
) -> float:
    return sum(
        mass * entropy(posterior)
        for mass, posterior in branch_beliefs(belief, answer_map).values()
    )


def count_path_dependent_roots(
    prior: Mapping[str, float],
    roots: Sequence[str],
    root_maps: Mapping[str, Mapping[str, str]],
    followups: Mapping[tuple[int, str], Sequence[str]],
) -> int:
    count = 0
    for root_index, root in enumerate(roots):
        branches = branch_beliefs(prior, root_maps[root])
        realizable_sets = {
            tuple(
                normalize_question(question)
                for question in followups[(root_index, outcome)]
            )
            for outcome, (mass, _posterior) in branches.items()
            if mass > 0.0
        }
        count += len(realizable_sets) >= 2
    return count


def score_tree(
    prior: Mapping[str, float],
    roots: Sequence[str],
    root_maps: Mapping[str, Mapping[str, str]],
    followups: Mapping[tuple[int, str], Sequence[str]],
    followup_maps: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    initial_entropy = entropy(prior)
    rows = []
    for root_index, root in enumerate(roots):
        root_branches = branch_beliefs(prior, root_maps[root])
        expected_after_root = sum(
            mass * entropy(posterior)
            for mass, posterior in root_branches.values()
        )
        expected_after_two = 0.0
        branch_rows = {}
        for outcome, (mass, posterior) in root_branches.items():
            candidates = list(followups[(root_index, outcome)])
            candidate_entropies = [
                expected_entropy_after_question(posterior, followup_maps[question])
                if mass > 0.0
                else 0.0
                for question in candidates
            ]
            selected_index = min(
                range(len(candidates)),
                key=lambda index: (candidate_entropies[index], index),
            )
            expected_after_two += mass * candidate_entropies[selected_index]
            branch_rows[outcome] = {
                "probability": mass,
                "posterior": posterior,
                "followups": candidates,
                "expected_entropies": candidate_entropies,
                "selected_followup_index": selected_index,
                "selected_followup": candidates[selected_index],
            }
        rows.append(
            {
                "root_index": root_index,
                "root_question": root,
                "immediate_eig_nats": initial_entropy - expected_after_root,
                "depth_two_eig_nats": initial_entropy - expected_after_two,
                "expected_entropy_after_root": expected_after_root,
                "expected_entropy_after_two": expected_after_two,
                "branches": branch_rows,
            }
        )
    myopic_index = max(
        range(len(rows)),
        key=lambda index: (rows[index]["immediate_eig_nats"], -index),
    )
    full_index = max(
        range(len(rows)),
        key=lambda index: (rows[index]["depth_two_eig_nats"], -index),
    )
    return {
        "initial_entropy_nats": initial_entropy,
        "roots": rows,
        "myopic_root_index": myopic_index,
        "full_root_index": full_index,
        "full_minus_myopic_depth_two_eig_nats": (
            rows[full_index]["depth_two_eig_nats"]
            - rows[myopic_index]["depth_two_eig_nats"]
        ),
    }


def root_question_messages(prior: Mapping[str, float]) -> list[dict[str, str]]:
    support = "\n".join(
        f"- {diagnosis}: {prior[diagnosis]:.6f}" for diagnosis in DIAGNOSES
    )
    return [
        {
            "role": "system",
            "content": (
                "You design diagnostic yes/no questions for a Bayesian medical "
                "identification task. No patient is currently revealed. Propose "
                "four distinct, atomic, broadly discriminative questions. Prefer "
                "questions whose different answers naturally motivate different "
                "follow-up questions. Do not ask combined questions and do not name "
                "a diagnosis. Return exactly four lines and no other text: "
                "Q1|<question> through Q4|<question>."
            ),
        },
        {
            "role": "user",
            "content": "Candidate diagnosis prior:\n" + support,
        },
    ]


def answer_matrix_messages(questions: Sequence[str]) -> list[dict[str, str]]:
    qtext = "\n".join(
        f"Q{index}|{question}" for index, question in enumerate(questions, start=1)
    )
    diagnoses = "\n".join(DIAGNOSES)
    return [
        {
            "role": "system",
            "content": (
                "You provide a frozen semantic response model for diagnostic "
                "questions. For every question and diagnosis, label whether a "
                "typical patient with that diagnosis would answer Yes, No, or Maybe. "
                "Use Maybe only when the feature is genuinely variable or the "
                "question cannot be determined from the diagnosis. Return every "
                "matrix cell exactly once as Q<number>|<diagnosis>|<Yes/No/Maybe>, "
                "ordered by question and then by the supplied diagnosis order. "
                "Return no markdown or explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Questions:\n{qtext}\n\nDiagnoses:\n{diagnoses}",
        },
    ]


def followup_question_messages(
    *,
    root_question: str,
    root_outcome: str,
    posterior: Mapping[str, float],
) -> list[dict[str, str]]:
    support = (
        "\n".join(
            f"- {diagnosis}: {posterior[diagnosis]:.6f}"
            for diagnosis in DIAGNOSES
            if posterior.get(diagnosis, 0.0) > 0.0
        )
        or "- No positive-mass diagnosis in this counterfactual branch."
    )
    return [
        {
            "role": "system",
            "content": (
                "You design the second question in a diagnostic Bayesian decision "
                "tree. The complete hypothetical first question and answer are "
                "provided. Propose two distinct atomic yes/no follow-ups for this "
                "specific branch. They must not repeat or paraphrase the first "
                "question, combine symptoms, or name a diagnosis. Return exactly "
                "two lines and no other text: Q1|<question> and Q2|<question>."
            ),
        },
        {
            "role": "user",
            "content": (
                f"First question: {root_question}\n"
                f"Hypothetical answer: {root_outcome}\n"
                f"Branch posterior:\n{support}"
            ),
        },
    ]


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot["adapter_requests"]),
        "http_attempts": int(snapshot["http_attempts"]),
        "retry_count": int(snapshot["retry_count"]),
        "reasoning_tokens": int(snapshot["adapter_reasoning_tokens"]),
        "forced_exits": int(snapshot["forced_exits"]),
        "adapter_cost_usd": float(snapshot["adapter_cost_usd"]),
        "generator": snapshot,
    }


def _build_model(config: Config) -> Any:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("VoI medical mechanics config selects the wrong model")
    return build_model_adapter(spec, config)


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_mechanics(
    config: Config,
    *,
    data_path: Path,
    raw_path: Path,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    prior = load_empirical_prior(data_path)
    model = model_adapter if model_adapter is not None else _build_model(config)
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "source_sha256": SOURCE_SHA256,
    }
    try:
        root_response = model.chat_complete_messages_batched(
            [root_question_messages(prior)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=config.openrouter_max_output_tokens,
        )[0]
        raw["root_questions"] = root_response
        _checkpoint(raw_path, raw)
        roots = parse_questions(root_response, ROOT_COUNT)

        root_map_response = model.chat_complete_messages_batched(
            [answer_matrix_messages(roots)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=config.openrouter_max_output_tokens,
        )[0]
        raw["root_answer_matrix"] = root_map_response
        _checkpoint(raw_path, raw)
        root_maps = parse_answer_matrix(root_map_response, roots)

        branch_keys = [
            (root_index, outcome)
            for root_index in range(ROOT_COUNT)
            for outcome in OUTCOMES
        ]
        root_branch_states = {
            (root_index, outcome): branch_beliefs(
                prior, root_maps[roots[root_index]]
            )[outcome]
            for root_index, outcome in branch_keys
        }
        followup_responses = model.chat_complete_messages_batched(
            [
                followup_question_messages(
                    root_question=roots[root_index],
                    root_outcome=outcome,
                    posterior=root_branch_states[(root_index, outcome)][1],
                )
                for root_index, outcome in branch_keys
            ],
            temperature=0.0,
            block_size=len(branch_keys),
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["followup_questions"] = {
            f"root_{root_index + 1}_{outcome.lower()}": response
            for (root_index, outcome), response in zip(
                branch_keys, followup_responses, strict=True
            )
        }
        _checkpoint(raw_path, raw)
        followups = {
            key: parse_questions(response, FOLLOWUPS_PER_BRANCH)
            for key, response in zip(branch_keys, followup_responses, strict=True)
        }
        for (root_index, _outcome), questions in followups.items():
            root_normalized = normalize_question(roots[root_index])
            if any(normalize_question(question) == root_normalized for question in questions):
                raise ValueError("follow-up question repeats its root")

        root_followup_questions = [
            [
                question
                for outcome in OUTCOMES
                for question in followups[(root_index, outcome)]
            ]
            for root_index in range(ROOT_COUNT)
        ]
        for questions in root_followup_questions:
            if len({normalize_question(question) for question in questions}) != len(
                questions
            ):
                raise ValueError("follow-up questions must be distinct within a root")
        followup_map_responses = model.chat_complete_messages_batched(
            [
                answer_matrix_messages(questions)
                for questions in root_followup_questions
            ],
            temperature=0.0,
            block_size=ROOT_COUNT,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["followup_answer_matrices"] = followup_map_responses
        _checkpoint(raw_path, raw)
        followup_maps: dict[str, dict[str, str]] = {}
        for questions, response in zip(
            root_followup_questions, followup_map_responses, strict=True
        ):
            parsed = parse_answer_matrix(response, questions)
            for question, values in parsed.items():
                normalized = normalize_question(question)
                existing = next(
                    (
                        prior_question
                        for prior_question in followup_maps
                        if normalize_question(prior_question) == normalized
                    ),
                    None,
                )
                if existing is not None and followup_maps[existing] != values:
                    raise ValueError("duplicate follow-up has inconsistent answer maps")
                followup_maps[question] = values

        scores = score_tree(prior, roots, root_maps, followups, followup_maps)
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise MechanicsExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc

    positive_outcomes = [
        sum(
            mass > 0.0
            for mass, _posterior in branch_beliefs(prior, root_maps[root]).values()
        )
        for root in roots
    ]
    path_dependent_roots = count_path_dependent_roots(
        prior,
        roots,
        root_maps,
        followups,
    )
    immediate_scores = [row["immediate_eig_nats"] for row in scores["roots"]]
    depth_two_scores = [row["depth_two_eig_nats"] for row in scores["roots"]]
    generator = usage["generator"]
    gates = {
        "exact_source_rows_and_hash": True,
        "exact_18_requests": usage["physical_requests"] == EXPECTED_REQUESTS,
        "zero_transport_retries": (
            usage["http_attempts"] == EXPECTED_REQUESTS
            and usage["retry_count"] == 0
        ),
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "four_complete_root_maps": len(root_maps) == ROOT_COUNT,
        "all_roots_have_at_least_two_positive_outcomes": min(positive_outcomes) >= 2,
        "all_twelve_branch_question_sets_complete": len(followups) == 12,
        "all_roots_have_answer_conditioned_followups": path_dependent_roots == ROOT_COUNT,
        "immediate_score_range_at_least_0_05": (
            max(immediate_scores) - min(immediate_scores) >= MIN_SCORE_RANGE_NATS
        ),
        "depth_two_score_range_at_least_0_05": (
            max(depth_two_scores) - min(depth_two_scores) >= MIN_SCORE_RANGE_NATS
        ),
        "depth_two_changes_root": (
            scores["full_root_index"] != scores["myopic_root_index"]
        ),
        "depth_two_gain_at_least_0_03_nats": (
            scores["full_minus_myopic_depth_two_eig_nats"]
            >= MIN_DEPTH_TWO_GAIN_NATS
        ),
        "cost_at_most_0_35": usage["adapter_cost_usd"] <= MAX_COST_USD,
        "adapter_reports_nonreasoning_model": (
            generator.get("model") == MODEL_ID
            and generator.get("reasoning_enabled") is False
            and int(generator.get("reasoning_tokens", 0)) == 0
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "source_sha256": SOURCE_SHA256,
            "source_rows": SOURCE_ROWS,
            "diagnoses": list(DIAGNOSES),
            "root_count": ROOT_COUNT,
            "outcomes": list(OUTCOMES),
            "followups_per_branch": FOLLOWUPS_PER_BRANCH,
            "expected_requests": EXPECTED_REQUESTS,
            "hidden_patient_exposed": False,
            "reasoning_requested": False,
            "repairs_or_scientific_retries": 0,
        },
        "prior": prior,
        "root_questions": roots,
        "root_answer_maps": root_maps,
        "followup_questions": {
            f"root_{root_index + 1}_{outcome.lower()}": questions
            for (root_index, outcome), questions in followups.items()
        },
        "followup_answer_maps": followup_maps,
        "scores": scores,
        "metrics": {
            "positive_outcomes_per_root": positive_outcomes,
            "path_dependent_roots": path_dependent_roots,
            "immediate_score_range_nats": max(immediate_scores) - min(immediate_scores),
            "depth_two_score_range_nats": max(depth_two_scores) - min(depth_two_scores),
            "myopic_root_index": scores["myopic_root_index"],
            "full_root_index": scores["full_root_index"],
            "full_minus_myopic_depth_two_eig_nats": (
                scores["full_minus_myopic_depth_two_eig_nats"]
            ),
        },
        "gates": gates,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 12
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        payload = run_mechanics(config, data_path=args.data, raw_path=raw_path)
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, MechanicsExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        _checkpoint(args.output_dir / "MECHANICS_FAILURE.json", failure)
        raise
    output = args.output_dir / "MECHANICS.json"
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
