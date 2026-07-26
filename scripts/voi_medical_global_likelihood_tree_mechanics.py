#!/usr/bin/env python3
"""Score a medical question tree with one global likelihood map per action."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.voi_medical_future_tree_mechanics import (
    DIAGNOSES,
    FOLLOWUPS_PER_BRANCH,
    MIN_DEPTH_TWO_GAIN_NATS,
    MIN_SCORE_RANGE_NATS,
    MODEL_ID,
    OUTCOMES,
    ROOT_COUNT,
    SOURCE_ROWS,
    SOURCE_SHA256,
    MechanicsExecutionError,
    _checkpoint,
    _usage_snapshot,
    answer_matrix_messages,
    branch_beliefs,
    count_path_dependent_roots,
    followup_question_messages,
    load_empirical_prior,
    normalize_question,
    parse_answer_matrix,
    parse_questions,
    root_question_messages,
    score_tree,
    sha256_file,
)


INTERFACE_VERSION = "voi-medical-global-likelihood-tree-mechanics-2"
PROJECTED_COST_USD = 0.10
MAX_COST_USD = 0.25
BASE_REQUESTS = 1 + ROOT_COUNT + ROOT_COUNT * len(OUTCOMES)


def unique_new_questions(
    roots: Sequence[str],
    followups: Mapping[tuple[int, str], Sequence[str]],
) -> list[str]:
    seen = {normalize_question(question) for question in roots}
    result = []
    for root_index in range(len(roots)):
        for outcome in OUTCOMES:
            for question in followups[(root_index, outcome)]:
                normalized = normalize_question(question)
                if normalized not in seen:
                    seen.add(normalized)
                    result.append(question)
    return result


def alias_global_maps(
    roots: Sequence[str],
    root_maps: Mapping[str, Mapping[str, str]],
    new_questions: Sequence[str],
    new_maps: Mapping[str, Mapping[str, str]],
    followups: Mapping[tuple[int, str], Sequence[str]],
) -> dict[str, dict[str, str]]:
    registry = {
        normalize_question(question): dict(root_maps[question]) for question in roots
    }
    registry.update(
        {
            normalize_question(question): dict(new_maps[question])
            for question in new_questions
        }
    )
    aliases = {}
    for questions in followups.values():
        for question in questions:
            aliases[question] = registry[normalize_question(question)]
    return aliases


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
        raise ValueError("VoI medical V2 config selects the wrong model")
    return build_model_adapter(spec, config)


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

        root_map_responses = model.chat_complete_messages_batched(
            [answer_matrix_messages([root]) for root in roots],
            temperature=0.0,
            block_size=ROOT_COUNT,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["root_answer_maps"] = root_map_responses
        _checkpoint(raw_path, raw)
        root_maps = {
            root: parse_answer_matrix(response, [root])[root]
            for root, response in zip(roots, root_map_responses, strict=True)
        }

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
            if any(
                normalize_question(question) == root_normalized
                for question in questions
            ):
                raise ValueError("follow-up question repeats its branch root")

        new_questions = unique_new_questions(roots, followups)
        new_map_responses = model.chat_complete_messages_batched(
            [answer_matrix_messages([question]) for question in new_questions],
            temperature=0.0,
            block_size=min(len(new_questions), config.openrouter_concurrency),
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["new_question_order"] = new_questions
        raw["new_question_answer_maps"] = new_map_responses
        _checkpoint(raw_path, raw)
        new_maps = {
            question: parse_answer_matrix(response, [question])[question]
            for question, response in zip(
                new_questions, new_map_responses, strict=True
            )
        }
        followup_maps = alias_global_maps(
            roots,
            root_maps,
            new_questions,
            new_maps,
            followups,
        )
        scores = score_tree(prior, roots, root_maps, followups, followup_maps)
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise MechanicsExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc

    expected_requests = BASE_REQUESTS + len(new_questions)
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
        "exact_dynamic_request_count": (
            usage["physical_requests"] == expected_requests
        ),
        "zero_transport_retries": (
            usage["http_attempts"] == expected_requests
            and usage["retry_count"] == 0
        ),
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "zero_forced_final_requests": (
            int(generator.get("forced_final_requests", 0)) == 0
        ),
        "four_complete_root_maps": len(root_maps) == ROOT_COUNT,
        "all_roots_have_at_least_two_positive_outcomes": min(positive_outcomes) >= 2,
        "all_twelve_branch_question_sets_complete": len(followups) == 12,
        "all_roots_have_answer_conditioned_followups": (
            path_dependent_roots == ROOT_COUNT
        ),
        "one_map_per_unique_question": (
            len(new_maps) == len(new_questions)
            and len(
                {
                    normalize_question(question)
                    for question in [*roots, *new_questions]
                }
            )
            == ROOT_COUNT + len(new_questions)
        ),
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
        "cost_at_most_0_25": usage["adapter_cost_usd"] <= MAX_COST_USD,
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
            "base_requests": BASE_REQUESTS,
            "unique_new_followup_questions": len(new_questions),
            "expected_requests": expected_requests,
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
        "unique_new_followup_questions": new_questions,
        "global_followup_answer_maps": followup_maps,
        "scores": scores,
        "metrics": {
            "positive_outcomes_per_root": positive_outcomes,
            "path_dependent_roots": path_dependent_roots,
            "unique_new_followup_questions": len(new_questions),
            "reused_question_occurrences": (
                ROOT_COUNT + ROOT_COUNT * len(OUTCOMES) * FOLLOWUPS_PER_BRANCH
                - ROOT_COUNT
                - len(new_questions)
            ),
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
