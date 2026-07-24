#!/usr/bin/env python3
"""Rescore frozen Detective trees with an answer-function-aligned likelihood."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.detective_cabed_depth_ranking import (
    BOOTSTRAP_SEED,
    ESTIMATOR_CONFIDENCE,
    RANKING_CASE_IDS,
    _argmax,
    _call_batch,
    _canonical_question,
    _target_from_question,
    _usage_snapshot,
    answer_prompt,
    bayes_update,
    Belief,
    DEFAULT_DATA_PATH,
    DeterministicMechanicsModel,
    immediate_eig,
    load_cases,
    parse_answer,
    spearman,
)


SOURCE_GATE_PATH = Path(
    "results/nonmyopic/detective_cabed_gemma_concise_ranking/"
    "detective-cabed-gemma-concise-ranking-20260724T192405Z/GATE.json"
)
SOURCE_GATE_SHA256 = (
    "561e80787e88bf19c8c340a62fe61436aeccea91199616ab958fef65bbb9aca9"
)
EXPECTED_UNIQUE_QUESTIONS = 241
EXPECTED_REQUESTS = 482
MIN_ROLE_DISTINCTION_FRACTION = 0.25


def load_source_gate(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_GATE_SHA256:
        raise ValueError(
            f"source gate hash mismatch: expected {SOURCE_GATE_SHA256}, got {digest}"
        )
    source = json.loads(payload)
    if source.get("status") != "gate_failed":
        raise ValueError("aligned diagnostic requires the frozen failed source gate")
    if tuple(source["protocol"]["case_ids"]) != RANKING_CASE_IDS:
        raise ValueError("source gate case IDs do not match frozen ranking cases")
    return source


def _question_cache(
    source: dict[str, Any],
) -> tuple[
    dict[tuple[int, str], str],
    dict[tuple[int, str], str],
]:
    questions: dict[tuple[int, str], str] = {}
    targets: dict[tuple[int, str], str] = {}
    for case_index, record in enumerate(source["records"]):
        hypotheses = record["hypotheses"]
        case_questions: list[str] = []
        for root in record["root_plans"]:
            case_questions.append(root["question"])
            for branch in root["branches"]:
                case_questions.extend(branch["questions"])
        for question in case_questions:
            canonical = _canonical_question(question)
            key = (case_index, canonical)
            target = _target_from_question(question, hypotheses)
            if key in questions and questions[key] != question:
                raise ValueError("canonical question collision has different text")
            questions[key] = question
            targets[key] = target
    if len(questions) != EXPECTED_UNIQUE_QUESTIONS:
        raise ValueError(
            f"expected {EXPECTED_UNIQUE_QUESTIONS} unique questions, got {len(questions)}"
        )
    return questions, targets


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def run_diagnostic(
    config: Config,
    *,
    model: Any,
    source: dict[str, Any],
    cases: Sequence[dict[str, Any]],
    raw_path: Path | None,
) -> dict[str, Any]:
    questions, targets = _question_cache(source)
    ordered_keys = sorted(questions)
    role_keys: list[tuple[int, str, str]] = []
    prompts: list[str] = []
    for key in ordered_keys:
        case_index, _canonical = key
        case = cases[RANKING_CASE_IDS[case_index]]
        target = targets[key]
        hypotheses = [suspect["name"] for suspect in case["suspects"]]
        innocent_assumption = next(
            hypothesis for hypothesis in hypotheses if hypothesis != target
        )
        role_keys.extend(
            [
                (case_index, key[1], "murderer"),
                (case_index, key[1], "innocent"),
            ]
        )
        prompts.extend(
            [
                answer_prompt(
                    case,
                    questions[key],
                    assumed_murderer=target,
                ),
                answer_prompt(
                    case,
                    questions[key],
                    assumed_murderer=innocent_assumption,
                ),
            ]
        )
    raw: dict[str, Any] = {
        "schema_version": 1,
        "source_gate_sha256": SOURCE_GATE_SHA256,
        "case_ids": list(RANKING_CASE_IDS),
    }
    responses = _call_batch(
        model,
        config,
        prompts,
        temperature=0.0,
        raw=raw,
        raw_key="counterfactual_role_answers",
        raw_path=raw_path,
    )
    labels = {
        key: parse_answer(response)
        for key, response in zip(role_keys, responses, strict=True)
    }

    def row_for(case_index: int, question: str) -> tuple[float, ...]:
        source_record = source["records"][case_index]
        canonical = _canonical_question(question)
        target = targets[(case_index, canonical)]
        murderer_label = labels[(case_index, canonical, "murderer")]
        innocent_label = labels[(case_index, canonical, "innocent")]
        return tuple(
            (
                ESTIMATOR_CONFIDENCE
                * (
                    1.0
                    if (
                        murderer_label
                        if hypothesis == target
                        else innocent_label
                    )
                    == "Yes"
                    else 0.0
                )
                + (1.0 - ESTIMATOR_CONFIDENCE) * 0.5
            )
            for hypothesis in source_record["hypotheses"]
        )

    def realized_answer(
        case_index: int,
        question: str,
        truth: str,
    ) -> str:
        canonical = _canonical_question(question)
        role = "murderer" if targets[(case_index, canonical)] == truth else "innocent"
        return labels[(case_index, canonical, role)]

    records: list[dict[str, Any]] = []
    for case_index, source_record in enumerate(source["records"]):
        belief = Belief.uniform(source_record["hypotheses"])
        truth = source_record["truth"]
        initial_truth_log_probability = belief.truth_log_probability(truth)
        rescored_roots: list[dict[str, Any]] = []
        for source_root in source_record["root_plans"]:
            root_question = source_root["question"]
            root_row = row_for(case_index, root_question)
            root_eig = immediate_eig(belief, root_row)
            branches: dict[str, dict[str, Any]] = {}
            continuation = 0.0
            for source_branch in source_root["branches"]:
                observation = source_branch["observation"]
                branch_belief, marginal = bayes_update(
                    belief,
                    root_row,
                    observation,
                )
                followup_rows = [
                    row_for(case_index, question)
                    for question in source_branch["questions"]
                ]
                followup_eigs = [
                    immediate_eig(branch_belief, row)
                    for row in followup_rows
                ]
                selected_index = _argmax(followup_eigs)
                continuation += marginal * followup_eigs[selected_index]
                branches[observation] = {
                    "probability": marginal,
                    "questions": source_branch["questions"],
                    "likelihoods": followup_rows,
                    "eig_scores": followup_eigs,
                    "selected_index": selected_index,
                }
            depth_two_score = root_eig + continuation
            root_observation = realized_answer(
                case_index,
                root_question,
                truth,
            )
            after_root, _ = bayes_update(
                belief,
                root_row,
                root_observation,
            )
            branch = branches[root_observation]
            followup_index = branch["selected_index"]
            followup_question = branch["questions"][followup_index]
            followup_observation = realized_answer(
                case_index,
                followup_question,
                truth,
            )
            after_followup, _ = bayes_update(
                after_root,
                branch["likelihoods"][followup_index],
                followup_observation,
            )
            rescored_roots.append(
                {
                    "question": root_question,
                    "immediate_eig": root_eig,
                    "depth_two_score": depth_two_score,
                    "root_answer": root_observation,
                    "followup_question": followup_question,
                    "followup_answer": followup_observation,
                    "truth_log_probability_gain": (
                        after_followup.truth_log_probability(truth)
                        - initial_truth_log_probability
                    ),
                    "final_entropy": after_followup.entropy(),
                }
            )
        realized = [
            root["truth_log_probability_gain"] for root in rescored_roots
        ]
        d1_index = _argmax(
            [root["immediate_eig"] for root in rescored_roots]
        )
        d2_index = _argmax(
            [root["depth_two_score"] for root in rescored_roots]
        )
        records.append(
            {
                "case_id": source_record["case_id"],
                "truth": truth,
                "hypotheses": source_record["hypotheses"],
                "root_plans": rescored_roots,
                "depth_one_selected_index": d1_index,
                "depth_two_selected_index": d2_index,
                "random_selected_index": source_record["random_selected_index"],
                "depth_one_truth_gain_spearman": spearman(
                    [root["immediate_eig"] for root in rescored_roots],
                    realized,
                ),
                "depth_two_truth_gain_spearman": spearman(
                    [root["depth_two_score"] for root in rescored_roots],
                    realized,
                ),
            }
        )

    role_distinct_count = sum(
        labels[(case_index, canonical, "murderer")]
        != labels[(case_index, canonical, "innocent")]
        for case_index, canonical in ordered_keys
    )
    role_distinction_fraction = role_distinct_count / len(ordered_keys)
    usage = _usage_snapshot(model)
    summary = summarize(
        records,
        role_distinct_count=role_distinct_count,
        role_distinction_fraction=role_distinction_fraction,
        usage=usage,
    )
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "source_gate_sha256": SOURCE_GATE_SHA256,
            "case_ids": list(RANKING_CASE_IDS),
            "unique_questions": len(ordered_keys),
            "requests_per_question": 2,
            "estimator_confidence": ESTIMATOR_CONFIDENCE,
            "temperature": 0.0,
            "source_questions_reused": True,
            "source_answers_reused": False,
            "fresh_cases_used": False,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def summarize(
    records: Sequence[dict[str, Any]],
    *,
    role_distinct_count: int,
    role_distinction_fraction: float,
    usage: dict[str, Any],
) -> dict[str, Any]:
    paired_correlations = [
        (
            float(record["depth_one_truth_gain_spearman"]),
            float(record["depth_two_truth_gain_spearman"]),
        )
        for record in records
        if record["depth_one_truth_gain_spearman"] is not None
        and record["depth_two_truth_gain_spearman"] is not None
    ]
    d1_correlations = [first for first, _second in paired_correlations]
    d2_correlations = [second for _first, second in paired_correlations]
    differences_vs_one = []
    differences_vs_random = []
    distinct_roots = 0
    wins_vs_one = 0
    for record in records:
        roots = record["root_plans"]
        d1 = roots[record["depth_one_selected_index"]][
            "truth_log_probability_gain"
        ]
        d2 = roots[record["depth_two_selected_index"]][
            "truth_log_probability_gain"
        ]
        random_gain = roots[record["random_selected_index"]][
            "truth_log_probability_gain"
        ]
        differences_vs_one.append(d2 - d1)
        differences_vs_random.append(d2 - random_gain)
        distinct_roots += int(
            record["depth_one_selected_index"]
            != record["depth_two_selected_index"]
        )
        wins_vs_one += int(d2 > d1)
    mean_d1_correlation = _mean(d1_correlations)
    mean_d2_correlation = _mean(d2_correlations)
    correlation_advantage = _mean(
        [second - first for first, second in paired_correlations]
    )
    mean_vs_one = _mean(differences_vs_one)
    mean_vs_random = _mean(differences_vs_random)
    gates = {
        "all_twelve_cases_complete": len(records) == len(RANKING_CASE_IDS),
        "exact_request_count": int(usage.get("requests", -1)) == EXPECTED_REQUESTS,
        "zero_reasoning_tokens": int(usage.get("reasoning_tokens", -1)) == 0,
        "zero_forced_exits": int(usage.get("forced_exits", -1)) == 0,
        "role_distinction_at_least_quarter": (
            role_distinction_fraction >= MIN_ROLE_DISTINCTION_FRACTION
        ),
        "at_least_nine_rankable_cases": len(paired_correlations) >= 9,
        "depth_two_distinct_at_least_four": distinct_roots >= 4,
        "depth_two_mean_spearman_at_least_point_two": (
            math.isfinite(mean_d2_correlation)
            and mean_d2_correlation >= 0.20
        ),
        "depth_two_spearman_advantage_at_least_point_one": (
            math.isfinite(correlation_advantage)
            and correlation_advantage >= 0.10
        ),
        "depth_two_mean_gain_over_one_positive": mean_vs_one > 0.0,
        "depth_two_wins_at_least_six": wins_vs_one >= 6,
        "depth_two_mean_gain_over_random_positive": mean_vs_random > 0.0,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "num_cases": len(records),
        "unique_questions": EXPECTED_UNIQUE_QUESTIONS,
        "role_distinct_question_count": role_distinct_count,
        "role_distinction_fraction": role_distinction_fraction,
        "rankable_case_count": len(paired_correlations),
        "depth_two_distinct_root_count": distinct_roots,
        "mean_depth_one_truth_gain_spearman": mean_d1_correlation,
        "mean_depth_two_truth_gain_spearman": mean_d2_correlation,
        "mean_depth_two_spearman_advantage": correlation_advantage,
        "mean_truth_gain_depth_two_vs_one": mean_vs_one,
        "depth_two_win_count_vs_one": wins_vs_one,
        "mean_truth_gain_depth_two_vs_random": mean_vs_random,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-gate", type=Path, default=SOURCE_GATE_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.10
    config.openrouter_run_budget_usd = 0.50
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = args.output_dir / "RAW_RESPONSES.json"
    try:
        source = load_source_gate(args.source_gate)
        cases = load_cases(args.data_path)
        if args.dry_run:
            model = DeterministicMechanicsModel()
        else:
            if len(config.model_pairs) != 1:
                raise ValueError("aligned diagnostic requires one model pair")
            pair = config.model_pairs[0]
            if pair.questioner != pair.answerer:
                raise ValueError("aligned diagnostic requires one shared model")
            model = build_model_adapter(pair.questioner, config=config)
        payload = run_diagnostic(
            config,
            model=model,
            source=source,
            cases=cases,
            raw_path=raw_path,
        )
    except Exception as exc:
        (args.output_dir / "DIAGNOSTIC_FAILURE.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "failed_closed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / "DIAGNOSTIC.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
