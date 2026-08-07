#!/usr/bin/env python3
"""Producer core for the sealed RegretBench SMC depth-two policy.

This module has no CLI and never constructs a network adapter. A future dated
executor may pass an already budget-authorized adapter only after the frozen
predecessor chain succeeds.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from scripts import regretbench_deepseek_dynamic_depth2_policy as scorer
from scripts import regretbench_deepseek_smc_dynamic_depth2_policy as core
from scripts import regretbench_deepseek_support_recovery as primary
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-smc-dynamic-depth2-experiment-1"
BRANCH_DRAWS = 2
SMOKE_ANNOTATION_SEED_START = 202608290000
SMOKE_BRANCH_SEED_START = 202608291000
DEVELOPMENT_ANNOTATION_SEED_START = 202608300000
DEVELOPMENT_BRANCH_SEED_START = 202608310000
TRUTH_SEED_START = 202608320000
ACTUAL_FIRST_SEED_START = 202608340000
ACTUAL_FINAL_SEED_START = 202608350000
BOOTSTRAP_SEED = 202608360000
NAIVE_FIRST_SEED_START = 202608370000
NAIVE_SECOND_SEED_START = 202608380000
NAIVE_FIRST_SUPPORT_SEED_START = 202608390000
NAIVE_FINAL_SUPPORT_SEED_START = 202608400000
NAIVE_SMOKE_FIRST_SEED_START = 202608292000
NAIVE_SMOKE_SECOND_SEED_START = 202608293000
NAIVE_SMOKE_REDRAW_SEED_START = 202608294000
PLANNING_REQUESTS = 8_256
SMOKE_BUDGET_USD = 0.20
DEVELOPMENT_BUDGET_USD = 3.50


class StructuredAdapter(Protocol):
    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages: Sequence[list[dict[str, str]]],
        seeds: Sequence[int],
        *,
        temperature: float,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


def branch_seed(task_index: int, particle_index: int, draw: int) -> int:
    if task_index not in range(64):
        raise ValueError("task index is outside the development cohort")
    if particle_index not in range(core.PARTICLES):
        raise ValueError("particle index is outside the parent population")
    if draw not in range(BRANCH_DRAWS):
        raise ValueError("draw index is outside the frozen schedule")
    return DEVELOPMENT_BRANCH_SEED_START + task_index * 16 + particle_index * 2 + draw


def actual_first_seed(task_index: int) -> int:
    if task_index not in range(64):
        raise ValueError("task index is outside the development cohort")
    return ACTUAL_FIRST_SEED_START + task_index


def actual_final_seed(task_index: int) -> int:
    if task_index not in range(64):
        raise ValueError("task index is outside the development cohort")
    return ACTUAL_FINAL_SEED_START + task_index


def _call(
    adapter: StructuredAdapter,
    messages: Sequence[list[dict[str, str]]],
    seeds: Sequence[int],
    response_format: dict[str, Any],
) -> list[str]:
    if len(messages) != len(seeds):
        raise ValueError("message and seed counts differ")
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        messages,
        seeds,
        temperature=core.TEMPERATURE,
        response_format=response_format,
        max_new_tokens=core.MAX_TOKENS,
    )
    if len(responses) != len(messages):
        raise ValueError("adapter returned the wrong response count")
    return list(responses)


def _call_naive(
    adapter: StructuredAdapter,
    messages: Sequence[list[dict[str, str]]],
    seeds: Sequence[int],
) -> list[str]:
    if len(messages) != len(seeds):
        raise ValueError("naive message and seed counts differ")
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        messages,
        seeds,
        temperature=0.7,
        response_format=scorer.naive_response_format(),
        max_new_tokens=scorer.NAIVE_MAX_TOKENS,
    )
    if len(responses) != len(messages):
        raise ValueError("naive adapter returned the wrong response count")
    return list(responses)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_banked_parent_contexts(
    *, stage: str, primary_dir: Path
) -> list[dict[str, Any]]:
    cigs = primary.load_stage_cigs(stage)
    raw_path = primary_dir / "private" / "RAW_RESPONSES.json"
    controls_path = primary_dir / "private" / "CONTROLS.json"
    raw = _load_json(raw_path)
    controls = _load_json(controls_path)
    roots = raw.get("root") or []
    if len(roots) != len(cigs):
        raise ValueError("banked parent response count changed")
    by_id = {row["task_id"]: row for row in controls.get("roots", [])}
    contexts = []
    for index, (cig, raw_root) in enumerate(zip(cigs, roots, strict=True)):
        control = by_id.get(cig.cig_id)
        if control is None:
            raise ValueError(f"banked parent control is missing: {cig.cig_id}")
        parsed_root = primary.parse_support(raw_root)
        if parsed_root["questions"][0] != control["question"]:
            raise ValueError("banked selected root question changed")
        parent = core.smc_support.parse_parent_population(raw_root)
        contexts.append(
            {
                "task_index": index,
                "cig": cig,
                "raw_parent": raw_root,
                "parent": parent,
                "questions": parsed_root["questions"],
                "control": control,
                "raw_parent_sha256": hashlib.sha256(raw_root.encode()).hexdigest(),
                "primary_raw_sha256": primary.sha256_file(raw_path),
                "primary_controls_sha256": primary.sha256_file(controls_path),
            }
        )
    return contexts


def _annotation_block(
    *,
    adapter: StructuredAdapter,
    contexts: Sequence[Mapping[str, Any]],
    seed_start: int,
) -> tuple[list[dict[str, Any]], list[str], list[int], list[dict[str, Any]]]:
    messages = []
    privacy = []
    for context in contexts:
        request, audit = core.annotation_messages_for(
            context["cig"], context["parent"], context["questions"]
        )
        messages.append(request)
        privacy.append(audit)
    seeds = [seed_start + index for index in range(len(contexts))]
    raw = _call(adapter, messages, seeds, core.annotation_response_format())
    supports = [
        core.parse_parent_annotation(
            response, context["parent"], context["questions"]
        )
        for response, context in zip(raw, contexts, strict=True)
    ]
    return supports, raw, seeds, privacy


def _informative_questions(support: Mapping[str, Any]) -> int:
    return sum(
        scorer.question_eig(support, index) > 1e-12
        for index in range(core.QUESTIONS)
    )


def run_smoke(
    *,
    output_dir: Path,
    adapter: StructuredAdapter,
    primary_smoke_dir: Path,
    smc_result_path: Path,
    smc_verification_path: Path,
    smc_daily_result_path: Path,
    smc_ledger_path: Path,
    daily_budget_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    core.validate_protocol_binding()
    predecessor = core.validate_smc_support_predecessor(
        result_path=smc_result_path,
        verification_path=smc_verification_path,
        daily_result_path=smc_daily_result_path,
        ledger_path=smc_ledger_path,
    )
    contexts = load_banked_parent_contexts(
        stage="smoke", primary_dir=primary_smoke_dir
    )
    initial, raw_initial, initial_seeds, privacy = _annotation_block(
        adapter=adapter,
        contexts=contexts,
        seed_start=SMOKE_ANNOTATION_SEED_START,
    )
    branch_messages = []
    branch_seeds = []
    first_mappings = []
    first_reply_matches = []
    truths = []
    for index, (context, support) in enumerate(
        zip(contexts[:3], initial[:3], strict=True)
    ):
        cig = context["cig"]
        truth_index, truth = primary.sample_truth(
            cig, primary.STAGES["smoke"]["truth_seed_start"] + index
        )
        if truth_index != context["control"]["truth_index"]:
            raise ValueError("banked smoke truth control changed")
        truths.append(truth)
        question = support["questions"][0]
        mapping = primary.map_and_answer(cig, question, truth)
        if mapping != context["control"]["mapping"]:
            raise ValueError("banked smoke environment mapping changed")
        aliases = str((truth.slots or {})["answer_aliases"])
        first_mappings.append(mapping)
        first_reply_matches.append(
            bool(
                scorer.truth_consistent_reply_indexes(
                    support, 0, mapping["answer"], aliases
                )
            )
            and mapping["supported"]
        )
        dialogue = [
            {"role": "assistant", "content": question},
            {"role": "user", "content": mapping["answer"]},
        ]
        conditioned, conditioned_audit = core.transition_messages_for(
            cig, dialogue, support
        )
        blind, blind_audit = core.transition_messages_for(cig, [], support)
        seed = SMOKE_BRANCH_SEED_START + index
        branch_messages.extend([conditioned, blind])
        branch_seeds.extend([seed, seed])
        privacy.extend([conditioned_audit, blind_audit])
    raw_branches = _call(
        adapter,
        branch_messages,
        branch_seeds,
        core.transition_response_format(),
    )
    branches = [
        core.parse_enriched_transition(raw_branches[2 * index + arm], initial[index])
        for index in range(3)
        for arm in range(2)
    ]
    second_mappings = []
    second_reply_matches = []
    for context, truth, conditioned in zip(
        contexts[:3], truths, branches[::2], strict=True
    ):
        second = scorer.select_question(conditioned)
        mapping = primary.map_and_answer(
            context["cig"], conditioned["questions"][second], truth
        )
        second_mappings.append(mapping)
        second_reply_matches.append(
            bool(scorer.matching_reply_indexes(conditioned, second, mapping["answer"]))
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    checkpoint(
        private / "RAW_RESPONSES.json",
        {
            "annotation_seeds": initial_seeds,
            "annotations": raw_initial,
            "branch_seeds": branch_seeds,
            "branches": raw_branches,
        },
    )
    usage = summarize_usage(adapter.usage_snapshot())
    gates = {
        "exact_ten_responses": len(initial) + len(branches) == 10,
        "exact_ten_requests": usage["adapter_requests"] == 10,
        "exact_ten_http_attempts": usage["http_attempts"] == 10,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_annotations_exact_without_regeneration": all(
            support["diagnostic"]["parent_index_permutation_exact"] is True
            and support["diagnostic"]["initial_hypotheses_regenerated"] is False
            and support["diagnostic"]["initial_questions_regenerated"] is False
            for support in initial
        ),
        "all_transitions_have_exact_lineage_and_retention": all(
            support["diagnostic"]["parent_index_permutation_exact"] is True
            and core.MIN_RETAINED
            <= support["diagnostic"]["retained_count"]
            <= core.MAX_RETAINED
            for support in branches
        ),
        "every_initial_has_two_informative_roots": all(
            _informative_questions(support) >= 2 for support in initial
        ),
        "every_branch_has_an_informative_followup": all(
            _informative_questions(support) >= 1 for support in branches
        ),
        "all_three_first_questions_supported": all(
            mapping["supported"] for mapping in first_mappings
        ),
        "all_three_exact_first_replies_match_truth_consistent_likelihoods": all(
            first_reply_matches
        ),
        "all_three_second_questions_supported": all(
            mapping["supported"] for mapping in second_mappings
        ),
        "all_three_second_actions_are_novel": all(
            scorer.distinct_supported_actions(first, second)
            for first, second in zip(first_mappings, second_mappings, strict=True)
        ),
        "all_three_exact_second_replies_match_generated_likelihoods": all(
            second_reply_matches
        ),
        "all_privacy_and_parent_provenance_audits_pass": len(privacy) == 10
        and all(item["passed"] for item in privacy),
        "within_smoke_budget": float(usage["run_cost_usd"])
        <= SMOKE_BUDGET_USD + 1e-12,
    }
    gates["all_pass"] = all(gates.values())
    status = "passed" if gates["all_pass"] else "mechanics_failed"
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "authorizes": "smc_policy_development_only" if status == "passed" else "nothing",
        "protocol": {
            "stage": "smoke",
            "model": core.MODEL_ID,
            "reasoning": "disabled_excluded",
            "expected_requests": 10,
            "protocol_sha256": core.PROTOCOL_SHA256,
            "predecessor": predecessor,
            "efficacy_used_for_authorization": False,
            "policy_endpoint_opened": False,
            "confirmation_opened": False,
        },
        "daily_budget_status": dict(daily_budget_status or {}),
        "usage": usage,
        "gates": gates,
        "supports": {
            "annotations": [row["diagnostic"] for row in initial],
            "transitions": [row["diagnostic"] for row in branches],
        },
    }
    checkpoint(output_dir / "RESULT.json", result)
    checkpoint(private / "PRIVACY.json", {"audits": privacy})
    return result


def validate_policy_smoke(path: Path) -> dict[str, Any]:
    result = _load_json(path)
    protocol = result.get("protocol") or {}
    if (
        result.get("interface_version") != INTERFACE_VERSION
        or result.get("status") != "passed"
        or result.get("authorizes") != "smc_policy_development_only"
        or protocol.get("stage") != "smoke"
        or protocol.get("model") != core.MODEL_ID
        or protocol.get("expected_requests") != 10
        or protocol.get("protocol_sha256") != core.PROTOCOL_SHA256
        or protocol.get("efficacy_used_for_authorization") is not False
        or protocol.get("policy_endpoint_opened") is not False
        or protocol.get("confirmation_opened") is not False
        or result.get("gates", {}).get("all_pass") is not True
    ):
        raise ValueError("SMC enriched smoke does not authorize development")
    return {"path": str(path), "sha256": primary.sha256_file(path)}


def validate_naive_smoke(path: Path) -> dict[str, Any]:
    result = _load_json(path)
    protocol = result.get("protocol") or {}
    if (
        result.get("interface_version") != INTERFACE_VERSION
        or result.get("status") != "passed"
        or result.get("authorizes") != "smc_policy_development_only"
        or protocol.get("stage") != "naive_smoke"
        or protocol.get("model") != scorer.NAIVE_MODEL_ID
        or protocol.get("reasoning_effort") != scorer.NAIVE_REASONING_EFFORT
        or protocol.get("expected_requests") != 10
        or protocol.get("protocol_sha256") != core.PROTOCOL_SHA256
        or protocol.get("efficacy_used_for_authorization") is not False
        or protocol.get("policy_endpoint_opened") is not False
        or protocol.get("confirmation_opened") is not False
        or result.get("gates", {}).get("all_pass") is not True
    ):
        raise ValueError("SMC naive-thinking smoke does not authorize baseline")
    return {"path": str(path), "sha256": primary.sha256_file(path)}


def _naive_usage(adapter: StructuredAdapter) -> dict[str, Any]:
    snapshot = adapter.usage_snapshot()
    usage = summarize_usage(snapshot)
    usage["forced_final_requests"] = int(
        snapshot.get("forced_final_requests", 0)
    )
    usage["forced_final_successes"] = int(
        snapshot.get("forced_final_successes", 0)
    )
    return usage


def run_naive_smoke(
    *,
    output_dir: Path,
    adapter: StructuredAdapter,
    policy_smoke_result: Path,
    daily_budget_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    core.validate_protocol_binding()
    enriched_smoke = validate_policy_smoke(policy_smoke_result)
    cigs = primary.load_stage_cigs("smoke")
    privacy = []
    first_messages = []
    for cig in cigs:
        request, audit = scorer.naive_messages_for(cig, [])
        first_messages.append(request)
        privacy.append(audit)
    first_seeds = [NAIVE_SMOKE_FIRST_SEED_START + index for index in range(4)]
    raw_first = _call_naive(adapter, first_messages, first_seeds)
    first_questions = [scorer.parse_naive_question(raw) for raw in raw_first]
    second_messages = []
    first_mappings = []
    truths = []
    for index, (cig, question) in enumerate(
        zip(cigs, first_questions, strict=True)
    ):
        _, truth = primary.sample_truth(
            cig, primary.STAGES["smoke"]["truth_seed_start"] + index
        )
        truths.append(truth)
        mapping = primary.map_and_answer(cig, question, truth)
        first_mappings.append(mapping)
        request, audit = scorer.naive_messages_for(
            cig,
            [
                {"role": "assistant", "content": question},
                {"role": "user", "content": mapping["answer"]},
            ],
        )
        second_messages.append(request)
        privacy.append(audit)
    second_seeds = [NAIVE_SMOKE_SECOND_SEED_START + index for index in range(4)]
    raw_second = _call_naive(adapter, second_messages, second_seeds)
    second_questions = [
        scorer.parse_naive_question(raw) for raw in raw_second
    ]
    second_mappings = [
        primary.map_and_answer(cig, question, truth)
        for cig, question, truth in zip(
            cigs, second_questions, truths, strict=True
        )
    ]
    redraw_messages = []
    for cig in cigs[:2]:
        request, audit = scorer.naive_messages_for(cig, [])
        redraw_messages.append(request)
        privacy.append(audit)
    redraw_seeds = [NAIVE_SMOKE_REDRAW_SEED_START + index for index in range(2)]
    raw_redraw = _call_naive(adapter, redraw_messages, redraw_seeds)
    redraw_questions = [
        scorer.parse_naive_question(raw) for raw in raw_redraw
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    checkpoint(
        private / "RAW_RESPONSES.json",
        {
            "first": raw_first,
            "second": raw_second,
            "redraw": raw_redraw,
            "first_seeds": first_seeds,
            "second_seeds": second_seeds,
            "redraw_seeds": redraw_seeds,
        },
    )
    usage = _naive_usage(adapter)
    gates = {
        "exact_ten_questions": len(first_questions)
        + len(second_questions)
        + len(redraw_questions)
        == 10,
        "exact_ten_requests": usage["adapter_requests"] == 10,
        "exact_ten_http_attempts": usage["http_attempts"] == 10,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "positive_reasoning_tokens": usage["adapter_reasoning_tokens"] > 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "zero_forced_final_requests": usage["forced_final_requests"] == 0,
        "all_eight_executed_questions_supported": all(
            row["supported"] for row in [*first_mappings, *second_mappings]
        ),
        "all_four_second_actions_are_novel": all(
            scorer.distinct_supported_actions(first, second)
            for first, second in zip(
                first_mappings, second_mappings, strict=True
            )
        ),
        "all_privacy_audits_pass": len(privacy) == 10
        and all(row["passed"] for row in privacy),
        "within_naive_smoke_budget": float(usage["run_cost_usd"])
        <= SMOKE_BUDGET_USD + 1e-12,
    }
    gates["all_pass"] = all(gates.values())
    status = "passed" if gates["all_pass"] else "mechanics_failed"
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "authorizes": "smc_policy_development_only" if status == "passed" else "nothing",
        "protocol": {
            "stage": "naive_smoke",
            "model": scorer.NAIVE_MODEL_ID,
            "reasoning_effort": scorer.NAIVE_REASONING_EFFORT,
            "expected_requests": 10,
            "protocol_sha256": core.PROTOCOL_SHA256,
            "enriched_smoke": enriched_smoke,
            "efficacy_used_for_authorization": False,
            "policy_endpoint_opened": False,
            "confirmation_opened": False,
        },
        "daily_budget_status": dict(daily_budget_status or {}),
        "usage": usage,
        "gates": gates,
        "question_sha256": [
            hashlib.sha256(question.encode()).hexdigest()
            for question in [
                *first_questions,
                *second_questions,
                *redraw_questions,
            ]
        ],
    }
    checkpoint(output_dir / "RESULT.json", result)
    checkpoint(private / "PRIVACY.json", {"audits": privacy})
    return result


def build_development_planning_tree(
    *,
    output_dir: Path,
    adapter: StructuredAdapter,
    primary_development_dir: Path,
) -> dict[str, Any]:
    """Build and score the frozen 8,256-call tree without realized truth access."""

    core.validate_protocol_binding()
    contexts = load_banked_parent_contexts(
        stage="development", primary_dir=primary_development_dir
    )
    if len(contexts) != 64:
        raise ValueError("development parent cohort changed")
    initial, raw_initial, initial_seeds, privacy = _annotation_block(
        adapter=adapter,
        contexts=contexts,
        seed_start=DEVELOPMENT_ANNOTATION_SEED_START,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    checkpoint(
        private / "RAW_INITIAL.json",
        {"seeds": initial_seeds, "responses": raw_initial},
    )

    messages = []
    seeds = []
    manifest = []
    for task_index, (context, support) in enumerate(
        zip(contexts, initial, strict=True)
    ):
        for root in range(core.QUESTIONS):
            question = support["questions"][root]
            for particle_index, particle in enumerate(support["hypotheses"]):
                dialogue = [
                    {"role": "assistant", "content": question},
                    {
                        "role": "user",
                        "content": particle["predicted_replies"][root],
                    },
                ]
                for draw in range(BRANCH_DRAWS):
                    seed = branch_seed(task_index, particle_index, draw)
                    conditioned, conditioned_audit = core.transition_messages_for(
                        context["cig"], dialogue, support
                    )
                    blind, blind_audit = core.transition_messages_for(
                        context["cig"], [], support
                    )
                    messages.extend([conditioned, blind])
                    seeds.extend([seed, seed])
                    privacy.extend([conditioned_audit, blind_audit])
                    manifest.append(
                        {
                            "task_index": task_index,
                            "root_index": root,
                            "hypothesis_index": particle_index,
                            "draw": draw,
                            "seed": seed,
                            "conditioned_dispatch_index": len(messages) - 2,
                            "blind_dispatch_index": len(messages) - 1,
                        }
                    )
    if len(messages) != PLANNING_REQUESTS - 64 or len(manifest) != 4_096:
        raise ValueError("development branch request schedule changed")
    raw_branches = _call(
        adapter, messages, seeds, core.transition_response_format()
    )
    checkpoint(
        private / "RAW_BRANCHES.json",
        {"manifest": manifest, "paired_seeds": seeds, "responses": raw_branches},
    )
    by_task: list[list[dict[str, Any]]] = [[] for _ in contexts]
    transitions = []
    for index, row in enumerate(manifest):
        parent = initial[row["task_index"]]
        conditioned = core.parse_enriched_transition(
            raw_branches[2 * index], parent
        )
        blind = core.parse_enriched_transition(raw_branches[2 * index + 1], parent)
        transitions.extend([conditioned, blind])
        by_task[row["task_index"]].append(
            {
                "task_index": row["task_index"],
                "root_index": row["root_index"],
                "hypothesis_index": row["hypothesis_index"],
                "draw": row["draw"],
                "seed": row["seed"],
                "conditioned": conditioned,
                "blind": blind,
            }
        )
    planning = []
    for task_index, support in enumerate(initial):
        conditioned = scorer.dynamic_root_risks(
            support, by_task[task_index], arm="conditioned"
        )
        by_draw = [
            scorer.dynamic_root_risks_for_draw(
                support, by_task[task_index], arm="conditioned", draw=draw
            )
            for draw in range(BRANCH_DRAWS)
        ]
        blind = scorer.dynamic_root_risks(
            support, by_task[task_index], arm="blind"
        )
        refresh = scorer.myopic_refresh_brier_root_risks(
            support, by_task[task_index]
        )
        myopic_brier = scorer.myopic_brier_root_risks(support)
        fixed = scorer.fixed_root_risks(support)
        selected = scorer.choose_roots(
            support,
            conditioned,
            blind,
            refresh,
            myopic_brier,
            fixed,
            random_seed=202608330000 + task_index,
        )
        renamed = {
            ("random" if name == "random" else f"smc_{name}"): root
            for name, root in selected.items()
        }
        planning.append(
            {
                "task_id": contexts[task_index]["cig"].cig_id,
                "conditioned": conditioned,
                "conditioned_by_draw": by_draw,
                "selection_stability": scorer.dynamic_selection_stability(
                    conditioned, by_draw
                ),
                "blind": blind,
                "myopic_refresh_brier": refresh,
                "myopic_brier": myopic_brier,
                "fixed": fixed,
                "root_eig": [
                    scorer.question_eig(support, root)
                    for root in range(core.QUESTIONS)
                ],
                "selected": renamed,
            }
        )
    usage = summarize_usage(adapter.usage_snapshot())
    exact_schedule = all(
        row["task_index"] == task
        and row["root_index"] == root
        and row["hypothesis_index"] == particle
        and row["draw"] == draw
        and row["seed"] == branch_seed(task, particle, draw)
        and row["conditioned_dispatch_index"] == 2 * index
        and row["blind_dispatch_index"] == 2 * index + 1
        for index, row in enumerate(manifest)
        for task, root, particle, draw in [
            (
                index // 64,
                (index % 64) // 16,
                (index % 16) // 2,
                index % 2,
            )
        ]
    )
    gates = {
        "exact_64_annotations": len(initial) == 64,
        "exact_8192_transitions": len(transitions) == 8_192,
        "exact_8256_requests": usage["adapter_requests"] == PLANNING_REQUESTS,
        "exact_8256_http_attempts": usage["http_attempts"] == PLANNING_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_annotations_exact_without_regeneration": all(
            row["diagnostic"]["parent_index_permutation_exact"] is True
            and row["diagnostic"]["initial_hypotheses_regenerated"] is False
            for row in initial
        ),
        "all_transitions_exact_lineage_and_retention": all(
            row["diagnostic"]["parent_index_permutation_exact"] is True
            and core.MIN_RETAINED
            <= row["diagnostic"]["retained_count"]
            <= core.MAX_RETAINED
            for row in transitions
        ),
        "every_initial_has_two_informative_roots": all(
            _informative_questions(row) >= 2 for row in initial
        ),
        "at_least_90pct_transitions_have_informative_followup": sum(
            _informative_questions(row) >= 1 for row in transitions
        )
        >= 0.9 * len(transitions),
        "conditioned_blind_adjacency_and_crn_exact": exact_schedule
        and all(seeds[2 * index] == seeds[2 * index + 1] for index in range(4_096)),
        "all_privacy_and_parent_provenance_audits_pass": len(privacy)
        == PLANNING_REQUESTS
        and all(item["passed"] for item in privacy),
        "within_development_budget_so_far": float(usage["run_cost_usd"])
        <= DEVELOPMENT_BUDGET_USD + 1e-12,
    }
    gates["all_pass"] = all(gates.values())
    checkpoint(
        private / "FROZEN_SELECTIONS.json",
        {
            "selected_roots": [row["selected"] for row in planning],
            "hidden_truth_accessed": False,
            "planning_requests": PLANNING_REQUESTS,
        },
    )
    return {
        "interface_version": INTERFACE_VERSION,
        "contexts": contexts,
        "initial_supports": initial,
        "branch_rows": by_task,
        "planning": planning,
        "privacy": privacy,
        "usage": usage,
        "gates": gates,
    }


def _scorer_policy_name(name: str) -> str:
    return name[4:] if name.startswith("smc_") else name


def _scorer_task(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **task,
        "selected_roots": {
            _scorer_policy_name(name): root
            for name, root in task["selected_roots"].items()
        },
        "policies": {
            _scorer_policy_name(name): metrics
            for name, metrics in task["policies"].items()
        },
    }


def _public_science(science: Mapping[str, Any]) -> dict[str, Any]:
    output = json.loads(json.dumps(science))
    for field in ("comparisons", "root_disagreements"):
        rows = output.get(field)
        if isinstance(rows, dict):
            output[field] = {
                (name if name == "random" else f"smc_{name}"): value
                for name, value in rows.items()
            }
    return output


def run_realized_primary(
    *,
    output_dir: Path,
    adapter: StructuredAdapter,
    tree: Mapping[str, Any],
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    """Execute selected roots after the planning tree has been frozen."""

    if not tree.get("gates", {}).get("all_pass"):
        raise ValueError("realized execution requires a clean planning tree")
    contexts = list(tree["contexts"])
    initial = list(tree["initial_supports"])
    planning = list(tree["planning"])
    if len(contexts) != 64 or len(initial) != 64 or len(planning) != 64:
        raise ValueError("realized execution requires all 64 frozen tasks")
    frozen_path = output_dir / "private" / "FROZEN_SELECTIONS.json"
    frozen = _load_json(frozen_path)
    if (
        frozen.get("hidden_truth_accessed") is not False
        or frozen.get("planning_requests") != PLANNING_REQUESTS
        or frozen.get("selected_roots")
        != [row["selected"] for row in planning]
    ):
        raise ValueError("policy selections were not frozen before truth access")

    truths = []
    first_messages = []
    first_seeds = []
    first_manifest = []
    actual_privacy = []
    for task_index, (context, support, plan) in enumerate(
        zip(contexts, initial, planning, strict=True)
    ):
        truth_index, truth = primary.sample_truth(
            context["cig"], TRUTH_SEED_START + task_index
        )
        truths.append((truth_index, truth))
        for root in sorted(set(plan["selected"].values())):
            question = support["questions"][root]
            mapping = primary.map_and_answer(context["cig"], question, truth)
            dialogue = [
                {"role": "assistant", "content": question},
                {"role": "user", "content": mapping["answer"]},
            ]
            request, audit = core.transition_messages_for(
                context["cig"], dialogue, support
            )
            first_messages.append(request)
            first_seeds.append(actual_first_seed(task_index))
            actual_privacy.append(audit)
            first_manifest.append(
                {
                    "task_index": task_index,
                    "root_index": root,
                    "seed": actual_first_seed(task_index),
                    "first_mapping": mapping,
                    "dialogue": dialogue,
                }
            )
    raw_first = _call(
        adapter,
        first_messages,
        first_seeds,
        core.transition_response_format(),
    )
    first_paths = {}
    final_messages = []
    final_seeds = []
    final_manifest = []
    for manifest, raw in zip(first_manifest, raw_first, strict=True):
        task_index = manifest["task_index"]
        root = manifest["root_index"]
        support = core.parse_enriched_transition(raw, initial[task_index])
        second = scorer.select_question(support)
        second_question = support["questions"][second]
        second_mapping = primary.map_and_answer(
            contexts[task_index]["cig"],
            second_question,
            truths[task_index][1],
        )
        dialogue = [
            *manifest["dialogue"],
            {"role": "assistant", "content": second_question},
            {"role": "user", "content": second_mapping["answer"]},
        ]
        posterior_parent, represented = core.posterior_parent_after_reply(
            support, second, second_mapping["answer"]
        )
        request, audit = core.transition_messages_for(
            contexts[task_index]["cig"], dialogue, posterior_parent
        )
        final_messages.append(request)
        final_seeds.append(actual_final_seed(task_index))
        actual_privacy.append(audit)
        first_paths[(task_index, root)] = {
            "support": support,
            "posterior_parent": posterior_parent,
            "first_mapping": manifest["first_mapping"],
            "second_index": second,
            "second_mapping": second_mapping,
            "second_reply_represented": represented,
            "dialogue": dialogue,
        }
        final_manifest.append(
            {
                "task_index": task_index,
                "root_index": root,
                "seed": actual_final_seed(task_index),
                "second_reply_represented": represented,
            }
        )
    raw_final = _call(
        adapter,
        final_messages,
        final_seeds,
        core.transition_response_format(),
    )
    final_paths = {
        (manifest["task_index"], manifest["root_index"]): (
            core.parse_enriched_transition(
                raw,
                first_paths[(manifest["task_index"], manifest["root_index"])][
                    "posterior_parent"
                ],
            )
        )
        for manifest, raw in zip(final_manifest, raw_final, strict=True)
    }
    private = output_dir / "private"
    checkpoint(
        private / "RAW_ACTUAL_PRIMARY.json",
        {
            "first_manifest": first_manifest,
            "first_responses": raw_first,
            "final_manifest": final_manifest,
            "final_responses": raw_final,
        },
    )

    tasks = []
    private_controls = []
    actual_supports = []
    for task_index, (context, initial_support, plan) in enumerate(
        zip(contexts, initial, planning, strict=True)
    ):
        truth_index, truth = truths[task_index]
        aliases = str((truth.slots or {})["answer_aliases"])
        policies = {}
        for name, root in plan["selected"].items():
            path = first_paths[(task_index, root)]
            final = final_paths[(task_index, root)]
            metrics = scorer.realized_path_metrics(
                initial_support,
                path["support"],
                final,
                first_question_index=root,
                question_index=path["second_index"],
                observed_reply=path["second_mapping"]["answer"],
                aliases=aliases,
                first_mapping=path["first_mapping"],
                second_mapping=path["second_mapping"],
            )
            policies[name] = {
                "endpoint_mode": "aligned_generated_likelihood",
                "root_index": root,
                "second_question_index": path["second_index"],
                "first_supported": path["first_mapping"]["supported"],
                "second_supported": path["second_mapping"]["supported"],
                "second_action_novel": scorer.distinct_supported_actions(
                    path["first_mapping"], path["second_mapping"]
                ),
                "posterior_parent_update_applied": path[
                    "second_reply_represented"
                ],
                **metrics,
            }
        tasks.append(
            {
                "task_id": context["cig"].cig_id,
                "selected_roots": dict(plan["selected"]),
                "root_eig": plan["root_eig"],
                "conditioned_root_risks": plan["conditioned"],
                "conditioned_draw_root_risks": plan["conditioned_by_draw"],
                "dynamic_selection_stability": plan["selection_stability"],
                "blind_root_risks": plan["blind"],
                "myopic_refresh_brier_root_risks": plan[
                    "myopic_refresh_brier"
                ],
                "myopic_brier_root_risks": plan["myopic_brier"],
                "fixed_root_risks": plan["fixed"],
                "policies": policies,
            }
        )
        private_controls.append(
            {
                "task_id": context["cig"].cig_id,
                "truth_index": truth_index,
                "aliases": aliases,
                "selected_questions": {
                    name: {
                        "first": initial_support["questions"][root],
                        "second": first_paths[(task_index, root)]["support"][
                            "questions"
                        ][first_paths[(task_index, root)]["second_index"]],
                    }
                    for name, root in plan["selected"].items()
                },
            }
        )
    for key in sorted(first_paths):
        actual_supports.extend([first_paths[key]["support"], final_paths[key]])

    usage = summarize_usage(adapter.usage_snapshot())
    expected_requests = PLANNING_REQUESTS + len(first_manifest) + len(final_manifest)
    policy_names = sorted(tasks[0]["policies"])
    action_gates = {
        f"{name}_at_least_48_supported_first_actions": sum(
            task["policies"][name]["first_supported"] for task in tasks
        )
        >= 48
        for name in policy_names
    }
    action_gates.update(
        {
            f"{name}_at_least_40_valid_second_actions": sum(
                task["policies"][name]["valid_two_action_trajectory"]
                for task in tasks
            )
            >= 40
            for name in policy_names
        }
    )
    action_gates.update(
        {
            f"{name}_at_least_40_exact_second_replies_represented": sum(
                task["policies"][name]["second_reply_likelihood_matched"]
                for task in tasks
            )
            >= 40
            for name in policy_names
        }
    )
    seed_schedule = all(
        row["seed"] == actual_first_seed(row["task_index"])
        for row in first_manifest
    ) and all(
        row["seed"] == actual_final_seed(row["task_index"])
        for row in final_manifest
    )
    gates = {
        "planning_tree_all_pass": tree["gates"]["all_pass"] is True,
        "exact_primary_request_count": usage["adapter_requests"]
        == expected_requests,
        "exact_primary_http_attempt_count": usage["http_attempts"]
        == expected_requests,
        "primary_requests_within_frozen_maximum": expected_requests <= 8_768,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_actual_transitions_exact_lineage_and_retention": all(
            support["diagnostic"]["parent_index_permutation_exact"] is True
            and core.MIN_RETAINED
            <= support["diagnostic"]["retained_count"]
            <= core.MAX_RETAINED
            for support in actual_supports
        ),
        "selection_frozen_before_truth_access": frozen[
            "hidden_truth_accessed"
        ]
        is False,
        "realized_task_level_common_random_numbers_exact": seed_schedule,
        "all_primary_privacy_and_parent_provenance_audits_pass": len(
            tree["privacy"]
        )
        + len(actual_privacy)
        == expected_requests
        and all(item["passed"] for item in [*tree["privacy"], *actual_privacy]),
        "within_development_budget": float(usage["run_cost_usd"])
        <= DEVELOPMENT_BUDGET_USD + 1e-12,
        **action_gates,
    }
    gates["all_pass"] = all(gates.values())
    scorer_tasks = [_scorer_task(task) for task in tasks]
    science = (
        _public_science(
            scorer.scientific_summary(
                scorer_tasks, samples=bootstrap_samples
            )
        )
        if gates["all_pass"]
        else None
    )
    stability = scorer.draw_stability_summary(scorer_tasks)
    checkpoint(private / "CONTROLS_PRIMARY.json", {"tasks": private_controls})
    return {
        "interface_version": INTERFACE_VERSION,
        "tasks": tasks,
        "usage": usage,
        "expected_primary_requests": expected_requests,
        "actual_privacy": actual_privacy,
        "mechanics_gates": gates,
        "science": science,
        "draw_stability_diagnostic": stability,
        "status": (
            "mechanics_failed"
            if not gates["all_pass"]
            else (
                "passed"
                if science and science["gates"]["all_pass"]
                else "gated_null"
            )
        ),
    }


def run_naive_baseline(
    *,
    output_dir: Path,
    contexts: Sequence[Mapping[str, Any]],
    initial_supports: Sequence[Mapping[str, Any]],
    naive_adapter: StructuredAdapter,
    endpoint_adapter: StructuredAdapter,
) -> dict[str, Any]:
    """Run the non-gating Luna-thinking baseline and SMC fresh endpoints."""

    if len(contexts) != 64 or len(initial_supports) != 64:
        raise ValueError("naive baseline requires all 64 frozen tasks")
    naive_privacy = []
    endpoint_privacy = []
    first_messages = []
    first_seeds = [NAIVE_FIRST_SEED_START + index for index in range(64)]
    for context in contexts:
        request, audit = scorer.naive_messages_for(context["cig"], [])
        first_messages.append(request)
        naive_privacy.append(audit)
    raw_first_questions = _call_naive(
        naive_adapter, first_messages, first_seeds
    )
    first_questions = [
        scorer.parse_naive_question(raw) for raw in raw_first_questions
    ]

    truths = []
    first_support_messages = []
    first_support_seeds = []
    second_messages = []
    second_seeds = []
    first_manifest = []
    for task_index, (context, support, question) in enumerate(
        zip(contexts, initial_supports, first_questions, strict=True)
    ):
        truth_index, truth = primary.sample_truth(
            context["cig"], TRUTH_SEED_START + task_index
        )
        truths.append((truth_index, truth))
        mapping = primary.map_and_answer(context["cig"], question, truth)
        dialogue = [
            {"role": "assistant", "content": question},
            {"role": "user", "content": mapping["answer"]},
        ]
        endpoint_request, endpoint_audit = core.transition_messages_for(
            context["cig"], dialogue, support
        )
        second_request, second_audit = scorer.naive_messages_for(
            context["cig"], dialogue
        )
        first_support_messages.append(endpoint_request)
        first_support_seeds.append(NAIVE_FIRST_SUPPORT_SEED_START + task_index)
        endpoint_privacy.append(endpoint_audit)
        second_messages.append(second_request)
        second_seeds.append(NAIVE_SECOND_SEED_START + task_index)
        naive_privacy.append(second_audit)
        first_manifest.append(
            {
                "task_index": task_index,
                "truth_index": truth_index,
                "first_mapping": mapping,
                "dialogue": dialogue,
            }
        )
    raw_first_supports = _call(
        endpoint_adapter,
        first_support_messages,
        first_support_seeds,
        core.transition_response_format(),
    )
    first_supports = [
        core.parse_enriched_transition(raw, initial_supports[index])
        for index, raw in enumerate(raw_first_supports)
    ]
    raw_second_questions = _call_naive(
        naive_adapter, second_messages, second_seeds
    )
    second_questions = [
        scorer.parse_naive_question(raw) for raw in raw_second_questions
    ]

    final_messages = []
    final_seeds = []
    paths = []
    for manifest, context, first_support, second_question in zip(
        first_manifest,
        contexts,
        first_supports,
        second_questions,
        strict=True,
    ):
        task_index = manifest["task_index"]
        second_mapping = primary.map_and_answer(
            context["cig"], second_question, truths[task_index][1]
        )
        dialogue = [
            *manifest["dialogue"],
            {"role": "assistant", "content": second_question},
            {"role": "user", "content": second_mapping["answer"]},
        ]
        # Luna's free-form second action is not aligned to the four generated
        # reply partitions, so the final redraw receives unchanged SMC weights.
        request, audit = core.transition_messages_for(
            context["cig"], dialogue, first_support
        )
        final_messages.append(request)
        final_seeds.append(NAIVE_FINAL_SUPPORT_SEED_START + task_index)
        endpoint_privacy.append(audit)
        paths.append(
            {
                "first_support": first_support,
                "first_mapping": manifest["first_mapping"],
                "second_mapping": second_mapping,
                "dialogue": dialogue,
            }
        )
    raw_final_supports = _call(
        endpoint_adapter,
        final_messages,
        final_seeds,
        core.transition_response_format(),
    )
    final_supports = [
        core.parse_enriched_transition(raw, first_supports[index])
        for index, raw in enumerate(raw_final_supports)
    ]
    rows = []
    for task_index, (path, final) in enumerate(
        zip(paths, final_supports, strict=True)
    ):
        aliases = str((truths[task_index][1].slots or {})["answer_aliases"])
        raw_first_mass = scorer.truth_mass_for_aliases(
            path["first_support"], aliases
        )
        raw_final_mass = scorer.truth_mass_for_aliases(final, aliases)
        valid = scorer.distinct_supported_actions(
            path["first_mapping"], path["second_mapping"]
        )
        first_mass = (
            raw_first_mass if path["first_mapping"]["supported"] else 0.0
        )
        final_mass = raw_final_mass if valid else 0.0
        rows.append(
            {
                "endpoint_mode": "fresh_smc_regeneration_descriptive",
                "root_index": None,
                "second_question_index": None,
                "first_supported": path["first_mapping"]["supported"],
                "second_supported": path["second_mapping"]["supported"],
                "second_action_novel": valid,
                "valid_two_action_trajectory": valid,
                "raw_truth_mass_after_first": raw_first_mass,
                "truth_mass_after_first": first_mass,
                "raw_truth_mass_final": raw_final_mass,
                "truth_mass_final": final_mass,
                "brier": (1.0 - final_mass) ** 2,
                "log_loss": -math.log(
                    max(core.PROBABILITY_FLOOR, final_mass)
                ),
                "covered": final_mass > 0.0,
                "raw_fresh_truth_mass_final": raw_final_mass,
                "fresh_truth_mass_final": final_mass,
                "fresh_brier": (1.0 - final_mass) ** 2,
                "fresh_log_loss": -math.log(
                    max(core.PROBABILITY_FLOOR, final_mass)
                ),
                "fresh_covered": final_mass > 0.0,
            }
        )
    naive_usage = summarize_usage(naive_adapter.usage_snapshot())
    endpoint_usage = summarize_usage(endpoint_adapter.usage_snapshot())
    diagnostics = {
        "status": "available",
        "descriptive_only": True,
        "can_gate_or_abort_primary": False,
        "exact_128_luna_requests": naive_usage["adapter_requests"] == 128,
        "exact_128_luna_http_attempts": naive_usage["http_attempts"] == 128,
        "positive_luna_reasoning_tokens": naive_usage[
            "adapter_reasoning_tokens"
        ]
        > 0,
        "exact_128_deepseek_endpoint_requests": endpoint_usage[
            "adapter_requests"
        ]
        == 128,
        "exact_128_deepseek_endpoint_http_attempts": endpoint_usage[
            "http_attempts"
        ]
        == 128,
        "zero_endpoint_reasoning_tokens": endpoint_usage[
            "adapter_reasoning_tokens"
        ]
        == 0,
        "all_prompt_privacy_audits_pass": len(naive_privacy) == 128
        and all(item["passed"] for item in naive_privacy),
        "all_endpoint_privacy_and_provenance_audits_pass": len(
            endpoint_privacy
        )
        == 128
        and all(item["passed"] for item in endpoint_privacy),
        "at_least_48_supported_first_actions": sum(
            row["first_supported"] for row in rows
        )
        >= 48,
        "at_least_40_valid_second_actions": sum(
            row["valid_two_action_trajectory"] for row in rows
        )
        >= 40,
    }
    diagnostics["all_descriptive_diagnostics_pass"] = all(
        value
        for key, value in diagnostics.items()
        if key
        not in {"status", "descriptive_only", "can_gate_or_abort_primary"}
    )
    private = output_dir / "private"
    checkpoint(
        private / "RAW_NAIVE.json",
        {
            "first_question_seeds": first_seeds,
            "first_questions": raw_first_questions,
            "first_support_seeds": first_support_seeds,
            "first_supports": raw_first_supports,
            "second_question_seeds": second_seeds,
            "second_questions": raw_second_questions,
            "final_support_seeds": final_seeds,
            "final_supports": raw_final_supports,
        },
    )
    return {
        "status": "available",
        "rows": rows,
        "luna_usage": naive_usage,
        "endpoint_usage": endpoint_usage,
        "naive_privacy": naive_privacy,
        "endpoint_privacy": endpoint_privacy,
        "diagnostics": diagnostics,
    }


def _empty_usage() -> dict[str, Any]:
    return {
        "adapter_requests": 0,
        "http_attempts": 0,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def finalize_development_result(
    *,
    output_dir: Path,
    primary_result: Mapping[str, Any],
    policy_smoke: Mapping[str, Any],
    naive_smoke: Mapping[str, Any],
    naive_result: Mapping[str, Any] | None,
    naive_error: Mapping[str, str] | None = None,
    naive_usage_on_error: Mapping[str, Any] | None = None,
    endpoint_usage_on_error: Mapping[str, Any] | None = None,
    daily_budget_status: Mapping[str, Any] | None = None,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    """Assemble the public result without changing the primary decision rule."""

    tasks = json.loads(json.dumps(primary_result["tasks"]))
    primary_usage = dict(primary_result["usage"])
    if naive_result is not None and naive_error is None:
        rows = list(naive_result["rows"])
        if len(rows) != len(tasks):
            raise ValueError("naive baseline row count changed")
        for task, row in zip(tasks, rows, strict=True):
            task["policies"]["naive_thinking"] = row
        naive_usage = dict(naive_result["luna_usage"])
        endpoint_usage = dict(naive_result["endpoint_usage"])
        baseline = {
            "status": "available",
            "error": None,
            **dict(naive_result["diagnostics"]),
        }
    else:
        naive_usage = dict(naive_usage_on_error or _empty_usage())
        endpoint_usage = dict(endpoint_usage_on_error or _empty_usage())
        baseline = {
            "status": "unavailable",
            "descriptive_only": True,
            "can_gate_or_abort_primary": False,
            "error": dict(naive_error or {"error": "baseline disabled"}),
        }
    combined_cost = sum(
        float(row["run_cost_usd"])
        for row in (primary_usage, endpoint_usage, naive_usage)
    )
    combined_requests = sum(
        int(row["adapter_requests"])
        for row in (primary_usage, endpoint_usage, naive_usage)
    )
    combined_attempts = sum(
        int(row["http_attempts"])
        for row in (primary_usage, endpoint_usage, naive_usage)
    )
    mechanics = dict(primary_result["mechanics_gates"])
    mechanics.pop("all_pass", None)
    mechanics.update(
        {
            "primary_deepseek_requests_within_8768": int(
                primary_usage["adapter_requests"]
            )
            <= 8_768,
            "deepseek_requests_with_naive_endpoints_within_8896": int(
                primary_usage["adapter_requests"]
            )
            + int(endpoint_usage["adapter_requests"])
            <= 8_896,
            "combined_requests_within_9024": combined_requests <= 9_024,
            "combined_spend_within_350": combined_cost
            <= DEVELOPMENT_BUDGET_USD + 1e-12,
            "naive_baseline_is_descriptive_only": baseline[
                "descriptive_only"
            ]
            is True,
            "naive_baseline_cannot_gate_or_abort_primary": baseline[
                "can_gate_or_abort_primary"
            ]
            is False,
        }
    )
    mechanics["all_pass"] = all(mechanics.values())
    scorer_tasks = [_scorer_task(task) for task in tasks]
    science = (
        _public_science(
            scorer.scientific_summary(
                scorer_tasks, samples=bootstrap_samples
            )
        )
        if mechanics["all_pass"]
        else None
    )
    status = "mechanics_failed"
    if mechanics["all_pass"]:
        status = (
            "passed" if science and science["gates"]["all_pass"] else "gated_null"
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "authorizes": (
            "smc_confirmation_preregistration_only"
            if status == "passed"
            else "nothing"
        ),
        "protocol": {
            "stage": "development",
            "model": core.MODEL_ID,
            "reasoning": "disabled_excluded",
            "task_count": 64,
            "branch_draws": BRANCH_DRAWS,
            "planning_requests": PLANNING_REQUESTS,
            "actual_primary_deepseek_requests": primary_usage[
                "adapter_requests"
            ],
            "actual_naive_endpoint_requests": endpoint_usage[
                "adapter_requests"
            ],
            "actual_naive_luna_requests": naive_usage["adapter_requests"],
            "actual_combined_requests": combined_requests,
            "maximum_primary_deepseek_requests": 8_768,
            "maximum_deepseek_requests": 8_896,
            "maximum_requests": 9_024,
            "protocol_sha256": core.PROTOCOL_SHA256,
            "producer_core_sha256": primary.sha256_file(
                Path(__file__).resolve()
            ),
            "policy_smoke": dict(policy_smoke),
            "naive_smoke": dict(naive_smoke),
            "naive_model": scorer.NAIVE_MODEL_ID,
            "naive_reasoning_effort": scorer.NAIVE_REASONING_EFFORT,
            "naive_is_descriptive_only": True,
            "naive_can_gate_or_abort_primary": False,
            "primary_endpoint": "aligned_generated_likelihood_truth_mass",
            "fresh_smc_regeneration_endpoint": "secondary_descriptive",
            "naive_endpoint": "fresh_smc_regeneration_descriptive",
            "initial_hypotheses_regenerated": False,
            "conditioned_blind_same_seed": True,
            "conditioned_blind_adjacent": True,
            "simulated_common_seed_across_roots": True,
            "realized_common_seed_across_roots": True,
            "selection_frozen_before_truth_access": True,
            "hidden_cig_exposed_to_model": False,
            "confirmation_opened": False,
        },
        "daily_budget_status": dict(daily_budget_status or {}),
        "usage": {
            "deepseek_primary": primary_usage,
            "deepseek_naive_endpoint": endpoint_usage,
            "naive_luna": naive_usage,
            "combined_cost_usd": combined_cost,
            "combined_requests": combined_requests,
            "combined_http_attempts": combined_attempts,
        },
        "mechanics_gates": mechanics,
        "naive_baseline": baseline,
        "science": science,
        "draw_stability_diagnostic": primary_result[
            "draw_stability_diagnostic"
        ],
        "tasks": tasks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint(output_dir / "RESULT.json", result)
    return result
