#!/usr/bin/env python3
"""Producer core for the sealed RegretBench SMC depth-two policy.

This module has no CLI and never constructs a network adapter. A future dated
executor may pass an already budget-authorized adapter only after the frozen
predecessor chain succeeds.
"""

from __future__ import annotations

import hashlib
import json
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
