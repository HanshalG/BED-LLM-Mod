#!/usr/bin/env python3
"""Producer core for the dormant RegretBench factorized exact-10 smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from scripts import regretbench_deepseek_dynamic_depth2_policy as transport
from scripts import regretbench_deepseek_smc_dynamic_depth2_experiment as experiment
from scripts import regretbench_deepseek_smc_dynamic_depth2_policy as core
from scripts import regretbench_deepseek_support_recovery as primary
from scripts import regretbench_static_child_likelihood as static
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-factorized-static-likelihood-smoke-1"
PROTOCOL = primary.REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_FACTORIZED_STATIC_LIKELIHOOD_SMOKE_PREREGISTRATION_20260807.md"
)
PROTOCOL_SHA256 = "bfd7257295615bc0e80f5205709d95599455e44bd944faacd4907c96f99aa4d3"
INITIAL_SEED_START = 202608420000
TRANSITION_SEED_START = 202608421000
STATIC_SEED_START = 202608422000
EXPECTED_REQUESTS = 10
SMOKE_BUDGET_USD = 0.20


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol() -> None:
    if not PROTOCOL.is_file() or sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("factorized smoke protocol changed")


def _call(
    adapter: StructuredAdapter,
    messages: Sequence[list[dict[str, str]]],
    seeds: Sequence[int],
    response_format: dict[str, Any],
) -> list[str]:
    if len(messages) != len(seeds):
        raise ValueError("factorized smoke messages and seeds differ")
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        messages,
        seeds,
        temperature=core.TEMPERATURE,
        response_format=response_format,
        max_new_tokens=core.MAX_TOKENS,
    )
    if len(responses) != len(messages):
        raise ValueError("factorized smoke adapter returned wrong response count")
    return list(responses)


def _informative(support: Mapping[str, Any]) -> int:
    return sum(
        transport.question_eig(support, index) > 1e-12
        for index in range(core.QUESTIONS)
    )


def run_smoke(
    *,
    output_dir: Path,
    adapter: StructuredAdapter,
    primary_smoke_dir: Path,
    support_smoke_result: Path,
    support_development_result: Path,
    daily_budget_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_protocol()
    predecessor = transport.validate_support_predecessors(
        support_smoke_result=support_smoke_result,
        support_development_result=support_development_result,
    )
    contexts = experiment.load_banked_parent_contexts(
        stage="smoke", primary_dir=primary_smoke_dir
    )[:2]
    if len(contexts) != 2:
        raise ValueError("factorized smoke requires two banked contexts")

    initial_messages = []
    privacy = []
    for context in contexts:
        request, audit = core.annotation_messages_for(
            context["cig"], context["parent"], context["questions"]
        )
        initial_messages.append(request)
        privacy.append(audit)
    initial_seeds = [INITIAL_SEED_START + index for index in range(2)]
    raw_initial = _call(
        adapter,
        initial_messages,
        initial_seeds,
        core.annotation_response_format(),
    )
    initial = [
        core.parse_parent_annotation(raw, context["parent"], context["questions"])
        for raw, context in zip(raw_initial, contexts, strict=True)
    ]

    truths = []
    first_mappings = []
    transition_messages = []
    transition_seeds = []
    for index, (context, support) in enumerate(zip(contexts, initial, strict=True)):
        truth_index, truth = primary.sample_truth(
            context["cig"], primary.STAGES["smoke"]["truth_seed_start"] + index
        )
        if truth_index != context["control"]["truth_index"]:
            raise ValueError("factorized smoke truth control changed")
        question = support["questions"][0]
        mapping = primary.map_and_answer(context["cig"], question, truth)
        if mapping != context["control"]["mapping"]:
            raise ValueError("factorized smoke first mapping changed")
        truths.append(truth)
        first_mappings.append(mapping)
        dialogue = [
            {"role": "assistant", "content": question},
            {"role": "user", "content": mapping["answer"]},
        ]
        conditioned, conditioned_audit = core.transition_messages_for(
            context["cig"], dialogue, support
        )
        blind, blind_audit = core.transition_messages_for(
            context["cig"], [], support
        )
        transition_messages.extend([conditioned, blind])
        transition_seeds.extend([TRANSITION_SEED_START + index] * 2)
        privacy.extend([conditioned_audit, blind_audit])
    raw_transitions = _call(
        adapter,
        transition_messages,
        transition_seeds,
        core.transition_response_format(),
    )
    transitions = [
        core.parse_enriched_transition(raw_transitions[2 * task + arm], initial[task])
        for task in range(2)
        for arm in range(2)
    ]

    static_messages = []
    static_seeds = []
    for task, context in enumerate(contexts):
        for arm in range(2):
            request, audit = static.messages_for(
                context["cig"], transitions[2 * task + arm]
            )
            static_messages.append(request)
            static_seeds.append(STATIC_SEED_START + task)
            privacy.append(audit)
    raw_static = _call(
        adapter, static_messages, static_seeds, static.response_format()
    )
    factorized = [
        static.parse_annotation(raw, transition)
        for raw, transition in zip(raw_static, transitions, strict=True)
    ]

    second_mappings = []
    second_reply_matches = []
    conditioned_novel = []
    for task, (context, truth, first) in enumerate(
        zip(contexts, truths, first_mappings, strict=True)
    ):
        for arm in range(2):
            support = factorized[2 * task + arm]
            second = transport.select_question(support)
            mapping = primary.map_and_answer(
                context["cig"], support["questions"][second], truth
            )
            second_mappings.append(mapping)
            second_reply_matches.append(
                bool(
                    transport.matching_reply_indexes(
                        support, second, mapping["answer"]
                    )
                )
            )
            if arm == 0:
                conditioned_novel.append(
                    transport.distinct_supported_actions(first, mapping)
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    checkpoint(
        private / "RAW_RESPONSES.json",
        {
            "initial_seeds": initial_seeds,
            "initial": raw_initial,
            "transition_seeds": transition_seeds,
            "transitions": raw_transitions,
            "static_seeds": static_seeds,
            "static": raw_static,
        },
    )
    checkpoint(private / "PRIVACY.json", {"audits": privacy})
    usage = summarize_usage(adapter.usage_snapshot())
    gates = {
        "exact_ten_parsed_responses": len(initial) + len(transitions) + len(factorized)
        == EXPECTED_REQUESTS,
        "exact_two_initial_four_transition_four_static": len(initial) == 2
        and len(transitions) == 4
        and len(factorized) == 4,
        "exact_ten_requests": usage["adapter_requests"] == EXPECTED_REQUESTS,
        "exact_ten_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_initial_annotations_exact": all(
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
        "all_static_annotations_preserve_children": all(
            row["diagnostic"]["history_conditioned_child_particles_preserved"]
            is True
            and row["diagnostic"]["updated_likelihood_replies_discarded"] is True
            and row["diagnostic"]["static_annotation_index_permutation_exact"]
            is True
            for row in factorized
        ),
        "every_initial_has_two_informative_roots": all(
            _informative(row) >= 2 for row in initial
        ),
        "every_factorized_child_has_informative_followup": all(
            _informative(row) >= 1 for row in factorized
        ),
        "both_first_questions_supported": all(
            row["supported"] for row in first_mappings
        ),
        "all_four_second_questions_supported": all(
            row["supported"] for row in second_mappings
        ),
        "both_conditioned_second_actions_novel": all(conditioned_novel),
        "all_four_exact_second_replies_have_factorized_likelihood": all(
            second_reply_matches
        ),
        "transition_pairing_exact": transition_seeds
        == [TRANSITION_SEED_START, TRANSITION_SEED_START,
            TRANSITION_SEED_START + 1, TRANSITION_SEED_START + 1],
        "static_pairing_exact": static_seeds
        == [STATIC_SEED_START, STATIC_SEED_START,
            STATIC_SEED_START + 1, STATIC_SEED_START + 1],
        "all_privacy_audits_pass": len(privacy) == EXPECTED_REQUESTS
        and all(row["passed"] for row in privacy),
        "within_smoke_budget": float(usage["run_cost_usd"])
        <= SMOKE_BUDGET_USD + 1e-12,
    }
    gates["all_pass"] = all(gates.values())
    status = "passed" if gates["all_pass"] else "mechanics_failed"
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "authorizes": (
            "separate_factorized_policy_preregistration_only"
            if status == "passed"
            else "nothing"
        ),
        "protocol": {
            "model": core.MODEL_ID,
            "reasoning": "disabled_excluded",
            "expected_requests": EXPECTED_REQUESTS,
            "protocol_sha256": PROTOCOL_SHA256,
            "predecessor": predecessor,
            "efficacy_accessed": False,
            "development_opened": False,
            "confirmation_opened": False,
            "paid_execution_authorized_by_implementation": False,
        },
        "daily_budget_status": dict(daily_budget_status or {}),
        "usage": usage,
        "gates": gates,
        "diagnostics": {
            "initial": [row["diagnostic"] for row in initial],
            "transitions": [row["diagnostic"] for row in transitions],
            "factorized": [row["diagnostic"] for row in factorized],
        },
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result
