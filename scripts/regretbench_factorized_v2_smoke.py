#!/usr/bin/env python3
"""Run the fresh RegretBench factorized-v2 exact-10 mechanics smoke."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Protocol, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
REGRETBENCH_ROOT = REPO_ROOT / "external/RegretBench"
REGRETBENCH_SRC = REGRETBENCH_ROOT / "src"
for path in (REPO_ROOT, REGRETBENCH_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helpers import Config, ModelSpec
from regretbench.schemas.cig import CIG, load_cig
from scripts import regretbench_deepseek_dynamic_depth2_policy as policy
from scripts import regretbench_deepseek_smc_dynamic_depth2_policy as transition
from scripts import regretbench_deepseek_support_recovery as source_tools
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.number_game_qwen_history_blind_serving_smoke import (
    PerRequestSeedStructuredAdapter,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-factorized-v2-exact10-smoke-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
TEMPERATURE = 0.7
MAX_TOKENS = 2_400
EXPECTED_REQUESTS = 10
MAX_RETRIES = 4
CONCURRENCY = 10
RUN_BUDGET_USD = 0.20
PROJECTED_COST_USD = 0.04
ROOT_SEED_START = 202608510000
TRANSITION_SEED_START = 202608511000
STATIC_SEED_START = 202608512000

SOURCE_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_FACTORIZED_V2_SOURCE_PROTOCOL_20260811.md"
)
SOURCE_PROTOCOL_SHA256 = (
    "151751b576e2db14a6e21848d7d548a09f92172a6709e476ed903d41fea8c753"
)
SOURCE_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/regretbench_factorized_v2_source_audit/"
    "SOURCE_PROTOCOL_MANIFEST.json"
)
SOURCE_MANIFEST_SHA256 = (
    "831a8bcf8f38b183c080c5a369896366915ca8b8ada38294143d7f31a11c1999"
)
SOURCE_RESULT = REPO_ROOT / (
    "results/nonmyopic/regretbench_factorized_v2_source_audit/RESULT.json"
)
SOURCE_RESULT_SHA256 = (
    "54fc24022ccce75f5865173edf0049392862c413813fdd0e7884fbfc55e537a7"
)
SMOKE_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_FACTORIZED_V2_EXACT10_SMOKE_PROTOCOL_20260811.md"
)
SMOKE_PROTOCOL_SHA256 = (
    "3901343d7633e056b450a778b9077f218866567bb6fe5a16c6c81c851f85f3db"
)
MECHANICS_SPLIT_SHA256 = (
    "93fdbfb55d0ded5fdbed1cfa78f251a863590f07ae2eaece2e20bdfc2d01a353"
)


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


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_bindings() -> dict[str, str]:
    expected = {
        "source_protocol": (SOURCE_PROTOCOL, SOURCE_PROTOCOL_SHA256),
        "source_manifest": (SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256),
        "source_result": (SOURCE_RESULT, SOURCE_RESULT_SHA256),
        "smoke_protocol": (SMOKE_PROTOCOL, SMOKE_PROTOCOL_SHA256),
    }
    for name, (path, digest) in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"factorized-v2 {name} changed")
    source = _load(SOURCE_RESULT)
    manifest = _load(SOURCE_MANIFEST)
    if (
        source.get("status") != "source_protocol_pass"
        or source.get("authorizes") != "factorized_v2_exact10_smoke_only"
        or source.get("gates", {}).get("all_pass") is not True
        or manifest.get("splits", {}).get("mechanics", {}).get("ids_sha256")
        != MECHANICS_SPLIT_SHA256
    ):
        raise ValueError("factorized-v2 source does not authorize smoke")
    return {name: digest for name, (_, digest) in expected.items()}


def load_smoke_cigs() -> list[CIG]:
    validate_bindings()
    ids = _load(SOURCE_MANIFEST)["splits"]["mechanics"]["ids"][:2]
    if len(ids) != 2:
        raise ValueError("factorized-v2 smoke task count changed")
    cigs = [
        load_cig(REGRETBENCH_ROOT / "data/OpenDomainQA/test" / f"{cig_id}.json")
        for cig_id in ids
    ]
    if [cig.cig_id for cig in cigs] != ids:
        raise ValueError("factorized-v2 smoke task order changed")
    return cigs


def static_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "regretbench_factorized_v2_static_likelihood",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["particles"],
                "properties": {
                    "particles": {
                        "type": "array",
                        "minItems": 8,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["particle_index", "predicted_replies"],
                            "properties": {
                                "particle_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 7,
                                },
                                "predicted_replies": {
                                    "type": "array",
                                    "minItems": 5,
                                    "maxItems": 5,
                                    "items": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 240,
                                    },
                                },
                            },
                        },
                    }
                },
            },
        },
    }


STATIC_SYSTEM_PROMPT = """Predict how a cooperative user with each fixed semantic particle would answer each fixed clarification question. Return exactly one record for particle_index 0 through 7 and exactly five concise nonempty replies aligned to the supplied five questions. Do not revise, merge, remove, reweight, or add particles or questions. The first question is an already asked conditioning question, but its observed answer is intentionally withheld. Use only the public ambiguous prompt, fixed particle text, and fixed questions. Return only the required JSON object."""


def _public_children(child: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = child.get("hypotheses")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("child support must contain eight particles")
    result = []
    for index, row in enumerate(rows):
        interpretation = row.get("interpretation")
        answer = row.get("final_answer")
        if not isinstance(interpretation, str) or not interpretation.strip():
            raise ValueError("child interpretation is empty")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("child final answer is empty")
        result.append(
            {
                "particle_index": index,
                "interpretation": interpretation.strip(),
                "final_answer": answer.strip(),
            }
        )
    return result


def _structural_hash(particles: Any, questions: Any) -> str:
    return hashlib.sha256(
        canonical_json({"particles": particles, "questions": questions}).encode()
    ).hexdigest()


def static_messages_for(
    cig: CIG, child: Mapping[str, Any], conditioning_question: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    particles = _public_children(child)
    child_questions = transition._validate_questions(list(child["questions"]))
    questions = [conditioning_question.strip(), *child_questions]
    if len(questions) != 5 or any(not question.endswith("?") for question in questions):
        raise ValueError("static annotation requires five questions")
    base = source_tools.public_payload(cig, [])
    base_audit = source_tools.privacy_audit(cig, base)
    structural_hash = _structural_hash(particles, questions)
    payload = {
        **base,
        "child_particles": particles,
        "questions": questions,
        "structural_child_sha256": structural_hash,
        "likelihood_factorization": "static_particle_question_only_v2",
        "particle_source": "history_conditioned_children_with_history_removed",
    }
    forbidden = {
        "answer",
        "probability",
        "prior_weight",
        "revision_type",
        "parent_index",
        "predicted_replies",
        "intents",
        "mapping",
        "truth",
    }
    serialized = canonical_json(payload)
    audit = {
        "passed": base_audit["passed"] is True
        and not any(f'"{key}"' in serialized for key in forbidden),
        "payload_keys": sorted(payload),
        "payload_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "structural_child_sha256": structural_hash,
        "dialogue_excluded": payload["dialogue"] == [],
        "observed_answer_excluded": True,
        "particle_probabilities_excluded": True,
        "lineage_metadata_excluded": True,
        "old_predicted_replies_excluded": True,
        "hidden_truth_exposed": False,
    }
    return [
        {"role": "system", "content": STATIC_SYSTEM_PROMPT},
        {"role": "user", "content": serialized},
    ], audit


def parse_static_annotation(raw: str, child: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"particles"}:
        raise ValueError("static annotation has wrong top-level fields")
    rows = value["particles"]
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("static annotation must contain eight particles")
    by_index: dict[int, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "particle_index",
            "predicted_replies",
        }:
            raise ValueError("static annotation particle has wrong fields")
        index = row["particle_index"]
        replies = row["predicted_replies"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in range(8)
            or index in by_index
            or not isinstance(replies, list)
            or len(replies) != 5
            or any(not isinstance(reply, str) or not reply.strip() for reply in replies)
        ):
            raise ValueError("static annotation values are invalid")
        by_index[index] = [reply.strip() for reply in replies]
    if sorted(by_index) != list(range(8)):
        raise ValueError("static particle indexes are not an exact permutation")
    hypotheses = []
    for index, row in enumerate(child["hypotheses"]):
        hypotheses.append(
            {
                **{key: item for key, item in row.items() if key != "predicted_replies"},
                "conditioning_reply": by_index[index][0],
                "predicted_replies": by_index[index][1:],
            }
        )
    result = {
        "hypotheses": hypotheses,
        "questions": list(child["questions"]),
        "diagnostic": {
            **dict(child.get("diagnostic") or {}),
            "likelihood_factorization": "static_particle_question_only_v2",
            "history_conditioned_child_particles_preserved": True,
            "updated_likelihood_replies_discarded": True,
            "static_annotation_index_permutation_exact": True,
            "static_reply_count_per_particle": 5,
        },
    }
    return result


def condition_on_observed_reply(
    support: Mapping[str, Any], observed_reply: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = source_tools.normalize_text(observed_reply)
    matches = [
        index
        for index, row in enumerate(support["hypotheses"])
        if source_tools.normalize_text(row["conditioning_reply"]) == observed
    ]
    if not observed or not matches:
        raise ValueError("observed reply has no static child likelihood")
    mass = sum(float(support["hypotheses"][index]["probability"]) for index in matches)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("observed reply has zero child prior mass")
    hypotheses = []
    for index, row in enumerate(support["hypotheses"]):
        probability = float(row["probability"]) / mass if index in matches else 0.0
        hypotheses.append({**row, "probability": probability})
    conditioned = {
        "hypotheses": hypotheses,
        "questions": list(support["questions"]),
        "diagnostic": {
            **dict(support.get("diagnostic") or {}),
            "exact_observation_conditioning": True,
            "matching_particle_indexes": matches,
            "precondition_matching_mass": mass,
        },
    }
    predictive = sum(hypotheses[index]["probability"] for index in matches)
    return conditioned, {
        "matching_particle_indexes": matches,
        "precondition_matching_mass": mass,
        "posterior_predictive_matched_reply": predictive,
        "finite_normalized_weights": math.isfinite(predictive)
        and abs(sum(row["probability"] for row in hypotheses) - 1.0) <= 1e-12,
    }


def _call(
    adapter: StructuredAdapter,
    messages: Sequence[list[dict[str, str]]],
    seeds: Sequence[int],
    response_format: dict[str, Any],
) -> list[str]:
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        messages,
        seeds,
        temperature=TEMPERATURE,
        response_format=response_format,
        max_new_tokens=MAX_TOKENS,
    )
    if len(responses) != len(messages):
        raise ValueError("factorized-v2 adapter returned wrong response count")
    return list(responses)


def build_adapter(*, run_id: str, output_dir: Path) -> PerRequestSeedStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=245.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=MAX_RETRIES,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return PerRequestSeedStructuredAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65536),
        config,
    )


def _transport_gates(usage: Mapping[str, Any]) -> dict[str, bool]:
    accepted = usage["adapter_requests"]
    attempts = usage["http_attempts"]
    retries = usage["retry_count"]
    provider = usage["provider_error_retries"]
    return {
        "exact_ten_accepted_requests": accepted == EXPECTED_REQUESTS,
        "http_attempts_between_ten_and_fourteen": EXPECTED_REQUESTS
        <= attempts
        <= EXPECTED_REQUESTS + MAX_RETRIES,
        "attempts_equal_accepted_plus_retries": attempts == accepted + retries,
        "retries_within_four": 0 <= retries <= MAX_RETRIES,
        "provider_retries_are_subset": 0 <= provider <= retries,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
    }


def run_smoke(
    *,
    output_dir: Path,
    adapter: StructuredAdapter,
    daily_budget_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bindings = validate_bindings()
    cigs = load_smoke_cigs()
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    root_messages = []
    privacy = []
    for cig in cigs:
        messages, audit = policy.messages_for(cig, [])
        root_messages.append(messages)
        privacy.append(audit)
    root_seeds = [ROOT_SEED_START + index for index in range(2)]
    raw_roots = _call(adapter, root_messages, root_seeds, policy.enriched_response_format())
    raw_bank = {"root_seeds": root_seeds, "roots": raw_roots}
    checkpoint(private / "RAW_RESPONSES.json", raw_bank)
    checkpoint(private / "PRIVACY.json", {"audits": privacy})
    roots = [policy.parse_enriched_support(raw) for raw in raw_roots]

    transition_messages = []
    transition_seeds = []
    first_mappings = []
    truths = []
    for index, (cig, root) in enumerate(zip(cigs, roots, strict=True)):
        _, truth = source_tools.sample_truth(cig, ROOT_SEED_START + 10_000 + index)
        truths.append(truth)
        first = source_tools.map_and_answer(cig, root["questions"][0], truth)
        first_mappings.append(first)
        dialogue = [
            {"role": "assistant", "content": root["questions"][0]},
            {"role": "user", "content": first["answer"]},
        ]
        conditioned, conditioned_audit = transition.transition_messages_for(
            cig, dialogue, root
        )
        blind, blind_audit = transition.transition_messages_for(cig, [], root)
        transition_messages.extend([conditioned, blind])
        transition_seeds.extend([TRANSITION_SEED_START + index] * 2)
        privacy.extend([conditioned_audit, blind_audit])
    raw_transitions = _call(
        adapter,
        transition_messages,
        transition_seeds,
        transition.transition_response_format(),
    )
    raw_bank.update(
        {
            "transition_seeds": transition_seeds,
            "transitions": raw_transitions,
        }
    )
    checkpoint(private / "RAW_RESPONSES.json", raw_bank)
    checkpoint(private / "PRIVACY.json", {"audits": privacy})
    transitions = [
        transition.parse_enriched_transition(raw_transitions[2 * task + arm], roots[task])
        for task in range(2)
        for arm in range(2)
    ]

    static_messages = []
    static_seeds = []
    for task, cig in enumerate(cigs):
        for arm in range(2):
            messages, audit = static_messages_for(
                cig, transitions[2 * task + arm], roots[task]["questions"][0]
            )
            static_messages.append(messages)
            static_seeds.append(STATIC_SEED_START + task)
            privacy.append(audit)
    raw_static = _call(adapter, static_messages, static_seeds, static_response_format())
    raw_bank.update({"static_seeds": static_seeds, "static": raw_static})
    checkpoint(private / "RAW_RESPONSES.json", raw_bank)
    checkpoint(private / "PRIVACY.json", {"audits": privacy})
    factorized = [
        parse_static_annotation(raw, child)
        for raw, child in zip(raw_static, transitions, strict=True)
    ]

    conditioned_supports = []
    conditioning = []
    second_mappings = []
    second_matches = []
    novel_actions = []
    informative = []
    for task, (cig, truth, first) in enumerate(
        zip(cigs, truths, first_mappings, strict=True)
    ):
        conditioned, diagnostic = condition_on_observed_reply(
            factorized[2 * task], first["answer"]
        )
        conditioned_supports.append(conditioned)
        conditioning.append(diagnostic)
        task_supports = [conditioned, factorized[2 * task + 1]]
        informative.extend(
            [
                sum(policy.question_eig(support, index) > 1e-12 for index in range(4))
                for support in task_supports
            ]
        )
        second_index = policy.select_question(conditioned)
        second = source_tools.map_and_answer(
            cig, conditioned["questions"][second_index], truth
        )
        second_mappings.append(second)
        second_matches.append(
            bool(policy.matching_reply_indexes(conditioned, second_index, second["answer"]))
        )
        novel_actions.append(policy.distinct_supported_actions(first, second))

    usage = summarize_usage(adapter.usage_snapshot())
    gates = {
        **_transport_gates(usage),
        "exact_two_root_four_transition_four_static": len(roots) == 2
        and len(transitions) == 4
        and len(factorized) == 4,
        "all_roots_have_eight_unique_particles": all(
            row["diagnostic"]["valid_unique_count"] == 8 for row in roots
        ),
        "both_roots_have_two_informative_questions": all(
            row["diagnostic"]["informative_question_count"] >= 2 for row in roots
        ),
        "all_transitions_have_exact_lineage_and_retention": all(
            row["diagnostic"]["parent_index_permutation_exact"] is True
            and transition.MIN_RETAINED
            <= row["diagnostic"]["retained_count"]
            <= transition.MAX_RETAINED
            for row in transitions
        ),
        "all_static_annotations_preserve_children": all(
            row["diagnostic"]["history_conditioned_child_particles_preserved"] is True
            and row["diagnostic"]["updated_likelihood_replies_discarded"] is True
            and row["diagnostic"]["static_annotation_index_permutation_exact"] is True
            and row["diagnostic"]["static_reply_count_per_particle"] == 5
            for row in factorized
        ),
        "both_first_answers_mapper_supported": all(row["supported"] for row in first_mappings),
        "both_conditioned_answers_have_static_matches": all(
            bool(row["matching_particle_indexes"]) for row in conditioning
        ),
        "exact_conditioning_is_finite_and_normalized": all(
            row["finite_normalized_weights"]
            and abs(row["posterior_predictive_matched_reply"] - 1.0) <= 1e-12
            for row in conditioning
        ),
        "every_child_has_informative_future_question": all(value >= 1 for value in informative),
        "both_second_questions_mapper_supported": all(row["supported"] for row in second_mappings),
        "both_conditioned_second_actions_are_novel": all(novel_actions),
        "both_second_replies_have_positive_static_likelihood": all(second_matches),
        "transition_seed_pairing_exact": transition_seeds
        == [TRANSITION_SEED_START, TRANSITION_SEED_START, TRANSITION_SEED_START + 1, TRANSITION_SEED_START + 1],
        "static_seed_pairing_exact": static_seeds
        == [STATIC_SEED_START, STATIC_SEED_START, STATIC_SEED_START + 1, STATIC_SEED_START + 1],
        "all_privacy_audits_pass": len(privacy) == EXPECTED_REQUESTS
        and all(row["passed"] is True for row in privacy),
        "within_smoke_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD + 1e-12,
    }
    gates["all_pass"] = all(gates.values())
    status = "passed" if gates["all_pass"] else "mechanics_failed"
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "authorizes": (
            "separate_factorized_v2_policy_preregistration_only"
            if status == "passed"
            else "nothing"
        ),
        "protocol": {
            "model": MODEL_ID,
            "reasoning": "disabled_excluded",
            "expected_accepted_requests": EXPECTED_REQUESTS,
            "maximum_http_attempts": EXPECTED_REQUESTS + MAX_RETRIES,
            "bindings": bindings,
            "mechanics_split_sha256": MECHANICS_SPLIT_SHA256,
            "efficacy_accessed": False,
            "development_opened": False,
            "confirmation_opened": False,
            "daily_budget_status": dict(daily_budget_status or {}),
        },
        "gates": gates,
        "diagnostics": {
            "roots": [row["diagnostic"] for row in roots],
            "transitions": [row["diagnostic"] for row in transitions],
            "factorized": [row["diagnostic"] for row in factorized],
            "conditioning": conditioning,
            "informative_future_question_counts": informative,
        },
        "usage": usage,
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result
