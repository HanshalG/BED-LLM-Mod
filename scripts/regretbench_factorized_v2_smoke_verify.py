#!/usr/bin/env python3
"""Independently replay the factorized-v2 exact-10 smoke."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
REGRETBENCH_ROOT = REPO_ROOT / "external/RegretBench"
REGRETBENCH_SRC = REGRETBENCH_ROOT / "src"
for path in (REPO_ROOT, REGRETBENCH_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from regretbench.schemas.cig import load_cig
from scripts import regretbench_deepseek_dynamic_depth2_policy as policy
from scripts import regretbench_deepseek_result_verify as compare
from scripts import regretbench_deepseek_smc_dynamic_depth2_policy as transition
from scripts import regretbench_deepseek_support_recovery as source_tools


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-factorized-v2-exact10-smoke-verify-1"
PRODUCER_INTERFACE = "regretbench-factorized-v2-exact10-smoke-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
EXPECTED_REQUESTS = 10
MAX_RETRIES = 4
RUN_BUDGET_USD = 0.20
ROOT_SEED_START = 202608510000
TRANSITION_SEED_START = 202608511000
STATIC_SEED_START = 202608512000
SOURCE_PROTOCOL_SHA256 = "151751b576e2db14a6e21848d7d548a09f92172a6709e476ed903d41fea8c753"
SOURCE_MANIFEST_SHA256 = "831a8bcf8f38b183c080c5a369896366915ca8b8ada38294143d7f31a11c1999"
SOURCE_RESULT_SHA256 = "54fc24022ccce75f5865173edf0049392862c413813fdd0e7884fbfc55e537a7"
SMOKE_PROTOCOL_SHA256 = "3901343d7633e056b450a778b9077f218866567bb6fe5a16c6c81c851f85f3db"
MECHANICS_SPLIT_SHA256 = "93fdbfb55d0ded5fdbed1cfa78f251a863590f07ae2eaece2e20bdfc2d01a353"
SOURCE_MANIFEST = REPO_ROOT / "results/nonmyopic/regretbench_factorized_v2_source_audit/SOURCE_PROTOCOL_MANIFEST.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cigs():
    manifest = _load(SOURCE_MANIFEST)
    ids = manifest["splits"]["mechanics"]["ids"][:2]
    return [
        load_cig(REGRETBENCH_ROOT / "data/OpenDomainQA/test" / f"{cig_id}.json")
        for cig_id in ids
    ]


def _public_children(child: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = child.get("hypotheses")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("child support must contain eight particles")
    return [
        {
            "particle_index": index,
            "interpretation": str(row["interpretation"]).strip(),
            "final_answer": str(row["final_answer"]).strip(),
        }
        for index, row in enumerate(rows)
    ]


def _structural_hash(particles: Any, questions: Any) -> str:
    return hashlib.sha256(
        _canonical({"particles": particles, "questions": questions}).encode()
    ).hexdigest()


def _static_audit(cig: Any, child: Mapping[str, Any], conditioning_question: str):
    particles = _public_children(child)
    questions = [conditioning_question, *transition._validate_questions(list(child["questions"]))]
    structural = _structural_hash(particles, questions)
    payload = {
        **source_tools.public_payload(cig, []),
        "child_particles": particles,
        "questions": questions,
        "structural_child_sha256": structural,
        "likelihood_factorization": "static_particle_question_only_v2",
        "particle_source": "history_conditioned_children_with_history_removed",
    }
    serialized = _canonical(payload)
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
    return {
        "passed": source_tools.privacy_audit(cig, source_tools.public_payload(cig, []))["passed"] is True
        and not any(f'"{key}"' in serialized for key in forbidden),
        "payload_keys": sorted(payload),
        "payload_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "structural_child_sha256": structural,
        "dialogue_excluded": payload["dialogue"] == [],
        "observed_answer_excluded": True,
        "particle_probabilities_excluded": True,
        "lineage_metadata_excluded": True,
        "old_predicted_replies_excluded": True,
        "hidden_truth_exposed": False,
    }


def _parse_static(raw: str, child: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"particles"}:
        raise ValueError("static annotation has wrong top-level fields")
    rows = value["particles"]
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("static annotation must contain eight particles")
    replies_by_index = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"particle_index", "predicted_replies"}:
            raise ValueError("static annotation particle has wrong fields")
        index = row["particle_index"]
        replies = row["predicted_replies"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in range(8)
            or index in replies_by_index
            or not isinstance(replies, list)
            or len(replies) != 5
            or any(not isinstance(reply, str) or not reply.strip() for reply in replies)
        ):
            raise ValueError("static annotation values are invalid")
        replies_by_index[index] = [reply.strip() for reply in replies]
    if sorted(replies_by_index) != list(range(8)):
        raise ValueError("static particle indexes are not an exact permutation")
    hypotheses = [
        {
            **{key: item for key, item in row.items() if key != "predicted_replies"},
            "conditioning_reply": replies_by_index[index][0],
            "predicted_replies": replies_by_index[index][1:],
        }
        for index, row in enumerate(child["hypotheses"])
    ]
    return {
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


def _condition(support: Mapping[str, Any], observed_reply: str):
    observed = source_tools.normalize_text(observed_reply)
    matches = [
        index
        for index, row in enumerate(support["hypotheses"])
        if source_tools.normalize_text(row["conditioning_reply"]) == observed
    ]
    if not matches:
        raise ValueError("observed reply has no static child likelihood")
    mass = sum(float(support["hypotheses"][index]["probability"]) for index in matches)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("observed reply has zero child prior mass")
    hypotheses = [
        {
            **row,
            "probability": float(row["probability"]) / mass if index in matches else 0.0,
        }
        for index, row in enumerate(support["hypotheses"])
    ]
    predictive = sum(hypotheses[index]["probability"] for index in matches)
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
    diagnostic = {
        "matching_particle_indexes": matches,
        "precondition_matching_mass": mass,
        "posterior_predictive_matched_reply": predictive,
        "finite_normalized_weights": math.isfinite(predictive)
        and abs(sum(row["probability"] for row in hypotheses) - 1.0) <= 1e-12,
    }
    return conditioned, diagnostic


def verify_smoke(run_dir: Path) -> dict[str, Any]:
    result = _load(run_dir / "RESULT.json")
    raw = _load(run_dir / "private/RAW_RESPONSES.json")
    saved_privacy = _load(run_dir / "private/PRIVACY.json")
    cigs = _cigs()
    root_seeds = [ROOT_SEED_START, ROOT_SEED_START + 1]
    transition_seeds = [
        TRANSITION_SEED_START,
        TRANSITION_SEED_START,
        TRANSITION_SEED_START + 1,
        TRANSITION_SEED_START + 1,
    ]
    static_seeds = [
        STATIC_SEED_START,
        STATIC_SEED_START,
        STATIC_SEED_START + 1,
        STATIC_SEED_START + 1,
    ]
    if raw.get("root_seeds") != root_seeds or raw.get("transition_seeds") != transition_seeds or raw.get("static_seeds") != static_seeds:
        raise ValueError("factorized-v2 seed schedule changed")
    raw_roots = raw.get("roots") or []
    raw_transitions = raw.get("transitions") or []
    raw_static = raw.get("static") or []
    if not (len(raw_roots) == 2 and len(raw_transitions) == len(raw_static) == 4):
        raise ValueError("factorized-v2 response schedule changed")
    roots = [policy.parse_enriched_support(item) for item in raw_roots]
    transitions = [
        transition.parse_enriched_transition(raw_transitions[2 * task + arm], roots[task])
        for task in range(2)
        for arm in range(2)
    ]
    factorized = [
        _parse_static(item, child)
        for item, child in zip(raw_static, transitions, strict=True)
    ]

    privacy = []
    for cig in cigs:
        _, root_audit = policy.messages_for(cig, [])
        privacy.append(root_audit)
    transition_privacy = []
    first_mappings = []
    truths = []
    for task, (cig, root) in enumerate(zip(cigs, roots, strict=True)):
        _, truth = source_tools.sample_truth(cig, ROOT_SEED_START + 10_000 + task)
        truths.append(truth)
        first = source_tools.map_and_answer(cig, root["questions"][0], truth)
        first_mappings.append(first)
        dialogue = [
            {"role": "assistant", "content": root["questions"][0]},
            {"role": "user", "content": first["answer"]},
        ]
        _, conditioned_audit = transition.transition_messages_for(cig, dialogue, root)
        _, blind_audit = transition.transition_messages_for(cig, [], root)
        transition_privacy.extend([conditioned_audit, blind_audit])
    privacy.extend(transition_privacy)
    for task, cig in enumerate(cigs):
        for arm in range(2):
            privacy.append(_static_audit(cig, transitions[2 * task + arm], roots[task]["questions"][0]))

    conditioning = []
    informative = []
    second_mappings = []
    second_matches = []
    novel_actions = []
    for task, (cig, truth, first) in enumerate(zip(cigs, truths, first_mappings, strict=True)):
        conditioned, diagnostic = _condition(factorized[2 * task], first["answer"])
        conditioning.append(diagnostic)
        informative.extend(
            [
                sum(policy.question_eig(support, index) > 1e-12 for index in range(4))
                for support in (conditioned, factorized[2 * task + 1])
            ]
        )
        second_index = policy.select_question(conditioned)
        second = source_tools.map_and_answer(cig, conditioned["questions"][second_index], truth)
        second_mappings.append(second)
        second_matches.append(bool(policy.matching_reply_indexes(conditioned, second_index, second["answer"])))
        novel_actions.append(policy.distinct_supported_actions(first, second))

    usage = result.get("usage") or {}
    accepted = usage.get("adapter_requests")
    attempts = usage.get("http_attempts")
    retries = usage.get("retry_count")
    provider = usage.get("provider_error_retries")
    gates = {
        "exact_ten_accepted_requests": accepted == EXPECTED_REQUESTS,
        "http_attempts_between_ten_and_fourteen": isinstance(attempts, int) and EXPECTED_REQUESTS <= attempts <= EXPECTED_REQUESTS + MAX_RETRIES,
        "attempts_equal_accepted_plus_retries": attempts == accepted + retries,
        "retries_within_four": isinstance(retries, int) and 0 <= retries <= MAX_RETRIES,
        "provider_retries_are_subset": isinstance(provider, int) and 0 <= provider <= retries,
        "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "exact_two_root_four_transition_four_static": len(roots) == 2 and len(transitions) == 4 and len(factorized) == 4,
        "all_roots_have_eight_unique_particles": all(row["diagnostic"]["valid_unique_count"] == 8 for row in roots),
        "both_roots_have_two_informative_questions": all(row["diagnostic"]["informative_question_count"] >= 2 for row in roots),
        "all_transitions_have_exact_lineage_and_retention": all(row["diagnostic"]["parent_index_permutation_exact"] is True and transition.MIN_RETAINED <= row["diagnostic"]["retained_count"] <= transition.MAX_RETAINED for row in transitions),
        "all_static_annotations_preserve_children": all(row["diagnostic"]["history_conditioned_child_particles_preserved"] is True and row["diagnostic"]["updated_likelihood_replies_discarded"] is True and row["diagnostic"]["static_annotation_index_permutation_exact"] is True and row["diagnostic"]["static_reply_count_per_particle"] == 5 for row in factorized),
        "both_first_answers_mapper_supported": all(row["supported"] for row in first_mappings),
        "both_conditioned_answers_have_static_matches": all(bool(row["matching_particle_indexes"]) for row in conditioning),
        "exact_conditioning_is_finite_and_normalized": all(row["finite_normalized_weights"] and abs(row["posterior_predictive_matched_reply"] - 1.0) <= 1e-12 for row in conditioning),
        "every_child_has_informative_future_question": all(value >= 1 for value in informative),
        "both_second_questions_mapper_supported": all(row["supported"] for row in second_mappings),
        "both_conditioned_second_actions_are_novel": all(novel_actions),
        "both_second_replies_have_positive_static_likelihood": all(second_matches),
        "transition_seed_pairing_exact": raw.get("transition_seeds") == transition_seeds,
        "static_seed_pairing_exact": raw.get("static_seeds") == static_seeds,
        "all_privacy_audits_pass": saved_privacy.get("audits") == privacy and len(privacy) == EXPECTED_REQUESTS and all(row["passed"] is True for row in privacy),
        "within_smoke_budget": float(usage.get("run_cost_usd", math.inf)) <= RUN_BUDGET_USD + 1e-12,
    }
    gates["all_pass"] = all(gates.values())
    expected_status = "passed" if gates["all_pass"] else "mechanics_failed"
    expected_authorizes = "separate_factorized_v2_policy_preregistration_only" if expected_status == "passed" else "nothing"
    diagnostics = {
        "roots": [row["diagnostic"] for row in roots],
        "transitions": [row["diagnostic"] for row in transitions],
        "factorized": [row["diagnostic"] for row in factorized],
        "conditioning": conditioning,
        "informative_future_question_counts": informative,
    }
    mismatches: list[str] = []
    compare._close(result.get("interface_version"), PRODUCER_INTERFACE, "$.interface_version", mismatches)
    compare._close(result.get("status"), expected_status, "$.status", mismatches)
    compare._close(result.get("authorizes"), expected_authorizes, "$.authorizes", mismatches)
    compare._close(result.get("gates"), gates, "$.gates", mismatches)
    compare._close(result.get("diagnostics"), diagnostics, "$.diagnostics", mismatches)
    protocol = result.get("protocol") or {}
    expected_bindings = {
        "source_protocol": SOURCE_PROTOCOL_SHA256,
        "source_manifest": SOURCE_MANIFEST_SHA256,
        "source_result": SOURCE_RESULT_SHA256,
        "smoke_protocol": SMOKE_PROTOCOL_SHA256,
    }
    for key, expected in {
        "model": MODEL_ID,
        "reasoning": "disabled_excluded",
        "expected_accepted_requests": EXPECTED_REQUESTS,
        "maximum_http_attempts": EXPECTED_REQUESTS + MAX_RETRIES,
        "bindings": expected_bindings,
        "mechanics_split_sha256": MECHANICS_SPLIT_SHA256,
        "efficacy_accessed": False,
        "development_opened": False,
        "confirmation_opened": False,
    }.items():
        compare._close(protocol.get(key), expected, f"$.protocol.{key}", mismatches)
    artifacts = {
        str(path.relative_to(run_dir)): _sha256_file(path)
        for path in sorted(run_dir.rglob("*.json"))
        if path.name != "VERIFICATION.json"
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "verified" if not mismatches else "verification_failed",
        "kind": "factorized_v2_exact10_smoke",
        "model_calls": 0,
        "cost_usd": 0.0,
        "result_status": result.get("status"),
        "checks": {
            "roots_reparsed": True,
            "lineages_reconstructed": True,
            "static_likelihoods_reparsed": True,
            "exact_conditioning_replayed": True,
            "privacy_recomputed": True,
            "truth_controls_replayed": True,
            "reported_result_matches_replay": not mismatches,
        },
        "mismatches": mismatches,
        "artifact_sha256": artifacts,
    }
