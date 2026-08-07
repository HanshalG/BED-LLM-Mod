#!/usr/bin/env python3
"""Independently replay a RegretBench factorized exact-10 smoke artifact."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from scripts import regretbench_deepseek_result_verify as base
from scripts import regretbench_deepseek_smc_dynamic_depth2_verify as smc


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-factorized-static-likelihood-smoke-verify-1"
PRODUCER_INTERFACE = "regretbench-factorized-static-likelihood-smoke-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
PROTOCOL_SHA256 = "bfd7257295615bc0e80f5205709d95599455e44bd944faacd4907c96f99aa4d3"
INITIAL_SEED_START = 202608420000
TRANSITION_SEED_START = 202608421000
STATIC_SEED_START = 202608422000
PRIMARY_SMOKE_TRUTH_SEED_START = 202608081100
EXPECTED_REQUESTS = 10
SMOKE_BUDGET_USD = 0.20


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_children(support: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = support.get("hypotheses")
    if not isinstance(rows, list) or len(rows) != smc.PARTICLES:
        raise ValueError("factorized child support must contain eight particles")
    output = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("factorized child is not an object")
        interpretation = row.get("interpretation")
        answer = row.get("final_answer")
        if not isinstance(interpretation, str) or not interpretation.strip():
            raise ValueError("factorized child interpretation is empty")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("factorized child answer is empty")
        output.append(
            {
                "particle_index": index,
                "interpretation": interpretation.strip(),
                "final_answer": answer.strip(),
            }
        )
    return output


def _structural_hash(
    particles: list[dict[str, Any]], questions: list[str]
) -> str:
    payload = {"particles": particles, "questions": questions}
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _parse_static(raw: str, child: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"particles"}:
        raise ValueError("static annotation has wrong top-level fields")
    rows = value["particles"]
    if not isinstance(rows, list) or len(rows) != smc.PARTICLES:
        raise ValueError("static annotation must contain eight particles")
    by_index: dict[int, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "particle_index",
            "predicted_replies",
        }:
            raise ValueError("static annotation particle has wrong fields")
        index = row["particle_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in range(smc.PARTICLES)
            or index in by_index
        ):
            raise ValueError("static annotation indexes are not an exact permutation")
        by_index[index] = smc._replies(row["predicted_replies"])
    if sorted(by_index) != list(range(smc.PARTICLES)):
        raise ValueError("static annotation indexes are not an exact permutation")

    particles = _public_children(child)
    questions = smc._questions(child.get("questions"))
    source_hash = smc._support_hash(child)
    hypotheses = [
        {**row, "predicted_replies": by_index[index]}
        for index, row in enumerate(child["hypotheses"])
    ]
    diagnostic = dict(child.get("diagnostic") or {})
    diagnostic.update(
        {
            "likelihood_factorization": "static_particle_question_only",
            "history_conditioned_child_particles_preserved": True,
            "updated_likelihood_replies_discarded": True,
            "static_annotation_index_permutation_exact": True,
            "source_child_support_sha256": source_hash,
            "structural_child_sha256": _structural_hash(particles, questions),
        }
    )
    return {
        "hypotheses": hypotheses,
        "questions": questions,
        "diagnostic": diagnostic,
    }


def _static_audit(cig: Any, child: Mapping[str, Any]) -> dict[str, Any]:
    particles = _public_children(child)
    questions = smc._questions(child.get("questions"))
    structural_hash = _structural_hash(particles, questions)
    payload = {
        "task_id": cig.cig_id,
        "prompt": cig.prompt,
        "dialogue": [],
        "child_particles": particles,
        "questions": questions,
        "structural_child_sha256": structural_hash,
        "likelihood_factorization": "static_particle_question_only",
        "particle_source": "history_conditioned_children_with_history_removed",
    }
    return {
        "passed": True,
        "payload_keys": sorted(payload),
        "payload_sha256": hashlib.sha256(_canonical(payload).encode()).hexdigest(),
        "structural_child_sha256": structural_hash,
        "dialogue_excluded": True,
        "particle_probabilities_excluded": True,
        "lineage_metadata_excluded": True,
        "updated_predicted_replies_excluded": True,
        "hidden_truth_exposed": False,
    }


def _informative(support: Mapping[str, Any]) -> int:
    return sum(
        base._question_eig(support, index) > 1e-12
        for index in range(smc.QUESTIONS)
    )


def verify_smoke(
    run_dir: Path,
    *,
    primary_dir: Path,
    expected_predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay raw responses without importing the factorized producer or core."""

    result = _load(run_dir / "RESULT.json")
    raw = _load(run_dir / "private/RAW_RESPONSES.json")
    privacy = _load(run_dir / "private/PRIVACY.json")
    primary_raw = _load(primary_dir / "private/RAW_RESPONSES.json")
    primary_controls = _load(primary_dir / "private/CONTROLS.json")
    cigs = base._stage_cigs("smoke")[:2]
    raw_parents = (primary_raw.get("root") or [])[:2]
    raw_initial = raw.get("initial") or []
    raw_transitions = raw.get("transitions") or []
    raw_static = raw.get("static") or []
    if not (
        len(cigs) == len(raw_parents) == len(raw_initial) == 2
        and len(raw_transitions) == len(raw_static) == 4
    ):
        raise ValueError("factorized smoke response schedule changed")

    initial_seeds = [INITIAL_SEED_START + index for index in range(2)]
    transition_seeds = [
        seed for index in range(2) for seed in (TRANSITION_SEED_START + index,) * 2
    ]
    static_seeds = [
        seed for index in range(2) for seed in (STATIC_SEED_START + index,) * 2
    ]
    if raw.get("initial_seeds") != initial_seeds:
        raise ValueError("factorized initial seed schedule changed")
    if raw.get("transition_seeds") != transition_seeds:
        raise ValueError("factorized transition seed schedule changed")
    if raw.get("static_seeds") != static_seeds:
        raise ValueError("factorized static seed schedule changed")

    parents = [smc._parse_parent(value) for value in raw_parents]
    initial = [
        smc._parse_annotation(value, parent)
        for value, parent in zip(raw_initial, parents, strict=True)
    ]
    transitions = [
        smc._parse_transition(raw_transitions[2 * task + arm], initial[task])
        for task in range(2)
        for arm in range(2)
    ]
    factorized = [
        _parse_static(value, child)
        for value, child in zip(raw_static, transitions, strict=True)
    ]

    control_by_id = {
        row["task_id"]: row for row in primary_controls.get("roots", [])
    }
    annotation_audits = [
        smc._annotation_audit(cig, parent)
        for cig, parent in zip(cigs, parents, strict=True)
    ]
    transition_audits = []
    static_audits = []
    first_mappings = []
    second_mappings = []
    second_matches = []
    conditioned_novel = []
    for task, cig in enumerate(cigs):
        control = control_by_id.get(cig.cig_id)
        if control is None:
            raise ValueError("factorized parent control is missing")
        truth_index, truth = base._truth(cig, PRIMARY_SMOKE_TRUTH_SEED_START + task)
        first = base._map(cig, initial[task]["questions"][0], truth)
        if (
            control.get("truth_index") != truth_index
            or control.get("question") != initial[task]["questions"][0]
            or base._close(control.get("mapping"), first)
        ):
            raise ValueError("factorized parent control changed")
        first_mappings.append(first)
        dialogue = [
            {"role": "assistant", "content": initial[task]["questions"][0]},
            {"role": "user", "content": first["answer"]},
        ]
        transition_audits.extend(
            [
                smc._transition_audit(cig, dialogue, initial[task]),
                smc._transition_audit(cig, [], initial[task]),
            ]
        )
        for arm in range(2):
            child = transitions[2 * task + arm]
            support = factorized[2 * task + arm]
            static_audits.append(_static_audit(cig, child))
            second_index = base._select_question(support)
            second = base._map(cig, support["questions"][second_index], truth)
            second_mappings.append(second)
            second_matches.append(
                bool(base._reply_indexes(support, second_index, second["answer"]))
            )
            if arm == 0:
                conditioned_novel.append(
                    first["supported"]
                    and second["supported"]
                    and first["facet"] is not None
                    and second["facet"] is not None
                    and first["facet"] != second["facet"]
                )

    audits = annotation_audits + transition_audits + static_audits
    usage = result.get("usage") or {}
    gates = {
        "exact_ten_parsed_responses": len(initial) + len(transitions) + len(factorized)
        == EXPECTED_REQUESTS,
        "exact_two_initial_four_transition_four_static": len(initial) == 2
        and len(transitions) == 4
        and len(factorized) == 4,
        "exact_ten_requests": usage.get("adapter_requests") == EXPECTED_REQUESTS,
        "exact_ten_http_attempts": usage.get("http_attempts") == EXPECTED_REQUESTS,
        "zero_retries": usage.get("retry_count") == 0,
        "zero_provider_error_retries": usage.get("provider_error_retries") == 0,
        "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "all_initial_annotations_exact": all(
            row["diagnostic"]["parent_index_permutation_exact"] is True
            and row["diagnostic"]["initial_hypotheses_regenerated"] is False
            for row in initial
        ),
        "all_transitions_exact_lineage_and_retention": all(
            row["diagnostic"]["parent_index_permutation_exact"] is True
            and smc.MIN_RETAINED <= row["diagnostic"]["retained_count"] <= smc.MAX_RETAINED
            for row in transitions
        ),
        "all_static_annotations_preserve_children": all(
            row["diagnostic"]["history_conditioned_child_particles_preserved"] is True
            and row["diagnostic"]["updated_likelihood_replies_discarded"] is True
            and row["diagnostic"]["static_annotation_index_permutation_exact"] is True
            for row in factorized
        ),
        "every_initial_has_two_informative_roots": all(_informative(row) >= 2 for row in initial),
        "every_factorized_child_has_informative_followup": all(_informative(row) >= 1 for row in factorized),
        "both_first_questions_supported": all(row["supported"] for row in first_mappings),
        "all_four_second_questions_supported": all(row["supported"] for row in second_mappings),
        "both_conditioned_second_actions_novel": all(conditioned_novel),
        "all_four_exact_second_replies_have_factorized_likelihood": all(second_matches),
        "transition_pairing_exact": raw.get("transition_seeds") == transition_seeds,
        "static_pairing_exact": raw.get("static_seeds") == static_seeds,
        "all_privacy_audits_pass": privacy.get("audits") == audits,
        "within_smoke_budget": float(usage.get("run_cost_usd", math.inf))
        <= SMOKE_BUDGET_USD + 1e-12,
    }
    gates["all_pass"] = all(gates.values())
    expected_status = "passed" if gates["all_pass"] else "mechanics_failed"
    expected_authorizes = (
        "separate_factorized_policy_preregistration_only"
        if expected_status == "passed"
        else "nothing"
    )
    diagnostics = {
        "initial": [row["diagnostic"] for row in initial],
        "transitions": [row["diagnostic"] for row in transitions],
        "factorized": [row["diagnostic"] for row in factorized],
    }
    mismatches: list[str] = []
    base._close(result.get("interface_version"), PRODUCER_INTERFACE, "$.interface_version", mismatches)
    base._close(result.get("status"), expected_status, "$.status", mismatches)
    base._close(result.get("authorizes"), expected_authorizes, "$.authorizes", mismatches)
    base._close(result.get("gates"), gates, "$.gates", mismatches)
    base._close(result.get("diagnostics"), diagnostics, "$.diagnostics", mismatches)
    protocol = result.get("protocol") or {}
    for key, expected in {
        "model": MODEL_ID,
        "reasoning": "disabled_excluded",
        "expected_requests": EXPECTED_REQUESTS,
        "protocol_sha256": PROTOCOL_SHA256,
        "predecessor": dict(expected_predecessor),
        "efficacy_accessed": False,
        "development_opened": False,
        "confirmation_opened": False,
        "paid_execution_authorized_by_implementation": False,
    }.items():
        base._close(protocol.get(key), expected, f"$.protocol.{key}", mismatches)

    artifacts = {
        str(path.relative_to(run_dir)): _sha256_file(path)
        for path in sorted(run_dir.rglob("*.json"))
        if path.name != "VERIFICATION.json"
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "verified" if not mismatches else "verification_failed",
        "kind": "factorized_static_likelihood_smoke",
        "model_calls": 0,
        "cost_usd": 0.0,
        "result_status": result.get("status"),
        "checks": {
            "banked_parents_reparsed": True,
            "annotations_reparsed": True,
            "lineages_reconstructed": True,
            "static_likelihoods_reparsed": True,
            "privacy_recomputed": True,
            "truth_controls_replayed": True,
            "reported_result_matches_replay": not mismatches,
        },
        "mismatches": mismatches,
        "artifact_sha256": artifacts,
    }
