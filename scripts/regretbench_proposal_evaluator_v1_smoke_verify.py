#!/usr/bin/env python3
"""Independently replay a proposal-evaluator v1 exact-20 smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
REGRETBENCH_ROOT = REPO_ROOT / "external/RegretBench"
REGRETBENCH_SRC = REGRETBENCH_ROOT / "src"
for path in (REPO_ROOT, REGRETBENCH_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from regretbench.mapping.semantic_action_mapper import SemanticActionMapper
from regretbench.schemas.cig import CIG, load_cig
from scripts import regretbench_deepseek_support_recovery as source_tools


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-proposal-evaluator-v1-independent-verifier-1"
PRODUCER_INTERFACE = "regretbench-proposal-evaluator-v1-exact20-smoke-1"
EXPECTED_REQUESTS = 20
MAX_RETRIES = 4
HYPOTHESES = 8
ROOT_PROPOSAL_SEED_START = 202608530000
ROOT_EVALUATOR_SEED_START = 202608531000
BRANCH_PROPOSAL_SEED_START = 202608532000
BRANCH_EVALUATOR_SEED_START = 202608533000
MIN_BRANCH_MASS = 0.10
SOURCE_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_PROPOSAL_EVALUATOR_V1_SOURCE_PROTOCOL_20260811.md"
)
SOURCE_PROTOCOL_SHA256 = (
    "9edb0a7828dfa74b660bea02fc80c50db05be4b6df571efe0911693ee41c190e"
)
SMOKE_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_PROPOSAL_EVALUATOR_V1_EXACT20_SMOKE_PROTOCOL_20260811.md"
)
SMOKE_PROTOCOL_SHA256 = (
    "c56939f2f45c1da2a5dd2dc4e6ab8fd7921cf3033febe085a4d3a0933cd5583b"
)
SOURCE_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/regretbench_proposal_evaluator_v1_source_audit/"
    "SOURCE_TASK_MANIFEST.json"
)
SOURCE_MANIFEST_SHA256 = (
    "d3c92a4edecc39b6524790f0e46b813ae8c04ad8507ac67535a95680b9afe7ad"
)
SOURCE_RESULT = REPO_ROOT / (
    "results/nonmyopic/regretbench_proposal_evaluator_v1_source_audit/RESULT.json"
)
SOURCE_RESULT_SHA256 = (
    "e3e627a69578f19671e57d9e0a731c8c1eb30d53984c5f238c6a83f91bc93be9"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def structural_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_bindings() -> None:
    for name, path, digest in (
        ("source protocol", SOURCE_PROTOCOL, SOURCE_PROTOCOL_SHA256),
        ("smoke protocol", SMOKE_PROTOCOL, SMOKE_PROTOCOL_SHA256),
        ("source manifest", SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256),
        ("source result", SOURCE_RESULT, SOURCE_RESULT_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"independent verifier {name} changed")
    source = load_object(SOURCE_RESULT)
    if (
        source.get("status") != "source_protocol_pass"
        or source.get("authorizes")
        != "proposal_evaluator_v1_exact20_smoke_only"
        or source.get("gates", {}).get("all_pass") is not True
    ):
        raise ValueError("independent verifier source gate is not passed")


def load_cigs() -> list[CIG]:
    manifest = load_object(SOURCE_MANIFEST)
    rows = manifest.get("successor_smoke_tasks", [])
    if len(rows) != 2:
        raise ValueError("independent verifier task count changed")
    cigs = []
    for row in rows:
        path = REGRETBENCH_ROOT / f"data/OpenDomainQA/test/{row['task_id']}.json"
        if sha256_file(path) != row["task_file_sha256"]:
            raise ValueError("independent verifier task file changed")
        cigs.append(load_cig(path))
    return cigs


def parse_proposal(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("independent proposal has wrong top-level fields")
    raw_hypotheses = value["hypotheses"]
    if not isinstance(raw_hypotheses, list) or len(raw_hypotheses) != HYPOTHESES:
        raise ValueError("independent proposal does not have eight hypotheses")
    hypotheses = []
    keys = set()
    for row in raw_hypotheses:
        if not isinstance(row, dict) or set(row) != {"interpretation", "final_answer"}:
            raise ValueError("independent proposal hypothesis fields changed")
        interpretation = row["interpretation"]
        answer = row["final_answer"]
        if (
            not isinstance(interpretation, str)
            or not interpretation.strip()
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ValueError("independent proposal hypothesis is invalid")
        key = (
            source_tools.normalize_text(interpretation),
            source_tools.normalize_text(answer),
        )
        if key in keys:
            raise ValueError("independent proposal hypotheses are duplicated")
        keys.add(key)
        hypotheses.append(
            {"interpretation": interpretation.strip(), "final_answer": answer.strip()}
        )
    questions = []
    question_keys = set()
    if not isinstance(value["questions"], list) or len(value["questions"]) != 4:
        raise ValueError("independent proposal does not have four questions")
    for question in value["questions"]:
        if not isinstance(question, str) or not question.strip().endswith("?"):
            raise ValueError("independent proposal question is invalid")
        cleaned = question.strip()
        key = source_tools.normalize_text(cleaned)
        if key in question_keys:
            raise ValueError("independent proposal questions are duplicated")
        question_keys.add(key)
        questions.append(cleaned)
    public = {"hypotheses": hypotheses, "questions": questions}
    return {
        **public,
        "structural_sha256": structural_hash(public),
    }


def parse_evaluation(
    raw: str, proposal: Mapping[str, Any], questions: Sequence[str]
) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"particles"}:
        raise ValueError("independent evaluator top-level fields changed")
    rows = value["particles"]
    if not isinstance(rows, list) or len(rows) != HYPOTHESES:
        raise ValueError("independent evaluator does not have eight particles")
    by_index = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "particle_index",
            "prior_weight",
            "predicted_replies",
        }:
            raise ValueError("independent evaluator particle fields changed")
        index = row["particle_index"]
        weight = row["prior_weight"]
        replies = row["predicted_replies"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in range(HYPOTHESES)
            or index in by_index
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
            or not isinstance(replies, list)
            or len(replies) != len(questions)
            or any(not isinstance(reply, str) or not reply.strip() for reply in replies)
        ):
            raise ValueError("independent evaluator particle is invalid")
        by_index[index] = (float(weight), [reply.strip() for reply in replies])
    if set(by_index) != set(range(HYPOTHESES)):
        raise ValueError("independent evaluator index permutation changed")
    total = sum(weight for weight, _ in by_index.values())
    if total <= 0:
        raise ValueError("independent evaluator has zero prior mass")
    hypotheses = []
    for index, original in enumerate(proposal["hypotheses"]):
        weight, replies = by_index[index]
        hypotheses.append(
            {
                "interpretation": original["interpretation"],
                "final_answer": original["final_answer"],
                "probability": weight / total,
                "predicted_replies": replies,
            }
        )
    return {"hypotheses": hypotheses, "questions": list(questions)}


def entropy(values: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in values if value > 0)


def eig(support: Mapping[str, Any], question_index: int) -> float:
    masses: dict[str, float] = {}
    for row in support["hypotheses"]:
        key = source_tools.normalize_text(row["predicted_replies"][question_index])
        masses[key] = masses.get(key, 0.0) + float(row["probability"])
    return entropy(list(masses.values()))


def action(cig: CIG, question: str) -> dict[str, Any]:
    parsed = SemanticActionMapper().map_question(cig, question)
    supported = parsed.facet is not None and parsed.semantic_action != "UNSUPPORTED"
    return {"supported": supported, "facet": parsed.facet if supported else None}


def select_root(cig: CIG, support: Mapping[str, Any]) -> dict[str, Any]:
    actions = [action(cig, question) for question in support["questions"]]
    indexes = [index for index, row in enumerate(actions) if row["supported"]]
    if not indexes:
        raise ValueError("independent root has no supported question")
    index = min(indexes, key=lambda item: (-eig(support, item), item))
    return {
        "index": index,
        "question": support["questions"][index],
        "eig": eig(support, index),
        "facet": actions[index]["facet"],
        "supported_count": len(indexes),
    }


def grouped_replies(support: Mapping[str, Any], question_index: int):
    groups: dict[str, dict[str, Any]] = {}
    for row in support["hypotheses"]:
        reply = row["predicted_replies"][question_index]
        key = source_tools.normalize_text(reply)
        groups.setdefault(key, {"key": key, "reply": reply, "mass": 0.0})
        groups[key]["mass"] += float(row["probability"])
    return sorted(groups.values(), key=lambda row: (-row["mass"], row["key"]))


def source_reply_real(cig: CIG, facet: str, reply: str) -> bool:
    target = source_tools.normalize_text(reply)
    values = {
        source_tools.normalize_text(str((intent.slots or {}).get(facet, "")))
        for intent in cig.intents
    }
    values.discard("")
    return target in values


def mass(support: Mapping[str, Any], question_index: int, reply: str) -> float:
    target = source_tools.normalize_text(reply)
    return sum(
        float(row["probability"])
        for row in support["hypotheses"]
        if source_tools.normalize_text(row["predicted_replies"][question_index])
        == target
    )


def condition(support: Mapping[str, Any], question_index: int, reply: str):
    target = source_tools.normalize_text(reply)
    indexes = [
        index
        for index, row in enumerate(support["hypotheses"])
        if source_tools.normalize_text(row["predicted_replies"][question_index])
        == target
    ]
    prior = sum(float(support["hypotheses"][index]["probability"]) for index in indexes)
    if not indexes or prior <= 0:
        raise ValueError("independent exact conditioning has zero mass")
    result = copy.deepcopy(support)
    for index, row in enumerate(result["hypotheses"]):
        row["probability"] = float(row["probability"]) / prior if index in indexes else 0.0
    return result, {
        "matching_particle_count": len(indexes),
        "prior_predictive_mass": prior,
        "posterior_probability_sum": sum(row["probability"] for row in result["hypotheses"]),
        "posterior_predictive_matched_reply": mass(result, question_index, reply),
    }


def select_child(cig: CIG, support: Mapping[str, Any], root_facet: str):
    actions = [action(cig, question) for question in support["questions"]]
    indexes = [
        index
        for index, row in enumerate(actions)
        if index > 0 and row["supported"] and row["facet"] != root_facet
    ]
    if not indexes:
        raise ValueError("independent child has no novel supported question")
    index = min(indexes, key=lambda item: (-eig(support, item), item))
    return {"index": index, "eig": eig(support, index), "facet": actions[index]["facet"]}


def proposal_payload(cig: CIG, dialogue: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    return source_tools.public_payload(cig, dialogue)


def evaluator_payload(
    cig: CIG, proposal: Mapping[str, Any], questions: Sequence[str]
) -> dict[str, Any]:
    public = {
        "hypotheses": proposal["hypotheses"],
        "questions": proposal["questions"],
    }
    return {
        "task_id": cig.cig_id,
        "prompt": cig.prompt,
        "particles": [
            {
                "particle_index": index,
                "interpretation": row["interpretation"],
                "final_answer": row["final_answer"],
            }
            for index, row in enumerate(proposal["hypotheses"])
        ],
        "questions": list(questions),
        "proposal_sha256": structural_hash(public),
        "source": "model_generated_semantic_proposal",
    }


def _privacy_hashes(
    cigs: Sequence[CIG],
    roots: Sequence[Mapping[str, Any]],
    root_selected: Sequence[Mapping[str, Any]],
    branches: Sequence[Sequence[Mapping[str, Any]]],
    proposals: Sequence[Mapping[str, Any]],
) -> list[str]:
    hashes = [structural_hash(proposal_payload(cig, [])) for cig in cigs]
    hashes.extend(
        structural_hash(evaluator_payload(cig, proposal, proposal["questions"]))
        for cig, proposal in zip(cigs, roots, strict=True)
    )
    for task, cig in enumerate(cigs):
        for branch in range(2):
            reply = branches[task][branch]["reply"]
            dialogue = [
                {"role": "assistant", "content": root_selected[task]["question"]},
                {"role": "user", "content": reply},
            ]
            hashes.extend(
                [
                    structural_hash(proposal_payload(cig, dialogue)),
                    structural_hash(proposal_payload(cig, [])),
                ]
            )
    for position, proposal in enumerate(proposals):
        task = position // 4
        questions = [root_selected[task]["question"], *proposal["questions"]]
        hashes.append(structural_hash(evaluator_payload(cigs[task], proposal, questions)))
    return hashes


def replay(run_dir: Path) -> dict[str, Any]:
    validate_bindings()
    raw_path = run_dir / "private/RAW_RESPONSES.json"
    privacy_path = run_dir / "private/PRIVACY.json"
    result_path = run_dir / "RESULT.json"
    raw = load_object(raw_path)
    privacy = load_object(privacy_path)
    result = load_object(result_path)
    expected_raw_keys = {
        "root_proposal_seeds",
        "root_proposals",
        "root_evaluator_seeds",
        "root_evaluations",
        "branch_proposal_layout",
        "branch_proposal_seeds",
        "branch_proposals",
        "branch_evaluator_layout",
        "branch_evaluator_seeds",
        "branch_evaluations",
    }
    if set(raw) != expected_raw_keys:
        raise ValueError("independent raw bank fields changed")
    cigs = load_cigs()
    roots = [parse_proposal(row) for row in raw["root_proposals"]]
    if len(roots) != 2 or len(raw["root_evaluations"]) != 2:
        raise ValueError("independent root response count changed")
    root_supports = [
        parse_evaluation(row, proposal, proposal["questions"])
        for row, proposal in zip(raw["root_evaluations"], roots, strict=True)
    ]
    selected = [select_root(cig, support) for cig, support in zip(cigs, root_supports, strict=True)]
    branches = []
    for cig, support, root in zip(cigs, root_supports, selected, strict=True):
        groups = grouped_replies(support, root["index"])
        if len(groups) < 2:
            raise ValueError("independent root branches changed")
        chosen = groups[:2]
        if any(row["mass"] < MIN_BRANCH_MASS for row in chosen):
            raise ValueError("independent root branch mass changed")
        if not all(source_reply_real(cig, root["facet"], row["reply"]) for row in chosen):
            raise ValueError("independent root branch source validity changed")
        branches.append(chosen)

    expected_root_proposal_seeds = [ROOT_PROPOSAL_SEED_START + index for index in range(2)]
    expected_root_evaluator_seeds = [ROOT_EVALUATOR_SEED_START + index for index in range(2)]
    expected_branch_seeds = [
        seed
        for task in range(2)
        for branch in range(2)
        for seed in [BRANCH_PROPOSAL_SEED_START + 10 * task + branch] * 2
    ]
    expected_evaluator_seeds = [
        seed
        for task in range(2)
        for branch in range(2)
        for seed in [BRANCH_EVALUATOR_SEED_START + 10 * task + branch] * 2
    ]
    expected_layout = [
        {"task": task, "branch": branch, "arm": arm}
        for task in range(2)
        for branch in range(2)
        for arm in ("conditioned", "answer_free")
    ]
    if (
        raw["root_proposal_seeds"] != expected_root_proposal_seeds
        or raw["root_evaluator_seeds"] != expected_root_evaluator_seeds
        or raw["branch_proposal_seeds"] != expected_branch_seeds
        or raw["branch_evaluator_seeds"] != expected_evaluator_seeds
        or raw["branch_proposal_layout"] != expected_layout
        or raw["branch_evaluator_layout"] != expected_layout
    ):
        raise ValueError("independent seed or layout replay changed")
    proposals = [parse_proposal(row) for row in raw["branch_proposals"]]
    if len(proposals) != 8 or len(raw["branch_evaluations"]) != 8:
        raise ValueError("independent branch response count changed")
    supports = []
    for position, (row, proposal) in enumerate(
        zip(raw["branch_evaluations"], proposals, strict=True)
    ):
        task = position // 4
        supports.append(
            parse_evaluation(row, proposal, [selected[task]["question"], *proposal["questions"]])
        )

    saved_audits = privacy.get("audits", [])
    expected_hashes = _privacy_hashes(cigs, roots, selected, branches, proposals)
    if (
        len(saved_audits) != EXPECTED_REQUESTS
        or [row.get("payload_sha256") for row in saved_audits] != expected_hashes
        or not all(row.get("passed") is True for row in saved_audits)
    ):
        raise ValueError("independent prompt privacy replay changed")

    own_opposite = []
    own_blind = []
    own_masses = []
    action_changes = []
    structural_changes = []
    updates = []
    task_diagnostics = []
    for task, cig in enumerate(cigs):
        task_rows = []
        conditioned_positions = [4 * task, 4 * task + 2]
        blind_positions = [4 * task + 1, 4 * task + 3]
        for branch in range(2):
            reply = branches[task][branch]["reply"]
            own = supports[conditioned_positions[branch]]
            opposite = supports[conditioned_positions[1 - branch]]
            blind = supports[blind_positions[branch]]
            own_mass = mass(own, 0, reply)
            opposite_mass = mass(opposite, 0, reply)
            blind_mass = mass(blind, 0, reply)
            conditioned_posterior, conditioned_update = condition(own, 0, reply)
            blind_posterior, blind_update = condition(blind, 0, reply)
            conditioned_second = select_child(cig, conditioned_posterior, selected[task]["facet"])
            blind_second = select_child(cig, blind_posterior, selected[task]["facet"])
            changed = conditioned_second["facet"] != blind_second["facet"]
            structural = (
                proposals[conditioned_positions[branch]]["structural_sha256"]
                != proposals[blind_positions[branch]]["structural_sha256"]
                and proposals[conditioned_positions[branch]]["structural_sha256"]
                != proposals[conditioned_positions[1 - branch]]["structural_sha256"]
            )
            own_opposite.append(own_mass - opposite_mass)
            own_blind.append(own_mass - blind_mass)
            own_masses.append(own_mass)
            action_changes.append(changed)
            structural_changes.append(structural)
            updates.extend([conditioned_update, blind_update])
            task_rows.append(
                {
                    "branch_index": branch,
                    "root_branch_mass": branches[task][branch]["mass"],
                    "conditioned_own_mass": own_mass,
                    "opposite_conditioned_mass": opposite_mass,
                    "paired_answer_free_mass": blind_mass,
                    "own_minus_opposite": own_mass - opposite_mass,
                    "own_minus_answer_free": own_mass - blind_mass,
                    "conditioned_matching_particles": conditioned_update["matching_particle_count"],
                    "answer_free_matching_particles": blind_update["matching_particle_count"],
                    "conditioned_second_eig": conditioned_second["eig"],
                    "answer_free_second_eig": blind_second["eig"],
                    "selected_child_facet_changed": changed,
                    "conditioned_support_changed_from_controls": structural,
                    "conditioned_proposal_sha256": proposals[conditioned_positions[branch]][
                        "structural_sha256"
                    ],
                    "answer_free_proposal_sha256": proposals[blind_positions[branch]][
                        "structural_sha256"
                    ],
                }
            )
        task_diagnostics.append(
            {
                "task_index": task,
                "root_eig": selected[task]["eig"],
                "root_supported_question_count": selected[task]["supported_count"],
                "root_proposal_sha256": roots[task]["structural_sha256"],
                "branches": task_rows,
            }
        )

    usage = result.get("usage", {})
    transport = {
        "exact_twenty_accepted_requests": usage.get("adapter_requests") == 20,
        "attempts_between_twenty_and_twenty_four": 20
        <= usage.get("http_attempts", -1)
        <= 24,
        "attempts_equal_accepted_plus_retries": usage.get("http_attempts")
        == usage.get("adapter_requests", -2) + usage.get("retry_count", -3),
        "at_most_four_retries": 0 <= usage.get("retry_count", -1) <= 4,
        "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "within_smoke_budget": 0 <= usage.get("run_cost_usd", -1) <= 0.20 + 1e-12,
    }
    own_opposite_mean = sum(own_opposite) / 4
    own_blind_mean = sum(own_blind) / 4
    mechanics = {
        "exact_two_untouched_tasks": len(cigs) == 2,
        "exact_two_root_proposals_and_evaluations": len(roots) == 2 and len(root_supports) == 2,
        "exact_eight_branch_proposals_and_evaluations": len(proposals) == 8 and len(supports) == 8,
        "all_proposals_exclude_weights_likelihoods_and_lineage": True,
        "all_evaluators_are_exact_index_permutations": True,
        "all_prompt_privacy_audits_pass": True,
        "proposal_pair_seeds_are_adjacent_and_equal": True,
        "evaluator_pair_seeds_are_adjacent_and_equal": True,
        "all_root_eig_positive": all(row["eig"] > 0 for row in selected),
        "all_root_branch_masses_at_least_010": all(
            row["mass"] >= 0.10 for task in branches for row in task
        ),
        "all_own_masses_at_least_010": all(value >= 0.10 for value in own_masses),
        "at_least_three_positive_own_minus_opposite": sum(value > 0 for value in own_opposite) >= 3,
        "mean_own_minus_opposite_at_least_010": own_opposite_mean >= 0.10,
        "at_least_three_positive_own_minus_answer_free": sum(value > 0 for value in own_blind) >= 3,
        "mean_own_minus_answer_free_at_least_005": own_blind_mean >= 0.05,
        "all_exact_updates_normalized": all(
            abs(row["posterior_probability_sum"] - 1.0) <= 1e-9
            and abs(row["posterior_predictive_matched_reply"] - 1.0) <= 1e-9
            for row in updates
        ),
        "at_least_one_selected_child_facet_changes": any(action_changes),
        "at_least_one_conditioned_support_changes_from_both_controls": any(structural_changes),
        "all_selected_child_scores_positive": all(
            row["conditioned_second_eig"] > 0 and row["answer_free_second_eig"] > 0
            for task in task_diagnostics
            for row in task["branches"]
        ),
    }
    expected_gates = {**transport, **mechanics}
    expected_gates["all_pass"] = all(expected_gates.values())
    expected_status = "passed" if expected_gates["all_pass"] else "mechanics_failed"
    expected_signal = {
        "own_minus_opposite": own_opposite,
        "own_minus_opposite_mean": own_opposite_mean,
        "own_minus_answer_free": own_blind,
        "own_minus_answer_free_mean": own_blind_mean,
        "selected_child_facet_change_count": sum(action_changes),
        "structural_change_count": sum(structural_changes),
    }
    mismatches = []
    comparisons = {
        "producer_interface": result.get("interface_version") == PRODUCER_INTERFACE,
        "status": result.get("status") == expected_status,
        "authorizes": result.get("authorizes")
        == (
            "separate_proposal_evaluator_v1_development_preregistration_only"
            if expected_status == "passed"
            else "nothing"
        ),
        "gates": canonical_json(result.get("gates")) == canonical_json(expected_gates),
        "signal_summary": canonical_json(result.get("signal_summary"))
        == canonical_json(expected_signal),
        "task_diagnostics": canonical_json(result.get("task_diagnostics"))
        == canonical_json(task_diagnostics),
        "endpoints_closed": result.get("endpoint_outcomes_opened") is False
        and result.get("development_opened") is False
        and result.get("confirmation_opened") is False,
    }
    mismatches.extend(name for name, passed in comparisons.items() if not passed)
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "verified" if not mismatches else "rejected",
        "mismatches": mismatches,
        "comparisons": comparisons,
        "artifact_sha256": {
            "raw_responses": sha256_file(raw_path),
            "privacy": sha256_file(privacy_path),
            "result": sha256_file(result_path),
        },
        "accepted_requests_replayed": EXPECTED_REQUESTS,
        "model_calls_made": 0,
        "endpoint_outcomes_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = replay(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
