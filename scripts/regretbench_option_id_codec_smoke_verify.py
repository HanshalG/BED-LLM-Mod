#!/usr/bin/env python3
"""Independently replay the RegretBench option-ID codec serving gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from rapidfuzz import fuzz


REPO_ROOT = Path(__file__).resolve().parents[1]
REGRETBENCH_ROOT = REPO_ROOT / "external/RegretBench"
REGRETBENCH_SRC = REGRETBENCH_ROOT / "src"
for path in (REPO_ROOT, REGRETBENCH_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from regretbench.mapping.semantic_action_mapper import SemanticActionMapper
from regretbench.schemas.cig import load_cig
from scripts import regretbench_deepseek_support_recovery as source_tools


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-option-id-codec-exact8-smoke-1"
HYPOTHESES = 8
QUESTIONS = 4
OPTIONS = 4
EXPECTED_REQUESTS = 8
MAX_RETRIES = 4
RUN_BUDGET_USD = 0.10
MIN_MUTUAL_INFORMATION = 0.05
MIN_TOP_OPTION_MASS = 0.10
OTHER_LABEL = "Other / none of these"
PROPOSAL_SEEDS = [202608540000, 202608540001]
EVALUATOR_SEEDS = [202608541000, 202608541001]
CODEC_SEEDS = [202608542000, 202608542001, 202608542010, 202608542011]
PUBLIC_TASKS = REPO_ROOT / (
    "results/nonmyopic/regretbench_option_id_codec_source_audit/"
    "CODEC_PUBLIC_TASKS.json"
)
PUBLIC_TASKS_SHA256 = (
    "7c79dbabcb07eb9f5935e03f089deb151927cde91b2315798f407a6a8959c11a"
)
EXPECTED_BINDINGS = {
    "source_protocol": "9f60d684327b4a8520669d341ad4d2490644e5409167c4b58e9f8d7bfdfeed9c",
    "codec_protocol": "5b574fc14551f4645b6b6a69d6b32a905b148a3cb210dbff971d33753969445f",
    "source_audit": "a07e854718e17717e4db16c5054c7c4ee421016c29de3d313fab2b5f7edd3b24",
    "source_manifest": "00526cefefda1633d0265be5af0d0f49b91665955335332b3e60ec78616fe273",
    "source_result": "f39ed160d340dad75884daac96c5b75b9545ffcc4efa7a5aa899063245fb2ff1",
    "public_tasks": PUBLIC_TASKS_SHA256,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def structural_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_tasks() -> list[dict[str, Any]]:
    if sha256_file(PUBLIC_TASKS) != PUBLIC_TASKS_SHA256:
        raise ValueError("independent option-ID public task binding changed")
    public = load_object(PUBLIC_TASKS)
    if (
        public.get("source_values_included") is not False
        or public.get("action_metadata_in_model_prompts") is not False
        or public.get("endpoint_outcomes_included") is not False
    ):
        raise ValueError("independent option-ID public task privacy changed")
    rows = public.get("tasks")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("independent option-ID task count changed")
    return rows


def parse_proposal(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("independent option-ID proposal fields changed")
    if not isinstance(value["hypotheses"], list) or len(value["hypotheses"]) != HYPOTHESES:
        raise ValueError("independent option-ID hypothesis count changed")
    hypotheses = []
    seen = set()
    for row in value["hypotheses"]:
        if not isinstance(row, dict) or set(row) != {"interpretation", "final_answer"}:
            raise ValueError("independent option-ID hypothesis fields changed")
        interpretation = row["interpretation"]
        answer = row["final_answer"]
        if not isinstance(interpretation, str) or not interpretation.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError("independent option-ID hypothesis invalid")
        key = (source_tools.normalize_text(interpretation), source_tools.normalize_text(answer))
        if key in seen:
            raise ValueError("independent option-ID hypothesis duplicated")
        seen.add(key)
        hypotheses.append({"interpretation": interpretation.strip(), "final_answer": answer.strip()})
    if not isinstance(value["questions"], list) or len(value["questions"]) != QUESTIONS:
        raise ValueError("independent option-ID question count changed")
    questions = []
    seen_questions = set()
    for row in value["questions"]:
        if not isinstance(row, dict) or set(row) != {"question", "options"}:
            raise ValueError("independent option-ID question fields changed")
        question = row["question"]
        if not isinstance(question, str) or not question.strip().endswith("?"):
            raise ValueError("independent option-ID question invalid")
        question = question.strip()
        key = source_tools.normalize_text(question)
        if key in seen_questions:
            raise ValueError("independent option-ID question duplicated")
        seen_questions.add(key)
        options = row["options"]
        if not isinstance(options, list) or len(options) != OPTIONS:
            raise ValueError("independent option-ID option count changed")
        clean_options = []
        labels = set()
        for index, option in enumerate(options):
            if not isinstance(option, dict) or set(option) != {"option_id", "label"} or option["option_id"] != index:
                raise ValueError("independent option-ID option fields changed")
            label = option["label"]
            if not isinstance(label, str) or not label.strip():
                raise ValueError("independent option-ID option label invalid")
            label = label.strip()
            label_key = source_tools.normalize_text(label)
            if label_key in labels:
                raise ValueError("independent option-ID option label duplicated")
            labels.add(label_key)
            clean_options.append({"option_id": index, "label": label})
        if clean_options[3]["label"] != OTHER_LABEL:
            raise ValueError("independent option-ID other label changed")
        questions.append({"question": question, "options": clean_options})
    public = {"hypotheses": hypotheses, "questions": questions}
    return {
        **public,
        "diagnostic": {
            "codec_mode": "strict_json",
            "valid_unique_count": HYPOTHESES,
            "question_count": QUESTIONS,
            "structural_sha256": structural_hash(public),
            "contains_weights_likelihoods_or_source_values": False,
        },
    }


def parse_evaluation(raw: str, proposal: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"particles"}:
        raise ValueError("independent option-ID evaluator fields changed")
    rows = value["particles"]
    if not isinstance(rows, list) or len(rows) != HYPOTHESES:
        raise ValueError("independent option-ID evaluator count changed")
    by_index = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"particle_index", "prior_weight", "option_likelihoods"}:
            raise ValueError("independent option-ID evaluator particle fields changed")
        index, weight, likelihoods = row["particle_index"], row["prior_weight"], row["option_likelihoods"]
        if isinstance(index, bool) or not isinstance(index, int) or index not in range(HYPOTHESES) or index in by_index:
            raise ValueError("independent option-ID evaluator index invalid")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or float(weight) < 0:
            raise ValueError("independent option-ID prior invalid")
        if not isinstance(likelihoods, list) or len(likelihoods) != QUESTIONS:
            raise ValueError("independent option-ID likelihood count changed")
        normalized = []
        for vector in likelihoods:
            if not isinstance(vector, list) or len(vector) != OPTIONS or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or float(x) < 0 for x in vector):
                raise ValueError("independent option-ID likelihood invalid")
            total = sum(float(x) for x in vector)
            if total <= 0:
                raise ValueError("independent option-ID likelihood has zero mass")
            normalized.append([float(x) / total for x in vector])
        by_index[index] = {"weight": float(weight), "likelihoods": normalized}
    if set(by_index) != set(range(HYPOTHESES)):
        raise ValueError("independent option-ID evaluator permutation changed")
    total = sum(row["weight"] for row in by_index.values())
    if total <= 0:
        raise ValueError("independent option-ID prior has zero mass")
    hypotheses = []
    for index, original in enumerate(proposal["hypotheses"]):
        row = by_index[index]
        hypotheses.append({
            **original,
            "probability": row["weight"] / total,
            "option_likelihoods": row["likelihoods"],
        })
    return {
        "hypotheses": hypotheses,
        "questions": proposal["questions"],
        "diagnostic": {
            "particle_index_permutation_exact": True,
            "prior_sum_before_normalization": total,
            "normalized_probability_sum": sum(row["probability"] for row in hypotheses),
            "all_likelihood_vectors_normalized": True,
            "proposal_sha256": proposal["diagnostic"]["structural_sha256"],
        },
    }


def entropy(values: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in values if value > 0)


def option_masses(support: Mapping[str, Any], question: int) -> list[float]:
    return [sum(row["probability"] * row["option_likelihoods"][question][option] for row in support["hypotheses"]) for option in range(OPTIONS)]


def mutual_information(support: Mapping[str, Any], question: int) -> float:
    masses = option_masses(support, question)
    conditional = sum(row["probability"] * entropy(row["option_likelihoods"][question]) for row in support["hypotheses"])
    value = entropy(masses) - conditional
    return max(0.0, value) if value > -1e-12 else value


def public_map(task: Mapping[str, Any], question: str) -> dict[str, Any]:
    normalized = source_tools.normalize_text(question)
    if not normalized or not question.strip().endswith("?"):
        return {"supported": False, "facet": None, "confidence": 0.0}
    best_facet, best_score = None, 0.0
    for facet in task["semantic_facets"]:
        candidates = [facet.replace("_", " ")] + [row["text"] for row in task["reference_questions"] if row["semantic_action"] == f"ask:{facet}"]
        if normalized in [source_tools.normalize_text(item) for item in candidates]:
            return {"supported": True, "facet": facet, "confidence": 1.0}
        score = max((fuzz.token_sort_ratio(question, item) / 100.0 for item in candidates), default=0.0)
        if source_tools.normalize_text(facet.replace("_", " ")) in normalized:
            score = max(score, 0.82)
        if score > best_score:
            best_facet, best_score = facet, score
    return {"supported": best_facet is not None and best_score >= 0.45, "facet": best_facet if best_score >= 0.45 else None, "confidence": best_score}


def select(task: Mapping[str, Any], support: Mapping[str, Any]) -> dict[str, Any]:
    mapped = [public_map(task, row["question"]) for row in support["questions"]]
    candidates = [index for index, row in enumerate(mapped) if row["supported"]]
    if not candidates:
        raise ValueError("independent option-ID proposal has no supported question")
    index = min(candidates, key=lambda item: (-mutual_information(support, item), item))
    masses = option_masses(support, index)
    top = sorted(range(OPTIONS), key=lambda option: (-masses[option], option))[:2]
    return {"index": index, "question": support["questions"][index], "facet": mapped[index]["facet"], "public_mapper_confidence": mapped[index]["confidence"], "mutual_information": mutual_information(support, index), "predictive_option_masses": masses, "top_options": top}


def selected_values(task: Mapping[str, Any], selected: Mapping[str, Any]):
    path = REGRETBENCH_ROOT / f"data/OpenDomainQA/test/{task['task_id']}.json"
    if sha256_file(path) != task["task_file_sha256"]:
        raise ValueError("independent option-ID selected CIG changed")
    cig = load_cig(path)
    if cig.cig_id != task["task_id"] or cig.prompt != task["prompt"]:
        raise ValueError("independent option-ID selected CIG identity changed")
    parsed = SemanticActionMapper().map_question(cig, selected["question"]["question"])
    if parsed.facet is None or parsed.semantic_action == "UNSUPPORTED" or parsed.facet != selected["facet"]:
        raise ValueError("independent option-ID mappers disagree")
    values = sorted({str((intent.slots or {}).get(parsed.facet, "")).strip() for intent in cig.intents if str((intent.slots or {}).get(parsed.facet, "")).strip()}, key=source_tools.normalize_text)
    if len(values) < 2:
        raise ValueError("independent option-ID selected facet has too few values")
    return values, float(parsed.confidence)


def parse_mapping(raw: str, count: int) -> list[int]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"mappings"} or not isinstance(value["mappings"], list) or len(value["mappings"]) != count:
        raise ValueError("independent option-ID codec fields changed")
    by_index = {}
    for row in value["mappings"]:
        if not isinstance(row, dict) or set(row) != {"value_index", "option_id"}:
            raise ValueError("independent option-ID codec row fields changed")
        index, option = row["value_index"], row["option_id"]
        if isinstance(index, bool) or not isinstance(index, int) or index not in range(count) or index in by_index or isinstance(option, bool) or not isinstance(option, int) or option not in range(OPTIONS):
            raise ValueError("independent option-ID codec row invalid")
        by_index[index] = option
    if set(by_index) != set(range(count)):
        raise ValueError("independent option-ID codec permutation changed")
    return [by_index[index] for index in range(count)]


def proposal_audit(task: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"task_id": task["task_id"], "prompt": task["prompt"]}
    return {"passed": True, "interface_role": "proposal", "payload_keys": sorted(payload), "payload_sha256": structural_hash(payload), "source_values_present": False, "action_metadata_present": False, "endpoint_present": False}


def evaluator_audit(task: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"task_id": task["task_id"], "prompt": task["prompt"], "particles": [{"particle_index": index, "interpretation": row["interpretation"], "final_answer": row["final_answer"]} for index, row in enumerate(proposal["hypotheses"])], "questions": proposal["questions"], "proposal_sha256": proposal["diagnostic"]["structural_sha256"]}
    return {"passed": True, "interface_role": "likelihood_evaluator", "payload_keys": sorted(payload), "payload_sha256": structural_hash(payload), "source_values_present": False, "action_metadata_present": False, "dialogue_or_observation_present": False, "endpoint_present": False}


def codec_audit(task: Mapping[str, Any], selected: Mapping[str, Any], values: Sequence[str], replicate: int) -> dict[str, Any]:
    payload = {"task_id": task["task_id"], "prompt": task["prompt"], "question": selected["question"], "values": [{"value_index": index, "value": value} for index, value in enumerate(values)]}
    return {"passed": True, "interface_role": "environment_codec", "payload_keys": sorted(payload), "payload_sha256": structural_hash(payload), "value_count": len(values), "intent_descriptions_present": False, "value_multiplicity_present": False, "final_answers_or_aliases_present": False, "truth_or_endpoint_present": False, "replicate": replicate}


def replay(run_dir: Path) -> dict[str, Any]:
    tasks = load_tasks()
    raw = load_object(run_dir / "private/RAW_RESPONSES.json")
    privacy = load_object(run_dir / "private/PRIVACY.json")
    ordering = load_object(run_dir / "private/ORDERING.json")
    result = load_object(run_dir / "RESULT.json")
    if set(raw) != {"proposal_seeds", "proposals", "evaluator_seeds", "evaluations", "codec_layout", "codec_seeds", "codec_responses"}:
        raise ValueError("independent option-ID raw bank fields changed")
    if raw["proposal_seeds"] != PROPOSAL_SEEDS or raw["evaluator_seeds"] != EVALUATOR_SEEDS or raw["codec_seeds"] != CODEC_SEEDS:
        raise ValueError("independent option-ID seeds changed")
    if raw["codec_layout"] != [{"task_index": task, "replicate": replicate} for task in range(2) for replicate in range(2)]:
        raise ValueError("independent option-ID codec layout changed")
    if not all(isinstance(raw[name], list) and len(raw[name]) == count for name, count in (("proposals", 2), ("evaluations", 2), ("codec_responses", 4))):
        raise ValueError("independent option-ID response counts changed")
    proposals = [parse_proposal(value) for value in raw["proposals"]]
    supports = [parse_evaluation(value, proposal) for value, proposal in zip(raw["evaluations"], proposals, strict=True)]
    selections = [select(task, support) for task, support in zip(tasks, supports, strict=True)]
    root_audits = [proposal_audit(task) for task in tasks] + [evaluator_audit(task, proposal) for task, proposal in zip(tasks, proposals, strict=True)]
    expected_ordering = {"root_requests_completed": 4, "expected_root_requests": 4, "source_values_loaded": True, "root_payload_sha256": [row["payload_sha256"] for row in root_audits], "source_values_loaded_after_root_requests": True}
    if canonical_json(ordering) != canonical_json(expected_ordering):
        raise ValueError("independent option-ID ordering replay changed")
    values_and_confidence = [selected_values(task, selected) for task, selected in zip(tasks, selections, strict=True)]
    values = [row[0] for row in values_and_confidence]
    mappings = [parse_mapping(raw["codec_responses"][2 * task + replicate], len(values[task])) for task in range(2) for replicate in range(2)]
    expected_privacy = list(root_audits)
    for task in range(2):
        for replicate in range(2):
            expected_privacy.append(codec_audit(tasks[task], selections[task], values[task], replicate))
    if canonical_json(privacy) != canonical_json({"audits": expected_privacy}):
        raise ValueError("independent option-ID privacy replay changed")
    diagnostics, agreement, used_ok, non_other_ok, other_ok, top_ok = [], [], [], [], [], []
    for task, selected in enumerate(selections):
        first, second = mappings[2 * task], mappings[2 * task + 1]
        used = sorted(set(first))
        non_other = [option for option in used if option != 3]
        top_realized = all(option in used for option in selected["top_options"])
        agreement.append(first == second)
        used_ok.append(len(used) >= 2)
        non_other_ok.append(len(non_other) >= 2)
        other_ok.append(sum(option == 3 for option in first) <= 1)
        top_ok.append(top_realized)
        diagnostics.append({"task_index": task, "proposal_sha256": proposals[task]["diagnostic"]["structural_sha256"], "selected_question_index": selected["index"], "selected_mutual_information_nats": selected["mutual_information"], "public_mapper_confidence": selected["public_mapper_confidence"], "full_mapper_confidence": values_and_confidence[task][1], "full_mapper_agrees": True, "source_value_count": len(values[task]), "codec_replicates_agree": first == second, "used_option_ids": used, "used_option_count": len(used), "used_non_other_option_count": len(non_other), "other_mapped_value_count": sum(option == 3 for option in first), "top_predictive_option_ids": selected["top_options"], "top_predictive_option_masses": [selected["predictive_option_masses"][option] for option in selected["top_options"]], "top_predictive_options_realized": top_realized, "private_mapping_sha256": structural_hash(first)})
    usage = result.get("usage") or {}
    transport = {"exact_eight_accepted_requests": usage.get("adapter_requests") == EXPECTED_REQUESTS, "attempts_between_eight_and_twelve": EXPECTED_REQUESTS <= usage.get("http_attempts", -1) <= EXPECTED_REQUESTS + MAX_RETRIES, "attempts_equal_accepted_plus_retries": usage.get("http_attempts") == usage.get("adapter_requests", -1) + usage.get("retry_count", -1), "at_most_four_retries": usage.get("retry_count", 999) <= MAX_RETRIES, "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0, "zero_forced_exits": usage.get("forced_exits") == 0, "within_codec_budget": usage.get("run_cost_usd", math.inf) <= RUN_BUDGET_USD + 1e-12}
    mechanics = {"exact_two_codec_tasks": True, "exact_two_proposals_and_evaluations": True, "exact_four_codec_responses": True, "all_proposals_exclude_weights_likelihoods_and_source_values": True, "all_evaluators_normalized": all(abs(row["diagnostic"]["normalized_probability_sum"] - 1.0) <= 1e-9 for row in supports), "all_prompt_privacy_audits_pass": True, "all_selected_questions_public_mapper_supported": all(row["facet"] is not None for row in selections), "all_full_mappers_agree_post_selection": True, "all_selected_mutual_information_at_least_005": all(row["mutual_information"] >= MIN_MUTUAL_INFORMATION for row in selections), "all_top_two_predictive_masses_at_least_010": all(row["predictive_option_masses"][option] >= MIN_TOP_OPTION_MASS for row in selections for option in row["top_options"]), "all_selected_facets_have_at_least_two_values": all(len(row) >= 2 for row in values), "all_codec_replicates_agree": all(agreement), "all_codec_mappings_use_at_least_two_options": all(used_ok), "all_codec_mappings_use_at_least_two_non_other_options": all(non_other_ok), "all_codec_mappings_use_other_at_most_once": all(other_ok), "all_top_predictive_options_are_realized": all(top_ok), "source_values_loaded_only_after_four_root_calls": True}
    gates = {**transport, **mechanics}
    gates["all_pass"] = all(gates.values())
    expected_status = "passed" if gates["all_pass"] else "codec_failed"
    expected_authorizes = "separate_option_id_mechanics_preregistration_only" if gates["all_pass"] else "nothing"
    comparisons = {"schema_version": result.get("schema_version") == SCHEMA_VERSION, "interface_version": result.get("interface_version") == INTERFACE_VERSION, "bindings": result.get("bindings") == EXPECTED_BINDINGS, "schedule": result.get("schedule") == {"proposals": 2, "evaluators": 2, "codec_calls": 4, "expected_accepted_requests": 8}, "task_diagnostics": canonical_json(result.get("task_diagnostics")) == canonical_json(diagnostics), "gates": canonical_json(result.get("gates")) == canonical_json(gates), "status": result.get("status") == expected_status, "authorizes": result.get("authorizes") == expected_authorizes, "sealed_outputs": all(result.get(name) is False for name in ("source_values_publicly_reported", "codec_mappings_publicly_reported", "policy_endpoint_opened", "mechanics_opened", "development_opened", "confirmation_opened"))}
    mismatches = [name for name, passed in comparisons.items() if not passed]
    return {"schema_version": SCHEMA_VERSION, "interface_version": "regretbench-option-id-codec-independent-replay-1", "status": "verified" if not mismatches else "rejected", "mismatches": mismatches, "comparisons": comparisons, "replayed_gates": gates, "model_calls_made": 0, "endpoint_outcomes_opened": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = replay(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
