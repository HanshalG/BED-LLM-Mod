#!/usr/bin/env python3
"""Independently replay the RegretBench typed-action serving gate."""

from __future__ import annotations

import argparse
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

from scripts import regretbench_deepseek_support_recovery as source_tools
from scripts import regretbench_typed_action_source_audit as typed_source


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-typed-action-exact8-smoke-1"
HYPOTHESES = 8
ACTIONS = 4
OPTIONS = 4
EXPECTED_REQUESTS = 8
MAX_RETRIES = 4
RUN_BUDGET_USD = 0.10
MIN_MUTUAL_INFORMATION = 0.05
MIN_TOP_OPTION_MASS = 0.10
OTHER_LABEL = "Other / none of these"
PROPOSAL_SEEDS = [202608550000, 202608550001]
EVALUATOR_SEEDS = [202608551000, 202608551001]
CODEC_SEEDS = [202608552000, 202608552001, 202608552010, 202608552011]
PUBLIC_TASKS = REPO_ROOT / (
    "results/nonmyopic/regretbench_typed_action_source_audit/"
    "TYPED_PUBLIC_TASKS.json"
)
PUBLIC_TASKS_SHA256 = (
    "12de3ae0e4953da606279c77af5aeabc7f85a6f202bd68952dfa2a6bb9e8526b"
)
EXPECTED_BINDINGS = {
    "source_protocol": "4301b0473ce7a3611af60dfc28a8b24b8e1a34d5d9a1592a10b9b7d2f207d93e",
    "codec_protocol": "79a2fef1d8244007afece2c5f69484be6d3e8e3a4795b69e7982aad1d3bcc740",
    "source_audit": "8e930df9015b909a4c0abd83ae4f01aefd320893efc4bfb9aa5c47cdc4e3ee86",
    "source_manifest": "276c0e5d43b8415b1f2e8e8ceb4d97d3340cc5b02b56ba389a61ddc9a12dbcec",
    "source_result": "df404209abe7287db5de5936185f907a96e6da43983c37737be554a4fe5f9a3d",
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
        raise ValueError("independent typed-action public task binding changed")
    public = load_object(PUBLIC_TASKS)
    if (
        public.get("source_values_included") is not False
        or public.get("action_metadata_in_model_prompts") is not True
        or public.get("endpoint_outcomes_included") is not False
    ):
        raise ValueError("independent typed-action public task privacy changed")
    rows = public.get("tasks")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("independent typed-action task count changed")
    return rows


def parse_proposal(raw: str, task: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "actions"}:
        raise ValueError("independent typed-action proposal fields changed")
    if not isinstance(value["hypotheses"], list) or len(value["hypotheses"]) != HYPOTHESES:
        raise ValueError("independent typed-action hypothesis count changed")
    hypotheses = []
    seen = set()
    for row in value["hypotheses"]:
        if not isinstance(row, dict) or set(row) != {"interpretation", "final_answer"}:
            raise ValueError("independent typed-action hypothesis fields changed")
        interpretation = row["interpretation"]
        answer = row["final_answer"]
        if not isinstance(interpretation, str) or not interpretation.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError("independent typed-action hypothesis invalid")
        key = (source_tools.normalize_text(interpretation), source_tools.normalize_text(answer))
        if key in seen:
            raise ValueError("independent typed-action hypothesis duplicated")
        seen.add(key)
        hypotheses.append({"interpretation": interpretation.strip(), "final_answer": answer.strip()})
    if not isinstance(value["actions"], list) or len(value["actions"]) != ACTIONS:
        raise ValueError("independent typed-action action count changed")
    by_action_id = {}
    for row in value["actions"]:
        if not isinstance(row, dict) or set(row) != {"action_id", "options"}:
            raise ValueError("independent typed-action action fields changed")
        action_id = row["action_id"]
        if not isinstance(action_id, str) or not action_id or action_id in by_action_id:
            raise ValueError("independent typed-action action ID invalid")
        options = row["options"]
        if not isinstance(options, list) or len(options) != OPTIONS:
            raise ValueError("independent typed-action option count changed")
        clean_options = []
        labels = set()
        for index, option in enumerate(options):
            if not isinstance(option, dict) or set(option) != {"option_id", "label"} or option["option_id"] != index:
                raise ValueError("independent typed-action option fields changed")
            label = option["label"]
            if not isinstance(label, str) or not label.strip():
                raise ValueError("independent typed-action option label invalid")
            label = label.strip()
            label_key = source_tools.normalize_text(label)
            if label_key in labels:
                raise ValueError("independent typed-action option label duplicated")
            labels.add(label_key)
            clean_options.append({"option_id": index, "label": label})
        if clean_options[3]["label"] != OTHER_LABEL:
            raise ValueError("independent typed-action other label changed")
        by_action_id[action_id] = clean_options
    allowed = [row["action_id"] for row in task["actions"]]
    if set(by_action_id) != set(allowed):
        raise ValueError("independent typed-action action permutation changed")
    actions = [
        {
            "action_id": row["action_id"],
            "question": row["question"],
            "public_order": row["public_order"],
            "options": by_action_id[row["action_id"]],
        }
        for row in task["actions"]
    ]
    public = {"hypotheses": hypotheses, "actions": actions}
    return {
        **public,
        "diagnostic": {
            "codec_mode": "strict_json",
            "valid_unique_count": HYPOTHESES,
            "action_count": ACTIONS,
            "structural_sha256": structural_hash(public),
            "contains_weights_likelihoods_or_source_values": False,
            "model_emitted_question_text": False,
        },
    }


def parse_evaluation(raw: str, proposal: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"particles"}:
        raise ValueError("independent typed-action evaluator fields changed")
    rows = value["particles"]
    if not isinstance(rows, list) or len(rows) != HYPOTHESES:
        raise ValueError("independent typed-action evaluator count changed")
    by_index = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"particle_index", "prior_weight", "option_likelihoods"}:
            raise ValueError("independent typed-action evaluator particle fields changed")
        index, weight, likelihoods = row["particle_index"], row["prior_weight"], row["option_likelihoods"]
        if isinstance(index, bool) or not isinstance(index, int) or index not in range(HYPOTHESES) or index in by_index:
            raise ValueError("independent typed-action evaluator index invalid")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or float(weight) < 0:
            raise ValueError("independent typed-action prior invalid")
        if not isinstance(likelihoods, list) or len(likelihoods) != ACTIONS:
            raise ValueError("independent typed-action likelihood count changed")
        normalized = []
        for vector in likelihoods:
            if not isinstance(vector, list) or len(vector) != OPTIONS or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or float(x) < 0 for x in vector):
                raise ValueError("independent typed-action likelihood invalid")
            total = sum(float(x) for x in vector)
            if total <= 0:
                raise ValueError("independent typed-action likelihood has zero mass")
            normalized.append([float(x) / total for x in vector])
        by_index[index] = {"weight": float(weight), "likelihoods": normalized}
    if set(by_index) != set(range(HYPOTHESES)):
        raise ValueError("independent typed-action evaluator permutation changed")
    total = sum(row["weight"] for row in by_index.values())
    if total <= 0:
        raise ValueError("independent typed-action prior has zero mass")
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
        "actions": proposal["actions"],
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


def select(support: Mapping[str, Any]) -> dict[str, Any]:
    index = min(range(ACTIONS), key=lambda item: (-mutual_information(support, item), item))
    masses = option_masses(support, index)
    top = sorted(range(OPTIONS), key=lambda option: (-masses[option], option))[:2]
    return {"index": index, "action": support["actions"][index], "action_id": support["actions"][index]["action_id"], "mutual_information": mutual_information(support, index), "predictive_option_masses": masses, "top_options": top}


def selected_values(task: Mapping[str, Any], selected: Mapping[str, Any]):
    path = REGRETBENCH_ROOT / f"data/OpenDomainQA/test/{task['task_id']}.json"
    if sha256_file(path) != task["task_file_sha256"]:
        raise ValueError("independent typed-action selected CIG changed")
    cig = load_object(path)
    if cig["cig_id"] != task["task_id"] or cig["prompt"] != task["prompt"]:
        raise ValueError("independent typed-action selected CIG identity changed")
    replay = [{"action_id": row["action_id"], "question": row["question"], "public_order": index} for index, row in enumerate(typed_source.executable_actions(cig))]
    if replay != task["actions"]:
        raise ValueError("independent typed-action full CIG replay changed")
    facet = selected["action_id"]
    values = sorted({str(typed_source.original._slots(intent).get(facet, "")).strip() for intent in cig["intents"] if str(typed_source.original._slots(intent).get(facet, "")).strip()}, key=source_tools.normalize_text)
    if len(values) < 2:
        raise ValueError("independent typed-action selected action has too few values")
    return values


def parse_mapping(raw: str, count: int) -> list[int]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"mappings"} or not isinstance(value["mappings"], list) or len(value["mappings"]) != count:
        raise ValueError("independent typed-action fields changed")
    by_index = {}
    for row in value["mappings"]:
        if not isinstance(row, dict) or set(row) != {"value_index", "option_id"}:
            raise ValueError("independent typed-action row fields changed")
        index, option = row["value_index"], row["option_id"]
        if isinstance(index, bool) or not isinstance(index, int) or index not in range(count) or index in by_index or isinstance(option, bool) or not isinstance(option, int) or option not in range(OPTIONS):
            raise ValueError("independent typed-action row invalid")
        by_index[index] = option
    if set(by_index) != set(range(count)):
        raise ValueError("independent typed-action permutation changed")
    return [by_index[index] for index in range(count)]


def proposal_audit(task: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"task_id": task["task_id"], "prompt": task["prompt"], "allowed_actions": task["actions"]}
    return {"passed": True, "interface_role": "proposal", "payload_keys": sorted(payload), "payload_sha256": structural_hash(payload), "source_values_present": False, "action_metadata_present": True, "endpoint_present": False}


def evaluator_audit(task: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"task_id": task["task_id"], "prompt": task["prompt"], "particles": [{"particle_index": index, "interpretation": row["interpretation"], "final_answer": row["final_answer"]} for index, row in enumerate(proposal["hypotheses"])], "allowed_actions": task["actions"], "actions": proposal["actions"], "proposal_sha256": proposal["diagnostic"]["structural_sha256"]}
    return {"passed": True, "interface_role": "likelihood_evaluator", "payload_keys": sorted(payload), "payload_sha256": structural_hash(payload), "source_values_present": False, "action_metadata_present": True, "dialogue_or_observation_present": False, "endpoint_present": False}


def codec_audit(task: Mapping[str, Any], selected: Mapping[str, Any], values: Sequence[str], replicate: int) -> dict[str, Any]:
    payload = {"task_id": task["task_id"], "prompt": task["prompt"], "action": selected["action"], "values": [{"value_index": index, "value": value} for index, value in enumerate(values)]}
    return {"passed": True, "interface_role": "environment_codec", "payload_keys": sorted(payload), "payload_sha256": structural_hash(payload), "value_count": len(values), "intent_descriptions_present": False, "value_multiplicity_present": False, "final_answers_or_aliases_present": False, "truth_or_endpoint_present": False, "replicate": replicate}


def replay(run_dir: Path) -> dict[str, Any]:
    tasks = load_tasks()
    raw = load_object(run_dir / "private/RAW_RESPONSES.json")
    privacy = load_object(run_dir / "private/PRIVACY.json")
    ordering = load_object(run_dir / "private/ORDERING.json")
    result = load_object(run_dir / "RESULT.json")
    if set(raw) != {"proposal_seeds", "proposals", "evaluator_seeds", "evaluations", "codec_layout", "codec_seeds", "codec_responses"}:
        raise ValueError("independent typed-action raw bank fields changed")
    if raw["proposal_seeds"] != PROPOSAL_SEEDS or raw["evaluator_seeds"] != EVALUATOR_SEEDS or raw["codec_seeds"] != CODEC_SEEDS:
        raise ValueError("independent typed-action seeds changed")
    if raw["codec_layout"] != [{"task_index": task, "replicate": replicate} for task in range(2) for replicate in range(2)]:
        raise ValueError("independent typed-action layout changed")
    if not all(isinstance(raw[name], list) and len(raw[name]) == count for name, count in (("proposals", 2), ("evaluations", 2), ("codec_responses", 4))):
        raise ValueError("independent typed-action response counts changed")
    proposals = [parse_proposal(value, task) for value, task in zip(raw["proposals"], tasks, strict=True)]
    supports = [parse_evaluation(value, proposal) for value, proposal in zip(raw["evaluations"], proposals, strict=True)]
    selections = [select(support) for support in supports]
    root_audits = [proposal_audit(task) for task in tasks] + [evaluator_audit(task, proposal) for task, proposal in zip(tasks, proposals, strict=True)]
    expected_ordering = {"root_requests_completed": 4, "expected_root_requests": 4, "source_values_loaded": True, "root_payload_sha256": [row["payload_sha256"] for row in root_audits], "source_values_loaded_after_root_requests": True}
    if canonical_json(ordering) != canonical_json(expected_ordering):
        raise ValueError("independent typed-action ordering replay changed")
    values = [selected_values(task, selected) for task, selected in zip(tasks, selections, strict=True)]
    mappings = [parse_mapping(raw["codec_responses"][2 * task + replicate], len(values[task])) for task in range(2) for replicate in range(2)]
    expected_privacy = list(root_audits)
    for task in range(2):
        for replicate in range(2):
            expected_privacy.append(codec_audit(tasks[task], selections[task], values[task], replicate))
    if canonical_json(privacy) != canonical_json({"audits": expected_privacy}):
        raise ValueError("independent typed-action privacy replay changed")
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
        diagnostics.append({"task_index": task, "proposal_sha256": proposals[task]["diagnostic"]["structural_sha256"], "selected_action_index": selected["index"], "selected_action_id": selected["action_id"], "selected_mutual_information_nats": selected["mutual_information"], "full_cig_action_replays": True, "source_value_count": len(values[task]), "codec_replicates_agree": first == second, "used_option_ids": used, "used_option_count": len(used), "used_non_other_option_count": len(non_other), "other_mapped_value_count": sum(option == 3 for option in first), "top_predictive_option_ids": selected["top_options"], "top_predictive_option_masses": [selected["predictive_option_masses"][option] for option in selected["top_options"]], "top_predictive_options_realized": top_realized, "private_mapping_sha256": structural_hash(first)})
    usage = result.get("usage") or {}
    transport = {"exact_eight_accepted_requests": usage.get("adapter_requests") == EXPECTED_REQUESTS, "attempts_between_eight_and_twelve": EXPECTED_REQUESTS <= usage.get("http_attempts", -1) <= EXPECTED_REQUESTS + MAX_RETRIES, "attempts_equal_accepted_plus_retries": usage.get("http_attempts") == usage.get("adapter_requests", -1) + usage.get("retry_count", -1), "at_most_four_retries": usage.get("retry_count", 999) <= MAX_RETRIES, "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0, "zero_forced_exits": usage.get("forced_exits") == 0, "within_codec_budget": usage.get("run_cost_usd", math.inf) <= RUN_BUDGET_USD + 1e-12}
    mechanics = {"exact_two_codec_tasks": True, "exact_two_proposals_and_evaluations": True, "exact_four_codec_responses": True, "all_proposals_exclude_weights_likelihoods_and_source_values": True, "all_proposals_use_exact_typed_action_permutations": all([row["action_id"] for row in proposal["actions"]] == [row["action_id"] for row in task["actions"]] for task, proposal in zip(tasks, proposals, strict=True)), "no_proposal_emits_question_text": all(proposal["diagnostic"]["model_emitted_question_text"] is False for proposal in proposals), "all_evaluators_normalized": all(abs(row["diagnostic"]["normalized_probability_sum"] - 1.0) <= 1e-9 for row in supports), "all_prompt_privacy_audits_pass": True, "all_selected_actions_replay_from_full_cig": True, "all_selected_mutual_information_at_least_005": all(row["mutual_information"] >= MIN_MUTUAL_INFORMATION for row in selections), "all_top_two_predictive_masses_at_least_010": all(row["predictive_option_masses"][option] >= MIN_TOP_OPTION_MASS for row in selections for option in row["top_options"]), "all_selected_actions_have_at_least_two_values": all(len(row) >= 2 for row in values), "all_codec_replicates_agree": all(agreement), "all_codec_mappings_use_at_least_two_options": all(used_ok), "all_codec_mappings_use_at_least_two_non_other_options": all(non_other_ok), "all_codec_mappings_use_other_at_most_once": all(other_ok), "all_top_predictive_options_are_realized": all(top_ok), "source_values_loaded_only_after_four_root_calls": True}
    gates = {**transport, **mechanics}
    gates["all_pass"] = all(gates.values())
    expected_status = "passed" if gates["all_pass"] else "calibration_failed"
    expected_authorizes = "separate_typed_action_mechanics_preregistration_only" if gates["all_pass"] else "nothing"
    comparisons = {"schema_version": result.get("schema_version") == SCHEMA_VERSION, "interface_version": result.get("interface_version") == INTERFACE_VERSION, "bindings": result.get("bindings") == EXPECTED_BINDINGS, "schedule": result.get("schedule") == {"proposals": 2, "evaluators": 2, "codec_calls": 4, "expected_accepted_requests": 8}, "task_diagnostics": canonical_json(result.get("task_diagnostics")) == canonical_json(diagnostics), "gates": canonical_json(result.get("gates")) == canonical_json(gates), "status": result.get("status") == expected_status, "authorizes": result.get("authorizes") == expected_authorizes, "sealed_outputs": all(result.get(name) is False for name in ("source_values_publicly_reported", "codec_mappings_publicly_reported", "policy_endpoint_opened", "mechanics_opened", "development_opened", "confirmation_opened"))}
    mismatches = [name for name, passed in comparisons.items() if not passed]
    return {"schema_version": SCHEMA_VERSION, "interface_version": "regretbench-typed-action-independent-replay-1", "status": "verified" if not mismatches else "rejected", "mismatches": mismatches, "comparisons": comparisons, "replayed_gates": gates, "model_calls_made": 0, "endpoint_outcomes_opened": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = replay(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
