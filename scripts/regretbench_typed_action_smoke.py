#!/usr/bin/env python3
"""Run the RegretBench typed-action exact-8 serving gate."""

from __future__ import annotations

import argparse
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
from scripts import regretbench_deepseek_support_recovery as source_tools
from scripts import regretbench_typed_action_source_audit as typed_source
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.number_game_qwen_history_blind_serving_smoke import (
    PerRequestSeedStructuredAdapter,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-typed-action-exact8-smoke-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
PROPOSAL_TEMPERATURE = 0.7
EVALUATOR_TEMPERATURE = 0.7
CODEC_TEMPERATURE = 0.0
PROPOSAL_MAX_TOKENS = 2_400
EVALUATOR_MAX_TOKENS = 2_000
CODEC_MAX_TOKENS = 800
EXPECTED_REQUESTS = 8
MAX_RETRIES = 4
CONCURRENCY = 8
RUN_BUDGET_USD = 0.10
PROJECTED_COST_USD = 0.02
HYPOTHESES = 8
ACTIONS = 4
OPTIONS = 4
PROPOSAL_SEED_START = 202608550000
EVALUATOR_SEED_START = 202608551000
CODEC_SEED_START = 202608552000
MIN_MUTUAL_INFORMATION = 0.05
MIN_TOP_OPTION_MASS = 0.10
OTHER_LABEL = "Other / none of these"

SOURCE_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_TYPED_ACTION_SOURCE_PROTOCOL_20260811.md"
)
SOURCE_PROTOCOL_SHA256 = (
    "4301b0473ce7a3611af60dfc28a8b24b8e1a34d5d9a1592a10b9b7d2f207d93e"
)
CODEC_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_TYPED_ACTION_EXACT8_PROTOCOL_20260811.md"
)
CODEC_PROTOCOL_SHA256 = (
    "79a2fef1d8244007afece2c5f69484be6d3e8e3a4795b69e7982aad1d3bcc740"
)
SOURCE_AUDIT = REPO_ROOT / "scripts/regretbench_typed_action_source_audit.py"
SOURCE_AUDIT_SHA256 = (
    "8e930df9015b909a4c0abd83ae4f01aefd320893efc4bfb9aa5c47cdc4e3ee86"
)
SOURCE_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/regretbench_typed_action_source_audit/"
    "SOURCE_PROTOCOL_MANIFEST.json"
)
SOURCE_MANIFEST_SHA256 = (
    "276c0e5d43b8415b1f2e8e8ceb4d97d3340cc5b02b56ba389a61ddc9a12dbcec"
)
SOURCE_RESULT = REPO_ROOT / (
    "results/nonmyopic/regretbench_typed_action_source_audit/RESULT.json"
)
SOURCE_RESULT_SHA256 = (
    "df404209abe7287db5de5936185f907a96e6da43983c37737be554a4fe5f9a3d"
)
PUBLIC_TASKS = REPO_ROOT / (
    "results/nonmyopic/regretbench_typed_action_source_audit/"
    "TYPED_PUBLIC_TASKS.json"
)
PUBLIC_TASKS_SHA256 = (
    "12de3ae0e4953da606279c77af5aeabc7f85a6f202bd68952dfa2a6bb9e8526b"
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


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structural_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_bindings() -> dict[str, str]:
    expected = {
        "source_protocol": (SOURCE_PROTOCOL, SOURCE_PROTOCOL_SHA256),
        "codec_protocol": (CODEC_PROTOCOL, CODEC_PROTOCOL_SHA256),
        "source_audit": (SOURCE_AUDIT, SOURCE_AUDIT_SHA256),
        "source_manifest": (SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256),
        "source_result": (SOURCE_RESULT, SOURCE_RESULT_SHA256),
        "public_tasks": (PUBLIC_TASKS, PUBLIC_TASKS_SHA256),
    }
    for name, (path, digest) in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"typed-action {name} changed")
    source = load_object(SOURCE_RESULT)
    public = load_object(PUBLIC_TASKS)
    if (
        source.get("status") != "source_protocol_pass"
        or source.get("authorizes") != "typed_action_exact8_only"
        or source.get("gates", {}).get("all_pass") is not True
        or len(public.get("tasks", [])) != 2
        or public.get("source_values_included") is not False
        or public.get("action_metadata_in_model_prompts") is not True
        or public.get("endpoint_outcomes_included") is not False
    ):
        raise ValueError("typed-action source does not authorize serving gate")
    return {name: digest for name, (_, digest) in expected.items()}


def load_public_tasks() -> list[dict[str, Any]]:
    validate_bindings()
    rows = load_object(PUBLIC_TASKS)["tasks"]
    result = []
    for index, row in enumerate(rows):
        if (
            set(row)
            != {
                "task_index",
                "task_id",
                "prompt",
                "task_file_sha256",
                "actions",
            }
            or row["task_index"] != index
            or not isinstance(row["task_id"], str)
            or not isinstance(row["prompt"], str)
            or not isinstance(row["actions"], list)
            or len(row["actions"]) != ACTIONS
            or [action.get("public_order") for action in row["actions"]]
            != list(range(ACTIONS))
            or len({action.get("action_id") for action in row["actions"]}) != ACTIONS
        ):
            raise ValueError("typed-action public task manifest is malformed")
        result.append(dict(row))
    return result


def load_selected_cig(task: Mapping[str, Any]) -> dict[str, Any]:
    path = REGRETBENCH_ROOT / f"data/OpenDomainQA/test/{task['task_id']}.json"
    if sha256_file(path) != task["task_file_sha256"]:
        raise ValueError("typed-action selected CIG file changed")
    cig = load_object(path)
    if cig["cig_id"] != task["task_id"] or cig["prompt"] != task["prompt"]:
        raise ValueError("typed-action selected CIG identity changed")
    return cig


def proposal_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "regretbench_typed_action_proposal",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses", "actions"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": HYPOTHESES,
                        "maxItems": HYPOTHESES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["interpretation", "final_answer"],
                            "properties": {
                                "interpretation": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 800,
                                },
                                "final_answer": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 240,
                                },
                            },
                        },
                    },
                    "actions": {
                        "type": "array",
                        "minItems": ACTIONS,
                        "maxItems": ACTIONS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["action_id", "options"],
                            "properties": {
                                "action_id": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 120,
                                },
                                "options": {
                                    "type": "array",
                                    "minItems": OPTIONS,
                                    "maxItems": OPTIONS,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["option_id", "label"],
                                        "properties": {
                                            "option_id": {
                                                "type": "integer",
                                                "minimum": 0,
                                                "maximum": OPTIONS - 1,
                                            },
                                            "label": {
                                                "type": "string",
                                                "minLength": 1,
                                                "maxLength": 160,
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def evaluator_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "regretbench_typed_action_likelihood_evaluator",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["particles"],
                "properties": {
                    "particles": {
                        "type": "array",
                        "minItems": HYPOTHESES,
                        "maxItems": HYPOTHESES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "particle_index",
                                "prior_weight",
                                "option_likelihoods",
                            ],
                            "properties": {
                                "particle_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": HYPOTHESES - 1,
                                },
                                "prior_weight": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 100,
                                },
                                "option_likelihoods": {
                                    "type": "array",
                                    "minItems": ACTIONS,
                                    "maxItems": ACTIONS,
                                    "items": {
                                        "type": "array",
                                        "minItems": OPTIONS,
                                        "maxItems": OPTIONS,
                                        "items": {
                                            "type": "number",
                                            "minimum": 0,
                                            "maximum": 100,
                                        },
                                    },
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def codec_response_format(value_count: int) -> dict[str, Any]:
    if value_count < 2:
        raise ValueError("codec requires at least two source values")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"regretbench_typed_action_environment_codec_{value_count}",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mappings"],
                "properties": {
                    "mappings": {
                        "type": "array",
                        "minItems": value_count,
                        "maxItems": value_count,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["value_index", "option_id"],
                            "properties": {
                                "value_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": value_count - 1,
                                },
                                "option_id": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": OPTIONS - 1,
                                },
                            },
                        },
                    }
                },
            },
        },
    }


PROPOSAL_SYSTEM_PROMPT = """Construct a finite semantic belief proposal for the public ambiguous factual question. Return exactly eight plausible, meaningfully distinct interpretations with one concise final factual answer each. For every allowed action_id, return exactly one action record with option IDs 0, 1, and 2 carrying distinct substantive natural-language labels, followed by exact option ID 3 labelled exactly 'Other / none of these'. Copy action_id values exactly and emit no question text. Options should cover plausible user meanings without revealing final factual answers. Do not emit weights, likelihoods, predicted replies, source values, lineage, or retention labels. Return only the required JSON object."""

EVALUATOR_SYSTEM_PROMPT = """Evaluate a fixed semantic proposal without changing it. For each particle_index, assign a nonnegative proposal-local prior weight and, for each fixed typed action in order, four nonnegative likelihood scores aligned exactly to option IDs 0,1,2,3. Each likelihood vector must have positive total mass. Use only the public prompt, allowed actions, and fixed model-generated proposal. Do not emit text replies, source values, dialogue, observations, lineage, or endpoints. Return only the required JSON object."""

CODEC_SYSTEM_PROMPT = """Map each indexed environment value to exactly one of the fixed answer option IDs for the fixed clarification question. Use the semantic meaning of the question, option labels, and value. Return one mapping for every value_index and no other fields. Do not infer or discuss hidden intents, final answers, frequencies, policy scores, or endpoints. Return only the required JSON object."""


def proposal_messages_for(task: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    payload = {
        "task_id": task["task_id"],
        "prompt": task["prompt"],
        "allowed_actions": task["actions"],
    }
    audit = {
        "passed": True,
        "interface_role": "proposal",
        "payload_keys": sorted(payload),
        "payload_sha256": structural_hash(payload),
        "source_values_present": False,
        "action_metadata_present": True,
        "endpoint_present": False,
    }
    return [
        {"role": "system", "content": PROPOSAL_SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(payload)},
    ], audit


def parse_proposal(raw: str, task: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "actions"}:
        raise ValueError("typed-action proposal has wrong top-level fields")
    hypotheses = []
    seen = set()
    rows = value["hypotheses"]
    if not isinstance(rows, list) or len(rows) != HYPOTHESES:
        raise ValueError("typed-action proposal must contain eight hypotheses")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"interpretation", "final_answer"}:
            raise ValueError("typed-action proposal hypothesis fields changed")
        interpretation = row["interpretation"]
        answer = row["final_answer"]
        if (
            not isinstance(interpretation, str)
            or not interpretation.strip()
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ValueError("typed-action proposal hypothesis is invalid")
        key = (
            source_tools.normalize_text(interpretation),
            source_tools.normalize_text(answer),
        )
        if key in seen:
            raise ValueError("typed-action proposal hypotheses are duplicated")
        seen.add(key)
        hypotheses.append(
            {"interpretation": interpretation.strip(), "final_answer": answer.strip()}
        )
    by_action_id = {}
    rows = value["actions"]
    if not isinstance(rows, list) or len(rows) != ACTIONS:
        raise ValueError("typed-action proposal must contain four actions")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"action_id", "options"}:
            raise ValueError("typed-action proposal action fields changed")
        action_id = row["action_id"]
        options = row["options"]
        if (
            not isinstance(action_id, str)
            or not action_id
            or action_id in by_action_id
        ):
            raise ValueError("typed-action proposal action ID is invalid")
        if not isinstance(options, list) or len(options) != OPTIONS:
            raise ValueError("typed-action proposal must contain four options")
        parsed_options = []
        labels = set()
        for index, option in enumerate(options):
            if not isinstance(option, dict) or set(option) != {"option_id", "label"}:
                raise ValueError("typed-action proposal option fields changed")
            if option["option_id"] != index:
                raise ValueError("typed-action proposal IDs are not ordered 0 through 3")
            label = option["label"]
            if not isinstance(label, str) or not label.strip():
                raise ValueError("typed-action proposal label is empty")
            label = label.strip()
            key = source_tools.normalize_text(label)
            if key in labels:
                raise ValueError("typed-action proposal labels are duplicated")
            labels.add(key)
            parsed_options.append({"option_id": index, "label": label})
        if parsed_options[3]["label"] != OTHER_LABEL:
            raise ValueError("typed-action proposal final label is not exact other option")
        by_action_id[action_id] = parsed_options
    allowed = [row["action_id"] for row in task["actions"]]
    if set(by_action_id) != set(allowed):
        raise ValueError("typed-action proposal action IDs are not an exact permutation")
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
            "valid_unique_count": len(hypotheses),
            "action_count": len(actions),
            "structural_sha256": structural_hash(public),
            "contains_weights_likelihoods_or_source_values": False,
            "model_emitted_question_text": False,
        },
    }


def evaluator_messages_for(
    task: Mapping[str, Any], proposal: Mapping[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    payload = {
        "task_id": task["task_id"],
        "prompt": task["prompt"],
        "particles": [
            {
                "particle_index": index,
                "interpretation": row["interpretation"],
                "final_answer": row["final_answer"],
            }
            for index, row in enumerate(proposal["hypotheses"])
        ],
        "allowed_actions": task["actions"],
        "actions": proposal["actions"],
        "proposal_sha256": proposal["diagnostic"]["structural_sha256"],
    }
    audit = {
        "passed": True,
        "interface_role": "likelihood_evaluator",
        "payload_keys": sorted(payload),
        "payload_sha256": structural_hash(payload),
        "source_values_present": False,
        "action_metadata_present": True,
        "dialogue_or_observation_present": False,
        "endpoint_present": False,
    }
    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(payload)},
    ], audit


def parse_evaluation(raw: str, proposal: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"particles"}:
        raise ValueError("typed-action evaluator has wrong top-level fields")
    rows = value["particles"]
    if not isinstance(rows, list) or len(rows) != HYPOTHESES:
        raise ValueError("typed-action evaluator must contain eight particles")
    by_index = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "particle_index",
            "prior_weight",
            "option_likelihoods",
        }:
            raise ValueError("typed-action evaluator particle fields changed")
        index = row["particle_index"]
        weight = row["prior_weight"]
        likelihoods = row["option_likelihoods"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in range(HYPOTHESES)
            or index in by_index
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
            or not isinstance(likelihoods, list)
            or len(likelihoods) != ACTIONS
        ):
            raise ValueError("typed-action evaluator particle is invalid")
        normalized = []
        for vector in likelihoods:
            if (
                not isinstance(vector, list)
                or len(vector) != OPTIONS
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    or float(item) < 0
                    for item in vector
                )
            ):
                raise ValueError("typed-action likelihood vector is invalid")
            total = sum(float(item) for item in vector)
            if total <= 0:
                raise ValueError("typed-action likelihood vector has zero mass")
            normalized.append([float(item) / total for item in vector])
        by_index[index] = {"prior_weight": float(weight), "likelihoods": normalized}
    if set(by_index) != set(range(HYPOTHESES)):
        raise ValueError("typed-action evaluator indexes are not an exact permutation")
    prior_total = sum(row["prior_weight"] for row in by_index.values())
    if prior_total <= 0:
        raise ValueError("typed-action evaluator prior has zero mass")
    hypotheses = []
    for index, original in enumerate(proposal["hypotheses"]):
        evaluated = by_index[index]
        hypotheses.append(
            {
                "interpretation": original["interpretation"],
                "final_answer": original["final_answer"],
                "probability": evaluated["prior_weight"] / prior_total,
                "option_likelihoods": evaluated["likelihoods"],
            }
        )
    return {
        "hypotheses": hypotheses,
        "actions": proposal["actions"],
        "diagnostic": {
            "particle_index_permutation_exact": True,
            "prior_sum_before_normalization": prior_total,
            "normalized_probability_sum": sum(row["probability"] for row in hypotheses),
            "all_likelihood_vectors_normalized": True,
            "proposal_sha256": proposal["diagnostic"]["structural_sha256"],
        },
    }


def entropy(values: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in values if value > 0)


def predictive_option_masses(
    support: Mapping[str, Any], question_index: int
) -> list[float]:
    return [
        sum(
            float(row["probability"])
            * float(row["option_likelihoods"][question_index][option])
            for row in support["hypotheses"]
        )
        for option in range(OPTIONS)
    ]


def categorical_mutual_information(
    support: Mapping[str, Any], question_index: int
) -> float:
    predictive = predictive_option_masses(support, question_index)
    conditional_entropy = sum(
        float(row["probability"])
        * entropy(row["option_likelihoods"][question_index])
        for row in support["hypotheses"]
    )
    value = entropy(predictive) - conditional_entropy
    return max(0.0, value) if value > -1e-12 else value


def select_action(support: Mapping[str, Any]) -> dict[str, Any]:
    index = min(
        range(ACTIONS),
        key=lambda item: (-categorical_mutual_information(support, item), item),
    )
    masses = predictive_option_masses(support, index)
    top_options = sorted(range(OPTIONS), key=lambda option: (-masses[option], option))[:2]
    return {
        "index": index,
        "action": support["actions"][index],
        "action_id": support["actions"][index]["action_id"],
        "mutual_information": categorical_mutual_information(support, index),
        "predictive_option_masses": masses,
        "top_options": top_options,
    }


def selected_source_values(
    task: Mapping[str, Any], selected: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    cig = load_selected_cig(task)
    replay = [
        {
            "action_id": row["action_id"],
            "question": row["question"],
            "public_order": index,
        }
        for index, row in enumerate(typed_source.executable_actions(cig))
    ]
    if replay != task["actions"]:
        raise ValueError("typed-action public actions do not replay from full CIG")
    facet = selected["action_id"]
    values = sorted(
        {
            str(typed_source.original._slots(intent).get(facet, "")).strip()
            for intent in cig["intents"]
            if str(typed_source.original._slots(intent).get(facet, "")).strip()
        },
        key=source_tools.normalize_text,
    )
    if len(values) < 2:
        raise ValueError("typed-action selected facet has fewer than two source values")
    return cig, values, {
        "action_replay_exact": True,
        "selected_action_id_present": facet
        in {row["action_id"] for row in replay},
    }


def codec_messages_for(
    task: Mapping[str, Any],
    selected: Mapping[str, Any],
    values: Sequence[str],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    payload = {
        "task_id": task["task_id"],
        "prompt": task["prompt"],
        "action": selected["action"],
        "values": [
            {"value_index": index, "value": value}
            for index, value in enumerate(values)
        ],
    }
    audit = {
        "passed": True,
        "interface_role": "environment_codec",
        "payload_keys": sorted(payload),
        "payload_sha256": structural_hash(payload),
        "value_count": len(values),
        "intent_descriptions_present": False,
        "value_multiplicity_present": False,
        "final_answers_or_aliases_present": False,
        "truth_or_endpoint_present": False,
    }
    return [
        {"role": "system", "content": CODEC_SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(payload)},
    ], audit


def parse_codec_mapping(raw: str, value_count: int) -> list[int]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"mappings"}:
        raise ValueError("typed-action has wrong top-level fields")
    rows = value["mappings"]
    if not isinstance(rows, list) or len(rows) != value_count:
        raise ValueError("typed-action mapping count changed")
    by_index = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"value_index", "option_id"}:
            raise ValueError("typed-action mapping fields changed")
        index = row["value_index"]
        option = row["option_id"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in range(value_count)
            or index in by_index
            or isinstance(option, bool)
            or not isinstance(option, int)
            or option not in range(OPTIONS)
        ):
            raise ValueError("typed-action mapping is invalid")
        by_index[index] = option
    if set(by_index) != set(range(value_count)):
        raise ValueError("typed-action indexes are not an exact permutation")
    return [by_index[index] for index in range(value_count)]


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
        openrouter_max_request_cost_usd=0.002,
        openrouter_max_output_tokens=PROPOSAL_MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return PerRequestSeedStructuredAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65_536),
        config,
    )


def call_batch(
    adapter: StructuredAdapter,
    messages: Sequence[list[dict[str, str]]],
    seeds: Sequence[int],
    *,
    temperature: float,
    response_format: dict[str, Any],
    max_tokens: int,
) -> list[str]:
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        messages,
        seeds,
        temperature=temperature,
        response_format=response_format,
        max_new_tokens=max_tokens,
    )
    if len(responses) != len(messages):
        raise ValueError("typed-action adapter returned wrong response count")
    return list(responses)


def usage_gates(adapter: StructuredAdapter) -> tuple[dict[str, Any], dict[str, bool]]:
    usage = summarize_usage(adapter.usage_snapshot())
    gates = {
        "exact_eight_accepted_requests": usage["adapter_requests"] == EXPECTED_REQUESTS,
        "attempts_between_eight_and_twelve": EXPECTED_REQUESTS
        <= usage["http_attempts"]
        <= EXPECTED_REQUESTS + MAX_RETRIES,
        "attempts_equal_accepted_plus_retries": usage["http_attempts"]
        == usage["adapter_requests"] + usage["retry_count"],
        "at_most_four_retries": usage["retry_count"] <= MAX_RETRIES,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_codec_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD + 1e-12,
    }
    return usage, gates


def run_smoke(
    *,
    output_dir: Path,
    adapter: StructuredAdapter,
    daily_budget_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bindings = validate_bindings()
    tasks = load_public_tasks()
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    raw_bank: dict[str, Any] = {}
    privacy = []

    proposal_messages = []
    for task in tasks:
        messages, audit = proposal_messages_for(task)
        proposal_messages.append(messages)
        privacy.append(audit)
    proposal_seeds = [PROPOSAL_SEED_START + index for index in range(2)]
    raw_proposals = call_batch(
        adapter,
        proposal_messages,
        proposal_seeds,
        temperature=PROPOSAL_TEMPERATURE,
        response_format=proposal_response_format(),
        max_tokens=PROPOSAL_MAX_TOKENS,
    )
    raw_bank.update({"proposal_seeds": proposal_seeds, "proposals": raw_proposals})
    checkpoint(private / "RAW_RESPONSES.json", raw_bank)
    checkpoint(private / "PRIVACY.json", {"audits": privacy})
    proposals = [
        parse_proposal(raw, task)
        for raw, task in zip(raw_proposals, tasks, strict=True)
    ]

    evaluator_messages = []
    for task, proposal in zip(tasks, proposals, strict=True):
        messages, audit = evaluator_messages_for(task, proposal)
        evaluator_messages.append(messages)
        privacy.append(audit)
    evaluator_seeds = [EVALUATOR_SEED_START + index for index in range(2)]
    raw_evaluations = call_batch(
        adapter,
        evaluator_messages,
        evaluator_seeds,
        temperature=EVALUATOR_TEMPERATURE,
        response_format=evaluator_response_format(),
        max_tokens=EVALUATOR_MAX_TOKENS,
    )
    raw_bank.update(
        {"evaluator_seeds": evaluator_seeds, "evaluations": raw_evaluations}
    )
    checkpoint(private / "RAW_RESPONSES.json", raw_bank)
    checkpoint(private / "PRIVACY.json", {"audits": privacy})
    supports = [
        parse_evaluation(raw, proposal)
        for raw, proposal in zip(raw_evaluations, proposals, strict=True)
    ]
    selections = [select_action(support) for support in supports]

    ordering = {
        "root_requests_completed": len(raw_proposals) + len(raw_evaluations),
        "expected_root_requests": 4,
        "source_values_loaded": False,
        "root_payload_sha256": [row["payload_sha256"] for row in privacy],
    }
    checkpoint(private / "ORDERING.json", ordering)

    source_values = []
    action_replay_audits = []
    for task, selected in zip(tasks, selections, strict=True):
        _, values, action_audit = selected_source_values(task, selected)
        source_values.append(values)
        action_replay_audits.append(action_audit)
    ordering.update(
        {
            "source_values_loaded": True,
            "source_values_loaded_after_root_requests": (
                ordering["root_requests_completed"]
                == ordering["expected_root_requests"]
            ),
        }
    )
    checkpoint(private / "ORDERING.json", ordering)

    codec_messages = []
    codec_seeds = []
    codec_layout = []
    for task_index, (task, selected, values) in enumerate(
        zip(tasks, selections, source_values, strict=True)
    ):
        messages, audit = codec_messages_for(task, selected, values)
        for replicate in range(2):
            codec_messages.append(messages)
            codec_seeds.append(CODEC_SEED_START + 10 * task_index + replicate)
            codec_layout.append({"task_index": task_index, "replicate": replicate})
            privacy.append(dict(audit, replicate=replicate))
    raw_codec = []
    for task_index in range(2):
        start = 2 * task_index
        responses = call_batch(
            adapter,
            codec_messages[start : start + 2],
            codec_seeds[start : start + 2],
            temperature=CODEC_TEMPERATURE,
            response_format=codec_response_format(len(source_values[task_index])),
            max_tokens=CODEC_MAX_TOKENS,
        )
        raw_codec.extend(responses)
        raw_bank.update(
            {
                "codec_layout": codec_layout[: len(raw_codec)],
                "codec_seeds": codec_seeds[: len(raw_codec)],
                "codec_responses": list(raw_codec),
            }
        )
        checkpoint(private / "RAW_RESPONSES.json", raw_bank)
        checkpoint(private / "PRIVACY.json", {"audits": privacy})
    raw_bank.update(
        {
            "codec_layout": codec_layout,
            "codec_seeds": codec_seeds,
            "codec_responses": raw_codec,
        }
    )
    checkpoint(private / "RAW_RESPONSES.json", raw_bank)
    checkpoint(private / "PRIVACY.json", {"audits": privacy})

    mappings = [
        parse_codec_mapping(raw_codec[2 * task + replicate], len(source_values[task]))
        for task in range(2)
        for replicate in range(2)
    ]
    task_diagnostics = []
    agreement_checks = []
    used_option_checks = []
    non_other_checks = []
    other_count_checks = []
    top_realized_checks = []
    for task_index, selected in enumerate(selections):
        first = mappings[2 * task_index]
        second = mappings[2 * task_index + 1]
        agreement = first == second
        used = sorted(set(first))
        non_other = [option for option in used if option != 3]
        other_count = sum(option == 3 for option in first)
        top_realized = all(option in used for option in selected["top_options"])
        agreement_checks.append(agreement)
        used_option_checks.append(len(used) >= 2)
        non_other_checks.append(len(non_other) >= 2)
        other_count_checks.append(other_count <= 1)
        top_realized_checks.append(top_realized)
        task_diagnostics.append(
            {
                "task_index": task_index,
                "proposal_sha256": proposals[task_index]["diagnostic"][
                    "structural_sha256"
                ],
                "selected_action_index": selected["index"],
                "selected_action_id": selected["action_id"],
                "selected_mutual_information_nats": selected["mutual_information"],
                "full_cig_action_replays": action_replay_audits[task_index][
                    "action_replay_exact"
                ],
                "source_value_count": len(source_values[task_index]),
                "codec_replicates_agree": agreement,
                "used_option_ids": used,
                "used_option_count": len(used),
                "used_non_other_option_count": len(non_other),
                "other_mapped_value_count": other_count,
                "top_predictive_option_ids": selected["top_options"],
                "top_predictive_option_masses": [
                    selected["predictive_option_masses"][option]
                    for option in selected["top_options"]
                ],
                "top_predictive_options_realized": top_realized,
                "private_mapping_sha256": structural_hash(first),
            }
        )

    usage, transport_gates = usage_gates(adapter)
    mechanics_gates = {
        "exact_two_codec_tasks": len(tasks) == 2,
        "exact_two_proposals_and_evaluations": len(proposals) == 2
        and len(supports) == 2,
        "exact_four_codec_responses": len(raw_codec) == 4,
        "all_proposals_exclude_weights_likelihoods_and_source_values": all(
            proposal["diagnostic"]["contains_weights_likelihoods_or_source_values"]
            is False
            for proposal in proposals
        ),
        "all_proposals_use_exact_typed_action_permutations": all(
            [row["action_id"] for row in proposal["actions"]]
            == [row["action_id"] for row in task["actions"]]
            for task, proposal in zip(tasks, proposals, strict=True)
        ),
        "no_proposal_emits_question_text": all(
            proposal["diagnostic"]["model_emitted_question_text"] is False
            for proposal in proposals
        ),
        "all_evaluators_normalized": all(
            abs(support["diagnostic"]["normalized_probability_sum"] - 1.0) <= 1e-9
            and support["diagnostic"]["all_likelihood_vectors_normalized"] is True
            for support in supports
        ),
        "all_prompt_privacy_audits_pass": len(privacy) == EXPECTED_REQUESTS
        and all(row.get("passed") is True for row in privacy),
        "all_selected_actions_replay_from_full_cig": all(
            row["action_replay_exact"] is True for row in action_replay_audits
        ),
        "all_selected_mutual_information_at_least_005": all(
            selected["mutual_information"] >= MIN_MUTUAL_INFORMATION
            for selected in selections
        ),
        "all_top_two_predictive_masses_at_least_010": all(
            selected["predictive_option_masses"][option] >= MIN_TOP_OPTION_MASS
            for selected in selections
            for option in selected["top_options"]
        ),
        "all_selected_actions_have_at_least_two_values": all(
            len(values) >= 2 for values in source_values
        ),
        "all_codec_replicates_agree": all(agreement_checks),
        "all_codec_mappings_use_at_least_two_options": all(used_option_checks),
        "all_codec_mappings_use_at_least_two_non_other_options": all(
            non_other_checks
        ),
        "all_codec_mappings_use_other_at_most_once": all(other_count_checks),
        "all_top_predictive_options_are_realized": all(top_realized_checks),
        "source_values_loaded_only_after_four_root_calls": ordering[
            "source_values_loaded_after_root_requests"
        ],
    }
    gates = {**transport_gates, **mechanics_gates}
    gates["all_pass"] = all(gates.values())
    status = "passed" if gates["all_pass"] else "calibration_failed"
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "authorizes": (
            "separate_typed_action_mechanics_preregistration_only"
            if gates["all_pass"]
            else "nothing"
        ),
        "bindings": bindings,
        "schedule": {
            "proposals": 2,
            "evaluators": 2,
            "codec_calls": 4,
            "expected_accepted_requests": EXPECTED_REQUESTS,
        },
        "task_diagnostics": task_diagnostics,
        "gates": gates,
        "usage": usage,
        "daily_budget_status": dict(daily_budget_status or {}),
        "source_values_publicly_reported": False,
        "codec_mappings_publicly_reported": False,
        "policy_endpoint_opened": False,
        "mechanics_opened": False,
        "development_opened": False,
        "confirmation_opened": False,
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    adapter = build_adapter(
        run_id="regretbench-typed-action-exact8",
        output_dir=args.output_dir,
    )
    result = run_smoke(output_dir=args.output_dir, adapter=adapter)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
