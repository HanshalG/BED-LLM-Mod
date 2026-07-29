#!/usr/bin/env python3
"""Qualify LogDx semantic support, likelihood, and follow-up serving."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import logdx_agent_chain_source_audit as source
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_predictive_risk_replication import (
    SeededStructuredAdapter,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "logdx-semantic-bed-serving-smoke-1"
SOURCE_AUDIT_SHA256 = (
    "7f6666a19b635087b78015a1ff98b8082c897c60a2eaca20ab8e73440931c4ff"
)
PLANNER_MODEL_ID = "openai/gpt-5.4-mini"
UPDATER_MODEL_ID = "google/gemini-2.5-flash"
PLANNER_SEED = 37_900
UPDATER_SEED = 38_000
PLANNER_TEMPERATURE = 0.7
UPDATER_TEMPERATURE = 0.0
NUM_CASES = 5
NUM_HYPOTHESES = 6
NUM_FIRST_QUERIES = 4
NUM_FOLLOWUPS = 3
EXPECTED_REQUESTS = NUM_CASES * 2
CONCURRENCY = 5
PLANNER_MAX_TOKENS = 3_000
UPDATER_MAX_TOKENS = 2_000
PROJECTED_COST_USD = 0.12
RUN_BUDGET_USD = 0.30
MIN_LIKELIHOOD_RANGE = 20
MIN_LIKELIHOOD_CASES = 4
MIN_PROBE_MATCH_CASES = 4
MIN_DEPENDENCY_CASES = 4
MIN_DEPENDENCY_TYPES = 2
CASE_SPECS = (
    ("v2/dev", "pip-pytest-network-github-v2-001"),
    ("dev", "mypy-pandas-001"),
    ("dev", "pytest-pandas-001"),
    ("dev", "cargo-tokio-001"),
    ("v2/dev", "pnpm-jest-config-v2-001"),
)
SAFE_METADATA_FIELDS = (
    "case_id",
    "repo",
    "source",
    "workflow_name",
    "job_name",
    "framework",
)


class StructuredModel(Protocol):
    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class ServingExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def canonical_text(value: str) -> str:
    return " ".join(value.casefold().split())


def strict_json_object(response: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not exact JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def planner_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "logdx_semantic_support",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses", "candidate_greps"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": NUM_HYPOTHESES,
                        "maxItems": NUM_HYPOTHESES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "description"],
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "minLength": 2,
                                    "maxLength": 2,
                                },
                                "description": {
                                    "type": "string",
                                    "minLength": 12,
                                    "maxLength": 600,
                                },
                            },
                        },
                    },
                    "candidate_greps": {
                        "type": "array",
                        "minItems": NUM_FIRST_QUERIES,
                        "maxItems": NUM_FIRST_QUERIES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "pattern"],
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "minLength": 2,
                                    "maxLength": 2,
                                },
                                "pattern": {
                                    "type": "string",
                                    "minLength": 4,
                                    "maxLength": 240,
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def updater_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "logdx_semantic_update",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["likelihoods", "followups"],
                "properties": {
                    "likelihoods": {
                        "type": "array",
                        "minItems": NUM_HYPOTHESES,
                        "maxItems": NUM_HYPOTHESES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "score"],
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "minLength": 2,
                                    "maxLength": 2,
                                },
                                "score": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 100,
                                },
                            },
                        },
                    },
                    "followups": {
                        "type": "array",
                        "minItems": NUM_FOLLOWUPS,
                        "maxItems": NUM_FOLLOWUPS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "tool", "argument"],
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "minLength": 2,
                                    "maxLength": 2,
                                },
                                "tool": {
                                    "type": "string",
                                    "enum": ["grep", "view_log_lines"],
                                },
                                "argument": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 240,
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def validate_pattern(pattern: Any, *, label: str) -> str:
    if not isinstance(pattern, str):
        raise ValueError(f"{label} must be a string")
    pattern = pattern.strip()
    if not 4 <= len(pattern) <= 240:
        raise ValueError(f"{label} has invalid length")
    try:
        re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"{label} is not a valid regex: {exc}") from exc
    if not source._literal_regex_alternatives(pattern):
        raise ValueError(f"{label} is generic or has no substantive literal")
    return pattern


def parse_planner_response(response: str) -> dict[str, Any]:
    value = strict_json_object(response, label="planner response")
    if set(value) != {"hypotheses", "candidate_greps"}:
        raise ValueError("planner response has wrong top-level fields")

    hypotheses = value["hypotheses"]
    if not isinstance(hypotheses, list) or len(hypotheses) != NUM_HYPOTHESES:
        raise ValueError("planner must return exactly six hypotheses")
    parsed_hypotheses = []
    for index, item in enumerate(hypotheses, start=1):
        if not isinstance(item, dict) or set(item) != {"id", "description"}:
            raise ValueError(f"hypotheses[{index - 1}] has wrong fields")
        if item["id"] != f"H{index}":
            raise ValueError("hypothesis IDs must be H1 through H6")
        description = item["description"]
        if (
            not isinstance(description, str)
            or not 12 <= len(description.strip()) <= 600
        ):
            raise ValueError(f"hypotheses[{index - 1}] has invalid description")
        parsed_hypotheses.append(
            {"id": item["id"], "description": description.strip()}
        )
    if len(
        {canonical_text(item["description"]) for item in parsed_hypotheses}
    ) != NUM_HYPOTHESES:
        raise ValueError("planner hypotheses are not unique")

    queries = value["candidate_greps"]
    if not isinstance(queries, list) or len(queries) != NUM_FIRST_QUERIES:
        raise ValueError("planner must return exactly four candidate greps")
    parsed_queries = []
    for index, item in enumerate(queries, start=1):
        if not isinstance(item, dict) or set(item) != {"id", "pattern"}:
            raise ValueError(f"candidate_greps[{index - 1}] has wrong fields")
        if item["id"] != f"Q{index}":
            raise ValueError("query IDs must be Q1 through Q4")
        parsed_queries.append(
            {
                "id": item["id"],
                "pattern": validate_pattern(
                    item["pattern"],
                    label=f"candidate_greps[{index - 1}].pattern",
                ),
            }
        )
    if len(
        {canonical_text(item["pattern"]) for item in parsed_queries}
    ) != NUM_FIRST_QUERIES:
        raise ValueError("planner query patterns are not unique")
    return {
        "hypotheses": parsed_hypotheses,
        "candidate_greps": parsed_queries,
    }


def followup_to_call(followup: dict[str, str]) -> dict[str, Any]:
    if followup["tool"] == "grep":
        return {
            "tool": "grep",
            "args": {
                "pattern": followup["argument"],
                "before": 2,
                "after": 8,
                "max_matches": 30,
            },
        }
    return {
        "tool": "view_log_lines",
        "args": {
            "center_line": int(followup["argument"]),
            "radius": 30,
        },
    }


def parse_updater_response(response: str) -> dict[str, Any]:
    value = strict_json_object(response, label="updater response")
    if set(value) != {"likelihoods", "followups"}:
        raise ValueError("updater response has wrong top-level fields")

    likelihoods = value["likelihoods"]
    if not isinstance(likelihoods, list) or len(likelihoods) != NUM_HYPOTHESES:
        raise ValueError("updater must return exactly six likelihoods")
    parsed_likelihoods = []
    for index, item in enumerate(likelihoods, start=1):
        if not isinstance(item, dict) or set(item) != {"id", "score"}:
            raise ValueError(f"likelihoods[{index - 1}] has wrong fields")
        if item["id"] != f"H{index}":
            raise ValueError("likelihood IDs must be H1 through H6")
        score = item["score"]
        if type(score) is not int or not 0 <= score <= 100:
            raise ValueError(f"likelihoods[{index - 1}] has invalid score")
        parsed_likelihoods.append({"id": item["id"], "score": score})

    followups = value["followups"]
    if not isinstance(followups, list) or len(followups) != NUM_FOLLOWUPS:
        raise ValueError("updater must return exactly three followups")
    parsed_followups = []
    for index, item in enumerate(followups, start=1):
        if (
            not isinstance(item, dict)
            or set(item) != {"id", "tool", "argument"}
        ):
            raise ValueError(f"followups[{index - 1}] has wrong fields")
        if item["id"] != f"F{index}":
            raise ValueError("followup IDs must be F1 through F3")
        tool = item["tool"]
        argument = item["argument"]
        if tool == "grep":
            argument = validate_pattern(
                argument,
                label=f"followups[{index - 1}].argument",
            )
        elif tool == "view_log_lines":
            if (
                not isinstance(argument, str)
                or not argument.isascii()
                or not argument.isdecimal()
                or int(argument) <= 0
            ):
                raise ValueError(
                    f"followups[{index - 1}] needs a positive line number"
                )
            argument = str(int(argument))
        else:
            raise ValueError(f"followups[{index - 1}] has invalid tool")
        parsed_followups.append(
            {"id": item["id"], "tool": tool, "argument": argument}
        )
    canonical_followups = {
        (item["tool"], canonical_text(item["argument"]))
        for item in parsed_followups
    }
    if len(canonical_followups) != NUM_FOLLOWUPS:
        raise ValueError("updater followups are not unique")
    return {
        "likelihoods": parsed_likelihoods,
        "followups": parsed_followups,
    }


def load_cases() -> list[dict[str, Any]]:
    source.verify_source()
    case_paths, _ = source.verify_manifests()
    cases = []
    for split, case_id in CASE_SPECS:
        case_dir = case_paths.get(case_id)
        if case_dir != source.SOURCE_ROOT / "cases" / split / case_id:
            raise ValueError(f"case split changed for {case_id}")
        metadata = json.loads((case_dir / "case.json").read_text())
        safe_metadata = {
            field: metadata[field]
            for field in SAFE_METADATA_FIELDS
            if field in metadata
        }
        if safe_metadata.get("case_id") != case_id:
            raise ValueError(f"safe metadata changed for {case_id}")
        initial_context_path = (
            source.SOURCE_ROOT
            / "results"
            / split
            / "rtk-log"
            / f"{case_id}.txt"
        )
        cases.append(
            {
                "split": split,
                "case_id": case_id,
                "safe_metadata": safe_metadata,
                "initial_context": initial_context_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ),
                "raw_log_path": case_dir / "raw.log",
            }
        )
    return cases


def planner_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    request = {
        "case_id": case["case_id"],
        "safe_metadata": case["safe_metadata"],
        "reduced_context_method": "rtk-log",
        "reduced_context": case["initial_context"],
    }
    return [
        {
            "role": "system",
            "content": (
                "You design active evidence acquisition for CI root-cause "
                "diagnosis. From only the safe metadata and reduced log context, "
                "generate six distinct, concrete root-cause hypotheses and four "
                "distinct case-specific regex searches over the hidden full log. "
                "Do not assume benchmark ground truth. Q1 is the fixed serving "
                "probe: make it broad enough to match likely evidence but targeted "
                "to distinctions among your hypotheses. Q2-Q4 should seek "
                "complementary evidence. Patterns must be valid Python-compatible "
                "case-insensitive regexes and must contain distinctive literals, "
                "not only generic words such as error, failed, or traceback."
            ),
        },
        {"role": "user", "content": source.canonical_json(request)},
    ]


def updater_messages(
    case: dict[str, Any],
    planner: dict[str, Any],
    observation: str,
) -> list[dict[str, str]]:
    probe = planner["candidate_greps"][0]
    request = {
        "case_id": case["case_id"],
        "safe_metadata": case["safe_metadata"],
        "initial_reduced_context": case["initial_context"],
        "hypotheses": planner["hypotheses"],
        "executed_probe": probe,
        "probe_observation": observation,
    }
    return [
        {
            "role": "system",
            "content": (
                "Update a semantic CI-diagnosis belief after a deterministic log "
                "query. For each supplied hypothesis, score the relative likelihood "
                "of seeing this exact probe observation if that hypothesis were "
                "true, from 0 to 100. Scores are likelihoods, not posterior "
                "probabilities, and need not sum to 100. Then propose exactly three "
                "valid follow-up actions. A grep argument is a Python-compatible "
                "regex. A view_log_lines argument is one positive base-10 line "
                "number copied from the observation. At least one follow-up must use "
                "a distinctive literal or line number that appears in the probe "
                "observation but was absent from the initial reduced context and "
                "prior probe arguments. Do not merely repeat the probe or use only "
                "generic failure words."
            ),
        },
        {"role": "user", "content": source.canonical_json(request)},
    ]


def _probe_call(planner: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool": "grep",
        "args": {
            "pattern": planner["candidate_greps"][0]["pattern"],
            "before": 2,
            "after": 8,
            "max_matches": 30,
        },
    }


def _usage(models: Sequence[StructuredModel]) -> dict[str, Any]:
    snapshots = [model.usage_snapshot() for model in models]
    totals: dict[str, Any] = {}
    for key in (
        "adapter_requests",
        "http_attempts",
        "retry_count",
        "provider_error_retries",
        "adapter_reasoning_tokens",
        "forced_exits",
        "adapter_prompt_tokens",
        "adapter_completion_tokens",
        "adapter_cost_usd",
    ):
        totals[key] = sum(float(item.get(key, 0) or 0) for item in snapshots)
    for key in (
        "adapter_requests",
        "http_attempts",
        "retry_count",
        "provider_error_retries",
        "adapter_reasoning_tokens",
        "forced_exits",
        "adapter_prompt_tokens",
        "adapter_completion_tokens",
    ):
        totals[key] = int(totals[key])
    totals["run_cost_usd"] = totals.pop("adapter_cost_usd")
    totals["models"] = snapshots
    return totals


def _checkpoint_private(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def case_metrics(
    *,
    case: dict[str, Any],
    planner: dict[str, Any],
    updater: dict[str, Any],
    probe_observation: str,
    tools_module: Any,
) -> dict[str, Any]:
    probe_call = _probe_call(planner)
    scores = [item["score"] for item in updater["likelihoods"]]
    dependencies = []
    execution_error_count = 0
    for followup in updater["followups"]:
        call = followup_to_call(followup)
        dependencies.extend(
            source.dependencies_for_call(
                call,
                initial_context=case["initial_context"],
                prior_observations=[probe_observation],
                prior_calls=[probe_call],
            )
        )
        observation = tools_module.dispatch_tool(
            call["tool"],
            call["args"],
            str(case["raw_log_path"]),
        )
        if observation.startswith("ERROR:"):
            execution_error_count += 1
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "hypothesis_count": len(planner["hypotheses"]),
        "unique_hypothesis_count": len(
            {
                canonical_text(item["description"])
                for item in planner["hypotheses"]
            }
        ),
        "first_query_count": len(planner["candidate_greps"]),
        "unique_first_query_count": len(
            {
                canonical_text(item["pattern"])
                for item in planner["candidate_greps"]
            }
        ),
        "probe_has_matches": not probe_observation.startswith(
            "No matches found"
        ),
        "probe_observation_tokens_estimate": max(
            1, len(probe_observation) // 4
        ),
        "likelihood_count": len(scores),
        "likelihood_range": max(scores) - min(scores),
        "unique_likelihood_count": len(set(scores)),
        "followup_count": len(updater["followups"]),
        "followup_execution_error_count": execution_error_count,
        "dependency_followup_count": len(dependencies),
        "dependency_types": sorted(
            {item["dependency_type"] for item in dependencies}
        ),
    }


def aggregate_gates(
    *,
    cases: Sequence[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, bool]:
    likelihood_cases = sum(
        case["likelihood_range"] >= MIN_LIKELIHOOD_RANGE
        and case["unique_likelihood_count"] >= 3
        for case in cases
    )
    dependency_cases = sum(
        case["dependency_followup_count"] >= 1 for case in cases
    )
    dependency_types = {
        dependency_type
        for case in cases
        for dependency_type in case["dependency_types"]
    }
    gates = {
        "exact_10_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "exact_10_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_five_planner_and_updater_schemas_parse": (
            len(cases) == NUM_CASES
        ),
        "all_cases_have_six_unique_hypotheses": all(
            case["unique_hypothesis_count"] == NUM_HYPOTHESES
            for case in cases
        ),
        "all_cases_have_four_unique_valid_regexes": all(
            case["unique_first_query_count"] == NUM_FIRST_QUERIES
            for case in cases
        ),
        "probe_matches_on_at_least_four_cases": (
            sum(case["probe_has_matches"] for case in cases)
            >= MIN_PROBE_MATCH_CASES
        ),
        "likelihood_dynamic_range_on_at_least_four_cases": (
            likelihood_cases >= MIN_LIKELIHOOD_CASES
        ),
        "all_fifteen_followups_execute_without_error": all(
            case["followup_count"] == NUM_FOLLOWUPS
            and case["followup_execution_error_count"] == 0
            for case in cases
        ),
        "observation_dependent_followup_on_at_least_four_cases": (
            dependency_cases >= MIN_DEPENDENCY_CASES
        ),
        "at_least_two_dependency_types": (
            len(dependency_types) >= MIN_DEPENDENCY_TYPES
        ),
        "cost_at_most_0_30": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def run_smoke(
    *,
    cases: Sequence[dict[str, Any]],
    planner_model: StructuredModel,
    updater_model: StructuredModel,
    raw_path: Path,
) -> dict[str, Any]:
    if len(cases) != NUM_CASES:
        raise ValueError(f"serving smoke requires exactly {NUM_CASES} cases")
    tools_module = source.load_agent_tools()
    raw: dict[str, Any] = {
        "case_ids": [case["case_id"] for case in cases],
        "planner_responses": [],
        "probe_observations": [],
        "updater_responses": [],
        "ground_truth_accessed": False,
        "diagnosis_evaluator_accessed": False,
        "policy_endpoint_accessed": False,
        "confirmation_accessed": False,
    }
    try:
        planner_responses = planner_model.chat_complete_messages_batched_structured(
            [planner_messages(case) for case in cases],
            temperature=PLANNER_TEMPERATURE,
            block_size=CONCURRENCY,
            response_format=planner_response_format(),
            max_new_tokens=PLANNER_MAX_TOKENS,
        )
        raw["planner_responses"] = list(planner_responses)
        _checkpoint_private(raw_path, raw)
        if len(planner_responses) != NUM_CASES:
            raise ValueError("planner response count changed")
        planners = [
            parse_planner_response(response) for response in planner_responses
        ]

        probe_observations = []
        for case, planner in zip(cases, planners, strict=True):
            probe_call = _probe_call(planner)
            observation = tools_module.dispatch_tool(
                probe_call["tool"],
                probe_call["args"],
                str(case["raw_log_path"]),
            )
            if observation.startswith("ERROR:"):
                raise ValueError(f"probe tool failed for {case['case_id']}")
            probe_observations.append(observation)
        raw["probe_observations"] = probe_observations
        _checkpoint_private(raw_path, raw)

        updater_responses = updater_model.chat_complete_messages_batched_structured(
            [
                updater_messages(case, planner, observation)
                for case, planner, observation in zip(
                    cases,
                    planners,
                    probe_observations,
                    strict=True,
                )
            ],
            temperature=UPDATER_TEMPERATURE,
            block_size=CONCURRENCY,
            response_format=updater_response_format(),
            max_new_tokens=UPDATER_MAX_TOKENS,
        )
        raw["updater_responses"] = list(updater_responses)
        _checkpoint_private(raw_path, raw)
        if len(updater_responses) != NUM_CASES:
            raise ValueError("updater response count changed")
        updaters = [
            parse_updater_response(response) for response in updater_responses
        ]
        public_cases = [
            case_metrics(
                case=case,
                planner=planner,
                updater=updater,
                probe_observation=observation,
                tools_module=tools_module,
            )
            for case, planner, updater, observation in zip(
                cases,
                planners,
                updaters,
                probe_observations,
                strict=True,
            )
        ]
        usage = _usage((planner_model, updater_model))
    except Exception as exc:
        _checkpoint_private(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage((planner_model, updater_model)),
        ) from exc

    gates = aggregate_gates(cases=public_cases, usage=usage)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_audit_sha256": SOURCE_AUDIT_SHA256,
            "planner_model": PLANNER_MODEL_ID,
            "updater_model": UPDATER_MODEL_ID,
            "planner_seed": PLANNER_SEED,
            "updater_seed": UPDATER_SEED,
            "planner_temperature": PLANNER_TEMPERATURE,
            "updater_temperature": UPDATER_TEMPERATURE,
            "initial_context_method": "rtk-log",
            "expected_requests": EXPECTED_REQUESTS,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "ground_truth_accessed": False,
            "diagnosis_evaluator_accessed": False,
            "policy_endpoint_accessed": False,
            "confirmation_accessed": False,
        },
        "metrics": {
            "case_count": len(public_cases),
            "probe_match_case_count": sum(
                case["probe_has_matches"] for case in public_cases
            ),
            "likelihood_dynamic_range_case_count": sum(
                case["likelihood_range"] >= MIN_LIKELIHOOD_RANGE
                and case["unique_likelihood_count"] >= 3
                for case in public_cases
            ),
            "dependency_case_count": sum(
                case["dependency_followup_count"] >= 1
                for case in public_cases
            ),
            "dependency_types": sorted(
                {
                    dependency_type
                    for case in public_cases
                    for dependency_type in case["dependency_types"]
                }
            ),
            "mean_likelihood_range": statistics.fmean(
                case["likelihood_range"] for case in public_cases
            ),
        },
        "cases": public_cases,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self, role: str) -> None:
        self.role = role
        self.requests = 0

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, response_format, max_new_tokens
        responses = []
        for messages in batch_messages:
            request = json.loads(messages[-1]["content"])
            if self.role == "planner":
                tokens = re.findall(
                    r"[A-Za-z][A-Za-z0-9_.:-]{6,}",
                    request["reduced_context"],
                )
                token = next(
                    (
                        item
                        for item in tokens
                        if canonical_text(item)
                        not in {"warning", "failure", "failed", "error"}
                    ),
                    "workflow",
                )
                responses.append(
                    json.dumps(
                        {
                            "hypotheses": [
                                {
                                    "id": f"H{index}",
                                    "description": (
                                        f"Distinct fixture root cause hypothesis "
                                        f"{index} involving {token}"
                                    ),
                                }
                                for index in range(1, NUM_HYPOTHESES + 1)
                            ],
                            "candidate_greps": [
                                {
                                    "id": f"Q{index}",
                                    "pattern": (
                                        re.escape(token)
                                        if index == 1
                                        else f"{re.escape(token)}|fixture_signal_{index}"
                                    ),
                                }
                                for index in range(1, NUM_FIRST_QUERIES + 1)
                            ],
                        }
                    )
                )
            else:
                case_index = self.requests + len(responses)
                initial = request["initial_reduced_context"]
                observation = request["probe_observation"]
                line_numbers = [
                    int(item)
                    for item in re.findall(r"(?m)^\s*(\d+)\s*:", observation)
                    if not source._line_prefix_present(initial, int(item))
                ]
                if line_numbers and case_index % 2 == 0:
                    first_followup = {
                        "id": "F1",
                        "tool": "view_log_lines",
                        "argument": str(line_numbers[0]),
                    }
                else:
                    probe_pattern = request["executed_probe"]["pattern"]
                    candidates = re.findall(
                        r"[A-Za-z][A-Za-z0-9_.:/-]{7,}",
                        source.observation_log_content(observation),
                    )
                    literal = next(
                        (
                            item
                            for item in candidates
                            if item.casefold() not in initial.casefold()
                            and item.casefold() not in probe_pattern.casefold()
                        ),
                        None,
                    )
                    if literal is not None:
                        first_followup = {
                            "id": "F1",
                            "tool": "grep",
                            "argument": re.escape(literal),
                        }
                    elif line_numbers:
                        first_followup = {
                            "id": "F1",
                            "tool": "view_log_lines",
                            "argument": str(line_numbers[0]),
                        }
                    else:
                        raise ValueError(
                            "fixture observation has no dependent followup"
                        )
                responses.append(
                    json.dumps(
                        {
                            "likelihoods": [
                                {"id": f"H{index}", "score": score}
                                for index, score in enumerate(
                                    (95, 75, 55, 35, 15, 5),
                                    start=1,
                                )
                            ],
                            "followups": [
                                first_followup,
                                {
                                    "id": "F2",
                                    "tool": "grep",
                                    "argument": "fixture_followup_signal_two",
                                },
                                {
                                    "id": "F3",
                                    "tool": "grep",
                                    "argument": "fixture_followup_signal_three",
                                },
                            ],
                        }
                    )
                )
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def _adapter(
    *,
    model: str,
    seed: int,
    run_id: str,
    output_dir: Path,
) -> SeededStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=500.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=PLANNER_MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return SeededStructuredAdapter(
        ModelSpec(model=model, backend="openrouter", max_model_len=131_072),
        config,
        request_seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"

    if args.dry_run:
        planner: StructuredModel = DeterministicFixtureModel("planner")
        updater: StructuredModel = DeterministicFixtureModel("updater")
    else:
        planner = _adapter(
            model=PLANNER_MODEL_ID,
            seed=PLANNER_SEED,
            run_id=f"{args.run_id}-planner",
            output_dir=args.output_dir,
        )
        updater = _adapter(
            model=UPDATER_MODEL_ID,
            seed=UPDATER_SEED,
            run_id=f"{args.run_id}-updater",
            output_dir=args.output_dir,
        )
    try:
        payload = run_smoke(
            cases=load_cases(),
            planner_model=planner,
            updater_model=updater,
            raw_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
            "ground_truth_accessed": False,
            "diagnosis_evaluator_accessed": False,
            "policy_endpoint_accessed": False,
            "confirmation_accessed": False,
        }
        if isinstance(exc, ServingExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        checkpoint(args.output_dir / "SERVING_FAILURE.json", failure)
        raise
    checkpoint(args.output_dir / "SERVING.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "metrics": payload["metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
