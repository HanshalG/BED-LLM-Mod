#!/usr/bin/env python3
"""Run the RegretBench DeepSeek path-dependent support-recovery gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REGRETBENCH_ROOT = REPO_ROOT / "external/RegretBench"
REGRETBENCH_SRC = REGRETBENCH_ROOT / "src"
for path in (REPO_ROOT, REGRETBENCH_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helpers import Config, ModelSpec
from regretbench.mapping.semantic_action_mapper import SemanticActionMapper
from regretbench.schemas.cig import CIG, load_cig
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.number_game_qwen_history_blind_serving_smoke import (
    PerRequestSeedStructuredAdapter,
)
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-support-recovery-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
TEMPERATURE = 0.7
MAX_TOKENS = 2_200
MIN_UNIQUE_HYPOTHESES = 4
QUESTIONS_PER_RESPONSE = 4
HYPOTHESES_PER_RESPONSE = 8
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 202608087000
SOURCE_RESULT = (
    REPO_ROOT / "results/nonmyopic/regretbench_llm_native_source_audit/RESULT.json"
)
SOURCE_MANIFEST = (
    REPO_ROOT
    / "results/nonmyopic/regretbench_llm_native_source_audit/"
    "SOURCE_PROTOCOL_MANIFEST.json"
)
PREREGISTRATION = (
    REPO_ROOT
    / "results/nonmyopic/"
    "REGRETBENCH_DEEPSEEK_SUPPORT_RECOVERY_PREREGISTRATION.md"
)
SOURCE_RESULT_SHA256 = (
    "d7a10f15ecf6779520fbb20712c8d43059d8b03168fa904fe72e82ffe87578de"
)
SOURCE_MANIFEST_SHA256 = (
    "8a46b40395487aae0857d1a61d6f680beef579414c4580acf09d7e0b30b38e97"
)
PREREGISTRATION_SHA256 = (
    "32825b97d9c73bbe940d2f0dfff5cb5deebd80f2cfec16812f227659a88a6dfe"
)
SPLIT_HASHES = {
    "mechanics": "707be5a1d1f86d6a0dc08ee61df77da1b9597093ac557d2e7706fcad8ef3b2f6",
    "development": "29d33b2fda0be7cc4eea6f4d9d4fe74fe200b5c580632dcb7ba844c3b825af69",
}
STAGES = {
    "smoke": {
        "split": "mechanics",
        "task_count": 4,
        "branch_count": 3,
        "expected_requests": 10,
        "truth_seed_start": 202608081100,
        "root_seed_start": 202608083000,
        "refresh_seed_start": 202608084000,
        "concurrency": 10,
        "run_budget_usd": 0.20,
        "projected_cost_usd": 0.02,
    },
    "development": {
        "split": "development",
        "task_count": 64,
        "branch_count": 64,
        "expected_requests": 192,
        "truth_seed_start": 202608082000,
        "root_seed_start": 202608085000,
        "refresh_seed_start": 202608086000,
        "concurrency": 24,
        "run_budget_usd": 0.50,
        "projected_cost_usd": 0.35,
    },
}
UNSUPPORTED_REPLY = "I cannot answer that clarification."


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.casefold())).strip()


def validate_source_bindings() -> dict[str, Any]:
    expected = {
        SOURCE_RESULT: SOURCE_RESULT_SHA256,
        SOURCE_MANIFEST: SOURCE_MANIFEST_SHA256,
        PREREGISTRATION: PREREGISTRATION_SHA256,
    }
    for path, digest in expected.items():
        if not path.exists() or sha256_file(path) != digest:
            raise ValueError(f"bound artifact changed: {path}")
    result = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if result.get("status") != "source_protocol_pass" or not result.get("gates", {}).get(
        "all_pass"
    ):
        raise ValueError("source audit does not authorize the support smoke")
    for split, digest in SPLIT_HASHES.items():
        if manifest["splits"][split]["ids_sha256"] != digest:
            raise ValueError(f"{split} split hash changed")
    return manifest


def load_stage_cigs(stage: str) -> list[CIG]:
    protocol = STAGES[stage]
    manifest = validate_source_bindings()
    ids = manifest["splits"][protocol["split"]]["ids"]
    if len(ids) != protocol["task_count"]:
        raise ValueError(f"{stage} task count changed")
    cigs = [
        load_cig(
            REGRETBENCH_ROOT
            / "data/OpenDomainQA/test"
            / f"{cig_id}.json"
        )
        for cig_id in ids
    ]
    if [cig.cig_id for cig in cigs] != ids:
        raise ValueError("loaded CIG order changed")
    return cigs


def validate_smoke_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol") or {}
    if (
        payload.get("status") != "passed"
        or payload.get("interface_version") != INTERFACE_VERSION
        or payload.get("authorizes") != "development_support_recovery_only"
        or protocol.get("stage") != "smoke"
        or protocol.get("model") != MODEL_ID
        or protocol.get("expected_requests") != STAGES["smoke"]["expected_requests"]
        or protocol.get("split_hash") != SPLIT_HASHES["mechanics"]
        or protocol.get("source_result_sha256") != SOURCE_RESULT_SHA256
        or protocol.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or protocol.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or protocol.get("efficacy_used_for_smoke_authorization") is not False
        or payload.get("science") is not None
        or not (payload.get("mechanics_gates") or {}).get("all_pass")
    ):
        raise ValueError("smoke result does not authorize development")
    return {"path": str(path), "sha256": sha256_file(path)}


def support_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "regretbench_support",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses", "questions"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": HYPOTHESES_PER_RESPONSE,
                        "maxItems": HYPOTHESES_PER_RESPONSE,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "interpretation",
                                "final_answer",
                                "prior_weight",
                            ],
                            "properties": {
                                "interpretation": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 320,
                                },
                                "final_answer": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 200,
                                },
                                "prior_weight": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 100,
                                },
                            },
                        },
                    },
                    "questions": {
                        "type": "array",
                        "minItems": QUESTIONS_PER_RESPONSE,
                        "maxItems": QUESTIONS_PER_RESPONSE,
                        "items": {
                            "type": "string",
                            "minLength": 2,
                            "maxLength": 240,
                        },
                    },
                },
            },
        },
    }


SYSTEM_PROMPT = """You maintain an explicit belief over interpretations of an ambiguous factual question.
Return exactly eight plausible, meaningfully distinct interpretations. For each, give the concise factual answer that would be correct under that interpretation and a nonnegative prior weight. Also return exactly four ranked, distinct, single-dimension clarification questions that a user could answer without giving the final factual answer. Rank the most useful question first. Do not ask for the entity name, the final answer, or an omnibus list. Use only the supplied prompt and dialogue. Return only the required JSON object."""


def public_payload(cig: CIG, dialogue: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    records = []
    for item in dialogue:
        if set(item) != {"role", "content"} or item["role"] not in {
            "assistant",
            "user",
        }:
            raise ValueError("dialogue record has invalid fields")
        content = str(item["content"]).strip()
        if not content:
            raise ValueError("dialogue content is empty")
        records.append({"role": str(item["role"]), "content": content})
    return {"task_id": cig.cig_id, "prompt": cig.prompt, "dialogue": records}


def privacy_audit(cig: CIG, payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != {"task_id", "prompt", "dialogue"}:
        raise ValueError("model payload contains forbidden fields")
    if payload["task_id"] != cig.cig_id or payload["prompt"] != cig.prompt:
        raise ValueError("model payload identity changed")
    visible = normalize_text(canonical_json(payload))
    hidden_strings: list[str] = []
    for intent in cig.intents:
        hidden_strings.append(intent.description)
        hidden_strings.extend((intent.slots or {}).values())
    hidden_strings.extend(variable.description for variable in cig.latent_variables)
    hidden_strings.extend(question.text for question in cig.reference_questions)
    accidental = []
    for hidden in hidden_strings:
        normalized = normalize_text(hidden)
        if len(normalized) >= 8 and normalized in visible:
            # Exact text is allowed only when it is already part of the visible
            # prompt or the realized environment dialogue.
            allowed_surface = normalize_text(
                str(payload["prompt"])
                + " "
                + " ".join(item["content"] for item in payload["dialogue"])
            )
            if normalized not in allowed_surface:
                accidental.append(hashlib.sha256(normalized.encode()).hexdigest())
    if accidental:
        raise ValueError("model payload leaked hidden source strings")
    return {
        "passed": True,
        "payload_keys": sorted(payload),
        "payload_sha256": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
    }


def messages_for(cig: CIG, dialogue: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    payload = public_payload(cig, dialogue)
    audit = privacy_audit(cig, payload)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(payload)},
    ], audit


def parse_support(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("support response has wrong top-level fields")
    raw_hypotheses = value["hypotheses"]
    if not isinstance(raw_hypotheses, list) or len(raw_hypotheses) != HYPOTHESES_PER_RESPONSE:
        raise ValueError("support response must contain exactly eight hypotheses")
    hypotheses = []
    seen = set()
    for index, item in enumerate(raw_hypotheses):
        if not isinstance(item, dict) or set(item) != {
            "interpretation",
            "final_answer",
            "prior_weight",
        }:
            raise ValueError(f"hypothesis {index} has wrong fields")
        interpretation = item["interpretation"]
        final_answer = item["final_answer"]
        weight = item["prior_weight"]
        if (
            not isinstance(interpretation, str)
            or not interpretation.strip()
            or not isinstance(final_answer, str)
            or not final_answer.strip()
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
        ):
            raise ValueError(f"hypothesis {index} has invalid values")
        key = (normalize_text(interpretation), normalize_text(final_answer))
        if key in seen:
            continue
        seen.add(key)
        hypotheses.append(
            {
                "interpretation": interpretation.strip(),
                "final_answer": final_answer.strip(),
                "prior_weight": float(weight),
            }
        )
    if len(hypotheses) < MIN_UNIQUE_HYPOTHESES:
        raise ValueError("support response has fewer than four unique hypotheses")
    weight_sum = sum(item["prior_weight"] for item in hypotheses)
    if weight_sum <= 0:
        raise ValueError("support weights sum to zero")
    for item in hypotheses:
        item["probability"] = item.pop("prior_weight") / weight_sum

    questions = value["questions"]
    if not isinstance(questions, list) or len(questions) != QUESTIONS_PER_RESPONSE:
        raise ValueError("support response must contain exactly four questions")
    cleaned_questions = []
    question_keys = set()
    for index, question in enumerate(questions):
        if not isinstance(question, str) or not question.strip().endswith("?"):
            raise ValueError(f"question {index} is not a question")
        cleaned = question.strip()
        key = normalize_text(cleaned)
        if not key or key in question_keys:
            raise ValueError("support response contains duplicate questions")
        question_keys.add(key)
        cleaned_questions.append(cleaned)
    return {
        "hypotheses": hypotheses,
        "questions": cleaned_questions,
        "diagnostic": {
            "codec_mode": "strict_json",
            "raw_hypothesis_count": len(raw_hypotheses),
            "valid_unique_count": len(hypotheses),
            "question_count": len(cleaned_questions),
        },
    }


def lexical_alias_match(generated: str, alias: str) -> bool:
    generated_norm = normalize_text(generated)
    alias_norm = normalize_text(alias)
    if not generated_norm or not alias_norm:
        return False
    if generated_norm == alias_norm:
        return True
    shorter = min((generated_norm, alias_norm), key=len)
    longer = max((generated_norm, alias_norm), key=len)
    return len(shorter) >= 8 and len(shorter.split()) >= 2 and shorter in longer


def truth_covered(support: Mapping[str, Any], aliases: str) -> bool:
    alternatives = [alias.strip() for alias in aliases.split("|") if alias.strip()]
    return any(
        lexical_alias_match(item["final_answer"], alias)
        for item in support["hypotheses"]
        for alias in alternatives
    )


def sample_truth(cig: CIG, seed: int) -> tuple[int, Any]:
    index = random.Random(seed).randrange(len(cig.intents))
    return index, cig.intents[index]


def map_and_answer(cig: CIG, question: str, truth: Any) -> dict[str, Any]:
    parsed = SemanticActionMapper().map_question(cig, question)
    supported = parsed.facet is not None and parsed.semantic_action != "UNSUPPORTED"
    answer = UNSUPPORTED_REPLY
    if supported:
        answer = str((truth.slots or {}).get(parsed.facet, "")).strip()
        supported = bool(answer)
    return {
        "supported": supported,
        "facet": parsed.facet if supported else None,
        "confidence": float(parsed.confidence),
        "method": parsed.method,
        "answer": answer if supported else UNSUPPORTED_REPLY,
    }


def build_adapter(
    *, stage: str, run_id: str, output_dir: Path
) -> PerRequestSeedStructuredAdapter:
    protocol = STAGES[stage]
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=275.0,
        openrouter_run_budget_usd=protocol["run_budget_usd"],
        openrouter_projected_cost_usd=protocol["projected_cost_usd"],
        openrouter_concurrency=protocol["concurrency"],
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=0.0015,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return PerRequestSeedStructuredAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65_536),
        config,
    )


def _call_batch(
    adapter: StructuredAdapter,
    messages: Sequence[list[dict[str, str]]],
    seeds: Sequence[int],
) -> list[str]:
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        messages,
        seeds,
        temperature=TEMPERATURE,
        response_format=support_response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    if len(responses) != len(messages):
        raise ValueError("adapter returned the wrong response count")
    return list(responses)


def _bootstrap(values: Sequence[float], *, samples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"mean": None, "ci95": [None, None], "probability_positive": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indexes].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "probability_positive": float(np.mean(means > 0.0)),
        "samples": samples,
        "seed": BOOTSTRAP_SEED,
    }


def _one_sided_sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def scientific_summary(rows: Sequence[Mapping[str, Any]], *, samples: int) -> dict[str, Any]:
    supported = [row for row in rows if row["supported"]]
    differences = [
        float(row["conditioned_covered"]) - float(row["blind_covered"])
        for row in supported
    ]
    root_missing = [row for row in supported if not row["root_covered"]]
    missing_differences = [
        float(row["conditioned_covered"]) - float(row["blind_covered"])
        for row in root_missing
    ]
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    bootstrap = _bootstrap(differences, samples=samples)
    missing_bootstrap = _bootstrap(missing_differences, samples=samples)
    gates = {
        "at_least_48_supported_tasks": len(supported) >= 48,
        "at_least_16_root_missing_supported_tasks": len(root_missing) >= 16,
        "at_least_8_changed_conditioning_outcomes": wins + losses >= 8,
        "conditioned_minus_blind_coverage_at_least_005": (
            bootstrap["mean"] is not None and bootstrap["mean"] >= 0.05
        ),
        "bootstrap_probability_positive_at_least_080": (
            bootstrap["probability_positive"] is not None
            and bootstrap["probability_positive"] >= 0.80
        ),
        "conditioned_recoveries_exceed_losses": wins > losses,
        "root_missing_recovery_difference_at_least_010": (
            missing_bootstrap["mean"] is not None
            and missing_bootstrap["mean"] >= 0.10
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "population": {
            "all_tasks": len(rows),
            "supported_tasks": len(supported),
            "root_missing_supported_tasks": len(root_missing),
        },
        "coverage": {
            "root": float(np.mean([row["root_covered"] for row in supported]))
            if supported
            else None,
            "conditioned": float(
                np.mean([row["conditioned_covered"] for row in supported])
            )
            if supported
            else None,
            "history_blind": float(
                np.mean([row["blind_covered"] for row in supported])
            )
            if supported
            else None,
            "conditioned_minus_history_blind": bootstrap,
            "root_missing_conditioned_minus_history_blind": missing_bootstrap,
        },
        "paired_outcomes": {
            "conditioned_recoveries": wins,
            "conditioned_losses": losses,
            "ties": len(differences) - wins - losses,
            "one_sided_exact_sign_p": _one_sided_sign_p(wins, losses),
        },
        "gates": gates,
    }


def mechanics_gates(
    *,
    stage: str,
    roots: Sequence[Mapping[str, Any]],
    branches: Sequence[Mapping[str, Any]],
    privacy: Sequence[Mapping[str, Any]],
    usage: Mapping[str, Any],
) -> dict[str, bool]:
    protocol = STAGES[stage]
    all_supports = [row["root_support"] for row in roots]
    for row in branches:
        all_supports.extend([row["conditioned_support"], row["blind_support"]])
    gates = {
        "exact_task_count": len(roots) == protocol["task_count"],
        "exact_branch_count": len(branches) == protocol["branch_count"],
        "exact_response_count": len(all_supports) == protocol["expected_requests"],
        "exact_accepted_requests": usage["adapter_requests"]
        == protocol["expected_requests"],
        "exact_http_attempts": usage["http_attempts"] == protocol["expected_requests"],
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_strict_supports_have_four_unique": all(
            support["diagnostic"]["codec_mode"] == "strict_json"
            and support["diagnostic"]["valid_unique_count"]
            >= MIN_UNIQUE_HYPOTHESES
            for support in all_supports
        ),
        "every_root_has_four_questions": all(
            root["root_support"]["diagnostic"]["question_count"]
            == QUESTIONS_PER_RESPONSE
            for root in roots
        ),
        "all_privacy_audits_pass": len(privacy) == protocol["expected_requests"]
        and all(item["passed"] for item in privacy),
        "within_run_budget": usage["run_cost_usd"]
        <= protocol["run_budget_usd"] + 1e-12,
    }
    if stage == "smoke":
        gates["all_three_selected_questions_supported"] = all(
            row["supported"] for row in branches
        )
    gates["all_pass"] = all(gates.values())
    return gates


def _public_root(row: Mapping[str, Any]) -> dict[str, Any]:
    support = row["root_support"]
    return {
        "task_id": row["cig"].cig_id,
        "truth_seed": row["truth_seed"],
        "root_seed": row["root_seed"],
        "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
        "root_valid_unique_count": support["diagnostic"]["valid_unique_count"],
        "supported": row["mapping"]["supported"],
        "mapping_confidence": row["mapping"]["confidence"],
        "mapping_method": row["mapping"]["method"],
        "root_covered": row["root_covered"],
    }


def run_stage(
    *,
    stage: str,
    output_dir: Path,
    run_id: str,
    adapter: StructuredAdapter | None = None,
    daily_budget_status: Mapping[str, Any] | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    smoke_result_path: Path | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    protocol = STAGES[stage]
    smoke_predecessor = None
    if stage == "development":
        if smoke_result_path is None:
            raise ValueError("development requires a passing smoke result")
        smoke_predecessor = validate_smoke_result(smoke_result_path)
    elif smoke_result_path is not None:
        raise ValueError("smoke stage cannot consume a smoke predecessor")
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    cigs = load_stage_cigs(stage)
    adapter = adapter or build_adapter(stage=stage, run_id=run_id, output_dir=output_dir)

    root_messages = []
    root_privacy = []
    for cig in cigs:
        messages, audit = messages_for(cig, [])
        root_messages.append(messages)
        root_privacy.append(audit)
    root_seeds = [protocol["root_seed_start"] + index for index in range(len(cigs))]
    raw_roots = _call_batch(adapter, root_messages, root_seeds)
    checkpoint(
        private_dir / "RAW_RESPONSES.json",
        {"stage": stage, "root": raw_roots, "branches": []},
    )
    roots = []
    for index, (cig, raw) in enumerate(zip(cigs, raw_roots, strict=True)):
        support = parse_support(raw)
        truth_seed = protocol["truth_seed_start"] + index
        truth_index, truth = sample_truth(cig, truth_seed)
        question = support["questions"][0]
        mapping = map_and_answer(cig, question, truth)
        aliases = str((truth.slots or {})["answer_aliases"])
        roots.append(
            {
                "cig": cig,
                "truth_index": truth_index,
                "truth_seed": truth_seed,
                "root_seed": root_seeds[index],
                "truth": truth,
                "aliases": aliases,
                "question": question,
                "mapping": mapping,
                "root_support": support,
                "root_covered": truth_covered(support, aliases),
            }
        )

    branch_messages = []
    branch_seeds = []
    branch_privacy = []
    for index, root in enumerate(roots[: protocol["branch_count"]]):
        dialogue = [
            {"role": "assistant", "content": root["question"]},
            {"role": "user", "content": root["mapping"]["answer"]},
        ]
        conditioned_messages, conditioned_audit = messages_for(root["cig"], dialogue)
        blind_messages, blind_audit = messages_for(root["cig"], [])
        matched_seed = protocol["refresh_seed_start"] + index
        branch_messages.extend([conditioned_messages, blind_messages])
        branch_seeds.extend([matched_seed, matched_seed])
        branch_privacy.extend([conditioned_audit, blind_audit])
    raw_branches = _call_batch(adapter, branch_messages, branch_seeds)
    checkpoint(
        private_dir / "RAW_RESPONSES.json",
        {"stage": stage, "root": raw_roots, "branches": raw_branches},
    )
    branches = []
    for index, root in enumerate(roots[: protocol["branch_count"]]):
        conditioned = parse_support(raw_branches[2 * index])
        blind = parse_support(raw_branches[2 * index + 1])
        branches.append(
            {
                "task_id": root["cig"].cig_id,
                "supported": root["mapping"]["supported"],
                "refresh_seed": branch_seeds[2 * index],
                "root_covered": root["root_covered"],
                "conditioned_covered": truth_covered(conditioned, root["aliases"]),
                "blind_covered": truth_covered(blind, root["aliases"]),
                "conditioned_support": conditioned,
                "blind_support": blind,
            }
        )

    usage = summarize_usage(adapter.usage_snapshot())
    gates = mechanics_gates(
        stage=stage,
        roots=roots,
        branches=branches,
        privacy=[*root_privacy, *branch_privacy],
        usage=usage,
    )
    public_rows = []
    branch_by_id = {row["task_id"]: row for row in branches}
    for root in roots:
        row = _public_root(root)
        branch = branch_by_id.get(root["cig"].cig_id)
        if branch is not None:
            row.update(
                {
                    "refresh_seed": branch["refresh_seed"],
                    "conditioned_valid_unique_count": branch["conditioned_support"][
                        "diagnostic"
                    ]["valid_unique_count"],
                    "blind_valid_unique_count": branch["blind_support"]["diagnostic"][
                        "valid_unique_count"
                    ],
                    "conditioned_covered": branch["conditioned_covered"],
                    "blind_covered": branch["blind_covered"],
                }
            )
        public_rows.append(row)

    science = (
        scientific_summary(public_rows, samples=bootstrap_samples)
        if stage == "development" and gates["all_pass"]
        else None
    )
    status = "passed" if gates["all_pass"] else "mechanics_failed"
    authorizes = "development_support_recovery_only" if stage == "smoke" and status == "passed" else "nothing"
    endpoint_accessed = stage == "development" and gates["all_pass"]
    if stage == "development" and gates["all_pass"]:
        status = "passed" if science and science["gates"]["all_pass"] else "gated_null"
        authorizes = "separately_preregistered_development_policy_only" if status == "passed" else "nothing"
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "authorizes": authorizes,
        "protocol": {
            "stage": stage,
            "model": MODEL_ID,
            "reasoning": "disabled_excluded",
            "temperature": TEMPERATURE,
            "expected_requests": protocol["expected_requests"],
            "split_hash": SPLIT_HASHES[protocol["split"]],
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "matched_refresh_seed": True,
            "conditioned_blind_dispatch_adjacent": True,
            "hidden_cig_exposed_to_model": False,
            "confirmation_opened": False,
            "policy_endpoint_opened": False,
            "support_recovery_endpoint_accessed": endpoint_accessed,
            "efficacy_used_for_smoke_authorization": False,
            "smoke_predecessor": smoke_predecessor,
        },
        "daily_budget_status": dict(daily_budget_status or {}),
        "usage": usage,
        "mechanics_gates": gates,
        "science": science,
        "tasks": public_rows,
    }
    checkpoint(output_dir / "RESULT.json", result)
    checkpoint(
        private_dir / "CONTROLS.json",
        {
            "stage": stage,
            "roots": [
                {
                    "task_id": root["cig"].cig_id,
                    "truth_index": root["truth_index"],
                    "question": root["question"],
                    "mapping": root["mapping"],
                    "aliases": root["aliases"],
                }
                for root in roots
            ],
            "privacy": [*root_privacy, *branch_privacy],
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--daily-ledger", type=Path, required=True)
    parser.add_argument("--smoke-result", type=Path)
    args = parser.parse_args()
    protocol = STAGES[args.stage]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter: PerRequestSeedStructuredAdapter | None = None
    try:
        live = read_live_credits()
        ledger = json.loads(args.daily_ledger.read_text(encoding="utf-8"))
        daily = require_budget(
            ledger,
            projected_cost_usd=protocol["projected_cost_usd"],
            total_usage_usd=live["total_usage_usd"],
        )
        if live["balance_usd"] + 1e-12 < protocol["projected_cost_usd"]:
            raise RuntimeError("live balance is below projected stage cost")
        daily.update(live)
        adapter = build_adapter(
            stage=args.stage,
            run_id=args.run_id,
            output_dir=output_dir,
        )
        result = run_stage(
            stage=args.stage,
            output_dir=output_dir,
            run_id=args.run_id,
            adapter=adapter,
            daily_budget_status=daily,
            smoke_result_path=args.smoke_result,
        )
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "authorizes": "nothing",
            "confirmation_opened": False,
            "policy_endpoint_opened": False,
        }
        if adapter is not None:
            failure["usage"] = summarize_usage(adapter.usage_snapshot())
        checkpoint(output_dir / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
