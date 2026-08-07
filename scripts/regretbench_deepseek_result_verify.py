#!/usr/bin/env python3
"""Independently replay RegretBench support and policy result artifacts.

This verifier makes no model calls and intentionally does not import either
experiment producer. It reconstructs the reported scientific endpoint from the
saved raw responses, frozen selections, and private truth controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REGRETBENCH_ROOT = REPO_ROOT / "external/RegretBench"
REGRETBENCH_SRC = REGRETBENCH_ROOT / "src"
for path in (REPO_ROOT, REGRETBENCH_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from regretbench.mapping.semantic_action_mapper import SemanticActionMapper
from regretbench.schemas.cig import CIG, load_cig
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-result-verify-1"
SOURCE_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/regretbench_llm_native_source_audit/"
    "SOURCE_PROTOCOL_MANIFEST.json"
)
SUPPORT_STAGES = {
    "smoke": {
        "split": "mechanics",
        "tasks": 4,
        "branches": 3,
        "requests": 10,
        "truth_seed": 202608081100,
        "root_seed": 202608083000,
        "refresh_seed": 202608084000,
        "budget": 0.20,
    },
    "development": {
        "split": "development",
        "tasks": 64,
        "branches": 64,
        "requests": 192,
        "truth_seed": 202608082000,
        "root_seed": 202608085000,
        "refresh_seed": 202608086000,
        "budget": 0.50,
    },
}
POLICY_NAMES = (
    "dynamic_depth2",
    "history_blind_depth2",
    "myopic_width",
    "fixed_depth2",
    "random",
)
HYPOTHESES = 8
QUESTIONS = 4
BRANCH_DRAWS = 2
PLANNING_REQUESTS = 8_256
MAX_PRIMARY_REQUESTS = 8_768
POLICY_BUDGET = 3.50
BOOTSTRAP_SEED = 202608150000
BRANCH_SEED_START = 202608100000
ACTUAL_FIRST_SEED_START = 202608110000
ACTUAL_FINAL_SEED_START = 202608120000
TRUTH_SEED_START = 202608130000
RANDOM_SEED_START = 202608140000
PROBABILITY_FLOOR = 1e-12
UNSUPPORTED_REPLY = "I cannot answer that clarification."
ACTION_AMENDMENT_SHA256 = (
    "8d375fca72f4da30265a9068df27aefceefb14c5474fff2661cbd1513f056160"
)
VALID_TRAJECTORY_AMENDMENT_SHA256 = (
    "57dacb5e671282b825f3fa375dfbd088dbd4d79f3387df59f5600f6e995edbf3"
)
OUTCOME_CRN_AMENDMENT_SHA256 = (
    "fd8533a151ce538eca74a354cfe5b807bec700af2c9a72fadb3ba2fb7c8f23e3"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(
        r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.casefold())
    ).strip()


def lexical_alias_match(generated: str, alias: str) -> bool:
    left = normalize_text(generated)
    right = normalize_text(alias)
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) == len(right):
        return False
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    return len(shorter) >= 8 and len(shorter.split()) >= 2 and shorter in longer


def _stage_cigs(stage: str) -> list[CIG]:
    spec = SUPPORT_STAGES[stage]
    manifest = _load(SOURCE_MANIFEST)
    ids = manifest["splits"][spec["split"]]["ids"]
    if len(ids) != spec["tasks"]:
        raise ValueError("source manifest task count changed")
    cigs = [
        load_cig(REGRETBENCH_ROOT / "data/OpenDomainQA/test" / f"{cig_id}.json")
        for cig_id in ids
    ]
    if [cig.cig_id for cig in cigs] != ids:
        raise ValueError("source manifest order changed")
    return cigs


def _truth(cig: CIG, seed: int) -> tuple[int, Any]:
    index = random.Random(seed).randrange(len(cig.intents))
    return index, cig.intents[index]


def _map(cig: CIG, question: str, truth: Any) -> dict[str, Any]:
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


def _distinct_actions(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    return bool(
        first["supported"]
        and second["supported"]
        and first["facet"] is not None
        and second["facet"] is not None
        and first["facet"] != second["facet"]
    )


def _payload_is_private(
    cig: CIG, dialogue: Sequence[Mapping[str, str]]
) -> bool:
    records = []
    for item in dialogue:
        if set(item) != {"role", "content"} or item["role"] not in {
            "assistant",
            "user",
        }:
            return False
        content = str(item["content"]).strip()
        if not content:
            return False
        records.append({"role": str(item["role"]), "content": content})
    payload = {"task_id": cig.cig_id, "prompt": cig.prompt, "dialogue": records}
    return set(payload) == {"task_id", "prompt", "dialogue"}


def _parse_support(raw: str, *, enriched: bool) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("support has wrong top-level fields")
    questions = value["questions"]
    if not isinstance(questions, list) or len(questions) != QUESTIONS:
        raise ValueError("support must contain four questions")
    clean_questions: list[str] = []
    question_keys: set[str] = set()
    for question in questions:
        if not isinstance(question, str) or not question.strip().endswith("?"):
            raise ValueError("support contains a non-question")
        clean = question.strip()
        key = normalize_text(clean)
        if not key or key in question_keys:
            raise ValueError("support contains duplicate questions")
        question_keys.add(key)
        clean_questions.append(clean)

    raw_hypotheses = value["hypotheses"]
    if not isinstance(raw_hypotheses, list) or len(raw_hypotheses) != HYPOTHESES:
        raise ValueError("support must contain eight hypotheses")
    expected = {"interpretation", "final_answer", "prior_weight"}
    if enriched:
        expected.add("predicted_replies")
    hypotheses: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_hypotheses:
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError("hypothesis has wrong fields")
        interpretation = item["interpretation"]
        answer = item["final_answer"]
        weight = item["prior_weight"]
        if (
            not isinstance(interpretation, str)
            or not interpretation.strip()
            or not isinstance(answer, str)
            or not answer.strip()
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
        ):
            raise ValueError("hypothesis has invalid values")
        key = (normalize_text(interpretation), normalize_text(answer))
        if key in seen:
            if enriched:
                raise ValueError("enriched support contains duplicate hypotheses")
            continue
        seen.add(key)
        row = {
            "interpretation": interpretation.strip(),
            "final_answer": answer.strip(),
            "probability": float(weight),
        }
        if enriched:
            replies = item["predicted_replies"]
            if (
                not isinstance(replies, list)
                or len(replies) != QUESTIONS
                or any(not isinstance(reply, str) or not reply.strip() for reply in replies)
            ):
                raise ValueError("hypothesis has invalid predicted replies")
            row["predicted_replies"] = [reply.strip() for reply in replies]
        hypotheses.append(row)
    minimum = HYPOTHESES if enriched else 4
    if len(hypotheses) < minimum:
        raise ValueError("support has too few unique hypotheses")
    total = sum(row["probability"] for row in hypotheses)
    if total <= 0:
        raise ValueError("support weights sum to zero")
    for row in hypotheses:
        row["probability"] /= total
    support = {"hypotheses": hypotheses, "questions": clean_questions}
    diagnostic = {
        "codec_mode": "strict_json",
        "raw_hypothesis_count": len(raw_hypotheses),
        "valid_unique_count": len(hypotheses),
        "question_count": len(clean_questions),
    }
    if enriched:
        diagnostic["informative_question_count"] = sum(
            _question_eig(support, index) > 1e-12 for index in range(QUESTIONS)
        )
    support["diagnostic"] = diagnostic
    return support


def _truth_mass(support: Mapping[str, Any], aliases: str) -> float:
    alternatives = [value.strip() for value in aliases.split("|") if value.strip()]
    return sum(
        row["probability"]
        for row in support["hypotheses"]
        if any(lexical_alias_match(row["final_answer"], alias) for alias in alternatives)
    )


def _covered(support: Mapping[str, Any], aliases: str) -> bool:
    return _truth_mass(support, aliases) > 0.0


def _bootstrap_positive(values: Sequence[float], samples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"mean": None, "ci95": [None, None], "probability_positive": None}
    rng = np.random.default_rng(202608087000)
    indexes = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indexes].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "probability_positive": float(np.mean(means > 0.0)),
        "samples": samples,
        "seed": 202608087000,
    }


def _support_science(rows: Sequence[Mapping[str, Any]], samples: int) -> dict[str, Any]:
    supported = [row for row in rows if row["supported"]]
    differences = [
        float(row["conditioned_covered"]) - float(row["blind_covered"])
        for row in supported
    ]
    missing = [row for row in supported if not row["root_covered"]]
    missing_differences = [
        float(row["conditioned_covered"]) - float(row["blind_covered"])
        for row in missing
    ]
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    bootstrap = _bootstrap_positive(differences, samples)
    missing_bootstrap = _bootstrap_positive(missing_differences, samples)
    n = wins + losses
    sign_p = (
        float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))
        if n
        else None
    )
    gates = {
        "at_least_48_supported_tasks": len(supported) >= 48,
        "at_least_16_root_missing_supported_tasks": len(missing) >= 16,
        "at_least_8_changed_conditioning_outcomes": wins + losses >= 8,
        "conditioned_minus_blind_coverage_at_least_005": bootstrap["mean"] is not None
        and bootstrap["mean"] >= 0.05,
        "bootstrap_probability_positive_at_least_080": bootstrap[
            "probability_positive"
        ]
        is not None
        and bootstrap["probability_positive"] >= 0.80,
        "conditioned_recoveries_exceed_losses": wins > losses,
        "root_missing_recovery_difference_at_least_010": missing_bootstrap["mean"]
        is not None
        and missing_bootstrap["mean"] >= 0.10,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "population": {
            "all_tasks": len(rows),
            "supported_tasks": len(supported),
            "root_missing_supported_tasks": len(missing),
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
            "history_blind": float(np.mean([row["blind_covered"] for row in supported]))
            if supported
            else None,
            "conditioned_minus_history_blind": bootstrap,
            "root_missing_conditioned_minus_history_blind": missing_bootstrap,
        },
        "paired_outcomes": {
            "conditioned_recoveries": wins,
            "conditioned_losses": losses,
            "ties": len(differences) - wins - losses,
            "one_sided_exact_sign_p": sign_p,
        },
        "gates": gates,
    }


def _close(left: Any, right: Any, path: str = "$", mismatches: list[str] | None = None) -> list[str]:
    output = mismatches if mismatches is not None else []
    if isinstance(left, bool) or isinstance(right, bool) or left is None or right is None:
        if left != right:
            output.append(path)
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=1e-11, abs_tol=1e-12):
            output.append(path)
    elif isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            output.append(path + ".<keys>")
        for key in sorted(set(left) & set(right)):
            _close(left[key], right[key], f"{path}.{key}", output)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            output.append(path + ".<length>")
        for index, (a, b) in enumerate(zip(left, right)):
            _close(a, b, f"{path}[{index}]", output)
    elif left != right:
        output.append(path)
    return output


def verify_support(run_dir: Path, *, stage: str) -> dict[str, Any]:
    spec = SUPPORT_STAGES[stage]
    result = _load(run_dir / "RESULT.json")
    raw = _load(run_dir / "private" / "RAW_RESPONSES.json")
    controls = _load(run_dir / "private" / "CONTROLS.json")
    cigs = _stage_cigs(stage)
    roots_raw = raw.get("root") or []
    branches_raw = raw.get("branches") or []
    if len(roots_raw) != spec["tasks"] or len(branches_raw) != 2 * spec["branches"]:
        raise ValueError("raw support response schedule changed")
    supports = [_parse_support(value, enriched=False) for value in roots_raw]
    control_by_id = {row["task_id"]: row for row in controls.get("roots", [])}
    rows = []
    for index, (cig, support) in enumerate(zip(cigs, supports, strict=True)):
        truth_seed = spec["truth_seed"] + index
        truth_index, truth = _truth(cig, truth_seed)
        question = support["questions"][0]
        mapping = _map(cig, question, truth)
        aliases = str((truth.slots or {})["answer_aliases"])
        control = control_by_id.get(cig.cig_id)
        if control is None or _close(
            control,
            {
                "task_id": cig.cig_id,
                "truth_index": truth_index,
                "question": question,
                "mapping": mapping,
                "aliases": aliases,
            },
        ):
            raise ValueError(f"private support control mismatch: {cig.cig_id}")
        row = {
            "task_id": cig.cig_id,
            "truth_seed": truth_seed,
            "root_seed": spec["root_seed"] + index,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "root_valid_unique_count": support["diagnostic"]["valid_unique_count"],
            "supported": mapping["supported"],
            "mapping_confidence": mapping["confidence"],
            "mapping_method": mapping["method"],
            "root_covered": _covered(support, aliases),
        }
        if index < spec["branches"]:
            conditioned = _parse_support(branches_raw[2 * index], enriched=False)
            blind = _parse_support(branches_raw[2 * index + 1], enriched=False)
            row.update(
                {
                    "refresh_seed": spec["refresh_seed"] + index,
                    "conditioned_valid_unique_count": conditioned["diagnostic"][
                        "valid_unique_count"
                    ],
                    "blind_valid_unique_count": blind["diagnostic"]["valid_unique_count"],
                    "conditioned_covered": _covered(conditioned, aliases),
                    "blind_covered": _covered(blind, aliases),
                }
            )
        rows.append(row)
    samples = int(((result.get("science") or {}).get("coverage") or {}).get(
        "conditioned_minus_history_blind", {}
    ).get("samples", 20_000))
    science_candidate = (
        _support_science(rows, samples) if stage == "development" else None
    )
    usage = result.get("usage") or {}
    all_support_count = len(roots_raw) + len(branches_raw)
    privacy = controls.get("privacy") or []
    mechanics = {
        "exact_task_count": len(rows) == spec["tasks"],
        "exact_branch_count": len(branches_raw) // 2 == spec["branches"],
        "exact_response_count": all_support_count == spec["requests"],
        "exact_accepted_requests": usage.get("adapter_requests") == spec["requests"],
        "exact_http_attempts": usage.get("http_attempts") == spec["requests"],
        "zero_retries": usage.get("retry_count") == 0,
        "zero_provider_error_retries": usage.get("provider_error_retries") == 0,
        "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "all_strict_supports_have_four_unique": all(
            _parse_support(value, enriched=False)["diagnostic"]["valid_unique_count"] >= 4
            for value in [*roots_raw, *branches_raw]
        ),
        "every_root_has_four_questions": all(
            support["diagnostic"]["question_count"] == 4 for support in supports
        ),
        "all_privacy_audits_pass": len(privacy) == spec["requests"]
        and all(item.get("passed") is True for item in privacy),
        "within_run_budget": float(usage.get("run_cost_usd", math.inf))
        <= spec["budget"] + 1e-12,
    }
    if stage == "smoke":
        mechanics["all_three_selected_questions_supported"] = all(
            row["supported"] for row in rows[: spec["branches"]]
        )
    mechanics["all_pass"] = all(mechanics.values())
    science = science_candidate if mechanics["all_pass"] else None
    expected_status = "passed" if mechanics["all_pass"] else "mechanics_failed"
    if stage == "development" and mechanics["all_pass"]:
        expected_status = "passed" if science and science["gates"]["all_pass"] else "gated_null"
    mismatches = []
    _close(result.get("tasks"), rows, "$.tasks", mismatches)
    _close(result.get("mechanics_gates"), mechanics, "$.mechanics_gates", mismatches)
    _close(result.get("science"), science, "$.science", mismatches)
    _close(result.get("status"), expected_status, "$.status", mismatches)
    return _verification(run_dir, "support", mismatches)


def _entropy(values: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in values if value > 0)


def _question_eig(
    support: Mapping[str, Any], question: int, indexes: Sequence[int] | None = None
) -> float:
    selected = list(indexes) if indexes is not None else list(range(HYPOTHESES))
    weights = [support["hypotheses"][index]["probability"] for index in selected]
    total = sum(weights)
    weights = [value / total for value in weights] if total > 0 else [1 / len(weights)] * len(weights)
    masses: dict[str, float] = {}
    for index, weight in zip(selected, weights, strict=True):
        reply = normalize_text(support["hypotheses"][index]["predicted_replies"][question])
        masses[reply] = masses.get(reply, 0.0) + weight
    return _entropy(list(masses.values()))


def _select_question(support: Mapping[str, Any], exclude: int | None = None) -> int:
    choices = [index for index in range(QUESTIONS) if index != exclude]
    return min(choices, key=lambda index: (-_question_eig(support, index), index))


def _branch_truth(support: Mapping[str, Any], truth_answer: str) -> dict[str, float]:
    truth_indexes = [
        index
        for index, row in enumerate(support["hypotheses"])
        if lexical_alias_match(row["final_answer"], truth_answer)
    ]
    second = _select_question(support)
    if not truth_indexes:
        return {"truth_mass": 0.0, "expected_brier": 1.0, "expected_log_loss": -math.log(PROBABILITY_FLOOR)}
    truth_mass = sum(support["hypotheses"][index]["probability"] for index in truth_indexes)
    all_mass: dict[str, float] = {}
    truth_outcome: dict[str, float] = {}
    for index, row in enumerate(support["hypotheses"]):
        outcome = normalize_text(row["predicted_replies"][second])
        all_mass[outcome] = all_mass.get(outcome, 0.0) + row["probability"]
        if index in truth_indexes:
            truth_outcome[outcome] = truth_outcome.get(outcome, 0.0) + row["probability"]
    brier = 0.0
    log_loss = 0.0
    for outcome, joint in truth_outcome.items():
        conditional = joint / truth_mass
        posterior = joint / all_mass[outcome]
        brier += conditional * (1.0 - posterior) ** 2
        log_loss += conditional * -math.log(max(PROBABILITY_FLOOR, posterior))
    return {"truth_mass": truth_mass, "expected_brier": brier, "expected_log_loss": log_loss}


def _dynamic_risks(initial: Mapping[str, Any], branches: Sequence[Mapping[str, Any]], arm: str) -> list[dict[str, float]]:
    lookup = {
        (row["root_index"], row["hypothesis_index"], row["draw"]): row[arm]
        for row in branches
    }
    output = []
    for root in range(QUESTIONS):
        brier = log_loss = coverage = 0.0
        for hypothesis, row in enumerate(initial["hypotheses"]):
            local = [_branch_truth(lookup[(root, hypothesis, draw)], row["final_answer"]) for draw in range(BRANCH_DRAWS)]
            weight = row["probability"]
            brier += weight * float(np.mean([item["expected_brier"] for item in local]))
            log_loss += weight * float(np.mean([item["expected_log_loss"] for item in local]))
            coverage += weight * float(np.mean([item["truth_mass"] > 0 for item in local]))
        output.append({"brier": brier, "log_loss": log_loss, "coverage": coverage})
    return output


def _fixed_risks(initial: Mapping[str, Any]) -> list[dict[str, float]]:
    hypotheses = initial["hypotheses"]
    output = []
    for root in range(QUESTIONS):
        brier = log_loss = 0.0
        for truth in hypotheses:
            first_reply = normalize_text(truth["predicted_replies"][root])
            first = [index for index, row in enumerate(hypotheses) if normalize_text(row["predicted_replies"][root]) == first_reply]
            second = min(
                [index for index in range(QUESTIONS) if index != root],
                key=lambda index: (-_question_eig(initial, index, first), index),
            )
            second_reply = normalize_text(truth["predicted_replies"][second])
            posterior = [index for index in first if normalize_text(hypotheses[index]["predicted_replies"][second]) == second_reply]
            denominator = sum(hypotheses[index]["probability"] for index in posterior)
            numerator = sum(
                hypotheses[index]["probability"]
                for index in posterior
                if lexical_alias_match(hypotheses[index]["final_answer"], truth["final_answer"])
            )
            mass = numerator / denominator if denominator else 0.0
            brier += truth["probability"] * (1.0 - mass) ** 2
            log_loss += truth["probability"] * -math.log(max(PROBABILITY_FLOOR, mass))
        output.append({"brier": brier, "log_loss": log_loss})
    return output


def _choose_roots(initial: Mapping[str, Any], conditioned: Sequence[Mapping[str, float]], blind: Sequence[Mapping[str, float]], fixed: Sequence[Mapping[str, float]], task: int) -> dict[str, int]:
    return {
        "dynamic_depth2": min(range(QUESTIONS), key=lambda index: (conditioned[index]["brier"], index)),
        "history_blind_depth2": min(range(QUESTIONS), key=lambda index: (blind[index]["brier"], index)),
        "myopic_width": min(range(QUESTIONS), key=lambda index: (-_question_eig(initial, index), index)),
        "fixed_depth2": min(range(QUESTIONS), key=lambda index: (fixed[index]["brier"], index)),
        "random": random.Random(RANDOM_SEED_START + task).randrange(QUESTIONS),
    }


def _terminal(support: Mapping[str, Any], question: int, observed: str, aliases: str) -> dict[str, Any]:
    outcome = [
        index
        for index, row in enumerate(support["hypotheses"])
        if normalize_text(row["predicted_replies"][question]) == normalize_text(observed)
    ]
    denominator = sum(support["hypotheses"][index]["probability"] for index in outcome)
    alternatives = [value.strip() for value in aliases.split("|") if value.strip()]
    numerator = sum(
        support["hypotheses"][index]["probability"]
        for index in outcome
        if any(lexical_alias_match(support["hypotheses"][index]["final_answer"], alias) for alias in alternatives)
    )
    mass = numerator / denominator if denominator else 0.0
    return {
        "reply_matched": bool(outcome),
        "matched_hypothesis_count": len(outcome),
        "truth_mass": mass,
        "brier": (1.0 - mass) ** 2,
        "log_loss": -math.log(max(PROBABILITY_FLOOR, mass)),
        "covered": mass > 0.0,
    }


def _path_metrics(
    first_support: Mapping[str, Any],
    final_support: Mapping[str, Any],
    *,
    question: int,
    observed: str,
    aliases: str,
    first_mapping: Mapping[str, Any],
    second_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    valid = _distinct_actions(first_mapping, second_mapping)
    raw_first = _truth_mass(first_support, aliases)
    raw_terminal = _terminal(first_support, question, observed, aliases)
    raw_terminal_mass = float(raw_terminal["truth_mass"])
    terminal_mass = raw_terminal_mass if valid else 0.0
    raw_fresh = _truth_mass(final_support, aliases)
    fresh_mass = raw_fresh if valid else 0.0
    first_mass = raw_first if first_mapping["supported"] else 0.0
    return {
        "valid_two_action_trajectory": valid,
        "raw_truth_mass_after_first": raw_first,
        "truth_mass_after_first": first_mass,
        "second_reply_likelihood_matched": raw_terminal["reply_matched"],
        "second_reply_matched_hypotheses": raw_terminal[
            "matched_hypothesis_count"
        ],
        "raw_truth_mass_final": raw_terminal_mass,
        "truth_mass_final": terminal_mass,
        "brier": (1.0 - terminal_mass) ** 2,
        "log_loss": -math.log(max(PROBABILITY_FLOOR, terminal_mass)),
        "covered": terminal_mass > 0.0,
        "raw_fresh_truth_mass_final": raw_fresh,
        "fresh_truth_mass_final": fresh_mass,
        "fresh_brier": (1.0 - fresh_mass) ** 2,
        "fresh_log_loss": -math.log(max(PROBABILITY_FLOOR, fresh_mass)),
        "fresh_covered": fresh_mass > 0.0,
    }


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0
        cursor = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    a, b = _rankdata(left), _rankdata(right)
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _paired(values: Sequence[float], samples: int, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indexes].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "probability_improvement": float(np.mean(means < 0.0)),
        "samples": samples,
        "seed": seed,
    }


def _comparison(tasks: Sequence[Mapping[str, Any]], baseline: str, samples: int, seed: int, prefix: str = "") -> dict[str, Any]:
    brier_key = prefix + "brier"
    log_key = prefix + "log_loss"
    brier = [task["policies"]["dynamic_depth2"][brier_key] - task["policies"][baseline][brier_key] for task in tasks]
    logs = [task["policies"]["dynamic_depth2"][log_key] - task["policies"][baseline][log_key] for task in tasks]
    return {
        "baseline": baseline,
        "brier_dynamic_minus_baseline": _paired(brier, samples, seed),
        "log_loss_dynamic_minus_baseline": _paired(logs, samples, seed + 1),
        "wins_ties_losses": {
            "wins": sum(value < -1e-12 for value in brier),
            "ties": sum(abs(value) <= 1e-12 for value in brier),
            "losses": sum(value > 1e-12 for value in brier),
        },
    }


def _correlation(predicted: Sequence[float], realized: Sequence[float], samples: int) -> dict[str, Any]:
    point = _spearman(predicted, realized)
    if point is None:
        return {"spearman": None, "ci95": [None, None], "probability_positive": None, "n": len(predicted)}
    rng = np.random.default_rng(BOOTSTRAP_SEED + 99)
    values = []
    for _ in range(samples):
        indexes = rng.integers(0, len(predicted), size=len(predicted))
        value = _spearman([predicted[index] for index in indexes], [realized[index] for index in indexes])
        if value is not None:
            values.append(value)
    if not values:
        return {
            "spearman": point,
            "ci95": [None, None],
            "probability_positive": None,
            "n": len(predicted),
        }
    array = np.asarray(values)
    return {
        "spearman": point,
        "ci95": [float(value) for value in np.quantile(array, [0.025, 0.975])],
        "probability_positive": float(np.mean(array > 0.0)),
        "n": len(predicted),
        "samples": samples,
        "seed": BOOTSTRAP_SEED + 99,
    }


def _policy_science(tasks: Sequence[Mapping[str, Any]], samples: int) -> dict[str, Any]:
    baselines = ["myopic_width", "history_blind_depth2", "fixed_depth2", "random"]
    comparisons = {name: _comparison(tasks, name, samples, BOOTSTRAP_SEED + index * 10) for index, name in enumerate(baselines)}
    fresh_names = [*baselines]
    if all("naive_thinking" in task["policies"] for task in tasks):
        fresh_names.append("naive_thinking")
    fresh = {name: _comparison(tasks, name, samples, BOOTSTRAP_SEED + 500 + index * 10, "fresh_") for index, name in enumerate(fresh_names)}
    disagreements = {name: sum(task["selected_roots"]["dynamic_depth2"] != task["selected_roots"][name] for task in tasks) for name in baselines}
    predicted = [
        task["conditioned_root_risks"][task["selected_roots"]["myopic_width"]]["brier"]
        - task["conditioned_root_risks"][task["selected_roots"]["dynamic_depth2"]]["brier"]
        for task in tasks
    ]
    changed = [index for index, task in enumerate(tasks) if task["selected_roots"]["dynamic_depth2"] != task["selected_roots"]["myopic_width"]]
    realized = [tasks[index]["policies"]["myopic_width"]["brier"] - tasks[index]["policies"]["dynamic_depth2"]["brier"] for index in changed]
    correlation = _correlation([predicted[index] for index in changed], realized, samples)
    myopic, blind, fixed = (comparisons[name] for name in ("myopic_width", "history_blind_depth2", "fixed_depth2"))
    gates = {
        "dynamic_myopic_differ_at_least_16": disagreements["myopic_width"] >= 16,
        "dynamic_blind_differ_at_least_12": disagreements["history_blind_depth2"] >= 12,
        "dynamic_fixed_differ_at_least_12": disagreements["fixed_depth2"] >= 12,
        "predicted_gain_over_myopic_at_least_001": float(np.mean(predicted)) >= 0.01,
        "dynamic_myopic_brier_gain_at_least_002": myopic["brier_dynamic_minus_baseline"]["mean"] <= -0.02,
        "dynamic_myopic_probability_at_least_090": myopic["brier_dynamic_minus_baseline"]["probability_improvement"] >= 0.90,
        "dynamic_myopic_wins_exceed_losses": myopic["wins_ties_losses"]["wins"] > myopic["wins_ties_losses"]["losses"],
        "dynamic_blind_brier_gain_at_least_0015": blind["brier_dynamic_minus_baseline"]["mean"] <= -0.015,
        "dynamic_blind_probability_at_least_080": blind["brier_dynamic_minus_baseline"]["probability_improvement"] >= 0.80,
        "dynamic_blind_wins_exceed_losses": blind["wins_ties_losses"]["wins"] > blind["wins_ties_losses"]["losses"],
        "dynamic_fixed_brier_gain_at_least_001": fixed["brier_dynamic_minus_baseline"]["mean"] <= -0.01,
        "dynamic_fixed_probability_at_least_080": fixed["brier_dynamic_minus_baseline"]["probability_improvement"] >= 0.80,
        "dynamic_fixed_wins_exceed_losses": fixed["wins_ties_losses"]["wins"] > fixed["wins_ties_losses"]["losses"],
        "dynamic_log_loss_nonworse_myopic": myopic["log_loss_dynamic_minus_baseline"]["mean"] <= 0.0,
        "dynamic_log_loss_nonworse_blind": blind["log_loss_dynamic_minus_baseline"]["mean"] <= 0.0,
        "dynamic_log_loss_nonworse_fixed": fixed["log_loss_dynamic_minus_baseline"]["mean"] <= 0.0,
        "predicted_realized_spearman_at_least_015": correlation["spearman"] is not None and correlation["spearman"] >= 0.15,
        "spearman_probability_positive_at_least_080": correlation["probability_positive"] is not None and correlation["probability_positive"] >= 0.80,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "comparisons": comparisons,
        "root_disagreements": disagreements,
        "mean_conditioned_predicted_gain_over_myopic": float(np.mean(predicted)),
        "predicted_to_realized_dynamic_myopic": correlation,
        "fresh_regeneration_comparisons_descriptive": fresh,
        "gates": gates,
    }


def _parse_naive(raw: str) -> str:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"question"}:
        raise ValueError("naive response has wrong fields")
    question = value["question"]
    if not isinstance(question, str) or not question.strip().endswith("?"):
        raise ValueError("naive response is not a question")
    return question.strip()


def verify_policy_smoke(run_dir: Path) -> dict[str, Any]:
    result = _load(run_dir / "RESULT.json")
    raw = _load(run_dir / "private" / "RAW_RESPONSES.json")
    cigs = _stage_cigs("smoke")
    initial_raw = raw.get("initial") or []
    branch_raw = raw.get("branches") or []
    if len(initial_raw) != 4 or len(branch_raw) != 6:
        raise ValueError("policy smoke raw response schedule changed")
    initial = [_parse_support(value, enriched=True) for value in initial_raw]
    branches = [_parse_support(value, enriched=True) for value in branch_raw]
    first_mappings = []
    second_mappings = []
    second_matches = []
    privacy_checks = [_payload_is_private(cig, []) for cig in cigs]
    for index in range(3):
        cig = cigs[index]
        _, truth = _truth(cig, SUPPORT_STAGES["smoke"]["truth_seed"] + index)
        first_question = initial[index]["questions"][0]
        first_mapping = _map(cig, first_question, truth)
        conditioned = branches[2 * index]
        second_index = _select_question(conditioned)
        second_mapping = _map(
            cig, conditioned["questions"][second_index], truth
        )
        first_mappings.append(first_mapping)
        second_mappings.append(second_mapping)
        second_matches.append(
            bool(
                [
                    row
                    for row in conditioned["hypotheses"]
                    if normalize_text(
                        row["predicted_replies"][second_index]
                    )
                    == normalize_text(second_mapping["answer"])
                ]
            )
        )
        privacy_checks.extend(
            [
                _payload_is_private(
                    cig,
                    [
                        {"role": "assistant", "content": first_question},
                        {
                            "role": "user",
                            "content": first_mapping["answer"],
                        },
                    ],
                ),
                _payload_is_private(cig, []),
            ]
        )
    usage = result.get("usage") or {}
    all_supports = [*initial, *branches]
    gates = {
        "exact_ten_responses": len(all_supports) == 10,
        "exact_ten_requests": usage.get("adapter_requests") == 10,
        "exact_ten_http_attempts": usage.get("http_attempts") == 10,
        "zero_retries": usage.get("retry_count") == 0,
        "zero_provider_error_retries": usage.get("provider_error_retries")
        == 0,
        "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "all_strict_and_exactly_eight_unique": all(
            support["diagnostic"]["valid_unique_count"] == 8
            for support in all_supports
        ),
        "every_initial_has_two_informative_roots": all(
            support["diagnostic"]["informative_question_count"] >= 2
            for support in initial
        ),
        "every_branch_has_an_informative_followup": all(
            support["diagnostic"]["informative_question_count"] >= 1
            for support in branches
        ),
        "all_three_first_questions_supported": all(
            row["supported"] for row in first_mappings
        ),
        "all_three_second_questions_supported": all(
            row["supported"] for row in second_mappings
        ),
        "all_three_second_actions_are_novel": all(
            _distinct_actions(first, second)
            for first, second in zip(
                first_mappings, second_mappings, strict=True
            )
        ),
        "all_three_exact_second_replies_match_generated_likelihoods": all(
            second_matches
        ),
        "all_privacy_audits_pass": len(privacy_checks) == 10
        and all(privacy_checks),
        "within_smoke_budget": float(
            usage.get("run_cost_usd", math.inf)
        )
        <= 0.20,
    }
    gates["all_pass"] = all(gates.values())
    expected_status = "passed" if gates["all_pass"] else "mechanics_failed"
    mismatches = []
    _close(
        (result.get("protocol") or {}).get(
            "distinct_action_amendment_sha256"
        ),
        ACTION_AMENDMENT_SHA256,
        "$.protocol.distinct_action_amendment_sha256",
        mismatches,
    )
    _close(
        (result.get("protocol") or {}).get(
            "valid_trajectory_amendment_sha256"
        ),
        VALID_TRAJECTORY_AMENDMENT_SHA256,
        "$.protocol.valid_trajectory_amendment_sha256",
        mismatches,
    )
    _close(
        (result.get("protocol") or {}).get("outcome_crn_amendment_sha256"),
        OUTCOME_CRN_AMENDMENT_SHA256,
        "$.protocol.outcome_crn_amendment_sha256",
        mismatches,
    )
    _close(result.get("supports"), [row["diagnostic"] for row in all_supports], "$.supports", mismatches)
    _close(result.get("gates"), gates, "$.gates", mismatches)
    _close(result.get("status"), expected_status, "$.status", mismatches)
    return _verification(run_dir, "policy_smoke", mismatches)


def _blind_crn_diagnostics(
    branches: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    groups: dict[tuple[int, int, int], list[Mapping[str, Any]]] = {}
    for task_rows in branches:
        for row in task_rows:
            key = (
                int(row["task_index"]),
                int(row["hypothesis_index"]),
                int(row["draw"]),
            )
            groups.setdefault(key, []).append(row)
    exact = 0
    for rows in groups.values():
        hashes = {
            hashlib.sha256(
                json.dumps(
                    row["blind"],
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            for row in rows
        }
        if (
            len(rows) == QUESTIONS
            and {int(row["root_index"]) for row in rows}
            == set(range(QUESTIONS))
            and len(hashes) == 1
        ):
            exact += 1
    return {
        "expected_group_count": 64 * HYPOTHESES * BRANCH_DRAWS,
        "observed_group_count": len(groups),
        "exact_group_count": exact,
        "exact_group_fraction": exact / len(groups) if groups else 0.0,
    }


def verify_policy(run_dir: Path) -> dict[str, Any]:
    result = _load(run_dir / "RESULT.json")
    initial_artifact = _load(run_dir / "private" / "RAW_INITIAL.json")
    branch_artifact = _load(run_dir / "private" / "RAW_BRANCHES.json")
    actual = _load(run_dir / "private" / "RAW_ACTUAL.json")
    frozen = _load(run_dir / "private" / "FROZEN_SELECTIONS.json")
    controls = _load(run_dir / "private" / "CONTROLS.json")
    cigs = _stage_cigs("development")
    initial = [_parse_support(raw, enriched=True) for raw in initial_artifact["responses"]]
    if len(initial) != 64 or initial_artifact["seeds"] != [202608089000 + index for index in range(64)]:
        raise ValueError("initial policy schedule changed")
    manifests = branch_artifact["manifest"]
    paired_seeds = branch_artifact["paired_seeds"]
    responses = branch_artifact["responses"]
    if len(manifests) != 4_096 or len(responses) != 8_192 or len(paired_seeds) != 8_192:
        raise ValueError("branch policy schedule changed")
    branches: list[list[dict[str, Any]]] = [[] for _ in range(64)]
    simulated: list[dict[str, Any]] = []
    seed_schedule_ok = True
    privacy_checks = [_payload_is_private(cig, []) for cig in cigs]
    for index, manifest in enumerate(manifests):
        task = int(manifest["task_index"])
        root = int(manifest["root_index"])
        hypothesis = int(manifest["hypothesis_index"])
        draw = int(manifest["draw"])
        expected_seed = BRANCH_SEED_START + task * 16 + hypothesis * 2 + draw
        seed_schedule_ok &= (
            int(manifest["seed"]) == expected_seed
            and paired_seeds[2 * index] == expected_seed
            and paired_seeds[2 * index + 1] == expected_seed
        )
        conditioned = _parse_support(responses[2 * index], enriched=True)
        blind = _parse_support(responses[2 * index + 1], enriched=True)
        simulated.extend([conditioned, blind])
        branches[task].append({**manifest, "conditioned": conditioned, "blind": blind})
        question = initial[task]["questions"][root]
        answer = initial[task]["hypotheses"][hypothesis]["predicted_replies"][root]
        privacy_checks.extend(
            [
                _payload_is_private(
                    cigs[task],
                    [
                        {"role": "assistant", "content": question},
                        {"role": "user", "content": answer},
                    ],
                ),
                _payload_is_private(cigs[task], []),
            ]
        )
    plans = []
    for task, support in enumerate(initial):
        conditioned = _dynamic_risks(support, branches[task], "conditioned")
        blind = _dynamic_risks(support, branches[task], "blind")
        fixed = _fixed_risks(support)
        selected = _choose_roots(support, conditioned, blind, fixed, task)
        plans.append({
            "conditioned": conditioned,
            "blind": blind,
            "fixed": fixed,
            "root_eig": [_question_eig(support, index) for index in range(QUESTIONS)],
            "selected": selected,
        })
    if frozen.get("selected_roots") != [plan["selected"] for plan in plans]:
        raise ValueError("frozen selections do not replay")
    if frozen.get("hidden_truth_accessed") is not False:
        raise ValueError("frozen selection claims hidden truth access")

    control_by_id = {row["task_id"]: row for row in controls["tasks"]}
    truths = []
    for task, cig in enumerate(cigs):
        truth_index, truth = _truth(cig, TRUTH_SEED_START + task)
        control = control_by_id[cig.cig_id]
        aliases = str((truth.slots or {})["answer_aliases"])
        if control["truth_index"] != truth_index or control["aliases"] != aliases:
            raise ValueError(f"policy truth control mismatch: {cig.cig_id}")
        truths.append((truth_index, truth, aliases))

    first_manifest = actual["first_manifest"]
    first_responses = actual["first_responses"]
    final_manifest = actual["final_manifest"]
    final_responses = actual["final_responses"]
    first_paths: dict[tuple[int, int], dict[str, Any]] = {}
    first_seed_ok = True
    for manifest, raw in zip(first_manifest, first_responses, strict=True):
        task, root = int(manifest["task_index"]), int(manifest["root_index"])
        first_seed_ok &= int(manifest["seed"]) == ACTUAL_FIRST_SEED_START + task
        support = _parse_support(raw, enriched=True)
        question = initial[task]["questions"][root]
        mapping = _map(cigs[task], question, truths[task][1])
        if _close(manifest["first_mapping"], mapping):
            raise ValueError("first realized mapping does not replay")
        if manifest["dialogue"] != [
            {"role": "assistant", "content": question},
            {"role": "user", "content": mapping["answer"]},
        ]:
            raise ValueError("first realized dialogue does not replay")
        privacy_checks.append(_payload_is_private(cigs[task], manifest["dialogue"]))
        second = _select_question(support)
        second_question = support["questions"][second]
        second_mapping = _map(cigs[task], second_question, truths[task][1])
        final_dialogue = [
            *manifest["dialogue"],
            {"role": "assistant", "content": second_question},
            {"role": "user", "content": second_mapping["answer"]},
        ]
        privacy_checks.append(_payload_is_private(cigs[task], final_dialogue))
        first_paths[(task, root)] = {
            "support": support,
            "first_mapping": mapping,
            "second": second,
            "second_mapping": second_mapping,
        }
    final_paths: dict[tuple[int, int], dict[str, Any]] = {}
    final_seed_ok = True
    for manifest, raw in zip(final_manifest, final_responses, strict=True):
        task, root = int(manifest["task_index"]), int(manifest["root_index"])
        final_seed_ok &= int(manifest["seed"]) == ACTUAL_FINAL_SEED_START + task
        final_paths[(task, root)] = _parse_support(raw, enriched=True)
    if set(first_paths) != set(final_paths):
        raise ValueError("first/final realized path keys differ")

    naive_available = not actual.get("naive_baseline_error") and bool(actual.get("naive_first_questions"))
    naive_rows: list[dict[str, Any]] = []
    if naive_available:
        first_questions = [_parse_naive(raw) for raw in actual["naive_first_questions"]]
        first_supports = [_parse_support(raw, enriched=True) for raw in actual["naive_first_support_responses"]]
        second_questions = [_parse_naive(raw) for raw in actual["naive_second_questions"]]
        final_supports = [_parse_support(raw, enriched=True) for raw in actual["naive_final_support_responses"]]
        if not all(len(values) == 64 for values in (first_questions, first_supports, second_questions, final_supports)):
            raise ValueError("naive artifact schedule changed")
        for task in range(64):
            first_mapping = _map(cigs[task], first_questions[task], truths[task][1])
            second_mapping = _map(cigs[task], second_questions[task], truths[task][1])
            raw_first_mass = _truth_mass(
                first_supports[task], truths[task][2]
            )
            raw_final_mass = _truth_mass(
                final_supports[task], truths[task][2]
            )
            valid = _distinct_actions(first_mapping, second_mapping)
            first_mass = raw_first_mass if first_mapping["supported"] else 0.0
            final_mass = raw_final_mass if valid else 0.0
            naive_rows.append({
                "endpoint_mode": "fresh_regeneration_descriptive",
                "root_index": None,
                "second_question_index": None,
                "first_supported": first_mapping["supported"],
                "second_supported": second_mapping["supported"],
                "second_action_novel": _distinct_actions(
                    first_mapping, second_mapping
                ),
                "valid_two_action_trajectory": valid,
                "raw_truth_mass_after_first": raw_first_mass,
                "truth_mass_after_first": first_mass,
                "raw_truth_mass_final": raw_final_mass,
                "truth_mass_final": final_mass,
                "brier": (1.0 - final_mass) ** 2,
                "log_loss": -math.log(max(PROBABILITY_FLOOR, final_mass)),
                "covered": final_mass > 0.0,
                "raw_fresh_truth_mass_final": raw_final_mass,
                "fresh_truth_mass_final": final_mass,
                "fresh_brier": (1.0 - final_mass) ** 2,
                "fresh_log_loss": -math.log(max(PROBABILITY_FLOOR, final_mass)),
                "fresh_covered": final_mass > 0.0,
            })

    tasks = []
    selected_questions_ok = True
    for task, (cig, support, plan) in enumerate(zip(cigs, initial, plans, strict=True)):
        aliases = truths[task][2]
        policies = {}
        selected_questions = {}
        for name, root in plan["selected"].items():
            path = first_paths[(task, root)]
            path_metrics = _path_metrics(
                path["support"],
                final_paths[(task, root)],
                question=path["second"],
                observed=path["second_mapping"]["answer"],
                aliases=aliases,
                first_mapping=path["first_mapping"],
                second_mapping=path["second_mapping"],
            )
            policies[name] = {
                "endpoint_mode": "aligned_generated_likelihood",
                "root_index": root,
                "second_question_index": path["second"],
                "first_supported": path["first_mapping"]["supported"],
                "second_supported": path["second_mapping"]["supported"],
                "second_action_novel": _distinct_actions(
                    path["first_mapping"], path["second_mapping"]
                ),
                **path_metrics,
            }
            selected_questions[name] = {
                "first": support["questions"][root],
                "second": path["support"]["questions"][path["second"]],
            }
        if naive_available:
            policies["naive_thinking"] = naive_rows[task]
            selected_questions["naive_thinking"] = {
                "first": _parse_naive(actual["naive_first_questions"][task]),
                "second": _parse_naive(actual["naive_second_questions"][task]),
            }
        selected_questions_ok &= control_by_id[cig.cig_id]["selected_questions"] == selected_questions
        tasks.append({
            "task_id": cig.cig_id,
            "selected_roots": plan["selected"],
            "root_eig": plan["root_eig"],
            "conditioned_root_risks": plan["conditioned"],
            "blind_root_risks": plan["blind"],
            "fixed_root_risks": plan["fixed"],
            "policies": policies,
        })
    samples = int(next(iter((result.get("science") or {}).get("comparisons", {}).values()))["brier_dynamic_minus_baseline"].get("samples", 20_000)) if result.get("science") else 20_000
    science_candidate = _policy_science(tasks, samples)
    usage = result["usage"]["deepseek_primary"]
    expected_requests = PLANNING_REQUESTS + len(first_manifest) + len(final_manifest)
    crn_diagnostics = _blind_crn_diagnostics(branches)
    primary_supports = [*initial, *simulated, *[_parse_support(raw, enriched=True) for raw in first_responses], *[_parse_support(raw, enriched=True) for raw in final_responses]]
    branch_groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in manifests:
        key = (
            int(row["task_index"]),
            int(row["hypothesis_index"]),
            int(row["draw"]),
        )
        branch_groups.setdefault(key, []).append(row)
    simulated_grouping_ok = len(branch_groups) == 64 * 8 * 2 and all(
        len(rows) == 4
        and {int(row["root_index"]) for row in rows} == set(range(4))
        and len({int(row["seed"]) for row in rows}) == 1
        for rows in branch_groups.values()
    )
    first_by_task: dict[int, list[dict[str, Any]]] = {}
    final_by_task: dict[int, list[dict[str, Any]]] = {}
    for row in first_manifest:
        first_by_task.setdefault(int(row["task_index"]), []).append(row)
    for row in final_manifest:
        final_by_task.setdefault(int(row["task_index"]), []).append(row)
    first_crn_ok = len(first_by_task) == 64 and all(
        len({int(row["seed"]) for row in rows}) == 1
        for rows in first_by_task.values()
    )
    final_crn_ok = len(final_by_task) == 64 and all(
        len({int(row["seed"]) for row in rows}) == 1
        for rows in final_by_task.values()
    )
    mechanics = {
        "exact_64_tasks": len(tasks) == 64,
        "expected_requests_within_frozen_maximum": PLANNING_REQUESTS <= expected_requests <= MAX_PRIMARY_REQUESTS,
        "exact_deepseek_response_count": len(primary_supports) == expected_requests,
        "exact_deepseek_accepted_requests": usage["adapter_requests"] == expected_requests,
        "exact_deepseek_http_attempts": usage["http_attempts"] == expected_requests,
        "deepseek_zero_retries": usage["retry_count"] == 0,
        "deepseek_zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "deepseek_zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "deepseek_zero_forced_exits": usage["forced_exits"] == 0,
        "all_supports_strict_and_exactly_eight_unique": all(row["diagnostic"]["valid_unique_count"] == 8 and row["diagnostic"]["question_count"] == 4 for row in primary_supports),
        "every_initial_has_two_informative_roots": all(row["diagnostic"]["informative_question_count"] >= 2 for row in initial),
        "ninety_percent_simulated_have_informative_followup": float(np.mean([row["diagnostic"]["informative_question_count"] >= 1 for row in simulated])) >= 0.90,
        "all_privacy_audits_pass": len(privacy_checks) == expected_requests
        and all(privacy_checks),
        "every_policy_has_48_supported_first_actions": all(sum(task["policies"][name]["first_supported"] for task in tasks) >= 48 for name in POLICY_NAMES),
        "every_policy_has_40_supported_second_actions": all(sum(task["policies"][name]["second_supported"] for task in tasks) >= 40 for name in POLICY_NAMES),
        "every_policy_has_40_novel_second_actions": all(sum(task["policies"][name]["second_action_novel"] for task in tasks) >= 40 for name in POLICY_NAMES),
        "every_policy_has_40_matchable_second_replies": all(sum(task["policies"][name]["second_reply_likelihood_matched"] for task in tasks) >= 40 for name in POLICY_NAMES),
        "within_combined_policy_budget": float(result["usage"]["combined_cost_usd"]) <= POLICY_BUDGET,
        "all_blind_crn_replays_exact": (
            crn_diagnostics["observed_group_count"]
            == crn_diagnostics["expected_group_count"]
            == crn_diagnostics["exact_group_count"]
        ),
        "conditioned_blind_pairs_share_exact_seed": seed_schedule_ok,
        "simulated_roots_use_task_hypothesis_draw_crn": simulated_grouping_ok,
        "simulated_crn_seed_formula_exact": seed_schedule_ok,
        "simulated_crn_seeds_distinct_across_groups": len({BRANCH_SEED_START + task * 16 + hypothesis * 2 + draw for task in range(64) for hypothesis in range(8) for draw in range(2)}) == 1024,
        "realized_first_refresh_uses_task_crn": first_crn_ok,
        "realized_first_seed_formula_exact": first_seed_ok,
        "realized_first_seeds_distinct_across_tasks": len({int(row["seed"]) for row in first_manifest}) == 64,
        "realized_final_refresh_uses_task_crn": final_crn_ok,
        "realized_final_seed_formula_exact": final_seed_ok,
        "realized_final_seeds_distinct_across_tasks": len({int(row["seed"]) for row in final_manifest}) == 64,
    }
    mechanics["all_pass"] = all(mechanics.values())
    science = science_candidate if mechanics["all_pass"] else None
    expected_status = "mechanics_failed" if not mechanics["all_pass"] else ("passed" if science["gates"]["all_pass"] else "gated_null")
    mismatches = []
    _close(
        (result.get("protocol") or {}).get(
            "distinct_action_amendment_sha256"
        ),
        ACTION_AMENDMENT_SHA256,
        "$.protocol.distinct_action_amendment_sha256",
        mismatches,
    )
    _close(
        (result.get("protocol") or {}).get(
            "valid_trajectory_amendment_sha256"
        ),
        VALID_TRAJECTORY_AMENDMENT_SHA256,
        "$.protocol.valid_trajectory_amendment_sha256",
        mismatches,
    )
    _close(
        (result.get("protocol") or {}).get("outcome_crn_amendment_sha256"),
        OUTCOME_CRN_AMENDMENT_SHA256,
        "$.protocol.outcome_crn_amendment_sha256",
        mismatches,
    )
    _close(result.get("tasks"), tasks, "$.tasks", mismatches)
    _close(
        result.get("crn_diagnostics"),
        crn_diagnostics,
        "$.crn_diagnostics",
        mismatches,
    )
    _close(result.get("science"), science, "$.science", mismatches)
    _close(result.get("mechanics_gates"), mechanics, "$.mechanics_gates", mismatches)
    _close(result.get("status"), expected_status, "$.status", mismatches)
    if not selected_questions_ok:
        mismatches.append("$.private.CONTROLS.selected_questions")
    if not seed_schedule_ok:
        mismatches.append("$.private.RAW_BRANCHES.seed_schedule")
    return _verification(run_dir, "policy", mismatches)


def _verification(run_dir: Path, kind: str, mismatches: Sequence[str]) -> dict[str, Any]:
    artifacts = {
        str(path.relative_to(run_dir)): sha256_file(path)
        for path in sorted(run_dir.rglob("*.json"))
        if path.name != "VERIFICATION.json"
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "verified" if not mismatches else "verification_failed",
        "kind": kind,
        "model_calls": 0,
        "cost_usd": 0.0,
        "result_status": _load(run_dir / "RESULT.json").get("status"),
        "checks": {
            "raw_artifacts_reparsed": True,
            "truth_controls_replayed": True,
            "scientific_endpoint_recomputed": True,
            "reported_result_matches_replay": not mismatches,
        },
        "mismatches": list(mismatches),
        "artifact_sha256": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("support", "policy_smoke", "policy"),
        required=True,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "development"))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.kind == "support":
        if args.stage is None:
            parser.error("--stage is required for support verification")
        verification = verify_support(run_dir, stage=args.stage)
    elif args.kind == "policy_smoke":
        if args.stage is not None:
            parser.error("--stage is not used for policy-smoke verification")
        verification = verify_policy_smoke(run_dir)
    else:
        if args.stage is not None:
            parser.error("--stage is not used for policy verification")
        verification = verify_policy(run_dir)
    checkpoint(run_dir / "VERIFICATION.json", verification)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
