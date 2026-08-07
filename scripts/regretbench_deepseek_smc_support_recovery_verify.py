#!/usr/bin/env python3
"""Independently replay the sealed RegretBench SMC support contingency.

This verifier makes no model calls and does not import the experiment producer
or its core. It reconstructs parent provenance, child lineage, privacy hashes,
truth coverage, bootstrap statistics, gates, and status from raw artifacts.
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
INTERFACE_VERSION = "regretbench-deepseek-smc-support-verify-1"
PRODUCER_INTERFACE = "regretbench-deepseek-smc-support-recovery-daily-1"
PRIMARY_INTERFACE = "regretbench-deepseek-support-recovery-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
TASKS = 64
PARENT_PARTICLES = 8
CHILD_PARTICLES = 8
QUESTIONS = 4
MIN_RETAINED = 2
MAX_RETAINED = 6
EXPECTED_REQUESTS = 128
TRUTH_SEED_START = 202608082000
BRANCH_SEED_START = 202608270000
BOOTSTRAP_SEED = 202608280000
RUN_BUDGET_USD = 0.50
PROTOCOL_SHA256 = (
    "b8eafe438c21e4793a59bd24a0991d20ecf4f53efa5a19fe21dc4eed7c22f807"
)
SOURCE_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/regretbench_llm_native_source_audit/"
    "SOURCE_PROTOCOL_MANIFEST.json"
)
PRIMARY_DIR = REPO_ROOT / (
    "results/nonmyopic/regretbench_deepseek_support_recovery/"
    "development-20260808"
)
UNSUPPORTED_REPLY = "I cannot answer that clarification."


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def normalize_text(text: str) -> str:
    return re.sub(
        r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.casefold())
    ).strip()


def _close(
    left: Any,
    right: Any,
    path: str = "$",
    mismatches: list[str] | None = None,
) -> list[str]:
    output = mismatches if mismatches is not None else []
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or left is None
        or right is None
    ):
        if left != right:
            output.append(path)
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(
            float(left), float(right), rel_tol=1e-11, abs_tol=1e-12
        ):
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


def _lexical_alias_match(generated: str, alias: str) -> bool:
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


def _covered(support: Mapping[str, Any], aliases: str) -> bool:
    alternatives = [value.strip() for value in aliases.split("|") if value.strip()]
    return any(
        _lexical_alias_match(row["final_answer"], alias)
        for row in support["hypotheses"]
        for alias in alternatives
    )


def _cigs() -> list[CIG]:
    manifest = _load(SOURCE_MANIFEST)
    ids = manifest["splits"]["development"]["ids"]
    if len(ids) != TASKS:
        raise ValueError("development cohort size changed")
    cigs = [
        load_cig(
            REGRETBENCH_ROOT / "data/OpenDomainQA/test" / f"{cig_id}.json"
        )
        for cig_id in ids
    ]
    if [cig.cig_id for cig in cigs] != ids:
        raise ValueError("development cohort order changed")
    return cigs


def _parse_parent(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("parent response has wrong top-level fields")
    hypotheses = value["hypotheses"]
    if not isinstance(hypotheses, list) or len(hypotheses) != PARENT_PARTICLES:
        raise ValueError("parent response must contain eight slots")
    particles = []
    for index, item in enumerate(hypotheses):
        if not isinstance(item, dict) or set(item) != {
            "interpretation",
            "final_answer",
            "prior_weight",
        }:
            raise ValueError("parent particle has wrong fields")
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
            raise ValueError("parent particle has invalid values")
        particles.append(
            {
                "parent_index": index,
                "interpretation": interpretation.strip(),
                "final_answer": answer.strip(),
                "probability": float(weight),
            }
        )
    total = sum(row["probability"] for row in particles)
    if total <= 0:
        raise ValueError("parent weights sum to zero")
    for row in particles:
        row["probability"] /= total
    questions = value["questions"]
    if not isinstance(questions, list) or len(questions) != QUESTIONS:
        raise ValueError("parent response must contain four questions")
    cleaned = []
    seen = set()
    for question in questions:
        if not isinstance(question, str) or not question.strip().endswith("?"):
            raise ValueError("parent response contains a non-question")
        text = question.strip()
        key = normalize_text(text)
        if not key or key in seen:
            raise ValueError("parent response contains duplicate questions")
        seen.add(key)
        cleaned.append(text)
    return {
        "hypotheses": particles,
        "questions": cleaned,
        "raw_parent_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }


def _parse_child(raw: str, parent: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("child response has wrong top-level fields")
    hypotheses = value["hypotheses"]
    if not isinstance(hypotheses, list) or len(hypotheses) != CHILD_PARTICLES:
        raise ValueError("child response must contain eight particles")
    children = []
    indexes = []
    seen = set()
    retained = 0
    for item in hypotheses:
        if not isinstance(item, dict) or set(item) != {
            "parent_index",
            "revision_type",
            "interpretation",
            "final_answer",
            "prior_weight",
        }:
            raise ValueError("child particle has wrong fields")
        parent_index = item["parent_index"]
        revision_type = item["revision_type"]
        interpretation = item["interpretation"]
        answer = item["final_answer"]
        weight = item["prior_weight"]
        if (
            isinstance(parent_index, bool)
            or not isinstance(parent_index, int)
            or parent_index not in range(PARENT_PARTICLES)
            or revision_type not in {"retained", "revised"}
            or not isinstance(interpretation, str)
            or not interpretation.strip()
            or not isinstance(answer, str)
            or not answer.strip()
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
        ):
            raise ValueError("child particle has invalid values")
        parent_row = parent["hypotheses"][parent_index]
        unchanged = (
            normalize_text(interpretation)
            == normalize_text(parent_row["interpretation"])
            and normalize_text(answer)
            == normalize_text(parent_row["final_answer"])
        )
        if (revision_type == "retained") != unchanged:
            raise ValueError("child revision label disagrees with lineage")
        retained += revision_type == "retained"
        key = (normalize_text(interpretation), normalize_text(answer))
        if key in seen:
            raise ValueError("child response contains duplicate hypotheses")
        seen.add(key)
        indexes.append(parent_index)
        children.append(
            {
                "parent_index": parent_index,
                "revision_type": revision_type,
                "interpretation": interpretation.strip(),
                "final_answer": answer.strip(),
                "probability": float(weight),
            }
        )
    if sorted(indexes) != list(range(PARENT_PARTICLES)):
        raise ValueError("child lineage is not an exact parent permutation")
    if not MIN_RETAINED <= retained <= MAX_RETAINED:
        raise ValueError("child retention count is outside the frozen range")
    total = sum(row["probability"] for row in children)
    if total <= 0:
        raise ValueError("child weights sum to zero")
    for row in children:
        row["probability"] /= total
    questions = value["questions"]
    if not isinstance(questions, list) or len(questions) != QUESTIONS:
        raise ValueError("child response must contain four questions")
    cleaned = []
    seen_questions = set()
    for question in questions:
        if not isinstance(question, str) or not question.strip().endswith("?"):
            raise ValueError("child response contains a non-question")
        text = question.strip()
        key = normalize_text(text)
        if not key or key in seen_questions:
            raise ValueError("child response contains duplicate questions")
        seen_questions.add(key)
        cleaned.append(text)
    return {
        "hypotheses": children,
        "questions": cleaned,
        "diagnostic": {
            "codec_mode": "strict_json",
            "valid_unique_count": CHILD_PARTICLES,
            "parent_index_permutation_exact": True,
            "retained_count": retained,
            "revised_count": CHILD_PARTICLES - retained,
            "question_count": QUESTIONS,
            "parent_population_sha256": parent["raw_parent_sha256"],
        },
    }


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


def _audit(
    cig: CIG,
    dialogue: Sequence[Mapping[str, str]],
    parent: Mapping[str, Any],
) -> dict[str, Any]:
    records = [
        {"role": str(item["role"]), "content": str(item["content"]).strip()}
        for item in dialogue
    ]
    payload = {
        "task_id": cig.cig_id,
        "prompt": cig.prompt,
        "dialogue": records,
        "parent_particles": parent["hypotheses"],
        "parent_population_sha256": parent["raw_parent_sha256"],
        "parent_source": "verified_primary_raw_root",
    }
    return {
        "passed": True,
        "payload_keys": sorted(payload),
        "payload_sha256": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
        "parent_population_sha256": parent["raw_parent_sha256"],
        "parent_source": "verified_primary_raw_root",
    }


def _bootstrap(values: Sequence[float], samples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not array.size:
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


def _science(rows: Sequence[Mapping[str, Any]], samples: int) -> dict[str, Any]:
    supported = [row for row in rows if row["supported"]]
    missing = [row for row in supported if not row["root_covered"]]
    covered = [row for row in supported if row["root_covered"]]
    differences = [
        float(row["conditioned_covered"]) - float(row["blind_covered"])
        for row in supported
    ]
    missing_differences = [
        float(row["conditioned_covered"]) - float(row["blind_covered"])
        for row in missing
    ]
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    conditioned_losses = sum(not row["conditioned_covered"] for row in covered)
    blind_losses = sum(not row["blind_covered"] for row in covered)
    overall = _bootstrap(differences, samples)
    missing_bootstrap = _bootstrap(missing_differences, samples)
    retention = (
        float(np.mean([row["conditioned_covered"] for row in covered]))
        if covered
        else None
    )
    gates = {
        "at_least_48_supported_tasks": len(supported) >= 48,
        "at_least_16_root_missing_supported_tasks": len(missing) >= 16,
        "conditioned_minus_blind_coverage_at_least_005": overall["mean"] is not None
        and overall["mean"] >= 0.05,
        "bootstrap_probability_positive_at_least_080": overall[
            "probability_positive"
        ]
        is not None
        and overall["probability_positive"] >= 0.80,
        "conditioned_recoveries_exceed_losses": wins > losses,
        "root_missing_recovery_difference_at_least_010": missing_bootstrap["mean"]
        is not None
        and missing_bootstrap["mean"] >= 0.10,
        "conditioned_root_covered_retention_at_least_090": retention is not None
        and retention >= 0.90,
        "conditioned_root_covered_losses_no_more_than_blind": conditioned_losses
        <= blind_losses,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "population": {
            "all_tasks": len(rows),
            "supported_tasks": len(supported),
            "root_missing_supported_tasks": len(missing),
            "root_covered_supported_tasks": len(covered),
        },
        "coverage": {
            "conditioned_minus_history_blind": overall,
            "root_missing_conditioned_minus_history_blind": missing_bootstrap,
            "conditioned_root_covered_retention": retention,
        },
        "paired_outcomes": {
            "conditioned_recoveries": wins,
            "conditioned_losses": losses,
            "ties": len(differences) - wins - losses,
            "conditioned_root_covered_losses": conditioned_losses,
            "blind_root_covered_losses": blind_losses,
        },
        "gates": gates,
    }


def verify(run_dir: Path, *, primary_dir: Path = PRIMARY_DIR) -> dict[str, Any]:
    result = _load(run_dir / "RESULT.json")
    raw = _load(run_dir / "private" / "RAW_RESPONSES.json")
    controls = _load(run_dir / "private" / "CONTROLS.json")
    primary_result = _load(primary_dir / "RESULT.json")
    primary_verification = _load(primary_dir / "VERIFICATION.json")
    primary_raw_path = primary_dir / "private" / "RAW_RESPONSES.json"
    primary_controls_path = primary_dir / "private" / "CONTROLS.json"
    primary_raw = _load(primary_raw_path)
    primary_controls = _load(primary_controls_path)
    if (
        primary_result.get("interface_version") != PRIMARY_INTERFACE
        or primary_result.get("status") != "gated_null"
        or primary_result.get("mechanics_gates", {}).get("all_pass") is not True
        or primary_verification.get("status") != "verified"
        or primary_verification.get("mismatches") != []
        or primary_verification.get("artifact_sha256", {}).get("RESULT.json")
        != sha256_file(primary_dir / "RESULT.json")
        or primary_verification.get("artifact_sha256", {}).get(
            "private/RAW_RESPONSES.json"
        )
        != sha256_file(primary_raw_path)
        or primary_verification.get("artifact_sha256", {}).get(
            "private/CONTROLS.json"
        )
        != sha256_file(primary_controls_path)
    ):
        raise ValueError("primary predecessor is not an independently verified null")
    roots_raw = primary_raw.get("root") or []
    branches_raw = raw.get("branches") or []
    if len(roots_raw) != TASKS or len(branches_raw) != EXPECTED_REQUESTS:
        raise ValueError("SMC raw response schedule changed")
    if raw.get("primary_raw_sha256") != sha256_file(primary_raw_path):
        raise ValueError("SMC parent source hash changed")
    if raw.get("interface_version") != PRODUCER_INTERFACE:
        raise ValueError("SMC raw-response interface changed")
    if controls.get("primary_controls_sha256") != sha256_file(
        primary_controls_path
    ):
        raise ValueError("SMC primary-control source hash changed")
    cigs = _cigs()
    primary_control_by_id = {
        row["task_id"]: row for row in primary_controls.get("roots", [])
    }
    rows = []
    audits = []
    supports = []
    for index, (cig, parent_raw) in enumerate(zip(cigs, roots_raw, strict=True)):
        parent = _parse_parent(parent_raw)
        truth_index, truth = _truth(cig, TRUTH_SEED_START + index)
        question = parent["questions"][0]
        mapping = _map(cig, question, truth)
        aliases = str((truth.slots or {})["answer_aliases"])
        primary_control = primary_control_by_id.get(cig.cig_id)
        expected_control = {
            "task_id": cig.cig_id,
            "truth_index": truth_index,
            "question": question,
            "mapping": mapping,
            "aliases": aliases,
        }
        if primary_control is None or _close(primary_control, expected_control):
            raise ValueError(f"primary private control mismatch: {cig.cig_id}")
        conditioned = _parse_child(branches_raw[2 * index], parent)
        blind = _parse_child(branches_raw[2 * index + 1], parent)
        supports.extend([conditioned, blind])
        dialogue = [
            {"role": "assistant", "content": question},
            {"role": "user", "content": mapping["answer"]},
        ]
        audits.extend([_audit(cig, dialogue, parent), _audit(cig, [], parent)])
        rows.append(
            {
                "task_id": cig.cig_id,
                "task_index": index,
                "supported": mapping["supported"],
                "root_covered": _covered(parent, aliases),
                "refresh_seed": BRANCH_SEED_START + index,
                "parent_population_sha256": parent["raw_parent_sha256"],
                "conditioned_dispatch_index": 2 * index,
                "blind_dispatch_index": 2 * index + 1,
                "conditioned_retained_count": conditioned["diagnostic"][
                    "retained_count"
                ],
                "blind_retained_count": blind["diagnostic"]["retained_count"],
                "conditioned_covered": _covered(conditioned, aliases),
                "blind_covered": _covered(blind, aliases),
            }
        )
    usage = result.get("usage") or {}
    exact_lineage = all(
        support["diagnostic"]["codec_mode"] == "strict_json"
        and support["diagnostic"]["valid_unique_count"] == CHILD_PARTICLES
        and support["diagnostic"]["parent_index_permutation_exact"] is True
        and MIN_RETAINED <= support["diagnostic"]["retained_count"] <= MAX_RETAINED
        and support["diagnostic"]["revised_count"]
        == CHILD_PARTICLES - support["diagnostic"]["retained_count"]
        and support["diagnostic"]["question_count"] == QUESTIONS
        for support in supports
    )
    mechanics = {
        "exact_64_tasks": len(rows) == TASKS,
        "unique_task_ids": len({row["task_id"] for row in rows}) == TASKS,
        "exact_128_child_responses": len(supports) == EXPECTED_REQUESTS,
        "exact_128_accepted_requests": usage.get("adapter_requests")
        == EXPECTED_REQUESTS,
        "exact_128_http_attempts": usage.get("http_attempts") == EXPECTED_REQUESTS,
        "zero_retries": usage.get("retry_count") == 0,
        "zero_provider_error_retries": usage.get("provider_error_retries") == 0,
        "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "all_child_lineages_strict_and_exact": exact_lineage,
        "conditioned_blind_parent_provenance_exact": all(
            row["parent_population_sha256"]
            == supports[2 * index]["diagnostic"]["parent_population_sha256"]
            == supports[2 * index + 1]["diagnostic"]["parent_population_sha256"]
            for index, row in enumerate(rows)
        ),
        "conditioned_blind_seed_formula_and_adjacency_exact": all(
            row["task_index"] == index
            and row["refresh_seed"] == BRANCH_SEED_START + index
            and row["conditioned_dispatch_index"] == 2 * index
            and row["blind_dispatch_index"] == 2 * index + 1
            for index, row in enumerate(rows)
        ),
        "at_least_48_supported_roots": sum(row["supported"] for row in rows) >= 48,
        "all_128_privacy_and_provenance_audits_pass": len(audits)
        == EXPECTED_REQUESTS
        and all(item["passed"] for item in audits),
        "within_run_budget": float(usage.get("run_cost_usd", math.inf))
        <= RUN_BUDGET_USD + 1e-12,
    }
    mechanics["all_pass"] = all(mechanics.values())
    samples = int(
        ((result.get("science") or {}).get("coverage") or {})
        .get("conditioned_minus_history_blind", {})
        .get("samples", 20_000)
    )
    science = _science(rows, samples) if mechanics["all_pass"] else None
    expected_status = "mechanics_failed"
    if mechanics["all_pass"]:
        expected_status = (
            "passed" if science and science["gates"]["all_pass"] else "gated_null"
        )
    expected_authorizes = (
        "separately_preregistered_smc_policy_only"
        if expected_status == "passed"
        else "nothing"
    )
    expected_protocol = {
        "model": MODEL_ID,
        "reasoning": "disabled_excluded",
        "temperature": 0.7,
        "max_tokens": 2_200,
        "expected_requests": EXPECTED_REQUESTS,
        "parent_particles": PARENT_PARTICLES,
        "child_particles": CHILD_PARTICLES,
        "matched_refresh_seed": True,
        "conditioned_blind_dispatch_adjacent": True,
        "initial_support_calls_repeated": False,
        "hidden_cig_exposed_to_model": False,
        "primary_policy_endpoint_opened": False,
        "primary_confirmation_opened": False,
        "smc_policy_endpoint_opened": False,
        "support_recovery_endpoint_accessed": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "primary_result_sha256": sha256_file(primary_dir / "RESULT.json"),
        "primary_verification_sha256": sha256_file(
            primary_dir / "VERIFICATION.json"
        ),
        "primary_raw_sha256": sha256_file(primary_raw_path),
        "primary_controls_sha256": sha256_file(primary_controls_path),
    }
    mismatches: list[str] = []
    _close(result.get("interface_version"), PRODUCER_INTERFACE, "$.interface_version", mismatches)
    _close(result.get("status"), expected_status, "$.status", mismatches)
    _close(result.get("authorizes"), expected_authorizes, "$.authorizes", mismatches)
    _close(result.get("protocol"), expected_protocol, "$.protocol", mismatches)
    _close(result.get("mechanics_gates"), mechanics, "$.mechanics_gates", mismatches)
    _close(result.get("science"), science, "$.science", mismatches)
    _close(result.get("tasks"), rows, "$.tasks", mismatches)
    _close(controls.get("privacy"), audits, "$.private.CONTROLS.privacy", mismatches)
    _close(
        controls.get("primary_raw_sha256"),
        sha256_file(primary_raw_path),
        "$.private.CONTROLS.primary_raw_sha256",
        mismatches,
    )
    artifacts = {
        str(path.relative_to(run_dir)): sha256_file(path)
        for path in sorted(run_dir.rglob("*.json"))
        if path.name != "VERIFICATION.json"
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "verified" if not mismatches else "verification_failed",
        "kind": "smc_support_recovery",
        "model_calls": 0,
        "cost_usd": 0.0,
        "result_status": result.get("status"),
        "checks": {
            "primary_null_revalidated": True,
            "raw_parent_particles_reparsed": True,
            "raw_child_particles_reparsed": True,
            "lineage_reconstructed": True,
            "truth_controls_replayed": True,
            "privacy_and_provenance_recomputed": True,
            "scientific_endpoint_recomputed": True,
            "reported_result_matches_replay": not mismatches,
        },
        "mismatches": mismatches,
        "artifact_sha256": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir.resolve())
    if args.output:
        checkpoint(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
