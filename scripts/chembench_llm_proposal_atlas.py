#!/usr/bin/env python3
"""Build and evaluate the frozen ChemBench LLM proposal-atlas gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.chembench_mopen.proposal_atlas import (
    core_family_jaccard,
    construct_opportunity_rows,
    private_task_record,
    proposal_induced_risk,
    public_task_record,
    random_typed_candidates,
    render_prompt,
    response_json_schema,
    retrieve_atlas_candidates,
    score_response,
    select_panel,
    standardized_features,
)
from environments.chembench_mopen.source import build_mixed_version_responses
from scripts.chembench_factored_mopen_oracle import make_factored_bank, sha256
from scripts.chembench_mopen_mechanics import INITIAL_SUPPORT_NAMES
from scripts.chembench_mopen_nonmyopic_opportunity import (
    EXECUTION_BUDGET,
    ValidationSlice,
    active_domains,
    frozen_assays,
    load_source,
    verify_source,
)


SCHEMA_VERSION = "chembench-llm-proposal-atlas-v1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
TEMPERATURE = 0.3
MAX_COMPLETION_TOKENS = 700
MAX_PROMPT_CHARS = 24_000
MAX_CONCURRENCY = 64
STAGE_CAP_USD = 0.75
DAILY_CAP_USD = 5.0
BASE_REQUEST_SEED = 202608370000
SECOND_ATLAS_SEED = 202608371000
SLICES = (
    ValidationSlice("easy", "v3", 2026081601),
    ValidationSlice("medium", "v3", 2026081602),
    ValidationSlice("hard", "v3", 2026081603),
)
PROTOCOL_BINDINGS = {
    Path("results/nonmyopic/CHEMBENCH_LLM_PROPOSAL_ATLAS_PROTOCOL_20260815.md"):
        "59ca8b498e0b01a6de5bea08414297edb340befd824de0b2e14e89c46e6a3483",
    Path(
        "results/nonmyopic/"
        "CHEMBENCH_LLM_PROPOSAL_ATLAS_SOURCE_SELECTION_CLARIFICATION_20260815.md"
    ):
        "5fb61a14e7bc4c6b1540ab79b81ef1281b4f868e0969b43c76f23f586dbf705c",
    Path("results/nonmyopic/CHEMBENCH_FACTORED_MOPEN_ORACLE_PROTOCOL_20260815.md"):
        "00d11d703098205122d3913ded06242f85d2c67f90fe7978130e483c3d39de66",
}
DEPENDENCY_BINDINGS = {
    Path(
        "results/nonmyopic/chembench_factored_mopen_oracle/"
        "factored-v1-20260815/RESULT.json.gz"
    ): "247f0b3996ce3ea58a71e37af8cc314dad1399d5f8987850b590b615ae6f3d2e",
    Path(
        "results/nonmyopic/chembench_factored_mopen_oracle/"
        "factored-v1-20260815/TRANSITION_BANK.json.gz"
    ): "4a48da4968407059806bdc70bee6f7dce987d36f63bbc5e7ab8f91b60ce280b3",
    Path(
        "results/nonmyopic/chembench_factored_mopen_oracle/"
        "factored-v1-20260815/VERIFICATION.json"
    ): "d97daa7dacbbeb7ab0a08907fd4e3785f92a319e0aa224f5ebdfc5f9059594f3",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def git_value(arguments: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def require_pushed_commit(required_commit: str) -> str:
    head = git_value(("rev-parse", "HEAD"))
    resolved = git_value(("rev-parse", required_commit))
    if head != resolved:
        raise RuntimeError(f"required commit is not HEAD: {resolved} != {head}")
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            head,
            "origin/codex/location-finding-llmstrategy",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return head


def verify_bindings() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path, expected in {**PROTOCOL_BINDINGS, **DEPENDENCY_BINDINGS}.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"binding mismatch for {path}: {actual} != {expected}")
        result[str(path)] = {"sha256": actual}
    return result


def build_source_context(source_root: Path) -> dict[str, Any]:
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    domains = active_domains(source)
    assays = frozen_assays()
    action_names = tuple(item.name for item in assays)
    banks = {}
    mixed_by_difficulty = {}
    rows = []
    for item in SLICES:
        mixed = build_mixed_version_responses(
            source,
            domains,
            INITIAL_SUPPORT_NAMES,
            difficulty=item.difficulty,
            initial_version="v2",
            truth_version=item.version,
            query_seed=item.query_seed,
            assays=assays,
        )
        bank = make_factored_bank(
            mixed.observation_means,
            mixed.target_log_rates,
            domains,
            action_names,
        )
        banks[item.difficulty] = bank
        mixed_by_difficulty[item.difficulty] = mixed
        rows.extend(construct_opportunity_rows(bank, item.difficulty, mixed.truth_indices))
    panel = select_panel(rows)
    return {
        "source_binding": source_binding,
        "domains": domains,
        "assays": assays,
        "banks": banks,
        "mixed": mixed_by_difficulty,
        "rows": tuple(rows),
        "panel": panel,
    }


def _request_plan(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        tasks,
        key=lambda task: (
            str(task["difficulty"]),
            int(task["history_length"]),
            str(task["task_id"]),
        ),
    )
    requests = []
    for position, task in enumerate(ordered):
        replicas = (0, 1) if task["split"] == "atlas" else (0,)
        for replica in replicas:
            seed = (
                BASE_REQUEST_SEED + position
                if replica == 0
                else SECOND_ATLAS_SEED + position
            )
            for arm in ("residual_aware", "history_blind"):
                prompt = task["prompts"][arm]
                request_id = f"{task['task_id']}:{arm}:r{replica}"
                requests.append(
                    {
                        "request_id": request_id,
                        "task_id": task["task_id"],
                        "task_position": position,
                        "split": task["split"],
                        "arm": arm,
                        "replicate": replica,
                        "seed": seed,
                        "prompt_sha256": payload_hash(prompt),
                    }
                )
    return sorted(requests, key=lambda item: item["request_id"])


def build_source_package(source_root: Path, implementation_commit: str) -> dict[str, Any]:
    bindings = verify_bindings()
    context = build_source_context(source_root)
    banks = context["banks"]
    public_tasks = []
    private_tasks = []
    for task_position, row in enumerate(context["panel"]):
        bank = banks[row.difficulty]
        public = public_task_record(
            bank,
            row,
            task_position=task_position,
            remaining_budget=EXECUTION_BUDGET - row.history_length,
        )
        public["prompts"] = {
            arm: render_prompt(bank, public, arm)
            for arm in ("residual_aware", "history_blind")
        }
        public["prompt_sha256"] = {
            arm: payload_hash(prompt) for arm, prompt in public["prompts"].items()
        }
        public_tasks.append(public)
        private_tasks.append(private_task_record(row, public))
    requests = _request_plan(public_tasks)
    hidden_names = {item["truth_model_name"] for item in private_tasks}
    prompt_text = canonical_json(
        [task["prompts"] for task in public_tasks]
    )
    outside_names = {
        name
        for name in context["domains"]
        if name not in set(INITIAL_SUPPORT_NAMES)
    }
    source_counts = {
        f"{difficulty}/{length}": sum(
            row.difficulty == difficulty and row.history_length == length
            for row in context["rows"]
        )
        for difficulty in ("easy", "medium", "hard")
        for length in (1, 2, 3)
    }
    family_count = len(
        {
            item["truth_signature"]["core_family"]
            for item in private_tasks
        }
    )
    conditions = {
        "four_tasks_per_stratum": all(
            sum(
                task["difficulty"] == difficulty
                and task["history_length"] == length
                for task in public_tasks
            )
            == 4
            for difficulty in ("easy", "medium", "hard")
            for length in (1, 2, 3)
        ),
        "thirty_six_unique_generators": len(hidden_names) == len(private_tasks) == 36,
        "at_least_ten_core_families": family_count >= 10,
        "all_latest_states_trigger": all(
            bool(task["residual_report"]["latest"]["expansion_triggered"])
            for task in public_tasks
        ),
        "truth_in_source_oracle_top_four": all(
            item["truth_model_index"] in item["oracle_proposal_indices"]
            for item in private_tasks
        ),
        "no_hidden_generator_name_in_prompts": not any(
            name in prompt_text for name in outside_names
        ),
        "request_counts_exact": sum(item["arm"] == "residual_aware" for item in requests)
        == 63
        and sum(item["arm"] == "history_blind" for item in requests) == 63,
        "paired_seed_identity": all(
            len({item["seed"] for item in requests if item["task_id"] == task["task_id"] and item["replicate"] == replica})
            == 1
            for task in public_tasks
            for replica in ((0, 1) if task["split"] == "atlas" else (0,))
        ),
        "prompt_lengths_bounded": all(
            len(prompt["system"]) + len(prompt["user"]) <= MAX_PROMPT_CHARS
            for task in public_tasks
            for prompt in task["prompts"].values()
        ),
    }
    if not all(conditions.values()):
        raise RuntimeError(f"proposal-atlas source gate failed: {conditions}")
    schema = response_json_schema(banks["easy"])
    if any(response_json_schema(bank) != schema for bank in banks.values()):
        raise AssertionError("response schema differs by difficulty")
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}-public-manifest",
        "implementation_commit": implementation_commit,
        "source": context["source_binding"],
        "bindings": bindings,
        "configuration": {
            "model": MODEL_ID,
            "reasoning": "disabled",
            "temperature": TEMPERATURE,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "max_concurrency": MAX_CONCURRENCY,
            "zero_retries": True,
            "stage_cap_usd": STAGE_CAP_USD,
            "daily_cap_usd": DAILY_CAP_USD,
            "atlas_neighbors": 3,
            "atlas_limit": 4,
        },
        "source_counts": source_counts,
        "source_gate": {"passed": True, "conditions": conditions},
        "response_schema": schema,
        "tasks": public_tasks,
        "requests": requests,
    }
    labels = {
        "schema_version": f"{SCHEMA_VERSION}-sealed-labels",
        "implementation_commit": implementation_commit,
        "tasks": private_tasks,
    }
    return {"manifest": manifest, "labels": labels, "context": context}


def _history_from_public(task: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    labels = {"low": 0, "mid": 1, "high": 2}
    return tuple(
        (int(item["action_index"]), labels[str(item["outcome"])])
        for item in task["history"]
    )


def _mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(record[key]) for record in records])) if records else 0.0


def evaluate_records(
    source_root: Path,
    manifest: Mapping[str, Any],
    labels: Mapping[str, Any],
    response_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate immutable response records; transport wrappers call this after serving."""

    implementation_commit = str(manifest["implementation_commit"])
    rebuilt = build_source_package(source_root, implementation_commit)
    if rebuilt["manifest"] != manifest or rebuilt["labels"] != labels:
        raise RuntimeError("proposal-atlas manifest or sealed labels do not replay")
    context = rebuilt["context"]
    banks = context["banks"]
    public_by_id = {str(item["task_id"]): item for item in manifest["tasks"]}
    private_by_id = {str(item["task_id"]): item for item in labels["tasks"]}
    row_by_id = {
        str(public["task_id"]): row
        for public, row in zip(manifest["tasks"], context["panel"], strict=True)
    }
    expected_by_id = {str(item["request_id"]): item for item in manifest["requests"]}
    actual_by_id = {str(item["request_id"]): item for item in response_records}
    if len(actual_by_id) != len(response_records) or set(actual_by_id) != set(expected_by_id):
        raise ValueError("response records do not exactly cover the request manifest")
    scored = []
    compiled_by_task: dict[str, list[list[int]]] = defaultdict(list)
    for request_id in sorted(expected_by_id):
        expected = expected_by_id[request_id]
        record = actual_by_id[request_id]
        for key in ("task_id", "arm", "replicate", "seed"):
            if record.get(key) != expected[key]:
                raise ValueError(f"response identity mismatch for {request_id}/{key}")
        if record.get("prompt_sha256") != expected["prompt_sha256"]:
            raise ValueError(f"response prompt hash mismatch for {request_id}")
        task_id = str(expected["task_id"])
        clean = bool(
            record.get("model_requested") == MODEL_ID
            and record.get("model_returned") == MODEL_ID
            and record.get("finish_reasons") == ["stop"]
            and isinstance(record.get("content"), str)
            and bool(record["content"].strip())
            and not bool(record.get("reasoning_present"))
            and record.get("error_type") is None
        )
        if bool(record.get("clean")) != clean:
            raise ValueError(f"response clean flag is inconsistent for {request_id}")
        raw = record.get("content") if clean else None
        bank = banks[str(public_by_id[task_id]["difficulty"])]
        truth = int(private_by_id[task_id]["truth_model_index"])
        metrics = score_response(bank, row_by_id[task_id].state, raw, truth)
        scored_record = {**expected, "clean": clean, "metrics": metrics}
        scored.append(scored_record)
        if expected["arm"] == "residual_aware" and public_by_id[task_id]["split"] == "atlas":
            compiled_by_task[task_id].append(metrics["compiled_candidate_indices"])
    by_arm = {}
    for arm in ("residual_aware", "history_blind"):
        arm_records = [item for item in scored if item["arm"] == arm]
        metrics = [item["metrics"] for item in arm_records]
        by_arm[arm] = {
            "requests": len(arm_records),
            "clean_rate": _mean(arm_records, "clean"),
            "schema_valid_rate": _mean(metrics, "schema_valid"),
            "item_compile_rate": _mean(metrics, "item_compile_rate"),
            "response_executable_rate": _mean(metrics, "response_executable"),
            "truth_recall_at_4": _mean(metrics, "truth_recall_at_4"),
            "core_family_recall_at_4": _mean(metrics, "core_family_recall_at_4"),
            "modifier_f1": _mean(metrics, "modifier_f1"),
        }
    paired = []
    grouped = defaultdict(dict)
    for item in scored:
        grouped[(item["task_id"], item["replicate"])][item["arm"]] = item
    for key, pair in grouped.items():
        if set(pair) != {"residual_aware", "history_blind"}:
            raise ValueError(f"unpaired semantic record: {key}")
        dynamic = float(pair["residual_aware"]["metrics"]["semantic_score"])
        blind = float(pair["history_blind"]["metrics"]["semantic_score"])
        paired.append(dynamic - blind)
    heldout = []
    atlas_tasks = [task for task in manifest["tasks"] if task["split"] == "atlas"]
    heldout_tasks = [task for task in manifest["tasks"] if task["split"] == "heldout"]
    standardized, feature_metadata = standardized_features(banks, atlas_tasks, heldout_tasks)
    for task in heldout_tasks:
        task_id = str(task["task_id"])
        bank = banks[str(task["difficulty"])]
        row = row_by_id[task_id]
        truth = int(private_by_id[task_id]["truth_model_index"])
        fresh_record = next(
            item for item in scored
            if item["task_id"] == task_id and item["arm"] == "residual_aware"
        )
        blind_record = next(
            item for item in scored
            if item["task_id"] == task_id and item["arm"] == "history_blind"
        )
        fresh = fresh_record["metrics"]["compiled_candidate_indices"]
        blind = blind_record["metrics"]["compiled_candidate_indices"]
        atlas = retrieve_atlas_candidates(
            task, atlas_tasks, standardized, compiled_by_task
        )
        atlas_candidates = atlas["candidate_indices"]
        random_candidates = random_typed_candidates(
            bank, row.state, task_position=int(task["task_position"])
        )
        oracle_candidates = private_by_id[task_id]["oracle_proposal_indices"]
        truth_signature = bank.compiler.signatures[truth]

        def recalls(candidates: Sequence[int]) -> dict[str, int]:
            return {
                "truth": int(truth in candidates),
                "core": int(
                    any(
                        bank.compiler.signatures[int(candidate)].core == truth_signature.core
                        for candidate in candidates
                    )
                ),
            }

        history = _history_from_public(task)
        heldout.append(
            {
                "task_id": task_id,
                "fresh": {**recalls(fresh), "candidates": fresh},
                "blind": {**recalls(blind), "candidates": blind},
                "atlas": {
                    **recalls(atlas_candidates),
                    **atlas,
                    "fresh_core_jaccard": core_family_jaccard(
                        bank, atlas_candidates, fresh
                    ),
                },
                "random": {**recalls(random_candidates), "candidates": list(random_candidates)},
                "oracle": {**recalls(oracle_candidates), "candidates": oracle_candidates},
                "risk": {
                    mode: proposal_induced_risk(bank, history, truth, candidates)
                    for mode, candidates in {
                        "fresh": fresh,
                        "blind": blind,
                        "atlas": atlas_candidates,
                        "random": random_candidates,
                        "oracle": oracle_candidates,
                    }.items()
                },
            }
        )
    heldout_summary = {
        "fresh_truth_recall": sum(item["fresh"]["truth"] for item in heldout),
        "fresh_core_recall": sum(item["fresh"]["core"] for item in heldout),
        "atlas_truth_recall": sum(item["atlas"]["truth"] for item in heldout),
        "atlas_core_recall": sum(item["atlas"]["core"] for item in heldout),
        "atlas_fresh_core_jaccard": float(
            np.mean([item["atlas"]["fresh_core_jaccard"] for item in heldout])
        ),
        "mean_risk": {
            mode: float(np.mean([item["risk"][mode]["risk"] for item in heldout]))
            for mode in ("fresh", "blind", "atlas", "random", "oracle")
        },
    }
    all_metrics = [item["metrics"] for item in scored]
    aware = by_arm["residual_aware"]
    blind = by_arm["history_blind"]
    risk = heldout_summary["mean_risk"]
    conditions = {
        "clean_transport_at_least_90pct": _mean(scored, "clean") >= 0.90,
        "strict_schema_at_least_90pct": _mean(all_metrics, "schema_valid") >= 0.90,
        "aware_item_compile_at_least_80pct": aware["item_compile_rate"] >= 0.80,
        "aware_executable_responses_at_least_70pct": aware["response_executable_rate"] >= 0.70,
        "aware_truth_recall_at_least_25pct": aware["truth_recall_at_4"] >= 0.25,
        "aware_core_recall_at_least_45pct": aware["core_family_recall_at_4"] >= 0.45,
        "aware_modifier_f1_at_least_45pct": aware["modifier_f1"] >= 0.45,
        "aware_truth_advantage_at_least_10pp": aware["truth_recall_at_4"] - blind["truth_recall_at_4"] >= 0.10,
        "aware_core_advantage_at_least_10pp": aware["core_family_recall_at_4"] - blind["core_family_recall_at_4"] >= 0.10,
        "paired_semantic_wins_exceed_losses": sum(value > 1e-12 for value in paired)
        > sum(value < -1e-12 for value in paired),
        "heldout_fresh_truth_at_least_2_of_9": heldout_summary["fresh_truth_recall"] >= 2,
        "heldout_fresh_core_at_least_4_of_9": heldout_summary["fresh_core_recall"] >= 4,
        "heldout_atlas_truth_at_least_2_of_9": heldout_summary["atlas_truth_recall"] >= 2,
        "heldout_atlas_core_at_least_4_of_9": heldout_summary["atlas_core_recall"] >= 4,
        "atlas_fresh_jaccard_at_least_20pct": heldout_summary["atlas_fresh_core_jaccard"] >= 0.20,
        "fresh_risk_beats_blind_and_random": risk["fresh"] < risk["blind"] and risk["fresh"] < risk["random"],
        "atlas_risk_beats_blind_and_random": risk["atlas"] < risk["blind"] and risk["atlas"] < risk["random"],
        "fresh_risk_within_125pct_oracle": risk["fresh"] <= 1.25 * risk["oracle"] + 1e-15,
        "atlas_risk_within_125pct_oracle": risk["atlas"] <= 1.25 * risk["oracle"] + 1e-15,
    }
    return {
        "schema_version": f"{SCHEMA_VERSION}-semantic-evaluation",
        "provisional_status": "passed_pending_independent_verification"
        if all(conditions.values())
        else "failed_closed",
        "manifest_sha256": payload_hash(manifest),
        "labels_sha256": payload_hash(labels),
        "requests": len(scored),
        "by_arm": by_arm,
        "paired_semantic": {
            "wins": sum(value > 1e-12 for value in paired),
            "ties": sum(abs(value) <= 1e-12 for value in paired),
            "losses": sum(value < -1e-12 for value in paired),
        },
        "feature_standardization": feature_metadata,
        "heldout": heldout,
        "heldout_summary": heldout_summary,
        "conditions": conditions,
        "scored_records": scored,
    }


def write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--required-commit", required=True)
    args = parser.parse_args()
    implementation_commit = require_pushed_commit(args.required_commit)
    package = build_source_package(args.source_root, implementation_commit)
    manifest = package["manifest"]
    labels = package["labels"]
    write_new(args.manifest, manifest)
    write_new(args.labels, labels)
    summary = {
        "schema_version": f"{SCHEMA_VERSION}-source-summary",
        "status": "passed",
        "model_calls": 0,
        "cost_usd": 0.0,
        "implementation_commit": implementation_commit,
        "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
        "labels": {"path": str(args.labels), "sha256": sha256(args.labels)},
        "source_gate": manifest["source_gate"],
        "source_counts": manifest["source_counts"],
    }
    write_new(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
