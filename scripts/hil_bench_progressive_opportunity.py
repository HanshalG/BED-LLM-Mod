#!/usr/bin/env python3
"""Audit HiL-Bench SQL for progressive, generated-support opportunity."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import subprocess
from typing import Any, Iterable


SCHEMA_VERSION = 1
HIL_COMMIT = "352d14c861f2531949dfa91848d4b2fe46b8a247"
SELECTION_SEED = 24392
EXCLUDED_IDS = tuple(f"sql_{index}" for index in range(11))
OPPORTUNITY_IDS = (
    "sql_81", "sql_34", "sql_58", "sql_71", "sql_95", "sql_67", "sql_13",
    "sql_11", "sql_43", "sql_41", "sql_14", "sql_27", "sql_46", "sql_83",
    "sql_80", "sql_91", "sql_22", "sql_36", "sql_19", "sql_42", "sql_35",
    "sql_79", "sql_97", "sql_39", "sql_44", "sql_88", "sql_98", "sql_38",
    "sql_84", "sql_32", "sql_49", "sql_16", "sql_54", "sql_56", "sql_65",
    "sql_90", "sql_64", "sql_18", "sql_30", "sql_59",
)
DEVELOPMENT_IDS = (
    "sql_48", "sql_89", "sql_72", "sql_45", "sql_53", "sql_17", "sql_93",
    "sql_21", "sql_23", "sql_76", "sql_61", "sql_20", "sql_78", "sql_47",
    "sql_26", "sql_52", "sql_66", "sql_12", "sql_28", "sql_60",
)
HOLDOUT_IDS = (
    "sql_85", "sql_73", "sql_50", "sql_55", "sql_70", "sql_77", "sql_40",
    "sql_86", "sql_96", "sql_25", "sql_69", "sql_62", "sql_33", "sql_29",
    "sql_24", "sql_68", "sql_51", "sql_15", "sql_74", "sql_31", "sql_87",
    "sql_94", "sql_99", "sql_82", "sql_37", "sql_92", "sql_75", "sql_57",
    "sql_63",
)
EXPECTED_TYPES = {"question", "business info", "schema"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "does", "for", "from", "has", "have", "how", "in", "info",
    "information", "is", "it", "may", "not", "of", "on", "or", "that", "the",
    "their", "there", "these", "this", "to", "used", "using", "value", "what",
    "when", "where", "which", "with", "would",
}
GATES = {
    "task_count": 40,
    "all_blocker_counts_between_3_and_5": True,
    "tasks_with_question_and_business": 34,
    "tasks_with_all_three_sources": 30,
    "business_blocker_count": 40,
    "business_blockers_with_novel_evidence_fraction": 0.60,
    "mean_business_novel_evidence_recall": 0.03,
}


def frozen_split() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    ids = [f"sql_{index}" for index in range(11, 100)]
    random.Random(SELECTION_SEED).shuffle(ids)
    return tuple(ids[:40]), tuple(ids[40:60]), tuple(ids[60:])


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 2 and token not in STOPWORDS
    }


def extract_problem(instruction: str) -> str:
    match = re.search(
        r"Answer the following question:\s*(.*?)\s*# Available Commands",
        instruction,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("instruction does not contain the SQL question")
    return " ".join(match.group(1).split())


def _novel_evidence(
    problem: str,
    blocker_description: str,
    business_info: Iterable[str],
) -> dict[str, float | int]:
    blocker_tokens = _tokens(blocker_description)
    if not blocker_tokens:
        raise ValueError("blocker description has no content tokens")
    problem_tokens = _tokens(problem)
    novel_target = blocker_tokens - problem_tokens
    best_novel = 0
    best_total = 0
    for document in business_info:
        document_tokens = _tokens(document)
        best_novel = max(best_novel, len(novel_target & document_tokens))
        best_total = max(best_total, len(blocker_tokens & document_tokens))
    return {
        "novel_token_count": best_novel,
        "novel_recall": best_novel / len(blocker_tokens),
        "best_total_recall": best_total / len(blocker_tokens),
    }


def analyze_task(
    task_id: str,
    problem: str,
    blockers: list[dict[str, Any]],
    business_info: list[str],
) -> dict[str, Any]:
    if not 3 <= len(blockers) <= 5:
        raise ValueError(f"{task_id} has an unexpected blocker count")
    types = [str(blocker.get("type", "")).strip().casefold() for blocker in blockers]
    if any(blocker_type not in EXPECTED_TYPES for blocker_type in types):
        raise ValueError(f"{task_id} has an unexpected blocker type")
    if not business_info or not all(isinstance(item, str) and item.strip() for item in business_info):
        raise ValueError(f"{task_id} has invalid business information")
    evidence = [
        _novel_evidence(problem, str(blocker["description"]), business_info)
        for blocker in blockers
        if str(blocker["type"]).strip().casefold() == "business info"
    ]
    return {
        "task_id": task_id,
        "blocker_count": len(blockers),
        "source_counts": dict(sorted(Counter(types).items())),
        "business_info_document_count": len(business_info),
        "business_evidence": evidence,
    }


def load_opportunity_tasks(hil_root: str | Path) -> list[dict[str, Any]]:
    root = Path(hil_root)
    if _git_head(root) != HIL_COMMIT:
        raise ValueError("HiL-Bench checkout commit does not match")
    if frozen_split() != (OPPORTUNITY_IDS, DEVELOPMENT_IDS, HOLDOUT_IDS):
        raise ValueError("frozen HiL-Bench split does not reproduce")
    sql_root = root / "harbor_sql"
    task_dirs = sorted(
        path for path in sql_root.glob("sql_*")
        if path.is_dir() and path.name[4:].isdigit()
    )
    if len(task_dirs) != 100:
        raise ValueError("HiL-Bench SQL release must contain exactly 100 tasks")
    records = []
    for task_id in OPPORTUNITY_IDS:
        shared = sql_root / task_id / "shared"
        instruction = (
            sql_root / task_id / "ask_human" / "instruction.md"
        ).read_text(encoding="utf-8")
        registry = json.loads(
            (shared / "ask-human-data" / "blocker_registry.json").read_text(
                encoding="utf-8"
            )
        )
        business = json.loads(
            (shared / "data" / "business_info.json").read_text(encoding="utf-8")
        )
        records.append(
            analyze_task(
                task_id,
                extract_problem(instruction),
                registry["blockers"],
                business["business_info"],
            )
        )
    return records


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    business_rows: list[dict[str, float | int]] = []
    blocker_counts: Counter[int] = Counter()
    tasks_question_business = 0
    tasks_all_three = 0
    for record in records:
        counts = record["source_counts"]
        source_counts.update(counts)
        blocker_counts[record["blocker_count"]] += 1
        business_rows.extend(record["business_evidence"])
        present = set(counts)
        tasks_question_business += int({"question", "business info"} <= present)
        tasks_all_three += int(EXPECTED_TYPES <= present)
    positive = sum(row["novel_token_count"] > 0 for row in business_rows)
    positive_fraction = positive / len(business_rows) if business_rows else 0.0
    mean_novel_recall = (
        sum(float(row["novel_recall"]) for row in business_rows)
        / len(business_rows)
        if business_rows else 0.0
    )
    metrics = {
        "task_count": len(records),
        "blocker_count_distribution": {
            str(key): value for key, value in sorted(blocker_counts.items())
        },
        "source_counts": dict(sorted(source_counts.items())),
        "tasks_with_question_and_business": tasks_question_business,
        "tasks_with_all_three_sources": tasks_all_three,
        "business_blocker_count": len(business_rows),
        "business_blockers_with_novel_evidence": positive,
        "business_blockers_with_novel_evidence_fraction": positive_fraction,
        "mean_business_novel_evidence_recall": mean_novel_recall,
        "mean_business_documents_per_task": (
            sum(record["business_info_document_count"] for record in records)
            / len(records)
            if records else 0.0
        ),
    }
    checks = {
        "task_count": metrics["task_count"] == GATES["task_count"],
        "all_blocker_counts_between_3_and_5": all(
            3 <= record["blocker_count"] <= 5 for record in records
        ),
        "tasks_with_question_and_business": (
            metrics["tasks_with_question_and_business"]
            >= GATES["tasks_with_question_and_business"]
        ),
        "tasks_with_all_three_sources": (
            metrics["tasks_with_all_three_sources"]
            >= GATES["tasks_with_all_three_sources"]
        ),
        "business_blocker_count": (
            metrics["business_blocker_count"] >= GATES["business_blocker_count"]
        ),
        "business_blockers_with_novel_evidence_fraction": (
            metrics["business_blockers_with_novel_evidence_fraction"]
            >= GATES["business_blockers_with_novel_evidence_fraction"]
        ),
        "mean_business_novel_evidence_recall": (
            metrics["mean_business_novel_evidence_recall"]
            >= GATES["mean_business_novel_evidence_recall"]
        ),
    }
    return {
        "metrics": metrics,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def _safe_task_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": record["task_id"],
            "blocker_count": record["blocker_count"],
            "source_counts": record["source_counts"],
            "business_info_document_count": record["business_info_document_count"],
            "business_blockers_with_novel_evidence": sum(
                row["novel_token_count"] > 0 for row in record["business_evidence"]
            ),
        }
        for record in records
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hil-root", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/nonmyopic/hil_bench_progressive_opportunity",
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    records = load_opportunity_tasks(args.hil_root)
    result = aggregate(records)
    run_id = args.run_id or (
        "hil-progressive-opportunity-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_root = Path(args.output_dir)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source": {
            "repository": "https://github.com/hilbenchauthors/hil-bench",
            "commit": HIL_COMMIT,
            "selection_seed": SELECTION_SEED,
            "excluded_ids": list(EXCLUDED_IDS),
            "opportunity_ids": list(OPPORTUNITY_IDS),
            "development_count": len(DEVELOPMENT_IDS),
            "holdout_count": len(HOLDOUT_IDS),
        },
        "gates": GATES,
        **result,
        "task_summaries": _safe_task_summary(records),
        "interpretation": (
            "A pass establishes a released progressive-evidence mechanism and "
            "authorizes only a separately preregistered serving/mechanism smoke. "
            "It is not evidence that non-myopic policy selection beats myopic."
        ),
        "api_requests": 0,
        "cost_usd": 0.0,
        "cluster_used": False,
    }
    (run_dir / "audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "LATEST_RUN_ID").write_text(run_id + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
