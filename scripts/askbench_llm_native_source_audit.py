#!/usr/bin/env python3
"""Audit and split AskBench AskMind for LLM-native non-myopic BED."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "askbench-llm-native-source-audit-1"
SOURCE_ROOT = REPO_ROOT / "external/askbench"
SOURCE_COMMIT = "f35da92feda34504f10413313554438e7abaeb08"
SOURCE_TREE = "bfa4bf929f5ebd61a465be4b12f42a212d38685a"
DATA_PATH = (
    SOURCE_ROOT
    / "ask_eval/data/ask_bench/ask_mind/test.jsonl"
)
DATA_SHA256 = (
    "406b9a48036374552d9e819c63d5c3ba25feea764c68f391e21c552f6221af82"
)
EVALUATOR_PATH = (
    SOURCE_ROOT / "ask_eval/ask_eval/evaluators/ask.py"
)
EVALUATOR_SHA256 = (
    "14d78c936b60bfe365db9d03162e35627e9b00e2bdc17f5946df812d29dff392"
)
PAPER_PATH = SOURCE_ROOT / "paper.pdf"
PAPER_SHA256 = (
    "835509e5d90fc550756c81b0bdbfbaf39ffe44c9a3380c56e690d7da7ffe5533"
)
SOURCE_TASK = "ask_mind_medqade"
SPLIT_SEED = 37300
DEVELOPMENT_SIZE = 10
HOLDOUT_SIZE = 40
MIN_ELIGIBLE = 60
MIN_REQUIRED_POINTS = 3
MAX_REQUIRED_POINTS = 8
EXPECTED_ANSWER_RE = re.compile(r"The answer is ([ABCD])\.")
OPTION_RE = re.compile(r"(?m)^([ABCD])\.\s+\S")
NO_MODIFICATION_MARKER = "no modifications were made"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_source() -> dict[str, Any]:
    observed = {
        "commit": _git_value("rev-parse", "HEAD"),
        "tree": _git_value("rev-parse", "HEAD^{tree}"),
        "data_sha256": sha256_file(DATA_PATH),
        "evaluator_sha256": sha256_file(EVALUATOR_PATH),
        "paper_sha256": sha256_file(PAPER_PATH),
    }
    expected = {
        "commit": SOURCE_COMMIT,
        "tree": SOURCE_TREE,
        "data_sha256": DATA_SHA256,
        "evaluator_sha256": EVALUATOR_SHA256,
        "paper_sha256": PAPER_SHA256,
    }
    if observed != expected:
        raise ValueError(
            f"AskBench source changed: expected {expected}, observed {observed}"
        )
    return observed


def load_rows(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        value["_public_row_sha256"] = hashlib.sha256(
            line.encode("utf-8")
        ).hexdigest()
        rows.append(value)
    return rows


def option_labels(question: str) -> tuple[str, ...]:
    return tuple(OPTION_RE.findall(question))


def eligibility_errors(row: dict[str, Any]) -> list[str]:
    errors = []
    required_types = {
        "id": str,
        "ori_question": str,
        "degraded_question": str,
        "degraded_info": str,
        "expected_answer": str,
        "required_points": list,
    }
    for key, expected_type in required_types.items():
        if not isinstance(row.get(key), expected_type):
            errors.append(f"invalid_{key}_type")
    if errors:
        return errors
    if row.get("source_task") != SOURCE_TASK:
        errors.append("wrong_source_task")
    if row["ori_question"] == row["degraded_question"]:
        errors.append("unchanged_question")
    if NO_MODIFICATION_MARKER in row["degraded_info"].lower():
        errors.append("explicit_no_modification")
    points = row["required_points"]
    if not all(isinstance(point, str) and point.strip() for point in points):
        errors.append("invalid_required_point")
    if not MIN_REQUIRED_POINTS <= len(points) <= MAX_REQUIRED_POINTS:
        errors.append("required_point_count")
    if option_labels(row["degraded_question"]) != ("A", "B", "C", "D"):
        errors.append("answer_options")
    if EXPECTED_ANSWER_RE.fullmatch(row["expected_answer"]) is None:
        errors.append("expected_answer_format")
    return errors


def selected_public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "row_sha256": row["_public_row_sha256"],
        "source_task": row["source_task"],
        "required_point_count": len(row["required_points"]),
        "answer_option_labels": list(option_labels(row["degraded_question"])),
    }


def split_rows(
    eligible: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(eligible, key=lambda row: row["id"])
    random.Random(SPLIT_SEED).shuffle(ordered)
    development = ordered[:DEVELOPMENT_SIZE]
    holdout = ordered[
        DEVELOPMENT_SIZE : DEVELOPMENT_SIZE + HOLDOUT_SIZE
    ]
    unused = ordered[DEVELOPMENT_SIZE + HOLDOUT_SIZE :]
    return development, holdout, unused


def control_flow_audit(evaluator_text: str) -> dict[str, bool]:
    return {
        "default_allows_three_assistant_turns": (
            'max_turns = config.getint("evaluatorconfig", "max_turns", fallback=3)'
            in evaluator_text
        ),
        "final_answer_forced_only_on_last_turn": (
            "if turn == max_turns and messages_for_llm:" in evaluator_text
            and 'last_message["content"] += "\\n" + FORCE_FINAL_ANSWER_PROMPT'
            in evaluator_text
        ),
        "simulator_receives_hidden_original_question": (
            '"my_real_question": sample_state["data"].get("ori_question", "")'
            in evaluator_text
        ),
        "simulator_receives_checklist": (
            '"checklist_points": sample_state["required_points"]'
            in evaluator_text
        ),
        "simulator_reveals_only_immediate_answer": (
            "ONLY answers the assistant's immediate question"
            in evaluator_text
        ),
        "judge_receives_released_expected_answer": (
            '.replace("<ground_truth_answer>", sample_state["data"]["expected_answer"])'
            in evaluator_text
        ),
    }


def candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "degraded_question": row["degraded_question"],
        "answer_option_labels": ["A", "B", "C", "D"],
    }


def hidden_state_separated(row: dict[str, Any]) -> bool:
    payload = json.dumps(candidate_payload(row), sort_keys=True)
    hidden_values = (
        row["ori_question"],
        row["expected_answer"],
        row["degraded_info"],
        json.dumps(row["required_points"], sort_keys=True),
    )
    return all(value not in payload for value in hidden_values)


def run_audit(*, output_path: Path) -> dict[str, Any]:
    source = verify_source()
    rows = load_rows()
    ids = [row.get("id") for row in rows]
    all_ids_unique = (
        all(isinstance(row_id, str) for row_id in ids)
        and len(ids) == len(set(ids))
    )
    error_counts = Counter()
    eligible = []
    for row in rows:
        errors = eligibility_errors(row)
        error_counts.update(errors)
        if not errors:
            eligible.append(row)
    development, holdout, unused = split_rows(eligible)
    selected = development + holdout
    selected_ids = [row["id"] for row in selected]
    control_flow = control_flow_audit(
        EVALUATOR_PATH.read_text(encoding="utf-8")
    )
    gates = {
        "source_hashes_match": bool(source),
        "at_least_60_eligible_rows": len(eligible) >= MIN_ELIGIBLE,
        "all_source_ids_unique": all_ids_unique,
        "selected_split_has_exact_sizes_and_unique_ids": (
            len(development) == DEVELOPMENT_SIZE
            and len(holdout) == HOLDOUT_SIZE
            and len(selected_ids) == len(set(selected_ids))
        ),
        "selected_rows_preserve_frozen_structure": all(
            not eligibility_errors(row) for row in selected
        ),
        "hidden_state_excluded_from_candidate_payload": all(
            hidden_state_separated(row) for row in selected
        ),
        "official_control_flow_matches_required_semantics": all(
            control_flow.values()
        ),
    }
    status = "pass" if all(gates.values()) else "fail"
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration": (
                "results/nonmyopic/"
                "ASKBENCH_LLM_NATIVE_SOURCE_AUDIT_PREREGISTRATION.md"
            ),
            "model_calls": 0,
            "cost_usd": 0.0,
            "split_seed": SPLIT_SEED,
        },
        "status": status,
        "source": source,
        "data": {
            "total_rows": len(rows),
            "source_task_counts": dict(
                sorted(Counter(row.get("source_task") for row in rows).items())
            ),
            "eligible_rows": len(eligible),
            "eligibility_rejection_counts": dict(sorted(error_counts.items())),
            "eligible_required_point_count_distribution": dict(
                sorted(
                    Counter(
                        len(row["required_points"]) for row in eligible
                    ).items()
                )
            ),
        },
        "splits": {
            "development": [
                selected_public_row(row) for row in development
            ],
            "holdout": [selected_public_row(row) for row in holdout],
            "unused_count": len(unused),
        },
        "control_flow": control_flow,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
    checkpoint(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=(
            REPO_ROOT
            / "results/nonmyopic/askbench_llm_native_source_audit/"
            "askbench-llm-native-source-audit-20260729/MANIFEST.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(output_path=args.output_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "data": result["data"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
