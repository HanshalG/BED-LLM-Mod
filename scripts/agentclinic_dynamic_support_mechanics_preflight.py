#!/usr/bin/env python3
"""Zero-call native-interface preflight for frozen AgentClinic mechanics cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import agentclinic_dynamic_support_source_audit as source


PROTOCOL_VERSION = "agentclinic-dynamic-support-mechanics-preflight-v1"
EXPECTED_MECHANICS_COUNT = 6
EXPECTED_MECHANICS_LIST_SHA256 = (
    "3c60e70097899ade4d1a84a8055d42eea6d8c6240582fe36655660e755cb2f69"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def select_mechanics_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {sha256_bytes(source.canonical_bytes(row)): row for row in rows}
    splits = source.split_case_ids(list(by_id))
    return [by_id[case_id] for case_id in splits["mechanics"]]


def answer_text(answer: dict[str, Any]) -> str:
    for key in ("answer", "text", "value"):
        value = answer.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def audit_mechanics(
    *,
    source_root: Path,
    source_protocol: Path,
    source_manifest_path: Path,
    source_audit_path: Path,
) -> dict[str, Any]:
    source_manifest = load_object(source_manifest_path)
    source_result = load_object(source_audit_path)
    rows = source.load_rows(source_root / source.SOURCE_FILE)
    mechanics = select_mechanics_rows(rows)
    mechanics_ids = [sha256_bytes(source.canonical_bytes(row)) for row in mechanics]

    answer_cardinalities: list[int] = []
    unique_answers = True
    exact_one_correct = True
    exact_gold_leak_free = True
    fixed_intake_nonempty = True
    native_channels_present = True
    for row in mechanics:
        answers = row.get("answers")
        if not isinstance(answers, list):
            answers = []
        texts = [answer_text(answer) for answer in answers if isinstance(answer, dict)]
        answer_cardinalities.append(len(texts))
        normalized_answers = [normalize_text(text) for text in texts]
        unique_answers &= bool(texts) and all(normalized_answers) and len(set(normalized_answers)) == len(texts)
        correct = [
            answer_text(answer)
            for answer in answers
            if isinstance(answer, dict) and answer.get("correct") is True
        ]
        exact_one_correct &= len(correct) == 1 and bool(correct[0])
        policy_visible = normalize_text(
            " ".join(str(row.get(field, "")) for field in ("question", "patient_info", "physical_exams"))
        )
        if len(correct) == 1 and correct[0]:
            exact_gold_leak_free &= normalize_text(correct[0]) not in policy_visible
        fixed_intake_nonempty &= bool(normalize_text(str(row.get("patient_info", ""))))
        native_channels_present &= bool(normalize_text(str(row.get("patient_info", "")))) and bool(
            normalize_text(str(row.get("physical_exams", "")))
        )

    actual_list_hash = sha256_bytes(canonical_bytes(mechanics_ids))
    source_binding = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "tree": subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "file_sha256": sha256_file(source_root / source.SOURCE_FILE),
    }
    source_artifacts_pass = (
        source_result.get("status") == "source_pass"
        and source_result.get("decision") == "mechanics_authorized"
        and source_manifest.get("ordered_split_case_id_sha256", {}).get("mechanics")
        == EXPECTED_MECHANICS_LIST_SHA256
        and source_binding
        == {
            "commit": source.SOURCE_COMMIT,
            "tree": source.SOURCE_TREE,
            "file_sha256": source.SOURCE_SHA256,
        }
    )
    gates = {
        "source_and_artifacts_immutable": source_artifacts_pass,
        "exact_unique_mechanics_cohort": len(mechanics) == EXPECTED_MECHANICS_COUNT
        and len(set(mechanics_ids)) == EXPECTED_MECHANICS_COUNT,
        "ordered_mechanics_hash_matches": actual_list_hash == EXPECTED_MECHANICS_LIST_SHA256,
        "at_least_four_answers_each": bool(answer_cardinalities)
        and min(answer_cardinalities) >= 4,
        "answers_nonempty_and_unique": unique_answers,
        "exactly_one_correct_answer_each": exact_one_correct,
        "gold_answer_not_verbatim_in_policy_context": exact_gold_leak_free,
        "fixed_intake_nonempty": fixed_intake_nonempty,
        "patient_and_test_channels_present": native_channels_present,
        "public_artifact_contains_no_case_values": True,
    }
    passed = all(gates.values())
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "mechanics_preflight_pass" if passed else "mechanics_preflight_failed_closed",
        "decision": "freeze_semantic_serving_protocol" if passed else "close_agentclinic_route",
        "source_protocol_sha256": sha256_file(source_protocol),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_audit_sha256": sha256_file(source_audit_path),
        "mechanics_count": len(mechanics),
        "answer_cardinality_min": min(answer_cardinalities) if answer_cardinalities else 0,
        "answer_cardinality_max": max(answer_cardinalities) if answer_cardinalities else 0,
        "gates": gates,
        "privacy": {
            "individual_case_ids_serialized": False,
            "source_values_serialized": False,
            "image_urls_serialized": False,
            "answer_choices_serialized": False,
            "diagnoses_serialized": False,
            "endpoint_values_opened": False,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_mechanics(
        source_root=args.source_root.resolve(),
        source_protocol=args.source_protocol.resolve(),
        source_manifest_path=args.source_manifest.resolve(),
        source_audit_path=args.source_audit.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "mechanics_preflight_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
