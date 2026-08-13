#!/usr/bin/env python3
"""Fail-closed value-blind source audit for MedChain dynamic support."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "medchain-dynamic-support-source-v1"
SOURCE_COMMIT = "35ab53e3c601831a7bca181c72ecad6b12b55ff3"
SOURCE_TREE = "ba425765442c69b51142ddd8d2731aa894eaf9b8"
EXPECTED_COUNT = 12163
MIN_ELIGIBLE_COUNT = 1000
MIN_ELIGIBLE_FRACTION = 0.90
MIN_DIAGNOSIS_LABELS = 100
SPLIT_SALT = "medchain-dynamic-support-v1|"
FIXED_SPLITS = (("mechanics", 6), ("opportunity", 30), ("development", 64), ("confirmation", 96))
BOUND_FILES = {
    "data": (
        "datasets/filtered_data_test_set.json",
        "cb947c9b7ff8979ebbaa816b3b0977ed04bc84b6257147166a7fe725b7dbdb87",
    ),
    "patient_interface": (
        "doctor_patient_interaction/wenzhen_main.py",
        "781aa0d9014c671f5fada3603d6292975e06b786b12d12d0f6df55a971677b08",
    ),
    "workflow": (
        "main.py",
        "ac85f53576e1c871b72291e55941e07c2df275eabbe115433fc152cd6b776954",
    ),
    "extraction_helper": (
        "utils/funtion_api.py",
        "ffa40c1899960c8b1cdd6ac022446280f84791badf0d0774e43593683bfeb3e0",
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def git_value(root: Path, expression: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", expression],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def normalized_scalars(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [item for child in value.values() for item in normalized_scalars(child)]
    if isinstance(value, list):
        return [item for child in value for item in normalized_scalars(child)]
    text = " ".join(str(value).split()).casefold()
    return [text] if text else []


def path_value(row: dict[str, Any], *path: str) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def field_nonempty(value: Any) -> bool:
    return bool(normalized_scalars(value))


def eligible_shape(row: dict[str, Any]) -> bool:
    intro = path_value(row, "\u3010\u75c5\u6848\u4ecb\u7ecd\u3011")
    if not isinstance(intro, dict):
        return False
    exam = intro.get("\u67e5\u4f53")
    return (
        field_nonempty(intro.get("\u4e3b\u8bc9"))
        and (field_nonempty(intro.get("\u73b0\u75c5\u53f2")) or field_nonempty(intro.get("\u65e2\u5f80\u53f2")))
        and isinstance(exam, dict)
        and field_nonempty(exam.get("\u4f53\u683c\u68c0\u67e5"))
        and field_nonempty(exam.get("\u8f85\u52a9\u68c0\u67e5"))
    )


def diagnosis_values(row: dict[str, Any]) -> list[str]:
    return normalized_scalars(path_value(row, "tags", "\u75c5\u79cd"))


def private_values(row: dict[str, Any]) -> list[Any]:
    intro = path_value(row, "\u3010\u75c5\u6848\u4ecb\u7ecd\u3011")
    exam = intro.get("\u67e5\u4f53") if isinstance(intro, dict) else None
    return [
        intro.get("\u73b0\u75c5\u53f2") if isinstance(intro, dict) else None,
        intro.get("\u65e2\u5f80\u53f2") if isinstance(intro, dict) else None,
        exam.get("\u4f53\u683c\u68c0\u67e5") if isinstance(exam, dict) else None,
        exam.get("\u8f85\u52a9\u68c0\u67e5") if isinstance(exam, dict) else None,
        path_value(row, "tags", "\u75c5\u79cd"),
    ]


def visible_private_separated(row: dict[str, Any]) -> bool:
    complaint = normalized_scalars(path_value(row, "\u3010\u75c5\u6848\u4ecb\u7ecd\u3011", "\u4e3b\u8bc9"))
    if not complaint:
        return False
    visible = tuple(complaint)
    return all(tuple(normalized_scalars(value)) != visible for value in private_values(row))


def split_ids(case_ids: list[str]) -> dict[str, list[str]]:
    ordered = sorted(case_ids, key=lambda value: sha256_bytes((SPLIT_SALT + value).encode()))
    if len(ordered) < sum(count for _, count in FIXED_SPLITS):
        raise ValueError("eligible population cannot cover fixed splits")
    result: dict[str, list[str]] = {}
    offset = 0
    for name, count in FIXED_SPLITS:
        result[name] = ordered[offset : offset + count]
        offset += count
    result["reserve"] = ordered[offset:]
    return result


def audit(source_root: Path, protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    hashes = {name: sha256_file(source_root / path) for name, (path, _) in BOUND_FILES.items()}
    expected_hashes = {name: expected for name, (_, expected) in BOUND_FILES.items()}
    raw = json.loads((source_root / BOUND_FILES["data"][0]).read_text(encoding="utf-8"))
    population_is_object = isinstance(raw, dict)
    rows = list(raw.items()) if population_is_object else []
    keys = [key for key, _ in rows]
    row_objects = all(isinstance(row, dict) for _, row in rows)
    eligible = [(key, row) for key, row in rows if isinstance(row, dict) and eligible_shape(row)]
    eligible_ids = [sha256_bytes(canonical_bytes([key, row])) for key, row in eligible]
    splits = split_ids(eligible_ids) if len(eligible_ids) >= sum(count for _, count in FIXED_SPLITS) else {}
    diagnosis_labels = {label for _, row in eligible for label in diagnosis_values(row)}
    patient_text = (source_root / BOUND_FILES["patient_interface"][0]).read_text(encoding="utf-8")
    workflow_text = (source_root / BOUND_FILES["workflow"][0]).read_text(encoding="utf-8")
    helper_text = (source_root / BOUND_FILES["extraction_helper"][0]).read_text(encoding="utf-8")
    patient_contract = {
        "doctor_receives_chief_complaint_only": 'system_message=f"\u4f60\u662f\u4e00\u4e2a\u533b\u751f\u3002\u5df2\u77e5\u75c5\u4eba\u7684\u4e3b\u8bc9\u5982\u4e0b' in patient_text and "chief_complaints" in patient_text,
        "patient_grounded_in_private_history_and_exams": all(value in patient_text for value in ("physical_exams", "inspections", "past_history", "current_history")),
        "patient_withholds_unasked_examinations": "\u9664\u975e\u533b\u751f\u660e\u786e\u63d0\u95ee\u4f53\u683c\u68c0\u67e5\u548c\u8f85\u52a9\u68c0\u67e5" in patient_text,
        "patient_forbidden_to_invent_missing_facts": "\u5207\u5fcc\u865a\u6784\u5185\u5bb9" in patient_text,
    }
    workflow_contract = {
        "extracts_separate_clinical_fields": all(value in helper_text for value in ("zhusu", "jiwangshi", "xianbingshi", "chati", "keshi", "jieguo")),
        "workflow_calls_separate_extractor": "extract_json_data(sample)" in workflow_text,
        "doctor_patient_intake_excludes_diagnosis": "run_conversation(case_name, intro)" in patient_text and "Correct_Diagnosis" not in patient_text,
        "diagnosis_used_downstream": "task4(case_message=case_message" in workflow_text,
    }
    gates = {
        "immutable_source_binding": git_value(source_root, "HEAD") == SOURCE_COMMIT and git_value(source_root, "HEAD^{tree}") == SOURCE_TREE and hashes == expected_hashes,
        "exact_unique_object_population": population_is_object and row_objects and len(rows) == EXPECTED_COUNT and len(keys) == len(set(keys)) and all(str(key).strip() for key in keys),
        "eligible_population_large_enough": len(eligible) >= MIN_ELIGIBLE_COUNT and len(eligible) / max(len(rows), 1) >= MIN_ELIGIBLE_FRACTION,
        "eligible_diagnoses_nonempty_and_diverse": len(eligible) > 0 and all(diagnosis_values(row) for _, row in eligible) and len(diagnosis_labels) >= MIN_DIAGNOSIS_LABELS,
        "visible_private_fields_separated": len(eligible) > 0 and all(visible_private_separated(row) for _, row in eligible),
        "released_patient_contract": all(patient_contract.values()),
        "released_workflow_contract": all(workflow_contract.values()),
        "eligible_partition_complete_disjoint": bool(splits) and sum(map(len, splits.values())) == len(eligible_ids) and len({item for values in splits.values() for item in values}) == len(eligible_ids),
    }
    passed = all(gates.values())
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source": {"repository": "https://github.com/ljwztc/MedChain", "commit": SOURCE_COMMIT, "tree": SOURCE_TREE, "bound_file_sha256": hashes},
        "population": {"total_count": len(rows), "eligible_count": len(eligible), "eligible_fraction": len(eligible) / max(len(rows), 1), "distinct_normalized_diagnosis_labels": len(diagnosis_labels)},
        "split_counts": {name: len(values) for name, values in splits.items()},
        "ordered_split_case_id_sha256": {name: sha256_bytes(canonical_bytes(values)) for name, values in splits.items()},
        "privacy": {"case_keys_serialized": False, "individual_case_ids_serialized": False, "source_values_serialized": False, "diagnoses_serialized": False, "endpoints_opened": False},
    }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "source_pass" if passed else "source_failed_closed",
        "decision": "mechanics_authorized" if passed else "close_exact_medchain_construction",
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
        "patient_contract": patient_contract,
        "workflow_contract": workflow_contract,
        "gates": gates,
        "privacy": manifest["privacy"],
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
        "authorizes": "mechanics_only" if passed else "nothing",
    }
    return manifest, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, result = audit(args.source_root.resolve(), args.protocol.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "SOURCE_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
