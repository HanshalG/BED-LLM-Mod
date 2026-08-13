#!/usr/bin/env python3
"""Fail-closed value-blind source audit for SymptomCheck dynamic support."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "symptomcheck-dynamic-support-source-v1"
SOURCE_COMMIT = "36c80b44fee77da271d97d38c43e49926dc95b8e"
SOURCE_TREE = "b0649fb98bb150889974974dd8737b4ced30ee58"
EXPECTED_COUNT = 400
MIN_DIAGNOSES = 100
EXPECTED_FIELDS = {
    "correct_diagnosis",
    "demographics",
    "presentation",
    "chief_complaints",
    "absent_findings",
    "physical_history",
    "family_history",
    "social_history",
}
SPLIT_SALT = "symptomcheck-dynamic-support-v1|"
SPLITS = (("mechanics", 6), ("opportunity", 30), ("development", 64), ("confirmation", 96), ("reserve", 204))
BOUND_FILES = {
    "data": ("symptomcheck_bench/vignettes/avey_vignettes.jsonl", "f516ee0fb17bdaefca7f53483fb4e38a1eb3c72cf52cf45d34dcf0484bf66976"),
    "simulator": ("symptomcheck_bench/simulator.py", "5400d243eb749b49335cde4c971303f68c8479c94a04c057a9f08c37699a22a7"),
    "vignette": ("symptomcheck_bench/vignette.py", "0721796828a3efc8f198466a1d433e0643928173d366815e4527908e411ad7c5"),
    "agent": ("symptomcheck_bench/agent.py", "117da867d1056612cde29918e3033c95b82573fe4868a5b953cc6f3559a75867"),
    "entrypoint": ("symptomcheck_bench/main.py", "622461ee9bfc2f84ca559f34501b499a2c981b09d901961a7443be002dfa24e2"),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def git_value(root: Path, expression: str) -> str:
    return subprocess.run(["git", "rev-parse", expression], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: row is not an object")
        rows.append(value)
    return rows


def normalized(value: Any) -> str:
    if isinstance(value, dict):
        return "|".join(f"{normalized(key)}:{normalized(item)}" for key, item in sorted(value.items()))
    if isinstance(value, list):
        return "|".join(normalized(item) for item in value)
    return " ".join(str(value).split()).casefold() if value is not None else ""


def structurally_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value) and any(structurally_nonempty(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and any(structurally_nonempty(item) for item in value)
    return True


def split_ids(case_ids: list[str]) -> dict[str, list[str]]:
    ordered = sorted(case_ids, key=lambda value: sha256_bytes((SPLIT_SALT + value).encode()))
    if len(ordered) != sum(count for _, count in SPLITS):
        raise ValueError("split counts do not cover the population")
    result = {}
    offset = 0
    for name, count in SPLITS:
        result[name] = ordered[offset : offset + count]
        offset += count
    return result


def audit(source_root: Path, protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    hashes = {name: sha256_file(source_root / path) for name, (path, _) in BOUND_FILES.items()}
    expected_hashes = {name: expected for name, (_, expected) in BOUND_FILES.items()}
    rows = load_rows(source_root / BOUND_FILES["data"][0])
    case_ids = [sha256_bytes(canonical_bytes(row)) for row in rows]
    splits = split_ids(case_ids) if len(rows) == EXPECTED_COUNT else {}
    diagnoses = {normalized(row.get("correct_diagnosis")) for row in rows if normalized(row.get("correct_diagnosis"))}
    exact_schema = all(set(row) == EXPECTED_FIELDS for row in rows)
    all_nonempty = all(all(structurally_nonempty(row.get(field)) for field in EXPECTED_FIELDS) for row in rows)
    visible_private_separated = all(
        normalized(row.get("demographics"))
        and all(normalized(row.get("demographics")) != normalized(row.get(field)) for field in EXPECTED_FIELDS - {"demographics"})
        for row in rows
    )
    simulator = (source_root / BOUND_FILES["simulator"][0]).read_text(encoding="utf-8")
    vignette = (source_root / BOUND_FILES["vignette"][0]).read_text(encoding="utf-8")
    agent = (source_root / BOUND_FILES["agent"][0]).read_text(encoding="utf-8")
    patient_contract = {
        "doctor_receives_demographics_only": "DEMOGRAPHICS: {self._vignette.demographics}" in agent and "self._vignette.current_history" not in agent.split("class Patient:", 1)[1].split("class Doctor:", 1)[1],
        "patient_receives_all_nondiagnostic_fields": all(value in agent for value in ("demographics", "current_history", "primary_complaints", "additional_information")) and all(value in vignette for value in ("absent_findings", "physical_history", "family_history", "social_history")),
        "patient_relevance_and_missing_information_contract": "sharing only the relevant information" in agent and "I don't know" in agent,
        "diagnosis_not_in_patient_prompt": "correct_diagnosis" not in agent,
    }
    simulator_contract = {
        "separate_doctor_and_patient_histories": all(value in simulator for value in ("self.chat_doctor", "self.chat_patient")),
        "alternating_patient_and_doctor_turns": all(value in simulator for value in ("out_patient = self.infer_patient()", "out_doctor = self.infer_doctor()")),
        "finite_dialogue_horizon": "self.max_len = 24" in simulator,
        "diagnosis_endpoint_accessor": "return self.vignette.correct_diagnosis" in simulator,
    }
    gates = {
        "immutable_source_binding": git_value(source_root, "HEAD") == SOURCE_COMMIT and git_value(source_root, "HEAD^{tree}") == SOURCE_TREE and hashes == expected_hashes,
        "exact_population_count": len(rows) == EXPECTED_COUNT,
        "single_exact_schema": exact_schema,
        "canonical_rows_unique_and_required_fields_nonempty": len(set(case_ids)) == len(rows) and all_nonempty,
        "diagnosis_population_diverse": len(diagnoses) >= MIN_DIAGNOSES,
        "visible_private_fields_separated": visible_private_separated,
        "released_patient_contract": all(patient_contract.values()),
        "released_simulator_contract": all(simulator_contract.values()),
        "partition_complete_disjoint": bool(splits) and sum(map(len, splits.values())) == len(rows) and len({item for values in splits.values() for item in values}) == len(rows),
    }
    passed = all(gates.values())
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source": {"repository": "https://github.com/medaks/medask-benchmarks", "commit": SOURCE_COMMIT, "tree": SOURCE_TREE, "bound_file_sha256": hashes},
        "population": {"count": len(rows), "distinct_normalized_diagnoses": len(diagnoses), "schema_fields": sorted(EXPECTED_FIELDS)},
        "split_counts": {name: len(values) for name, values in splits.items()},
        "ordered_split_case_id_sha256": {name: sha256_bytes(canonical_bytes(values)) for name, values in splits.items()},
        "privacy": {"individual_case_ids_serialized": False, "source_values_serialized": False, "diagnoses_serialized": False, "dialogues_serialized": False, "endpoints_opened": False},
    }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "source_pass" if passed else "source_failed_closed",
        "decision": "mechanics_authorized" if passed else "close_exact_symptomcheck_construction",
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
        "patient_contract": patient_contract,
        "simulator_contract": simulator_contract,
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
