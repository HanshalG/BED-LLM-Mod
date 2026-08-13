#!/usr/bin/env python3
"""Value-blind source admission for CAR-bench disambiguation BED."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "carbench-dynamic-belief-source-v1"
CODE_COMMIT = "9ed387d8de2dac20e5227d8e949bd33d20041fc5"
CODE_TREE = "9f0d800b901f94277b2c9e80333a47bc5ac345eb"
DATA_COMMIT = "1fcf24ad802c42e04a0d8fe05b5ca0d481a4e7af"
DATA_TREE = "fa920ff4d6b0b9079d6b78e011e87c02aac5b28f"
TRAIN_SHA256 = "9dace1484321e1b7b40967713e9715efae72ec83bd84848d24a4a0a6ed5c452f"
TEST_SHA256 = "0cb012e7a066f9d4f226dc42643b0a8ffa42e23ecf09a5e1564fc8c9233903d5"
EXPECTED_SCHEMA = {
    "actions",
    "calendar_id",
    "context_init_config",
    "disambiguation_element_internal",
    "disambiguation_element_note",
    "disambiguation_element_user",
    "instruction",
    "persona",
    "removed_part",
    "task_id",
    "task_type",
}
CODE_FILES = {
    "types": ("car_bench/types.py", "0789ba719a4089be62eebff748efa3b5f93efdde76d0c3ff5a5d66a292d27362"),
    "environment": ("car_bench/envs/car_voice_assistant/env.py", "c2ba3dac37b681ebb54e9380f7d90b0bd649026ac889a18911c9957413e8a819"),
    "user_simulator": ("car_bench/envs/user/user.py", "ffa1f77a6b54b2da87d4653bf7ab146c31b6dfd2d7aba06cad60dff6aa4c7583"),
    "reward": ("car_bench/envs/reward_calculators.py", "3dba6c02bc52a5584cbe7f1e11a270aa228264c2b3ee45b72188544e1f53ce64"),
    "tools": ("car_bench/envs/car_voice_assistant/tools/__init__.py", "0fbb9660d802aae62785674d194ec801a9b3875070758764ce3d1ca5232f74b5"),
    "wiki": ("car_bench/envs/car_voice_assistant/wiki.py", "471dd42ddcede9be8251291ef87e1e18db3d24c326fc892e07db2bcb41f4f93f"),
}
SPLIT_SALT = "carbench-dynamic-belief-v1|"
TRAIN_SPLITS = (("mechanics", 6), ("opportunity", 10), ("development", 15))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def git_value(root: Path, expression: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", expression], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: task must be an object")
        rows.append(value)
    return rows


def structured_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        if not value.strip():
            return False
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return True
        return structured_nonempty(parsed)
    if isinstance(value, dict):
        return bool(value) and all(str(key).strip() and structured_nonempty(item) for key, item in value.items())
    if isinstance(value, list):
        return bool(value) and all(structured_nonempty(item) for item in value)
    return True


def split_train(case_ids: list[str]) -> dict[str, list[str]]:
    ordered = sorted(case_ids, key=lambda value: sha256_bytes((SPLIT_SALT + value).encode()))
    if len(ordered) != sum(count for _, count in TRAIN_SPLITS):
        raise ValueError("train split counts do not cover population")
    result: dict[str, list[str]] = {}
    offset = 0
    for name, count in TRAIN_SPLITS:
        result[name] = ordered[offset : offset + count]
        offset += count
    return result


def task_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    allowed = ("disambiguation_internal", "disambiguation_user")
    return {value: sum(row.get("task_type") == value for row in rows) for value in allowed}


def appropriate_ambiguity(row: dict[str, Any]) -> bool:
    internal = row.get("disambiguation_element_internal")
    user = row.get("disambiguation_element_user")
    if row.get("task_type") == "disambiguation_internal":
        return structured_nonempty(internal) and user is None
    if row.get("task_type") == "disambiguation_user":
        return structured_nonempty(user) and internal is None
    return False


def audit(code_root: Path, data_root: Path, protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    train_path = data_root / "disambiguation_train.jsonl"
    test_path = data_root / "disambiguation_test.jsonl"
    train = load_rows(train_path)
    test = load_rows(test_path)
    all_rows = train + test
    train_ids = [sha256_bytes(canonical_bytes(row)) for row in train]
    test_ids = [sha256_bytes(canonical_bytes(row)) for row in test]
    splits = split_train(train_ids)
    code_hashes = {name: sha256_file(code_root / path) for name, (path, _) in CODE_FILES.items()}
    expected_code_hashes = {name: expected for name, (_, expected) in CODE_FILES.items()}
    code_text = {name: (code_root / path).read_text(encoding="utf-8") for name, (path, _) in CODE_FILES.items()}
    code_contract = {
        "task_types_bind_internal_and_user": all(value in code_text["types"] for value in ("DISAMBIGUATION_INTERNAL", "DISAMBIGUATION_USER")),
        "environment_is_stateful_and_loads_actions": all(value in code_text["environment"] for value in ("context_init_config", "Action(**a)", "load_data")),
        "user_simulator_has_distinct_failure_contract": all(value in code_text["user_simulator"] for value in ("DISAMBIGUATION_INTERNAL", "DISAMBIGUATION_USER", "DISAMBIGUATION_ERROR")),
        "reward_binds_state_tools_errors_policy_and_user": all(value in code_text["reward"] for value in ("calculate_state_based_reward", "calculate_tool_subset_reward", "calculate_tool_execution_reward", "calculate_end_conversation_reward", "policy_errors_during_runtime")),
        "native_tools_and_wiki_nonempty": "ALL_TOOLS" in code_text["tools"] and bool(code_text["wiki"].strip()),
    }
    counts = {"train": task_type_counts(train), "test": task_type_counts(test)}
    schemas_exact = all(set(row) == EXPECTED_SCHEMA for row in all_rows)
    required_nonempty = all(
        all(structured_nonempty(row.get(field)) for field in ("persona", "instruction", "context_init_config", "actions", "disambiguation_element_note"))
        and appropriate_ambiguity(row)
        for row in all_rows
    )
    ids = [str(row.get("task_id", "")).strip() for row in all_rows]
    gates = {
        "immutable_code_binding": git_value(code_root, "HEAD") == CODE_COMMIT and git_value(code_root, "HEAD^{tree}") == CODE_TREE and code_hashes == expected_code_hashes,
        "immutable_data_binding": git_value(data_root, "HEAD") == DATA_COMMIT and git_value(data_root, "HEAD^{tree}") == DATA_TREE and sha256_file(train_path) == TRAIN_SHA256 and sha256_file(test_path) == TEST_SHA256,
        "exact_official_population_counts": len(train) == 31 and len(test) == 25,
        "single_exact_task_schema": schemas_exact,
        "task_ids_unique_nonempty_disjoint": all(ids) and len(set(ids)) == len(ids),
        "both_disambiguation_types_in_each_split": all(counts[split][kind] > 0 for split in counts for kind in counts[split]) and sum(sum(value.values()) for value in counts.values()) == len(all_rows),
        "required_structural_fields_nonempty": required_nonempty,
        "released_code_contract": all(code_contract.values()),
        "train_partition_complete_disjoint": sum(len(value) for value in splits.values()) == len(train) and len({item for value in splits.values() for item in value}) == len(train),
    }
    passed = all(gates.values())
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "code": {"repository": "https://github.com/CAR-bench/car-bench", "commit": CODE_COMMIT, "tree": CODE_TREE, "bound_file_sha256": code_hashes},
        "data": {"repository": "https://huggingface.co/datasets/johanneskirmayr/car-bench-dataset", "commit": DATA_COMMIT, "tree": DATA_TREE, "train_sha256": TRAIN_SHA256, "test_sha256": TEST_SHA256},
        "population_counts": {"train": len(train), "test": len(test)},
        "task_type_counts": counts,
        "schema_fields": sorted(EXPECTED_SCHEMA),
        "train_split_counts": {name: len(value) for name, value in splits.items()},
        "ordered_split_case_id_sha256": {name: sha256_bytes(canonical_bytes(value)) for name, value in splits.items()},
        "confirmation_case_id_sha256": sha256_bytes(canonical_bytes(test_ids)),
        "privacy": {"individual_case_ids_serialized": False, "source_values_serialized": False, "actions_serialized": False, "endpoints_opened": False},
    }
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "source_pass" if passed else "source_failed_closed",
        "decision": "mechanics_authorized" if passed else "close_carbench_source_route",
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
        "code_contract": code_contract,
        "gates": gates,
        "privacy": manifest["privacy"],
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
        "authorizes": "mechanics_only" if passed else "nothing",
    }
    return manifest, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, result = audit(args.code_root.resolve(), args.data_root.resolve(), args.protocol.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "SOURCE_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
