#!/usr/bin/env python3
"""Metadata-only source admission for R2E-Gym Debug-BED."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VERSION = "r2e-gym-debug-bed-source-v1"
SALT = "r2e-gym-debug-bed-v1|"
R2E_COMMIT = "0d94c4eb9431cd195c55a7ea3abd54006c9a1735"
R2E_TREE = "fcf3cab14b8fe62b0cddcc52f3a3a5d4fb2855cf"
DATA_COMMIT = "8d3163011f01f9393bb3dc7700497a79a8686ae5"
DATA_TREE = "fa991fad5d79264d94047a50233b4240281e27a4"
DEBUGGYM_COMMIT = "cc3fe3ef4ce08919e522eb00ea1bea5689f3b53e"
DEBUGGYM_TREE = "54b04c0313b66d72c7f285dc255d2693164a7193"
EXPECTED_COLUMNS = {
    "repo_name", "docker_image", "execution_result_content", "expected_output_json",
    "modified_entity_summaries", "modified_files", "num_non_test_files",
    "num_non_test_func_methods", "num_non_test_lines", "parsed_commit_content",
    "problem_statement", "prompt", "relevant_files", "commit_hash",
}
ALLOWED_COLUMNS = (
    "repo_name", "docker_image", "commit_hash", "num_non_test_files",
    "num_non_test_func_methods", "num_non_test_lines",
)
SHARDS = (
    ("data/train-00000-of-00008.parquet", "03da135cddd3ac037bbae1c9c1ae78c81a0ef04d2b79d36d5290045d9fe08708", 51404349),
    ("data/train-00001-of-00008.parquet", "e0d3cf4b4dec1aab0263b73ab9f214012ad3296639036a6f9783b9fd3cb39a79", 120913910),
    ("data/train-00002-of-00008.parquet", "a48ab1c024cbbebc365d90407060e4e5addbd9b67aa1da47edc50abd05deee1c", 85528799),
    ("data/train-00003-of-00008.parquet", "fdef7fa3845df43b34c872ebd52da658c7c2f4cab3e68dab6dd6ca8bab117f43", 175814088),
    ("data/train-00004-of-00008.parquet", "e11d27b6a57f470033edb755695758d12b52e7e567a854b33befbc6701abc95d", 220723229),
    ("data/train-00005-of-00008.parquet", "f41eff181a1e075cab12a7ef08e5f2489fe8e0e4e542287b11f9a9a4c5868456", 175132057),
    ("data/train-00006-of-00008.parquet", "5c97e3099203912a4fc228d75c1f32174daef52f3810ccd08a53cc7ac63ba6a5", 61492145),
    ("data/train-00007-of-00008.parquet", "1ccfcbfbcefb75cb94318bcc2323abecbae54cc3d253e28f84c486718701626c", 53246653),
)
RUNTIME_FILES = {
    "r2e_docker_runtime": ("src/r2egym/agenthub/runtime/docker.py", "a83b1d1daea4181e97bce2a3ea19f85144b7776403792783dc0b977adc69eb0a"),
    "r2e_environment": ("src/r2egym/agenthub/environment/env.py", "5098d5237f3e761151285f4a9bb3d80fdd0cea2d34d1065d83ac1214f86ac718"),
    "r2e_log_parser": ("src/r2egym/repo_analysis/execution_log_parser.py", "395f637f4b8d68160948f95097f861506f123978852a9da77258aa3ba3fe1904"),
}
DEBUG_FILES = {
    "debuggym_r2e_adapter": ("debug_gym/gym/envs/r2egym.py", "e6aa7c331030c578b15a40dfb0e5ddad33bf40506df8013f0758e8a33a8e659a"),
    "debuggym_pdb": ("debug_gym/gym/tools/pdb.py", "ffd1453415ee0588bf72c62c3adb8ea238f9d2c004352aa884d8b947e79bc60b"),
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_value(root: Path, revision: str) -> str:
    return subprocess.check_output(["git", "rev-parse", revision], cwd=root, text=True).strip()


def split_hash(rows: list[dict[str, Any]]) -> str:
    return digest(canonical([row["commit_hash"] for row in rows]))


def _runtime_contract(r2e_root: Path, debuggym_root: Path) -> bool:
    docker = (r2e_root / RUNTIME_FILES["r2e_docker_runtime"][0]).read_text()
    env = (r2e_root / RUNTIME_FILES["r2e_environment"][0]).read_text()
    adapter = (debuggym_root / DEBUG_FILES["debuggym_r2e_adapter"][0]).read_text()
    pdb = (debuggym_root / DEBUG_FILES["debuggym_pdb"][0]).read_text()
    return all((
        "class DockerRuntime" in docker,
        "def run_tests(" in docker,
        "def reset(" in docker,
        "def run_action(" in env,
        "class R2EGymEnv" in adapter,
        'EVAL_COMMAND = "bash /root/run_tests.sh"' in adapter,
        "class Pdb" in pdb,
    ))


def audit(
    r2e_root: Path,
    data_root: Path,
    debuggym_root: Path,
    protocol: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    schemas: list[set[str]] = []
    shard_bindings = True
    for rel, expected_sha, expected_size in SHARDS:
        path = data_root / rel
        shard_bindings &= path.stat().st_size == expected_size and file_digest(path) == expected_sha
        parquet = pq.ParquetFile(path)
        schemas.append(set(parquet.schema_arrow.names))
        table = parquet.read(columns=list(ALLOWED_COLUMNS))
        columns = table.to_pydict()
        rows.extend(dict(zip(columns, values)) for values in zip(*columns.values()))

    commit_counts = Counter(str(row["commit_hash"] or "") for row in rows)
    image_counts = Counter(str(row["docker_image"] or "") for row in rows)
    commit_pattern = re.compile(r"^[0-9a-f]{40}$")
    eligible = [
        row for row in rows
        if str(row["repo_name"] or "").strip()
        and str(row["docker_image"] or "").strip()
        and commit_pattern.fullmatch(str(row["commit_hash"] or ""))
        and 1 <= int(row["num_non_test_files"]) <= 2
        and 2 <= int(row["num_non_test_func_methods"]) <= 4
        and 4 <= int(row["num_non_test_lines"]) <= 80
        and commit_counts[str(row["commit_hash"])] == 1
        and image_counts[str(row["docker_image"])] == 1
    ]
    ordered = sorted(
        eligible,
        key=lambda row: digest((SALT + row["commit_hash"] + "|" + row["docker_image"]).encode()),
    )
    splits = {
        "structural_screen": ordered[:16],
        "opportunity": ordered[16:80],
        "development": ordered[80:144],
        "confirmation": ordered[144:240],
        "reserve": ordered[240:],
    }
    selected = [row for values in splits.values() for row in values]
    selected_ids = [row["commit_hash"] for row in selected]
    runtime_hashes = all(
        file_digest(r2e_root / rel) == expected for rel, expected in RUNTIME_FILES.values()
    ) and all(file_digest(debuggym_root / rel) == expected for rel, expected in DEBUG_FILES.values())
    repo_counts = Counter(str(row["repo_name"]) for row in eligible)
    repo_size_histogram = dict(sorted(Counter(repo_counts.values()).items()))
    source_bindings = (
        git_value(r2e_root, "HEAD") == R2E_COMMIT
        and git_value(r2e_root, "HEAD^{tree}") == R2E_TREE
        and git_value(data_root, "HEAD") == DATA_COMMIT
        and git_value(data_root, "HEAD^{tree}") == DATA_TREE
        and git_value(debuggym_root, "HEAD") == DEBUGGYM_COMMIT
        and git_value(debuggym_root, "HEAD^{tree}") == DEBUGGYM_TREE
        and shard_bindings
        and runtime_hashes
    )
    gates = {
        "exact_source_and_runtime_bindings": source_bindings,
        "exact_population_and_schema": len(rows) == 4578 and all(schema == EXPECTED_COLUMNS for schema in schemas),
        "eligible_population_size": len(eligible) >= 256,
        "eligible_repository_diversity": len(repo_counts) >= 8,
        "complete_disjoint_split": len(selected_ids) == len(eligible)
        and len(selected_ids) == len(set(selected_ids))
        and len(splits["structural_screen"]) == 16
        and len(splits["opportunity"]) == 64
        and len(splits["development"]) == 64
        and len(splits["confirmation"]) == 96,
        "native_debugger_contract": _runtime_contract(r2e_root, debuggym_root),
    }
    manifest = {
        "protocol_version": VERSION,
        "bindings": {
            "r2e_commit": R2E_COMMIT,
            "r2e_tree": R2E_TREE,
            "dataset_revision": DATA_COMMIT,
            "dataset_tree": DATA_TREE,
            "debuggym_commit": DEBUGGYM_COMMIT,
            "debuggym_tree": DEBUGGYM_TREE,
            "ordered_shard_sha256": [value[1] for value in SHARDS],
            "runtime_file_sha256": {name: value[1] for name, value in {**RUNTIME_FILES, **DEBUG_FILES}.items()},
        },
        "population": {
            "row_count": len(rows),
            "schema_sha256": digest(canonical(sorted(EXPECTED_COLUMNS))),
            "eligible_count": len(eligible),
            "eligible_repository_count": len(repo_counts),
            "eligible_repository_size_histogram": repo_size_histogram,
        },
        "selection": {
            "salt_sha256": digest(SALT.encode()),
            "split_counts": {name: len(values) for name, values in splits.items()},
            "split_ordered_id_sha256": {name: split_hash(values) for name, values in splits.items()},
        },
        "privacy": {
            "projected_columns": list(ALLOWED_COLUMNS),
            "repositories_serialized": False,
            "docker_images_serialized": False,
            "commit_hashes_serialized": False,
            "row_offsets_serialized": False,
            "task_payload_columns_materialized": False,
            "task_payload_values_serialized": False,
            "endpoints_opened": False,
        },
    }
    gates["public_manifest_shape"] = set(manifest) == {"protocol_version", "bindings", "population", "selection", "privacy"}
    passed = all(gates.values())
    result = {
        "protocol_version": VERSION,
        "status": "source_pass" if passed else "source_failed_closed",
        "decision": "structural_screen_protocol_authorized" if passed else "close_exact_r2e_gym_debug_bed_source",
        "protocol_sha256": file_digest(protocol),
        "manifest_sha256": digest(canonical(manifest)),
        "gates": gates,
        "privacy": manifest["privacy"],
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "oatml_cluster_use": 0},
        "authorizes": "separately_frozen_exact_row_structural_screen_only" if passed else "nothing",
    }
    return manifest, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r2e-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--debuggym-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, result = audit(args.r2e_root, args.data_root, args.debuggym_root, args.protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "SOURCE_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
