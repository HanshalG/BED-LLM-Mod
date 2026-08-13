from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


EXPECTED_COMMIT = "ed58236332ad039b54f968145d7bed9ba988f262"
EXPECTED_TREE = "a6c2204c8f9e1bfcc4c6d96cc5f5605f980063fa"
SPLIT_SIZES = (("mechanics", 6), ("opportunity", 40), ("development", 64), ("confirmation", 96))
CSV_PATHS = {
    "fully_specified": Path("data/fully-specified.csv"),
    "underspecified": Path("data/underspecified.csv"),
    "interaction": Path("data/interaction.csv"),
}
EXPECTED_PATHS = (
    *CSV_PATHS.values(),
    Path("evaluation/benchmarks/swe_bench/data/full_summaries_verified.xlsx"),
    Path("evaluation/benchmarks/swe_bench/interact_run_infer.py"),
    Path("evaluation/benchmarks/swe_bench/prompt.py"),
    Path("LICENSE"),
)
SPLIT_SALT = "ambig-swe-20260813:"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_id_hash(ids: list[str]) -> str:
    return sha256_bytes(canonical_json(ids).encode("ascii"))


def projected_task_ids(path: Path) -> tuple[list[str], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if not header or header[0] != "instance_id":
            raise ValueError(f"{path}: first CSV column must be instance_id")
        ids = []
        for row_number, row in enumerate(reader, start=2):
            if not row or not row[0].strip():
                raise ValueError(f"{path}: empty instance_id at row {row_number}")
            ids.append(row[0].strip())
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate instance_id")
    return header, ids


def split_ids(common_ids: set[str]) -> dict[str, list[str]]:
    ordered = sorted(
        common_ids,
        key=lambda task_id: (sha256_bytes(f"{SPLIT_SALT}{task_id}".encode("utf-8")), task_id),
    )
    splits: dict[str, list[str]] = {}
    offset = 0
    for name, size in SPLIT_SIZES:
        splits[name] = ordered[offset : offset + size]
        offset += size
    splits["retained"] = ordered[offset:]
    return splits


def inventory_csvs(source_root: Path) -> dict[str, Any]:
    projected: dict[str, list[str]] = {}
    headers: dict[str, list[str]] = {}
    for name, relative_path in CSV_PATHS.items():
        header, ids = projected_task_ids(source_root / relative_path)
        headers[name] = header
        projected[name] = ids

    id_sets = {name: set(ids) for name, ids in projected.items()}
    common = set.intersection(*id_sets.values())
    all_same = all(values == id_sets["fully_specified"] for values in id_sets.values())
    splits = split_ids(common)
    return {
        "headers": headers,
        "row_counts": {name: len(ids) for name, ids in projected.items()},
        "ordered_id_sha256": {name: ordered_id_hash(ids) for name, ids in projected.items()},
        "sorted_id_set_sha256": {
            name: ordered_id_hash(sorted(ids)) for name, ids in projected.items()
        },
        "all_three_views_have_identical_id_sets": all_same,
        "aligned_task_count": len(common),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "split_ordered_id_sha256": {
            name: ordered_id_hash(ids) for name, ids in splits.items()
        },
    }


def git_output(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=source_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def source_binding(source_root: Path) -> dict[str, Any]:
    return {
        "repository": "https://github.com/sani903/InteractiveSWEAgents",
        "commit": git_output(source_root, "rev-parse", "HEAD"),
        "tree": git_output(source_root, "rev-parse", "HEAD^{tree}"),
        "working_tree_clean": git_output(source_root, "status", "--short") == "",
    }


def audit(source_root: Path, protocol_path: Path) -> dict[str, Any]:
    binding = source_binding(source_root)
    missing = [str(path) for path in EXPECTED_PATHS if not (source_root / path).is_file()]
    if missing:
        raise FileNotFoundError("missing expected Ambig-SWE files: " + ", ".join(missing))
    if binding["commit"] != EXPECTED_COMMIT:
        raise ValueError("Ambig-SWE commit mismatch")
    if binding["tree"] != EXPECTED_TREE:
        raise ValueError("Ambig-SWE tree mismatch")
    if not binding["working_tree_clean"]:
        raise ValueError("Ambig-SWE checkout is dirty")

    inventory = inventory_csvs(source_root)
    identical = inventory["all_three_views_have_identical_id_sets"]
    retained_nonempty = inventory["split_counts"]["retained"] > 0
    first_gate = identical and inventory["aligned_task_count"] >= 300 and retained_nonempty

    gates: dict[str, bool | None] = {
        "identical_unique_task_ids_across_three_views": identical,
        "at_least_300_aligned_tasks": inventory["aligned_task_count"] >= 300,
        "retained_split_nonempty": retained_nonempty,
        "shorter_hidden_initial_issue_on_all_mechanics": None,
        "at_least_three_missing_information_units_per_mechanics_task": None,
        "category_diversity_on_at_least_four_mechanics_tasks": None,
        "simulator_excludes_gold_patches_tests_and_outcomes": None,
        "unrestricted_forkable_history_conditioned_simulator": None,
        "policy_initial_observation_is_target_blind": None,
        "external_executable_patch_endpoint_available": None,
        "license_permits_research_use": None,
        "all_source_gates_pass": False,
    }
    decision = "continue_to_mechanics_content_audit" if first_gate else "gate_failed_before_content"

    return {
        "schema_version": 1,
        "status": "source_gate_pending" if first_gate else "source_gate_failed",
        "decision": decision,
        "source": binding,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "split_salt": SPLIT_SALT,
            "model_calls": 0,
            "cost_usd": 0.0,
        },
        "files": {
            str(path): {"sha256": sha256_file(source_root / path), "size_bytes": (source_root / path).stat().st_size}
            for path in EXPECTED_PATHS
        },
        "inventory": inventory,
        "gates": gates,
        "privacy": {
            "csv_projection": ["instance_id"],
            "task_text_retained_or_serialized": False,
            "mechanics_content_opened": False,
            "opportunity_content_opened": False,
            "development_content_opened": False,
            "confirmation_content_opened": False,
            "retained_content_opened": False,
            "gold_patches_opened": False,
            "tests_or_test_outcomes_opened": False,
            "saved_policy_outcomes_opened": False,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/nonmyopic/AMBIG_SWE_SOURCE_AUDIT_PROTOCOL_20260813.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nonmyopic/ambig_swe_source_audit/AUDIT.json"),
    )
    args = parser.parse_args()
    result = audit(args.source_root.resolve(), args.protocol.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(canonical_json({"status": result["status"], "decision": result["decision"]}))
    return 0 if result["status"] == "source_gate_pending" else 2


if __name__ == "__main__":
    raise SystemExit(main())
