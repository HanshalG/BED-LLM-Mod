#!/usr/bin/env python3
"""Freeze repository-disjoint CI-Repair-Bench BED source cohorts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "external/CI-REPAIR-BENCH"
PARQUET_PATH = SOURCE_ROOT / "ci_repair_dataset.parquet"
OUTPUT_DIR = REPO_ROOT / "results/nonmyopic/ci_repair_bed_source_manifest"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/CI_REPAIR_BED_SOURCE_PROTOCOL_20260811.md"
)
SOURCE_COMMIT = "938310f45d5b76dfed56e2cfd7be344aed5b3de1"
DATASET_COMMIT = "7f5a6e8799ff57b9590cc8e87ce220c880d82c7b"
PARQUET_SHA256 = (
    "11caa322466b50f0ec31348d6adca096c7f5aac2bc8b9ec8df7721cf5b53d3f9"
)
EXPECTED_ROWS = 567
EXPECTED_REPOSITORIES = 103
EXPECTED_ELIGIBLE_REPOSITORIES = 33
MIN_STATES = 4
MAX_STATES = 12
REPO_SALT = "ci-repair-bed-v2|repo|"
ROW_SALT = "ci-repair-bed-v2|row|"
SPLIT_REPOSITORIES = {
    "mechanics": 3,
    "opportunity": 10,
    "development": 8,
    "confirmation": 8,
    "reserve": 4,
}
EXPECTED_SELECTED_STATES = {
    "mechanics": 16,
    "opportunity": 69,
    "development": 58,
    "confirmation": 71,
    "reserve": 40,
}
PARQUET_COLUMNS = ("id", "repo_name")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sequence_sha256(values: Iterable[str]) -> str:
    return sha256_bytes("\n".join(values).encode("utf-8"))


def source_head(source_root: Path = SOURCE_ROOT) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()


def read_safe_rows(parquet_path: Path = PARQUET_PATH) -> list[dict[str, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment error is explicit
        raise RuntimeError("pyarrow is required to read the bound Parquet source") from exc
    table = pq.read_table(parquet_path, columns=list(PARQUET_COLUMNS))
    return [
        {"id": str(row["id"]), "repo_name": str(row["repo_name"])}
        for row in table.to_pylist()
    ]


def select_cohorts(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    ids = [row["id"] for row in rows]
    if any(not task_id for task_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("CI-Repair-Bench IDs must be nonempty and unique")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        repo_name = row["repo_name"]
        if not repo_name:
            raise ValueError("CI-Repair-Bench repository names must be nonempty")
        grouped[repo_name].append(row)
    eligible = [name for name, items in grouped.items() if len(items) >= MIN_STATES]
    eligible.sort(key=lambda name: (sha256_bytes((REPO_SALT + name).encode()), name))
    if len(eligible) != sum(SPLIT_REPOSITORIES.values()):
        raise ValueError("eligible repository count does not match frozen allocation")
    cohorts: dict[str, list[dict[str, str]]] = {}
    offset = 0
    for split_name, repository_count in SPLIT_REPOSITORIES.items():
        selected_repositories = eligible[offset : offset + repository_count]
        offset += repository_count
        cohort = []
        for repo_name in selected_repositories:
            selected = sorted(
                grouped[repo_name],
                key=lambda row: (
                    sha256_bytes((ROW_SALT + row["id"]).encode()),
                    row["id"],
                ),
            )[:MAX_STATES]
            cohort.extend(selected)
        cohorts[split_name] = cohort
    return cohorts


def public_manifest(rows: list[dict[str, str]]) -> dict[str, Any]:
    counts = Counter(row["repo_name"] for row in rows)
    cohorts = select_cohorts(rows)
    splits = {}
    for split_name, cohort in cohorts.items():
        by_repo: dict[str, list[str]] = defaultdict(list)
        for row in cohort:
            by_repo[row["repo_name"]].append(row["id"])
        repositories = []
        for repo_name in sorted(by_repo):
            selected_ids = by_repo[repo_name]
            repositories.append(
                {
                    "repo_name": repo_name,
                    "available_state_count": counts[repo_name],
                    "selected_state_count": len(selected_ids),
                    "selected_ids_sha256": sequence_sha256(selected_ids),
                }
            )
        selected_ids = [row["id"] for row in cohort]
        splits[split_name] = {
            "repository_count": len(repositories),
            "selected_state_count": len(cohort),
            "selected_ids_sha256": sequence_sha256(selected_ids),
            "repositories": repositories,
        }
    return {
        "schema_version": 1,
        "interface_version": "ci-repair-bed-source-manifest-1",
        "source": {
            "repository_commit": SOURCE_COMMIT,
            "dataset_commit": DATASET_COMMIT,
            "parquet_sha256": PARQUET_SHA256,
            "row_count": len(rows),
            "repository_count": len(counts),
            "eligible_repository_count": sum(
                count >= MIN_STATES for count in counts.values()
            ),
            "projected_columns": list(PARQUET_COLUMNS),
        },
        "selection": {
            "repository_salt": REPO_SALT,
            "row_salt": ROW_SALT,
            "minimum_states_per_repository": MIN_STATES,
            "maximum_selected_states_per_repository": MAX_STATES,
            "grouping_key": "repo_name",
        },
        "splits": splits,
        "selected_ids_serialized": False,
        "workflow_opened": False,
        "logs_opened": False,
        "diffs_opened": False,
        "changed_files_opened": False,
        "error_types_opened": False,
        "model_calls_made": 0,
        "cost_usd": 0.0,
    }


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if not PROTOCOL.is_file():
        raise ValueError("CI-Repair-Bench source protocol is missing")
    if source_head() != SOURCE_COMMIT:
        raise ValueError("CI-Repair-Bench source commit changed")
    if sha256_file(PARQUET_PATH) != PARQUET_SHA256:
        raise ValueError("CI-Repair-Bench Parquet changed")
    rows = read_safe_rows()
    manifest = public_manifest(rows)
    source = manifest["source"]
    split_counts = {
        name: split["selected_state_count"]
        for name, split in manifest["splits"].items()
    }
    gates = {
        "exact_source_commit": source_head() == SOURCE_COMMIT,
        "exact_parquet_sha256": sha256_file(PARQUET_PATH) == PARQUET_SHA256,
        "only_safe_metadata_columns_projected": tuple(source["projected_columns"])
        == PARQUET_COLUMNS,
        "exact_567_rows": source["row_count"] == EXPECTED_ROWS,
        "exact_103_repository_names": source["repository_count"]
        == EXPECTED_REPOSITORIES,
        "exact_33_eligible_repositories": source["eligible_repository_count"]
        == EXPECTED_ELIGIBLE_REPOSITORIES,
        "exact_repository_split_sizes": all(
            manifest["splits"][name]["repository_count"] == count
            for name, count in SPLIT_REPOSITORIES.items()
        ),
        "exact_selected_state_counts": split_counts == EXPECTED_SELECTED_STATES,
        "no_source_content_or_outcome_opened": not any(
            manifest[name]
            for name in (
                "workflow_opened",
                "logs_opened",
                "diffs_opened",
                "changed_files_opened",
                "error_types_opened",
            )
        ),
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "schema_version": 1,
        "interface_version": "ci-repair-bed-source-manifest-result-1",
        "status": "source_manifest_frozen" if gates["all_pass"] else "source_manifest_invalid",
        "authorizes": "zero_call_opportunity_audit_only" if gates["all_pass"] else "nothing",
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_manifest_sha256": sha256_bytes(canonical_json(manifest).encode()),
        "gates": gates,
        "model_calls_made": 0,
        "cost_usd": 0.0,
        "selected_source_content_opened": False,
        "development_or_confirmation_opened": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not gates["all_pass"]:
        raise ValueError("CI-Repair-Bench source manifest gates failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
