#!/usr/bin/env python3
"""Freeze a fresh RegretBench cohort for the option-ID reply codec."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import regretbench_llm_native_source_audit as original


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-option-id-codec-source-audit-1"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_OPTION_ID_CODEC_SOURCE_PROTOCOL_20260811.md"
)
PROTOCOL_SHA256 = (
    "d90ed1e3f575e0ef50a5df13caf16a1637a64b3946359562402921dcf3f4389a"
)
CODEC_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_OPTION_ID_CODEC_EXACT8_PROTOCOL_20260811.md"
)
CODEC_PROTOCOL_SHA256 = (
    "e626e402f85163be49df432ca2414d0745103bdf60ce989cc3b25a9279bc18ed"
)
ORIGINAL_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/regretbench_llm_native_source_audit/"
    "SOURCE_PROTOCOL_MANIFEST.json"
)
ORIGINAL_MANIFEST_SHA256 = (
    "8a46b40395487aae0857d1a61d6f680beef579414c4580acf09d7e0b30b38e97"
)
FACTORIZED_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/regretbench_factorized_v2_source_audit/"
    "SOURCE_PROTOCOL_MANIFEST.json"
)
FACTORIZED_MANIFEST_SHA256 = (
    "831a8bcf8f38b183c080c5a369896366915ca8b8ada38294143d7f31a11c1999"
)
OUTPUT_DIR = REPO_ROOT / "results/nonmyopic/regretbench_option_id_codec_source_audit"
SPLIT_SALT = "regretbench-option-id-codec-v1|"
SPLIT_SIZES = {
    "codec_calibration": 2,
    "mechanics": 2,
    "development": 64,
    "confirmation": 64,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def split_hash(ids: list[str]) -> str:
    return sha256_bytes("\n".join(ids).encode())


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_bindings() -> None:
    for name, path, digest in (
        ("source protocol", PROTOCOL, PROTOCOL_SHA256),
        ("codec protocol", CODEC_PROTOCOL, CODEC_PROTOCOL_SHA256),
        ("original manifest", ORIGINAL_MANIFEST, ORIGINAL_MANIFEST_SHA256),
        ("factorized manifest", FACTORIZED_MANIFEST, FACTORIZED_MANIFEST_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"option-ID source {name} changed")


def prior_ids() -> tuple[set[str], dict[str, int]]:
    original_manifest = load_object(ORIGINAL_MANIFEST)
    factorized_manifest = load_object(FACTORIZED_MANIFEST)
    original_ids = {
        cig_id
        for split in original_manifest["splits"].values()
        for cig_id in split["ids"]
    }
    factorized_ids = {
        cig_id
        for split in factorized_manifest["splits"].values()
        for cig_id in split["ids"]
    }
    if len(original_ids) != 132 or len(factorized_ids) != 132:
        raise ValueError("option-ID predecessor cohort size changed")
    if original_ids & factorized_ids:
        raise ValueError("option-ID predecessor cohorts unexpectedly overlap")
    return original_ids | factorized_ids, {
        "original": len(original_ids),
        "factorized_v2": len(factorized_ids),
        "union": len(original_ids | factorized_ids),
    }


def run_audit(*, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    validate_bindings()
    if original._source_commit() != original.SOURCE_COMMIT:
        raise ValueError("RegretBench source commit changed")
    files = sorted(original.TEST_ROOT.glob("*.json"))
    published = original._published_checksums()
    checksum_failures = [
        str(path.relative_to(original.DATASET_ROOT))
        for path in files
        if published.get(str(path.relative_to(original.DATASET_ROOT)))
        != sha256_file(path)
    ]
    rows = [(path, load_object(path)) for path in files]
    eligible = [(path, cig) for path, cig in rows if original.eligible(cig)]
    excluded, excluded_counts = prior_ids()
    fresh = [row for row in eligible if str(row[1]["cig_id"]) not in excluded]
    fresh.sort(
        key=lambda row: sha256_bytes(
            (SPLIT_SALT + str(row[1]["cig_id"])).encode()
        )
    )
    selected_count = sum(SPLIT_SIZES.values())
    selected = fresh[:selected_count]
    splits = {}
    offset = 0
    for name, size in SPLIT_SIZES.items():
        splits[name] = selected[offset : offset + size]
        offset += size
    ids_by_split = {
        name: [str(cig["cig_id"]) for _, cig in cohort]
        for name, cohort in splits.items()
    }
    selected_ids = [str(cig["cig_id"]) for _, cig in selected]
    prompts = [original._normalize(str(cig["prompt"])) for _, cig in selected]
    alias_leaks = []
    for _, cig in selected:
        prompt = original._normalize(str(cig["prompt"]))
        for intent in cig["intents"]:
            aliases = str(original._slots(intent).get("answer_aliases", ""))
            for alias in aliases.split("|"):
                normalized = original._normalize(alias)
                if len(normalized) >= 4 and normalized in prompt:
                    alias_leaks.append(str(cig["cig_id"]))
    gains = [original.fixed_support_depth_gain(cig) for _, cig in selected]
    gates = {
        "official_commit_matches": original._source_commit() == original.SOURCE_COMMIT,
        "exact_6286_test_files": len(files) == original.EXPECTED_TEST_FILES,
        "all_available_test_checksums_match": not checksum_failures,
        "exact_2419_eligible_cigs": len(eligible) == original.EXPECTED_ELIGIBLE,
        "exact_264_prior_tasks_excluded": excluded_counts["union"] == 264
        and not (set(selected_ids) & excluded),
        "exact_fresh_split_sizes": all(
            len(splits[name]) == size for name, size in SPLIT_SIZES.items()
        ),
        "fresh_splits_are_disjoint": len(selected_ids) == len(set(selected_ids)),
        "selected_prompts_are_unique": len(prompts) == len(set(prompts)),
        "no_selected_prompt_contains_answer_alias": not alias_leaks,
        "all_selected_fixed_support_depth_gains_are_zero": all(
            abs(value) <= 1e-12 for value in gains
        ),
    }
    gates["all_pass"] = all(gates.values())
    original_manifest = load_object(ORIGINAL_MANIFEST)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "source": {
            "repository": str(original.SOURCE_ROOT),
            "commit": original.SOURCE_COMMIT,
            "dataset": "OpenDomainQA/test",
            "available_test_file_count": len(files),
            "published_checksum_inventory_sha256": sha256_file(
                original.DATASET_ROOT / "SHA256SUMS"
            ),
            "original_manifest_sha256": ORIGINAL_MANIFEST_SHA256,
            "factorized_manifest_sha256": FACTORIZED_MANIFEST_SHA256,
        },
        "selection": {
            "salt": SPLIT_SALT,
            "eligible_count": len(eligible),
            "prior_excluded_counts": excluded_counts,
            "fresh_eligible_count": len(fresh),
            "selected_count": len(selected),
            "all_selected_ids_sha256": split_hash(selected_ids),
        },
        "splits": {
            name: {
                "size": len(ids_by_split[name]),
                "ids": ids_by_split[name],
                "ids_sha256": split_hash(ids_by_split[name]),
                "files_sha256": {
                    str(path.relative_to(original.DATASET_ROOT)): sha256_file(path)
                    for path, _ in splits[name]
                },
            }
            for name in SPLIT_SIZES
        },
        "fixed_support_control": {
            "selected_strict_positive_depth_gains": sum(value > 1e-12 for value in gains),
            "selected_max_depth_gain_nats": max(gains),
        },
        "policy_visibility": original_manifest["policy_visibility"],
        "codec_calibration_cannot_select_mechanics": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "SOURCE_PROTOCOL_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "source_protocol_pass" if gates["all_pass"] else "source_protocol_null",
        "authorizes": "option_id_codec_exact8_only" if gates["all_pass"] else "nothing",
        "gates": gates,
        "counts": {
            "available_test_files": len(files),
            "eligible_cigs": len(eligible),
            "prior_excluded_tasks": len(excluded),
            "fresh_selected_tasks": len(selected),
        },
        "split_hashes": {
            name: manifest["splits"][name]["ids_sha256"] for name in SPLIT_SIZES
        },
        "source_protocol_manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": PROTOCOL_SHA256,
        "codec_protocol_sha256": CODEC_PROTOCOL_SHA256,
        "model_calls_made": 0,
        "cost_usd": 0.0,
        "source_values_opened": False,
        "policy_endpoint_opened": False,
        "development_opened": False,
        "confirmation_opened": False,
    }
    (output_dir / "RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run_audit(output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gates"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
