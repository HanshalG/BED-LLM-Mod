#!/usr/bin/env python3
"""Freeze a fresh RegretBench cohort for the typed-action interface."""

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
INTERFACE_VERSION = "regretbench-typed-action-source-audit-1"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_TYPED_ACTION_SOURCE_PROTOCOL_20260811.md"
)
PROTOCOL_SHA256 = (
    "4301b0473ce7a3611af60dfc28a8b24b8e1a34d5d9a1592a10b9b7d2f207d93e"
)
CALIBRATION_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_TYPED_ACTION_EXACT8_PROTOCOL_20260811.md"
)
CALIBRATION_PROTOCOL_SHA256 = (
    "79a2fef1d8244007afece2c5f69484be6d3e8e3a4795b69e7982aad1d3bcc740"
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
OPTION_ID_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/regretbench_option_id_codec_source_audit/"
    "SOURCE_PROTOCOL_MANIFEST.json"
)
OPTION_ID_MANIFEST_SHA256 = (
    "00526cefefda1633d0265be5af0d0f49b91665955335332b3e60ec78616fe273"
)
OUTPUT_DIR = REPO_ROOT / "results/nonmyopic/regretbench_typed_action_source_audit"
SPLIT_SALT = "regretbench-typed-action-v1|"
SPLIT_SIZES = {
    "typed_calibration": 2,
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
        ("codec protocol", CALIBRATION_PROTOCOL, CALIBRATION_PROTOCOL_SHA256),
        ("original manifest", ORIGINAL_MANIFEST, ORIGINAL_MANIFEST_SHA256),
        ("factorized manifest", FACTORIZED_MANIFEST, FACTORIZED_MANIFEST_SHA256),
        ("option-ID manifest", OPTION_ID_MANIFEST, OPTION_ID_MANIFEST_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"typed-action source {name} changed")


def prior_ids() -> tuple[set[str], dict[str, int]]:
    original_manifest = load_object(ORIGINAL_MANIFEST)
    factorized_manifest = load_object(FACTORIZED_MANIFEST)
    option_id_manifest = load_object(OPTION_ID_MANIFEST)
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
    option_id_ids = {
        cig_id
        for split in option_id_manifest["splits"].values()
        for cig_id in split["ids"]
    }
    if (
        len(original_ids) != 132
        or len(factorized_ids) != 132
        or len(option_id_ids) != 132
    ):
        raise ValueError("typed-action predecessor cohort size changed")
    if (
        original_ids & factorized_ids
        or original_ids & option_id_ids
        or factorized_ids & option_id_ids
    ):
        raise ValueError("typed-action predecessor cohorts unexpectedly overlap")
    union = original_ids | factorized_ids | option_id_ids
    return union, {
        "original": len(original_ids),
        "factorized_v2": len(factorized_ids),
        "option_id": len(option_id_ids),
        "union": len(union),
    }


def executable_actions(cig: dict[str, Any]) -> list[dict[str, Any]]:
    references: dict[str, list[str]] = {}
    for row in cig.get("reference_questions", []):
        action = str(row.get("semantic_action", ""))
        text = str(row.get("text", "")).strip()
        if action.startswith("ask:") and text:
            references.setdefault(action[4:], []).append(text)
    actions = []
    facets = list((cig.get("semantic_action_schema") or {}).get("ask_facets", []))
    for public_order, facet in enumerate(facets):
        values = {
            str(original._slots(intent).get(facet, "")).strip()
            for intent in cig["intents"]
            if str(original._slots(intent).get(facet, "")).strip()
        }
        questions = references.get(facet, [])
        if len(values) < 2 or not questions:
            continue
        question = min(
            questions,
            key=lambda item: (original._normalize(item), item),
        )
        actions.append(
            {
                "action_id": facet,
                "question": question,
                "public_order": public_order,
                "private_distinct_value_count": len(values),
            }
        )
    return actions


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
    typed_eligible = [row for row in fresh if len(executable_actions(row[1])) == 4]
    typed_eligible.sort(
        key=lambda row: sha256_bytes(
            (SPLIT_SALT + str(row[1]["cig_id"])).encode()
        )
    )
    selected_count = sum(SPLIT_SIZES.values())
    selected = typed_eligible[:selected_count]
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
        "exact_396_prior_tasks_excluded": excluded_counts["union"] == 396
        and not (set(selected_ids) & excluded),
        "exact_891_fresh_typed_eligible_cigs": len(typed_eligible) == 891,
        "exact_fresh_split_sizes": all(
            len(splits[name]) == size for name, size in SPLIT_SIZES.items()
        ),
        "fresh_splits_are_disjoint": len(selected_ids) == len(set(selected_ids)),
        "selected_prompts_are_unique": len(prompts) == len(set(prompts)),
        "no_selected_prompt_contains_answer_alias": not alias_leaks,
        "all_selected_fixed_support_depth_gains_are_zero": all(
            abs(value) <= 1e-12 for value in gains
        ),
        "all_selected_have_exactly_four_executable_actions": all(
            len(executable_actions(cig)) == 4 for _, cig in selected
        ),
        "all_selected_actions_have_at_least_two_private_values": all(
            action["private_distinct_value_count"] >= 2
            for _, cig in selected
            for action in executable_actions(cig)
        ),
    }
    gates["all_pass"] = all(gates.values())
    original_manifest = load_object(ORIGINAL_MANIFEST)
    public_typed_tasks = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "tasks": [
            {
                "task_index": index,
                "task_id": str(cig["cig_id"]),
                "prompt": str(cig["prompt"]),
                "task_file_sha256": sha256_file(path),
                "actions": [
                    {
                        "action_id": action["action_id"],
                        "question": action["question"],
                        "public_order": order,
                    }
                    for order, action in enumerate(executable_actions(cig))
                ],
            }
            for index, (path, cig) in enumerate(splits["typed_calibration"])
        ],
        "source_values_included": False,
        "intent_descriptions_included": False,
        "canonical_questions_included": True,
        "action_metadata_in_model_prompts": True,
        "endpoint_outcomes_included": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    public_tasks_path = output_dir / "TYPED_PUBLIC_TASKS.json"
    public_tasks_path.write_text(
        json.dumps(public_typed_tasks, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
            "option_id_manifest_sha256": OPTION_ID_MANIFEST_SHA256,
        },
        "selection": {
            "salt": SPLIT_SALT,
            "eligible_count": len(eligible),
            "prior_excluded_counts": excluded_counts,
            "fresh_official_eligible_count": len(fresh),
            "fresh_typed_eligible_count": len(typed_eligible),
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
        "typed_calibration_cannot_select_mechanics": True,
        "typed_public_tasks_sha256": sha256_file(public_tasks_path),
        "source_values_serialized_in_public_artifacts": False,
    }
    manifest_path = output_dir / "SOURCE_PROTOCOL_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "source_protocol_pass" if gates["all_pass"] else "source_protocol_null",
        "authorizes": "typed_action_exact8_only" if gates["all_pass"] else "nothing",
        "gates": gates,
        "counts": {
            "available_test_files": len(files),
            "eligible_cigs": len(eligible),
            "prior_excluded_tasks": len(excluded),
            "fresh_typed_eligible_cigs": len(typed_eligible),
            "fresh_selected_tasks": len(selected),
        },
        "split_hashes": {
            name: manifest["splits"][name]["ids_sha256"] for name in SPLIT_SIZES
        },
        "source_protocol_manifest_sha256": sha256_file(manifest_path),
        "typed_public_tasks_sha256": sha256_file(public_tasks_path),
        "protocol_sha256": PROTOCOL_SHA256,
        "calibration_protocol_sha256": CALIBRATION_PROTOCOL_SHA256,
        "model_calls_made": 0,
        "cost_usd": 0.0,
        "source_values_materialized_in_public_artifacts": False,
        "policy_source_values_opened": False,
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
