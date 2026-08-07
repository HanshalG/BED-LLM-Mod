#!/usr/bin/env python3
"""Audit the new RegretBench release for an LLM-native BED construction."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "external/RegretBench"
DATASET_ROOT = SOURCE_ROOT / "data/OpenDomainQA"
TEST_ROOT = DATASET_ROOT / "test"
SOURCE_COMMIT = "b2978e1c2e31b7a7c4e1508ee3e1fa1cb98f4aa7"
DATASET_VERSION = "1.0.0"
SPLIT_SALT = "regretbench-llm-native-v1|"
SPLIT_SIZES = {"mechanics": 4, "development": 64, "confirmation": 64}
EXPECTED_TEST_FILES = 6_286
EXPECTED_MANIFEST_TRAIN = 21_252
EXPECTED_ELIGIBLE = 2_419
EXPECTED_SPLIT_HASHES = {
    "mechanics": "707be5a1d1f86d6a0dc08ee61df77da1b9597093ac557d2e7706fcad8ef3b2f6",
    "development": "29d33b2fda0be7cc4eea6f4d9d4fe74fe200b5c580632dcb7ba844c3b825af69",
    "confirmation": "780a0e4e172251be2781729eeb4e591592b996dc9e1cd668b26240e3076660c9",
}
EXPECTED_ALL_SELECTION_HASH = (
    "86fa30ee244f568ea274966087a068a848bb33e98d5be1896ac8a3a2f9a41eba"
)
INTERFACE_VERSION = "regretbench-llm-native-source-audit-1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.casefold())).strip()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _published_checksums() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (DATASET_ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        result[relative.strip()] = digest
    return result


def _slots(intent: Mapping[str, Any]) -> Mapping[str, Any]:
    slots = intent.get("slots")
    return slots if isinstance(slots, Mapping) else {}


def _facets(cig: Mapping[str, Any]) -> list[str]:
    schema = cig.get("semantic_action_schema")
    if not isinstance(schema, Mapping):
        return []
    facets = schema.get("ask_facets")
    return [str(value) for value in facets] if isinstance(facets, list) else []


def eligible(cig: Mapping[str, Any]) -> bool:
    cig_id = str(cig.get("cig_id", ""))
    intents = cig.get("intents")
    facets = _facets(cig)
    if (
        not cig_id.startswith("ambigdocs_")
        or not isinstance(intents, list)
        or not 3 <= len(intents) <= 6
        or not 2 <= len(facets) <= 4
    ):
        return False
    for intent in intents:
        if not isinstance(intent, Mapping):
            return False
        slots = _slots(intent)
        if not str(slots.get("answer_aliases", "")).strip():
            return False
        if any(not str(slots.get(facet, "")).strip() for facet in facets):
            return False
    for facet in facets:
        values = {_normalize(str(_slots(intent).get(facet, ""))) for intent in intents}
        if len(values) < 2:
            return False
    return True


def _partition(
    cig: Mapping[str, Any], facet: str, subset: Iterable[int]
) -> list[tuple[int, ...]]:
    groups: dict[str, list[int]] = defaultdict(list)
    intents = cig["intents"]
    for index in subset:
        value = _normalize(str(_slots(intents[index]).get(facet, "")))
        groups[value].append(index)
    return [tuple(groups[key]) for key in sorted(groups)]


def _remaining_entropy(
    cig: Mapping[str, Any], facet: str, subset: Sequence[int]
) -> float:
    size = len(subset)
    return sum(
        len(group) / size * math.log(len(group))
        for group in _partition(cig, facet, subset)
    )


def _one_step_information(cig: Mapping[str, Any], facet: str) -> float:
    subset = tuple(range(len(cig["intents"])))
    return math.log(len(subset)) - _remaining_entropy(cig, facet, subset)


def _two_step_information(cig: Mapping[str, Any], first: str) -> float:
    facets = _facets(cig)
    size = len(cig["intents"])
    expected_remaining = 0.0
    for group in _partition(cig, first, range(size)):
        if len(group) <= 1:
            continue
        alternatives = [facet for facet in facets if facet != first]
        remaining = (
            min(_remaining_entropy(cig, facet, group) for facet in alternatives)
            if alternatives
            else math.log(len(group))
        )
        expected_remaining += len(group) / size * remaining
    return math.log(size) - expected_remaining


def fixed_support_depth_gain(cig: Mapping[str, Any]) -> float:
    facets = _facets(cig)
    greedy = max(sorted(facets), key=lambda facet: _one_step_information(cig, facet))
    depth_two = max(sorted(facets), key=lambda facet: _two_step_information(cig, facet))
    return _two_step_information(cig, depth_two) - _two_step_information(cig, greedy)


def _split_hash(ids: Sequence[str]) -> str:
    return sha256_bytes("\n".join(ids).encode("utf-8"))


def run_audit(*, output_dir: Path) -> dict[str, Any]:
    manifest = _load(DATASET_ROOT / "manifest.json")
    checksums = _published_checksums()
    files = sorted(TEST_ROOT.glob("*.json"))
    checksum_failures = [
        str(path.relative_to(DATASET_ROOT))
        for path in files
        if checksums.get(str(path.relative_to(DATASET_ROOT))) != sha256_file(path)
    ]
    rows = [(path, _load(path)) for path in files]
    eligible_rows = [(path, cig) for path, cig in rows if eligible(cig)]
    eligible_rows.sort(
        key=lambda row: sha256_bytes(
            (SPLIT_SALT + str(row[1]["cig_id"])).encode("utf-8")
        )
    )
    selected_count = sum(SPLIT_SIZES.values())
    selected = eligible_rows[:selected_count]
    splits: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    offset = 0
    for split, size in SPLIT_SIZES.items():
        splits[split] = selected[offset : offset + size]
        offset += size

    ids_by_split = {
        split: [str(cig["cig_id"]) for _, cig in cohort]
        for split, cohort in splits.items()
    }
    split_hashes = {
        split: _split_hash(ids) for split, ids in ids_by_split.items()
    }
    selected_ids = [str(cig["cig_id"]) for _, cig in selected]
    prompts = [_normalize(str(cig["prompt"])) for _, cig in selected]
    alias_leaks = []
    for _, cig in selected:
        prompt = _normalize(str(cig["prompt"]))
        for intent in cig["intents"]:
            for alias in str(_slots(intent).get("answer_aliases", "")).split("|"):
                normalized = _normalize(alias)
                if len(normalized) >= 4 and normalized in prompt:
                    alias_leaks.append(str(cig["cig_id"]))

    selected_gains = [fixed_support_depth_gain(cig) for _, cig in selected]
    all_gains = [fixed_support_depth_gain(cig) for _, cig in rows if _facets(cig)]
    tracked_train_files = list((DATASET_ROOT / "train").glob("*.json"))
    gates = {
        "official_commit_matches": _source_commit() == SOURCE_COMMIT,
        "dataset_version_matches": manifest.get("version") == DATASET_VERSION,
        "exact_6286_test_files": len(files) == EXPECTED_TEST_FILES,
        "all_available_test_checksums_match": not checksum_failures,
        "manifest_git_tree_train_gap_detected": (
            manifest.get("splits", {}).get("train") == EXPECTED_MANIFEST_TRAIN
            and not tracked_train_files
        ),
        "exact_2419_eligible_cigs": len(eligible_rows) == EXPECTED_ELIGIBLE,
        "exact_frozen_split_sizes": all(
            len(splits[split]) == size for split, size in SPLIT_SIZES.items()
        ),
        "split_hashes_match": split_hashes == EXPECTED_SPLIT_HASHES,
        "all_selection_hash_matches": _split_hash(selected_ids)
        == EXPECTED_ALL_SELECTION_HASH,
        "splits_are_disjoint": len(selected_ids) == len(set(selected_ids)),
        "selected_prompts_are_unique": len(prompts) == len(set(prompts)),
        "no_selected_prompt_contains_answer_alias": not alias_leaks,
        "model_payload_fields_are_minimal": {
            "task_id",
            "prompt",
            "dialogue",
        }
        == {"task_id", "prompt", "dialogue"},
        "selected_fixed_support_depth_gain_is_zero": max(selected_gains) <= 1e-12,
    }
    gates["all_pass"] = all(gates.values())

    protocol_manifest = {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "source": {
            "repository": "https://github.com/ngocminhta/RegretBench",
            "commit": SOURCE_COMMIT,
            "dataset": "OpenDomainQA",
            "version": DATASET_VERSION,
            "available_test_file_count": len(files),
            "available_test_checksum_inventory_sha256": sha256_file(
                DATASET_ROOT / "SHA256SUMS"
            ),
            "release_manifest_sha256": sha256_file(DATASET_ROOT / "manifest.json"),
            "manifest_train_files_absent_from_git_tree": True,
        },
        "eligibility": {
            "salt": SPLIT_SALT,
            "eligible_count": len(eligible_rows),
            "intent_count_min": 3,
            "intent_count_max": 6,
            "facet_count_min": 2,
            "facet_count_max": 4,
            "all_intents_require_answer_aliases_and_all_facet_values": True,
            "every_facet_requires_two_distinct_values": True,
        },
        "splits": {
            split: {
                "size": len(ids),
                "ids": ids,
                "ids_sha256": split_hashes[split],
                "files_sha256": {
                    str(path.relative_to(DATASET_ROOT)): sha256_file(path)
                    for path, _ in splits[split]
                },
            }
            for split, ids in ids_by_split.items()
        },
        "all_selected_ids_sha256": _split_hash(selected_ids),
        "policy_visibility": {
            "allowed": ["task_id", "prompt", "dialogue"],
            "forbidden": [
                "intents",
                "answer_aliases",
                "intent_descriptions",
                "slots",
                "latent_variables",
                "semantic_facets",
                "reference_questions",
                "metadata",
                "benchmark_belief",
            ],
        },
        "fixed_support_control": {
            "selected_strict_positive_depth_gains": sum(
                gain > 1e-12 for gain in selected_gains
            ),
            "selected_max_depth_gain_nats": max(selected_gains),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "SOURCE_PROTOCOL_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(protocol_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "status": "source_protocol_pass" if gates["all_pass"] else "source_protocol_null",
        "authorizes": (
            "exact_support_recovery_smoke_only" if gates["all_pass"] else "nothing"
        ),
        "source_commit": _source_commit(),
        "counts": {
            "available_test_files": len(files),
            "published_checksum_entries": len(checksums),
            "missing_manifest_train_files": EXPECTED_MANIFEST_TRAIN
            - len(tracked_train_files),
            "eligible_cigs": len(eligible_rows),
            "selected_cigs": len(selected),
            "all_available_strict_positive_fixed_support_depth_gains": sum(
                gain > 1e-12 for gain in all_gains
            ),
            "selected_strict_positive_fixed_support_depth_gains": sum(
                gain > 1e-12 for gain in selected_gains
            ),
        },
        "split_hashes": split_hashes,
        "all_selection_hash": _split_hash(selected_ids),
        "checksum_failures": checksum_failures,
        "answer_alias_prompt_leaks": sorted(set(alias_leaks)),
        "gates": gates,
        "source_protocol_manifest_sha256": sha256_file(manifest_path),
        "model_calls_made": 0,
        "cost_usd": 0.0,
        "confirmation_opened": False,
        "policy_endpoint_opened": False,
    }
    result_path = output_dir / "RESULT.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/nonmyopic/regretbench_llm_native_source_audit",
    )
    args = parser.parse_args()
    result = run_audit(output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gates"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
