#!/usr/bin/env python3
"""Audit RegretBench for an environment-grounded depth-two action gap."""

from __future__ import annotations

from collections import defaultdict
import argparse
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
EXPECTED_TEST_FILES = 6286
EXPECTED_ELIGIBLE = 2419
EXPECTED_PRIOR_TASKS = 528
EXPECTED_FRESH_EXACT4 = 759
TOLERANCE = 1e-12
SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-horizon-opportunity-audit-1"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_HORIZON_OPPORTUNITY_AUDIT_PROTOCOL_20260811.md"
)
PROTOCOL_SHA256 = (
    "6f708ba0a2056d15acc602f10a62f4377bf794398dea9c0910753c8162611f61"
)
SOURCE_HELPER = REPO_ROOT / "scripts/regretbench_llm_native_source_audit.py"
SOURCE_HELPER_SHA256 = (
    "6906a6db52f26fb781c59b76d29ff095adaef0d85154c6b28d8a396402028163"
)
OUTPUT_DIR = REPO_ROOT / "results/nonmyopic/regretbench_horizon_opportunity_audit"
PRIOR_MANIFESTS = {
    "original": (
        REPO_ROOT
        / "results/nonmyopic/regretbench_llm_native_source_audit/"
        "SOURCE_PROTOCOL_MANIFEST.json",
        "8a46b40395487aae0857d1a61d6f680beef579414c4580acf09d7e0b30b38e97",
    ),
    "factorized_v2": (
        REPO_ROOT
        / "results/nonmyopic/regretbench_factorized_v2_source_audit/"
        "SOURCE_PROTOCOL_MANIFEST.json",
        "831a8bcf8f38b183c080c5a369896366915ca8b8ada38294143d7f31a11c1999",
    ),
    "option_id": (
        REPO_ROOT
        / "results/nonmyopic/regretbench_option_id_codec_source_audit/"
        "SOURCE_PROTOCOL_MANIFEST.json",
        "00526cefefda1633d0265be5af0d0f49b91665955335332b3e60ec78616fe273",
    ),
    "typed_action": (
        REPO_ROOT
        / "results/nonmyopic/regretbench_typed_action_source_audit/"
        "SOURCE_PROTOCOL_MANIFEST.json",
        "276c0e5d43b8415b1f2e8e8ceb4d97d3340cc5b02b56ba389a61ddc9a12dbcec",
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def slots(intent: Mapping[str, Any]) -> Mapping[str, Any]:
    value = intent.get("slots")
    return value if isinstance(value, Mapping) else {}


def facets(cig: Mapping[str, Any]) -> list[str]:
    schema = cig.get("semantic_action_schema")
    if not isinstance(schema, Mapping):
        return []
    values = schema.get("ask_facets")
    return [str(value) for value in values] if isinstance(values, list) else []


def eligible(cig: Mapping[str, Any]) -> bool:
    intents = cig.get("intents")
    actions = facets(cig)
    if (
        not str(cig.get("cig_id", "")).startswith("ambigdocs_")
        or not isinstance(intents, list)
        or not 3 <= len(intents) <= 6
        or not 2 <= len(actions) <= 4
    ):
        return False
    for intent in intents:
        if not isinstance(intent, Mapping):
            return False
        values = slots(intent)
        if not str(values.get("answer_aliases", "")).strip():
            return False
        if any(not str(values.get(action, "")).strip() for action in actions):
            return False
    return all(
        len({normalize(str(slots(intent)[action])) for intent in intents}) >= 2
        for action in actions
    )


def partition(
    cig: Mapping[str, Any], action: str, subset: Iterable[int]
) -> list[tuple[int, ...]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in subset:
        value = normalize(str(slots(cig["intents"][index])[action]))
        groups[value].append(index)
    return [tuple(groups[key]) for key in sorted(groups)]


def remaining_entropy(
    cig: Mapping[str, Any], action: str, subset: Sequence[int]
) -> float:
    size = len(subset)
    return sum(
        len(group) / size * math.log(len(group))
        for group in partition(cig, action, subset)
    )


def one_step_information(cig: Mapping[str, Any], action: str) -> float:
    subset = tuple(range(len(cig["intents"])))
    return math.log(len(subset)) - remaining_entropy(cig, action, subset)


def two_step_information(cig: Mapping[str, Any], first: str) -> float:
    actions = facets(cig)
    size = len(cig["intents"])
    expected_remaining = 0.0
    for group in partition(cig, first, range(size)):
        if len(group) <= 1:
            continue
        alternatives = [action for action in actions if action != first]
        residual = (
            min(remaining_entropy(cig, action, group) for action in alternatives)
            if alternatives
            else math.log(len(group))
        )
        expected_remaining += len(group) / size * residual
    return math.log(size) - expected_remaining


def depth_two_gain(cig: Mapping[str, Any]) -> float:
    actions = sorted(facets(cig))
    greedy = max(actions, key=lambda action: one_step_information(cig, action))
    depth_two = max(actions, key=lambda action: two_step_information(cig, action))
    return two_step_information(cig, depth_two) - two_step_information(cig, greedy)


def published_checksums() -> dict[str, str]:
    result = {}
    for line in (DATASET_ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        result[relative.strip()] = digest
    return result


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def prior_ids() -> tuple[set[str], dict[str, int], bool]:
    union: set[str] = set()
    counts = {}
    disjoint = True
    for name, (path, digest) in PRIOR_MANIFESTS.items():
        if sha256_file(path) != digest:
            raise ValueError(f"prior manifest changed: {name}")
        manifest = load_object(path)
        ids = {
            str(cig_id)
            for split in manifest["splits"].values()
            for cig_id in split["ids"]
        }
        counts[name] = len(ids)
        disjoint = disjoint and not bool(union & ids)
        union |= ids
    counts["union"] = len(union)
    return union, counts, disjoint


def has_exactly_four_executable_actions(cig: Mapping[str, Any]) -> bool:
    references = defaultdict(list)
    for row in cig.get("reference_questions", []):
        action = str(row.get("semantic_action", ""))
        text = str(row.get("text", "")).strip()
        if action.startswith("ask:") and text:
            references[action[4:]].append(text)
    return len(facets(cig)) == 4 and all(references[action] for action in facets(cig))


def run_audit(*, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("horizon-opportunity protocol changed")
    if sha256_file(SOURCE_HELPER) != SOURCE_HELPER_SHA256:
        raise ValueError("bound RegretBench source helper changed")
    files = sorted(TEST_ROOT.glob("*.json"))
    checksums = published_checksums()
    checksum_failures = [
        str(path.relative_to(DATASET_ROOT))
        for path in files
        if checksums.get(str(path.relative_to(DATASET_ROOT))) != sha256_file(path)
    ]
    rows = [(path, load_object(path)) for path in files]
    eligible_rows = [(path, cig) for path, cig in rows if eligible(cig)]
    gains = [depth_two_gain(cig) for _, cig in eligible_rows]
    strata: dict[str, dict[str, float | int]] = {}
    for facet_count in range(2, 5):
        for intent_count in range(3, 7):
            values = [
                gain
                for gain, (_, cig) in zip(gains, eligible_rows, strict=True)
                if len(facets(cig)) == facet_count
                and len(cig["intents"]) == intent_count
            ]
            strata[f"facets_{facet_count}_intents_{intent_count}"] = {
                "task_count": len(values),
                "strict_positive_gain_count": sum(
                    value > TOLERANCE for value in values
                ),
                "maximum_gain_nats": max(values, default=0.0),
            }
    excluded, excluded_counts, prior_disjoint = prior_ids()
    fresh_exact4 = [
        cig
        for _, cig in eligible_rows
        if str(cig["cig_id"]) not in excluded
        and has_exactly_four_executable_actions(cig)
    ]
    fresh_gains = [depth_two_gain(cig) for cig in fresh_exact4]
    gates = {
        "protocol_hash_matches": sha256_file(PROTOCOL) == PROTOCOL_SHA256,
        "source_helper_hash_matches": sha256_file(SOURCE_HELPER)
        == SOURCE_HELPER_SHA256,
        "official_commit_matches": source_commit() == SOURCE_COMMIT,
        "exact_6286_test_files": len(files) == EXPECTED_TEST_FILES,
        "all_available_test_checksums_match": not checksum_failures,
        "exact_2419_eligible_tasks": len(eligible_rows) == EXPECTED_ELIGIBLE,
        "four_prior_cohorts_are_disjoint": prior_disjoint,
        "exact_528_prior_tasks_excluded": len(excluded) == EXPECTED_PRIOR_TASKS,
        "exact_759_fresh_exact4_tasks": len(fresh_exact4) == EXPECTED_FRESH_EXACT4,
        "all_eligible_depth_two_gains_are_zero": all(
            abs(value) <= TOLERANCE for value in gains
        ),
        "all_fresh_exact4_depth_two_gains_are_zero": all(
            abs(value) <= TOLERANCE for value in fresh_gains
        ),
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "structural_opportunity_null" if gates["all_pass"] else "audit_failed"
        ),
        "authorizes": (
            "close_regretbench_primary_nonmyopic_route"
            if gates["all_pass"]
            else "nothing"
        ),
        "gates": gates,
        "population": {
            "test_file_count": len(files),
            "eligible_task_count": len(eligible_rows),
            "strict_positive_depth_two_gain_count": sum(
                value > TOLERANCE for value in gains
            ),
            "maximum_depth_two_gain_nats": max(gains, default=0.0),
            "tolerance": TOLERANCE,
        },
        "fresh_exact4_population": {
            "prior_cohort_counts": excluded_counts,
            "task_count": len(fresh_exact4),
            "strict_positive_depth_two_gain_count": sum(
                value > TOLERANCE for value in fresh_gains
            ),
            "maximum_depth_two_gain_nats": max(fresh_gains, default=0.0),
        },
        "strata": strata,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_commit": SOURCE_COMMIT,
        "source_helper_sha256": SOURCE_HELPER_SHA256,
        "model_calls_made": 0,
        "cost_usd": 0.0,
        "saved_llm_responses_opened": False,
        "policy_endpoint_opened": False,
        "development_opened": False,
        "confirmation_opened": False,
        "interpretation": (
            "The official fixed-support CIG has no depth-two first-action advantage; "
            "LLM-regeneration differences would lack an independent environment-grounded "
            "horizon opportunity."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
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
