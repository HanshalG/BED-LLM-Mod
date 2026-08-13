#!/usr/bin/env python3
"""Source admission for HiddenBench adaptive semantic elicitation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


VERSION = "hiddenbench-adaptive-elicitation-source-v1"
SALT = "hiddenbench-adaptive-elicitation-v1|"
COMMIT = "3be6ca16973e4fb751ffc0dfb7eb11f2d28335d1"
TREE = "e72388d2d7baf29fdef25807ac00d0bd93dc1ad1"
EXPECTED_TASK_KEYS = {
    "id", "name", "description", "shared_information", "hidden_information",
    "possible_answers", "correct_answer", "rationale",
}
REQUIRED_TASK_KEYS = EXPECTED_TASK_KEYS - {"rationale"}
BOUND_FILES = {
    "benchmark": ("data/benchmark.json", "2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3"),
    "simulator": ("src/hiddenbench/simulator.py", "1728834ea009073a8f7ca14dd4961f387e8f3c7f480a22981f5322c85a87d14f"),
    "loader": ("src/hiddenbench/benchmark.py", "3961483e5824af418fc3dc4a274a3c9ef61983ee3099cc0e6f619a9147b2daa9"),
    "system_prompt": ("prompts/system_prompt.txt", "38e8a41bb90836b488821729e3072557fa42d3fb69984a50c63bc91aa3c606fe"),
    "first_prompt": ("prompts/first_user_prompt.txt", "526ccec3cecec874ed63498e2eb22bffaa1abd69decedb676371ead9f9df1d8f"),
    "later_prompt": ("prompts/user_prompt.txt", "f83f6be2ba440e4f6fa34b88c92fa2fc84f9e4f6c48e66b4f2e0578bc0756205"),
    "license": ("LICENSE", "705dcb2b9b5abff9312bd885deae2d3fdcadfe65d4fc0cd0aa3b8b1d8545e354"),
}
SPLIT_SIZES = {
    "mechanics": 4,
    "opportunity": 12,
    "development": 16,
    "confirmation": 24,
    "reserve": 9,
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


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def unique_nonempty_strings(values: Any) -> bool:
    return (
        isinstance(values, list)
        and all(nonempty_string(value) for value in values)
        and len(values) == len(set(value.strip() for value in values))
    )


def split_tasks(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(tasks, key=lambda task: digest((SALT + str(task["id"])).encode()))
    splits: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for name, size in SPLIT_SIZES.items():
        splits[name] = ordered[offset:offset + size]
        offset += size
    return splits


def split_hash(tasks: list[dict[str, Any]]) -> str:
    return digest(canonical([str(task["id"]) for task in tasks]))


def simulator_contract(source: str) -> bool:
    required = (
        "num_agents = len(hidden_info)",
        "facts = list(task.shared_information)",
        "facts.append(hidden_info[index])",
        "assigned_hidden = [hidden_info[index]]",
        "previous_messages.append",
        "other.history[-1]['content']",
        "response = agent.chat(prompt)",
    )
    return all(token in source for token in required)


def audit(source_root: Path, protocol: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bindings_match = (
        git_value(source_root, "HEAD") == COMMIT
        and git_value(source_root, "HEAD^{tree}") == TREE
        and all(file_digest(source_root / relative) == expected for relative, expected in BOUND_FILES.values())
    )
    tasks = json.loads((source_root / BOUND_FILES["benchmark"][0]).read_text())
    schemas_valid = isinstance(tasks, list) and all(
        isinstance(task, dict)
        and REQUIRED_TASK_KEYS <= set(task) <= EXPECTED_TASK_KEYS
        for task in tasks
    )
    ids = [str(task.get("id", "")) for task in tasks] if isinstance(tasks, list) else []
    names = [str(task.get("name", "")) for task in tasks] if isinstance(tasks, list) else []
    population_valid = (
        schemas_valid
        and len(tasks) == 65
        and len(ids) == len(set(ids))
        and len(names) == len(set(names))
        and all(nonempty_string(task["name"]) and nonempty_string(task["description"]) for task in tasks)
    )
    task_contracts_valid = population_valid and all(
        unique_nonempty_strings(task["shared_information"])
        and len(task["shared_information"]) >= 3
        and unique_nonempty_strings(task["hidden_information"])
        and 3 <= len(task["hidden_information"]) <= 4
        and unique_nonempty_strings(task["possible_answers"])
        and 3 <= len(task["possible_answers"]) <= 4
        and nonempty_string(task["correct_answer"])
        and task["possible_answers"].count(task["correct_answer"]) == 1
        and ("rationale" not in task or nonempty_string(task["rationale"]))
        for task in tasks
    )
    splits = split_tasks(tasks) if population_valid else {name: [] for name in SPLIT_SIZES}
    selected_ids = [str(task["id"]) for values in splits.values() for task in values]
    split_valid = (
        {name: len(values) for name, values in splits.items()} == SPLIT_SIZES
        and len(selected_ids) == len(set(selected_ids)) == len(tasks)
    )
    native_channel = simulator_contract((source_root / BOUND_FILES["simulator"][0]).read_text())
    privacy = {
        "task_ids_serialized": False,
        "names_serialized": False,
        "descriptions_serialized": False,
        "facts_serialized": False,
        "answers_serialized": False,
        "rationales_serialized": False,
        "task_rows_serialized": False,
        "model_responses_opened": False,
        "planner_scores_opened": False,
        "endpoints_opened": False,
    }
    manifest = {
        "protocol_version": VERSION,
        "bindings": {
            "commit": COMMIT,
            "tree": TREE,
            "file_sha256": {name: expected for name, (_, expected) in BOUND_FILES.items()},
        },
        "population": {
            "task_count": len(tasks),
            "schema_sha256": digest(canonical(sorted(EXPECTED_TASK_KEYS))),
            "possible_answer_count_histogram": {
                str(size): sum(len(task["possible_answers"]) == size for task in tasks)
                for size in (3, 4)
            },
            "private_fact_count_histogram": {
                str(size): sum(len(task["hidden_information"]) == size for task in tasks)
                for size in (3, 4)
            },
        },
        "selection": {
            "salt_sha256": digest(SALT.encode()),
            "split_counts": {name: len(values) for name, values in splits.items()},
            "split_ordered_id_sha256": {name: split_hash(values) for name, values in splits.items()},
        },
        "privacy": privacy,
    }
    gates = {
        "exact_source_bindings": bindings_match,
        "exact_population_and_schema": population_valid,
        "semantic_task_contracts": task_contracts_valid,
        "native_private_information_channel": native_channel,
        "complete_disjoint_split": split_valid,
        "public_manifest_shape": set(manifest) == {"protocol_version", "bindings", "population", "selection", "privacy"},
        "public_manifest_is_aggregate_only": all(not value for value in privacy.values()),
    }
    passed = all(gates.values())
    result = {
        "protocol_version": VERSION,
        "status": "source_pass" if passed else "source_failed_closed",
        "decision": "mechanics_protocol_authorized" if passed else "close_exact_hiddenbench_source",
        "protocol_sha256": file_digest(protocol),
        "manifest_sha256": digest(canonical(manifest)),
        "gates": gates,
        "limitations": {
            "nonmyopic_opportunity_established": False,
            "llm_likelihood_calibrated": False,
            "policy_endpoint_opened": False,
            "published_multiagent_results_used_as_endpoint": False,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "oatml_cluster_use": 0},
        "authorizes": "separately_frozen_four_task_mechanics_only" if passed else "nothing",
    }
    return manifest, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest, result = audit(args.source_root, args.protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "SOURCE_AUDIT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "source_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
