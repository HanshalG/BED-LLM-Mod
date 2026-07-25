#!/usr/bin/env python3
"""Audit whether ClarifyBench exposes a native non-myopic BED interface."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable


SOURCE_REPOSITORY = "https://github.com/MananSuri27/ClarifyBench"
SOURCE_COMMIT = "a85d4f9df1fc87d05c713fd408f8a57d72bcc348"
CLASS_DIRECTORIES = {
    "ambiguous": "ClarifyBench_A",
    "explicit": "ClarifyBench_E",
    "infeasible": "ClarifyBench_I",
}
ALTERNATIVE_WORLD_KEYS = frozenset(
    {
        "alternative_intents",
        "alternative_user_intentions",
        "hypotheses",
        "intent_hypotheses",
        "latent_worlds",
        "possible_intents",
        "worlds",
    }
)
ANSWER_BRANCH_KEYS = frozenset(
    {
        "answer_map",
        "answer_branches",
        "answers_by_question",
        "clarification_responses",
        "response_map",
        "responses_by_question",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_source_hash(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def summarize_records(
    records: Iterable[tuple[str, str, dict[str, Any]]],
) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    follow_up_histogram: Counter[int] = Counter()
    tool_call_histogram: Counter[int] = Counter()
    records_with_follow_ups = 0
    records_with_turn_annotations = 0
    annotated_tool_calls = 0
    tool_calls = 0
    alternative_world_records = 0
    answer_branch_records = 0
    if_asked_records = 0
    record_count = 0

    for class_name, _record_id, record in records:
        record_count += 1
        class_counts[class_name] += 1
        follow_ups = record.get("potential_follow_ups", [])
        calls = record.get("ground_truth_tool_calls", [])
        if not isinstance(follow_ups, list):
            raise ValueError("potential_follow_ups must be a list")
        if not isinstance(calls, list):
            raise ValueError("ground_truth_tool_calls must be a list")
        follow_up_histogram[len(follow_ups)] += 1
        tool_call_histogram[len(calls)] += 1
        records_with_follow_ups += bool(follow_ups)
        call_turns = sum(
            isinstance(call, dict) and "turn" in call for call in calls
        )
        records_with_turn_annotations += call_turns > 0
        annotated_tool_calls += call_turns
        tool_calls += len(calls)
        keys = nested_keys(record)
        alternative_world_records += bool(keys & ALTERNATIVE_WORLD_KEYS)
        answer_branch_records += bool(keys & ANSWER_BRANCH_KEYS)
        if_asked_records += "if asked" in str(
            record.get("user_intention", "")
        ).casefold()

    return {
        "records": record_count,
        "class_counts": dict(sorted(class_counts.items())),
        "records_with_fixed_follow_ups": records_with_follow_ups,
        "fixed_follow_up_count_histogram": {
            str(key): follow_up_histogram[key]
            for key in sorted(follow_up_histogram)
        },
        "ground_truth_tool_calls": tool_calls,
        "ground_truth_tool_call_count_histogram": {
            str(key): tool_call_histogram[key]
            for key in sorted(tool_call_histogram)
        },
        "records_with_native_turn_annotations": (
            records_with_turn_annotations
        ),
        "native_turn_annotated_tool_calls": annotated_tool_calls,
        "records_with_alternative_world_support": (
            alternative_world_records
        ),
        "records_with_question_conditioned_answer_branches": (
            answer_branch_records
        ),
        "records_whose_intention_mentions_if_asked": if_asked_records,
    }


def function_method_calls(source: str, function_name: str) -> set[str]:
    tree = ast.parse(source)
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise ValueError(f"missing function {function_name}")
    return {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }


def function_assigns_attribute(
    source: str,
    function_name: str,
    attribute: str,
) -> bool:
    tree = ast.parse(source)
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise ValueError(f"missing function {function_name}")
    assignment_nodes = (
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
    )
    return any(
        isinstance(node, assignment_nodes)
        and any(
            isinstance(child, ast.Attribute) and child.attr == attribute
            for child in ast.walk(node)
        )
        for node in ast.walk(function)
    )


def audit_source(root: Path, *, verify_revision: bool = True) -> dict[str, Any]:
    root = root.resolve()
    if verify_revision:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != SOURCE_COMMIT:
            raise ValueError(
                f"ClarifyBench revision is {revision}, expected {SOURCE_COMMIT}"
            )
    else:
        revision = None

    records: list[tuple[str, str, dict[str, Any]]] = []
    source_paths: list[Path] = []
    for class_name, directory_name in CLASS_DIRECTORIES.items():
        directory = root / "ClarifyBench" / directory_name
        for path in sorted(directory.glob("*.json")):
            records.append(
                (class_name, path.stem, json.loads(path.read_text()))
            )
            source_paths.append(path)

    summary = summarize_records(records)
    main_path = root / "main.py"
    simulator_path = root / "llm" / "simulation.py"
    loader_path = root / "simulation" / "data_loader.py"
    main_source = main_path.read_text()
    simulator_source = simulator_path.read_text()
    loader_source = loader_path.read_text()
    run_calls = function_method_calls(main_source, "run_simulation")
    run_advances_turn = function_assigns_attribute(
        main_source,
        "run_simulation",
        "current_turn",
    )

    python_paths = sorted(root.rglob("*.py"))
    implementation_text = "\n".join(
        path.read_text(errors="replace") for path in python_paths
    ).casefold()
    license_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name.casefold().startswith("license")
            or path.name.casefold().startswith("copying")
        )
    )

    mentions_evpi = bool(re.search(r"\bevpi\b", implementation_text))
    mentions_pomdp = bool(re.search(r"\bpomdp\b", implementation_text))
    contains_bed_implementation = bool(
        re.search(
            r"\bclass\s+\w*(?:sage|pomdp)\w*\b"
            r"|\bdef\s+(?:calculate|compute|estimate)_evpi\b",
            implementation_text,
        )
    )
    mechanics = {
        "main_enumerates_fixed_follow_ups": (
            'all_requests = [("initial", user_query)] + '
            '[("follow_up", req) for req in follow_ups]'
        )
        in main_source,
        "run_simulation_calls_get_response_to_question": (
            "get_response_to_question" in run_calls
        ),
        "run_simulation_calls_get_next_request": (
            "get_next_request" in run_calls
        ),
        "run_simulation_advances_simulator_current_turn": run_advances_turn,
        "loader_defaults_missing_tool_call_turn_to_one": (
            'tc["turn"] = 1' in loader_source
        ),
        "simulator_has_llm_generated_clarification_answers": (
            "Generate a realistic user response to this SPECIFIC question."
            in simulator_source
        ),
        "shipped_python_mentions_evpi": mentions_evpi,
        "shipped_python_mentions_pomdp": mentions_pomdp,
        "shipped_python_contains_bed_implementation": (
            contains_bed_implementation
        ),
        "top_level_license_files": license_paths,
    }

    gates = {
        "multiple_latent_worlds_are_released": (
            summary["records_with_alternative_world_support"] > 0
        ),
        "question_conditioned_answer_branches_are_released": (
            summary[
                "records_with_question_conditioned_answer_branches"
            ]
            > 0
        ),
        "follow_up_sequence_is_endogenous": (
            not mechanics["main_enumerates_fixed_follow_ups"]
            and mechanics["run_simulation_calls_get_next_request"]
        ),
        "simulator_turn_state_tracks_main_request_sequence": (
            mechanics["run_simulation_advances_simulator_current_turn"]
            and summary["records_with_native_turn_annotations"]
            == summary["records"]
        ),
        "released_harness_contains_bed_policy": (
            mechanics["shipped_python_contains_bed_implementation"]
        ),
    }
    decision = "pass" if all(gates.values()) else "fail"

    return {
        "interface_version": "clarifybench-source-audit-1",
        "decision": decision,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": revision or SOURCE_COMMIT,
            "dataset_sha256": ordered_source_hash(source_paths, root),
            "main_sha256": sha256_file(main_path),
            "simulator_sha256": sha256_file(simulator_path),
            "loader_sha256": sha256_file(loader_path),
        },
        "corpus": summary,
        "mechanics": mechanics,
        "gates": gates,
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = audit_source(args.source_root)
    payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
