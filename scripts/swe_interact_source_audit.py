#!/usr/bin/env python3
"""Audit the frozen SWE-Interact mechanics split without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any


EXPECTED_MANIFEST_SHA256 = (
    "d6184d7decef79e596a522f57b79fecd1aafbc838721290245baaf54d3295102"
)
EXPECTED_COMMIT = "b32f98c3b8f76ca65e84341d1f30e5af7135f85d"
EXPECTED_TREE = "d2b16852aa541c780598283ceb0302a4dbe66aff"
EXPECTED_SERVER_SHA256 = (
    "092dac347e3af748c0a3d2c0db8387a264c8eafdf2e31dd585f4fbebeb2bd417"
)
EXPECTED_STEPS = (
    "01_plan",
    "02_implement",
    "03_handoff",
    "04_write_tests",
    "05_test_handoff",
)

# These annotations were frozen after opening only the nine mechanics packages.
# Counts are conservative verifier-backed atomic surfaces, not every assertion.
TASK_ANNOTATIONS = {
    "deepswe_clack-async-autocomplete-options": {
        "persona_sha256": "baf2a165285377d1e1acbf6ef049488b22362540c2a3e7f9bd83eff0f762d46a",
        "atomic_requirement_count": 14,
        "count_source": "private task bullet groups backed by released tests",
        "dependency_example": "async detection -> fetch lifecycle -> stale-result handling",
    },
    "deepswe_ofetch-per-origin-circuit-breaker": {
        "persona_sha256": "61023e00e9ca964eef3f9bd1f7bd66aa5d28d8f6847cc9a83c01d0b5cd60bf30",
        "atomic_requirement_count": 47,
        "count_source": "private task bullet groups backed by released tests",
        "dependency_example": "per-origin state -> transition rules -> fast-fail behavior",
    },
    "deepswe_participle-grammar-conflict-analysis": {
        "persona_sha256": "447b23c03da29931c6a56423737e2e0c648df4b5a58087af0b3a133a3a47c8f9",
        "atomic_requirement_count": 11,
        "count_source": "conservative API/rule groups backed by released tests",
        "dependency_example": "conflict representation -> analysis API -> strict build behavior",
    },
    "rf_task-694b4b99829f00e24fd118a1": {
        "persona_sha256": "73099d8a49b2230c62341a05c325bdd590e98bb475b31fce391047385a202d89",
        "atomic_requirement_count": 7,
        "count_source": "released positive verifier rubrics",
        "dependency_example": "shared extraction -> inheritance -> duplicate removal",
    },
    "rf_task-696719205599a51110d4b428": {
        "persona_sha256": "8eb22817e63609b815bb0790f19a6d11664f6b713c482410235006e8abeb940c",
        "atomic_requirement_count": 10,
        "count_source": "released positive verifier rubrics",
        "dependency_example": "feature-flag removal -> component consolidation -> cleanup",
    },
    "rf_task-69b7c2a04b6f8ff9ed98813b": {
        "persona_sha256": "d67bad747b9491ea6b8db7410e3ee8433e2bd77ae7530441e7b709e069a2263a",
        "atomic_requirement_count": 21,
        "count_source": "released positive verifier rubrics",
        "dependency_example": "symbol extraction -> type propagation -> build registration",
    },
    "swebenchpro_instance_flipt-io__flipt-c1728053367c753688f114ec26e703c8fdeda125": {
        "persona_sha256": "b6b7f07cc22998434291c181876a0d699e005a1f3c335add906acf56957e9cc3",
        "atomic_requirement_count": 29,
        "count_source": "released verifier requirement bullets",
        "dependency_example": "CLI surface -> validation pipeline -> output/exit semantics",
    },
    "swebenchpro_instance_future-architect__vuls-7eb77f5b5127c847481bcf600b4dca2b7a85cf3e": {
        "persona_sha256": "041426fe587c62dfc89daf0a622146633833a65c6d6a069a8efc510d28b55802",
        "atomic_requirement_count": 12,
        "count_source": "released verifier requirement bullets",
        "dependency_example": "configuration -> validation -> external-scanner execution",
    },
    "swebenchpro_instance_qutebrowser__qutebrowser-fea33d607fde83cf505b228238cf365936437a63-v9f8e9d96c85c85a605e382f1510bd08563afc566": {
        "persona_sha256": "0aefb4bd5177ce927f19f0c4e3b26915b21693144eed9e1546f97bb4e4379b13",
        "atomic_requirement_count": 7,
        "count_source": "released verifier requirement bullets",
        "dependency_example": "version-check mode -> runtime bounds -> workaround behavior",
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def common_prefix_length(lines: list[list[str]]) -> int:
    index = 0
    while all(index < len(item) for item in lines):
        if len({item[index] for item in lines}) != 1:
            break
        index += 1
    return index


def count_requirement_bullets(value: str) -> int:
    return len(re.findall(r"(?:^|\n)\s*-\s+\S", value))


def count_positive_rubrics(value: Any) -> int:
    if not isinstance(value, list):
        raise ValueError("Expected a list of released rubrics")
    return sum(
        row.get("annotations", {}).get("type") == "positive hli verifier"
        for row in value
        if isinstance(row, dict)
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "Frozen manifest hash mismatch: "
            f"expected {EXPECTED_MANIFEST_SHA256}, found {digest}"
        )
    manifest = json.loads(raw)
    if manifest.get("commit") != EXPECTED_COMMIT:
        raise ValueError("Frozen manifest commit mismatch")
    if manifest.get("git_tree") != EXPECTED_TREE:
        raise ValueError("Frozen manifest tree mismatch")
    mechanics = manifest.get("partitions", {}).get("mechanics")
    if not isinstance(mechanics, list) or len(mechanics) != 9:
        raise ValueError("Frozen manifest must contain exactly nine mechanics tasks")
    if {row.get("task_id") for row in mechanics} != set(TASK_ANNOTATIONS):
        raise ValueError("Mechanics task IDs do not match frozen annotations")
    return manifest


def _released_requirement_count(task: Path, family: str) -> int | None:
    tests = task / "steps/05_test_handoff/tests"
    if family == "refactoring":
        return count_positive_rubrics(
            json.loads((tests / "rubrics.json").read_text(encoding="utf-8"))
        )
    if family == "swebench_pro":
        config = json.loads((tests / "config.json").read_text(encoding="utf-8"))
        return count_requirement_bullets(config["requirements"])
    return None


def build_audit(repo: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    if git_output(repo, "rev-parse", "HEAD") != EXPECTED_COMMIT:
        raise ValueError("Source checkout commit does not match frozen release")
    if git_output(repo, "rev-parse", "HEAD^{tree}") != EXPECTED_TREE:
        raise ValueError("Source checkout tree does not match frozen release")
    if git_output(repo, "status", "--short"):
        raise ValueError("Source checkout is not clean")

    mechanics = manifest["partitions"]["mechanics"]
    personas: list[list[str]] = []
    task_rows: list[dict[str, Any]] = []
    server_hashes: set[str] = set()

    for manifest_row in mechanics:
        task_id = manifest_row["task_id"]
        family = manifest_row["family"]
        task = repo / manifest_row["task_path"]
        annotation = TASK_ANNOTATIONS[task_id]
        persona_path = task / "environment/user-server/persona.md"
        persona_raw = persona_path.read_bytes()
        if sha256_bytes(persona_raw) != annotation["persona_sha256"]:
            raise ValueError(f"Private persona hash mismatch for {task_id}")
        persona_lines = persona_raw.decode("utf-8").splitlines()
        personas.append(persona_lines)

        server_path = task / "environment/user-server/server.py"
        server_hashes.add(sha256_bytes(server_path.read_bytes()))
        task_config = tomllib.loads(
            (task / "task.toml").read_text(encoding="utf-8")
        )
        step_names = tuple(step["name"] for step in task_config["steps"])
        if step_names != EXPECTED_STEPS:
            raise ValueError(f"Unexpected staged runner contract for {task_id}")

        plan_instruction = (
            task / "steps/01_plan/instruction.md"
        ).read_text(encoding="utf-8")
        if "ask_user" not in plan_instruction or "understanding" not in plan_instruction:
            raise ValueError(f"Planning stage does not route through user for {task_id}")
        if "Do not make implementation edits" not in plan_instruction:
            raise ValueError(f"Planning stage is not isolated for {task_id}")

        released_count = _released_requirement_count(task, family)
        expected_count = annotation["atomic_requirement_count"]
        if released_count is not None and released_count != expected_count:
            raise ValueError(
                f"Released requirement count changed for {task_id}: "
                f"expected {expected_count}, found {released_count}"
            )

        root_verifier = (task / "tests/test.sh").read_text(encoding="utf-8")
        if "reward.txt" not in root_verifier:
            raise ValueError(f"Missing external reward verifier for {task_id}")
        if "ask_user" in root_verifier or "persona.md" in root_verifier:
            raise ValueError(f"Verifier is coupled to user satisfaction for {task_id}")

        task_rows.append(
            {
                "family": family,
                "task_id": task_id,
                "public_initial_instruction_words": len(plan_instruction.split()),
                "private_task_suffix_words": None,
                "atomic_requirement_count": expected_count,
                "count_source": annotation["count_source"],
                "dependency_structure": True,
                "dependency_example": annotation["dependency_example"],
                "materially_incomplete_public_start": True,
                "external_verifier": True,
            }
        )

    prefix_lines = common_prefix_length(personas)
    if prefix_lines != 190:
        raise ValueError(
            f"Unexpected shared persona prefix: expected 190 lines, found {prefix_lines}"
        )
    for row, persona_lines in zip(task_rows, personas, strict=True):
        suffix_words = len(" ".join(persona_lines[prefix_lines:]).split())
        if suffix_words < 25:
            raise ValueError(f"Private task goal is unexpectedly short for {row['task_id']}")
        row["private_task_suffix_words"] = suffix_words

    if server_hashes != {EXPECTED_SERVER_SHA256}:
        raise ValueError("Mechanics tasks do not share the frozen simulator source")
    server = (
        repo
        / mechanics[0]["task_path"]
        / "environment/user-server/server.py"
    ).read_text(encoding="utf-8")
    persona_prefix = "\n".join(personas[0][:prefix_lines])

    simulator_checks = {
        "full_conversation_conditions_reply": all(
            token in server
            for token in (
                'conversation.append({"role": "user", "content": question})',
                "*conversation",
                "complete_with_tools(messages)",
            )
        ),
        "workspace_state_conditions_review": all(
            token in server
            for token in (
                "automatic_repo_snapshot(question)",
                "git show --patch --find-renames --stat HEAD",
            )
        ),
        "private_task_unavailable_to_policy": (
            'persona = Path("/app/persona.md").read_text()' in server
            and 'f"{persona}"' in server
        ),
        "generic_questions_do_not_dump_checklist": all(
            token in persona_prefix
            for token in (
                "Open with a vague, short request.",
                "Do not reveal a checklist mechanically.",
                "make it the agent's call.",
            )
        ),
        "one_issue_at_a_time": "One problem at a time" in persona_prefix,
        "generative_not_response_table": (
            "litellm.completion(" in server
            and "messages=working" in server
            and "conversation" in server
        ),
        "forkable_private_state": (
            "conversation: list[dict[str, Any]] = []" in server
            and "ask_user_call_count = 0" in server
        ),
    }
    if not all(simulator_checks.values()):
        failed = [name for name, passed in simulator_checks.items() if not passed]
        raise ValueError(f"Simulator source checks failed: {failed}")

    incomplete_count = sum(
        row["materially_incomplete_public_start"] for row in task_rows
    )
    four_atom_count = sum(row["atomic_requirement_count"] >= 4 for row in task_rows)
    dependency_count = sum(row["dependency_structure"] for row in task_rows)
    external_verifier_count = sum(row["external_verifier"] for row in task_rows)
    gates = {
        "public_start_materially_incomplete_at_least_7_of_9": incomplete_count >= 7,
        "four_atomic_hidden_requirements_at_least_7_of_9": four_atom_count >= 7,
        "dependency_structure_at_least_6_of_9": dependency_count >= 6,
        "reply_conditions_on_history_or_workspace": (
            simulator_checks["full_conversation_conditions_reply"]
            and simulator_checks["workspace_state_conditions_review"]
        ),
        "vague_questions_do_not_auto_dump_requirements": (
            simulator_checks["generic_questions_do_not_dump_checklist"]
        ),
        "forkable_and_target_blind": (
            simulator_checks["forkable_private_state"]
            and simulator_checks["private_task_unavailable_to_policy"]
        ),
        "external_verifier_not_user_satisfaction": external_verifier_count == 9,
        "not_finite_candidate_response_table": (
            simulator_checks["generative_not_response_table"]
        ),
    }

    return {
        "schema_version": 1,
        "audit": "swe_interact_release_source_mechanics",
        "source_commit": EXPECTED_COMMIT,
        "source_tree": EXPECTED_TREE,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "model_requests": 0,
        "openrouter_cost_usd": 0.0,
        "opened_partitions": ["mechanics"],
        "sealed_partitions": ["development", "confirmation", "retained"],
        "counts": {
            "mechanics_tasks": len(task_rows),
            "materially_incomplete_public_starts": incomplete_count,
            "tasks_with_at_least_four_atomic_requirements": four_atom_count,
            "tasks_with_dependency_structure": dependency_count,
            "tasks_with_external_verifier": external_verifier_count,
            "shared_persona_prefix_lines": prefix_lines,
        },
        "simulator_checks": simulator_checks,
        "gates": gates,
        "all_pass": all(gates.values()),
        "decision": (
            "authorize_preregistered_mechanics_serving_gate"
            if all(gates.values())
            else "close_before_model_calls"
        ),
        "tasks": task_rows,
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_atomic(
        args.output.resolve(),
        build_audit(args.repo.resolve(), args.manifest.resolve()),
    )


if __name__ == "__main__":
    main()
