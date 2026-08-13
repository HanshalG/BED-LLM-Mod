#!/usr/bin/env python3
"""Audit NewtonBench's public source without executing tasks or target laws."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


EXPECTED_COMMIT = "912a4ba5f4356ddd06acc16e44460ca30be4abc2"
EXPECTED_TREE = "88b68ea14e1ee6bc17a237642ab9eccde52faa48"
EXPECTED_PROTOCOL_SHA256 = (
    "b408a36f94c4750c352662f4c7e424cf0b23ef973608e5e53dd94c49178e5ad4"
)
DOMAIN_NAMES = tuple(f"m{i}_" for i in range(12))


def git_text(repo: Path, path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"HEAD:{path}"], text=True
    )


def git_value(repo: Path, expression: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", expression], text=True
    ).strip()


def git_paths(repo: Path) -> list[str]:
    text = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD"],
        text=True,
    )
    return text.splitlines()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry_shape(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    registry_node = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "LAW_REGISTRY"
            for target in node.targets
        ):
            registry_node = node.value
            break
    if not isinstance(registry_node, ast.Dict):
        raise RuntimeError("LAW_REGISTRY is absent or not a literal mapping")
    versions: dict[str, list[str]] = {}
    for difficulty_node, values_node in zip(
        registry_node.keys, registry_node.values, strict=True
    ):
        if not isinstance(difficulty_node, ast.Constant) or not isinstance(
            difficulty_node.value, str
        ):
            raise RuntimeError("LAW_REGISTRY difficulty key is not a string literal")
        if not isinstance(values_node, ast.Dict):
            raise RuntimeError("LAW_REGISTRY difficulty value is not a literal mapping")
        version_names = []
        for version_node in values_node.keys:
            if not isinstance(version_node, ast.Constant) or not isinstance(
                version_node.value, str
            ):
                raise RuntimeError("LAW_REGISTRY version key is not a string literal")
            version_names.append(version_node.value)
        versions[difficulty_node.value] = sorted(version_names)
    difficulties = sorted(versions)
    return {
        "difficulty_count": len(difficulties),
        "difficulties": difficulties,
        "versions_per_difficulty": {
            key: len(value) for key, value in sorted(versions.items())
        },
        "complete_three_by_three_registry": (
            difficulties == ["easy", "hard", "medium"]
            and all(len(value) == 3 for value in versions.values())
        ),
    }


def audit(*, repo: Path, protocol_path: Path) -> dict[str, Any]:
    paths = git_paths(repo)
    module_dirs = sorted(
        path.rsplit("/", 1)[0]
        for path in paths
        if path.startswith("modules/m") and path.endswith("/laws.py")
    )
    readme = git_text(repo, "README.md")
    prompt = git_text(repo, "modules/common/prompts_base.py")
    license_text = git_text(repo, "LICENSE")
    registry_shapes = {
        module: registry_shape(git_text(repo, f"{module}/laws.py"))
        for module in module_dirs
    }
    commit = git_value(repo, "HEAD")
    tree = git_value(repo, "HEAD^{tree}")
    protocol_sha = sha256_file(protocol_path)

    bindings_pass = (
        commit == EXPECTED_COMMIT
        and tree == EXPECTED_TREE
        and protocol_sha == EXPECTED_PROTOCOL_SHA256
        and "MIT License" in license_text
        and len(module_dirs) == 12
        and "324 tasks" in readme
        and "12 physics domains" in readme
    )
    native_pass = (
        "<run_experiment>" in prompt
        and "<final_law>" in prompt
        and "interactive experimentation" in readme
    )
    arbitrary_batch = (
        "one or arbitrarily many experimental sets" in prompt
        and "noise-free" in prompt
        and "perfectly accurate and deterministic" in prompt
    )
    complete_registries = (
        len(registry_shapes) == 12
        and all(
            shape["complete_three_by_three_registry"]
            for shape in registry_shapes.values()
        )
    )
    if not (bindings_pass and native_pass and arbitrary_batch and complete_registries):
        raise RuntimeError("frozen NewtonBench source did not reach expected gate")

    return {
        "schema_version": 1,
        "interface_version": "newtonbench-llm-native-source-admission-v1",
        "status": "source_failed_closed",
        "decision": "close_exact_newtonbench_release_as_nonmyopic_llm_native_route",
        "authorizes": "nothing",
        "failure_gate": "finite_budget_horizon",
        "failure_reason": "one_action_accepts_arbitrarily_many_noiseless_experiments",
        "ordering": "stopped_before_task_execution_or_target_law_evaluation",
        "repository_binding": {
            "commit": commit,
            "tree": tree,
            "protocol_sha256": protocol_sha,
        },
        "aggregate_source": {
            "declared_tasks": 324,
            "domain_modules": len(module_dirs),
            "law_difficulties_per_domain": 3,
            "law_versions_per_difficulty": 3,
            "system_complexities": 3,
            "complete_source_law_registries_observed": complete_registries,
            "arbitrary_noiseless_batch_interface_observed": arbitrary_batch,
        },
        "gates": {
            "bindings_and_population": True,
            "native_sequential_experiment": True,
            "finite_budget_horizon": False,
            "open_support_ownership": None,
            "irreducible_llm_role": None,
            "exact_controls": None,
            "no_privileged_publication": True,
        },
        "access_accounting": {
            "tasks_executed": 0,
            "target_laws_invoked": 0,
            "saved_trajectories_or_results_read": 0,
            "candidate_laws_submitted": 0,
            "evaluation_outcomes_opened": 0,
            "model_calls": 0,
            "openrouter_cost_usd": 0.0,
            "cluster_jobs": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(repo=args.repo, protocol_path=args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
