#!/usr/bin/env python3
"""Audit CausaLab's source interface without reading graph or episode records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


EXPECTED_COMMIT = "42ba47fb88e60dc1eca17bd29c47ace4e8e9960e"
EXPECTED_TREE = "5700be5d041eda64c56d610cdaa9dfaa1e7736c1"
EXPECTED_PROTOCOL_SHA256 = (
    "41deda4b99dcbbc9cb60befcde9e94a975fec0b5bf98e6c86fd28cabf872fa7b"
)

SOURCE_PATHS = {
    "readme": "README.md",
    "license": "LICENSE.txt",
    "manifest": "release/causalab_dataset/manifest.json",
    "scenario": "discoveryworld/scenarios/reactor_lab.py",
    "causal_tool": "agents/recoma/causal_tool.py",
    "prompt": "agents/recoma/prompts/react_simple_memory_prompt_dsl.txt",
    "metrics": "causalab_reeval/metrics.py",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_value(repo: Path, expression: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", expression], text=True
    ).strip()


def git_text(repo: Path, path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"HEAD:{path}"], text=True
    )


def source_features(texts: Mapping[str, str]) -> dict[str, bool]:
    readme = texts["readme"]
    scenario = texts["scenario"]
    causal_tool = texts["causal_tool"]
    prompt = texts["prompt"]
    metrics = texts["metrics"]
    return {
        "hidden_scm_declared": "Each episode hides a freshly sampled structural causal model" in readme,
        "manipulator_and_transfer_objects": (
            (
                "property manipulator" in readme.lower()
                or "manipulator crystal" in readme.lower()
                or "*manipulator* crystal" in readme.lower()
            )
            and "reactor" in readme.lower()
            and (
                "held out" in readme.lower()
                or "held-out" in readme.lower()
            )
        ),
        "finite_intervention_budget": (
            'graph.budget = config_dict.get("budget", None)' in scenario
            and "budget exhausted" in readme
        ),
        "machine_parseable_hypothesis": (
            '"hypothesis"' in prompt
            and '"edges"' in prompt
            and '"freq_equation"' in prompt
            and '"coefficients"' in prompt
            and "ast.parse" in metrics
        ),
        "graph_binding": (
            "CAUSAL_GRAPH_CONFIG" in scenario and "CAUSAL_GRAPH_ID" in scenario
        ),
        "environment_seed_binding": "ENV_SEED" in scenario,
        "policy_can_omit_hidden_graph": "Previous State" in prompt,
        "complete_candidate_loader": (
            "def _load_candidates" in causal_tool
            and "for line in f:" in causal_tool
            and '"config": config' in causal_tool
            and "load_causal_graph_from_config(config, random.Random(0))" in causal_tool
        ),
        "exact_transition_simulator": (
            "def _simulate_transition" in causal_tool
            and "graph.compute_values" in causal_tool
            and "override_base_values=base_values" in causal_tool
        ),
        "deterministic_consistency_filter": (
            "def add_transition" in causal_tool
            and "self._transition_matches_graph" in causal_tool
            and "self.active_candidates" in causal_tool
        ),
        "candidate_bank_can_enter_prompt": (
            '"candidate_graphs"' in causal_tool
            and '"edges": config.get("edges", [])' in causal_tool
        ),
    }


def evaluate(
    *,
    commit: str,
    tree: str,
    protocol_sha256: str,
    license_text: str,
    manifest: Mapping[str, Any],
    features: Mapping[str, bool],
    source_blob_sha256: Mapping[str, str],
) -> dict[str, Any]:
    files = manifest.get("files")
    release_metadata_valid = (
        manifest.get("included_file_count") == 19
        and manifest.get("included_record_count") == 950
        and isinstance(files, list)
        and len(files) == 19
        and sum(item.get("records", 0) for item in files if isinstance(item, dict))
        == 950
    )
    bindings_pass = (
        commit == EXPECTED_COMMIT
        and tree == EXPECTED_TREE
        and protocol_sha256 == EXPECTED_PROTOCOL_SHA256
        and "Apache License" in license_text
        and "Version 2.0" in license_text
        and release_metadata_valid
    )
    native_pass = all(
        features[name]
        for name in (
            "hidden_scm_declared",
            "manipulator_and_transfer_objects",
            "finite_intervention_budget",
            "machine_parseable_hypothesis",
        )
    )
    replay_pass = all(
        features[name]
        for name in (
            "graph_binding",
            "environment_seed_binding",
            "policy_can_omit_hidden_graph",
        )
    )
    exact_filter_present = all(
        features[name]
        for name in (
            "complete_candidate_loader",
            "exact_transition_simulator",
            "deterministic_consistency_filter",
            "candidate_bank_can_enter_prompt",
        )
    )
    gates: dict[str, bool | None] = {
        "bindings_and_release": bindings_pass,
        "native_sequential_experiment": native_pass if bindings_pass else None,
        "replay_and_sealing": replay_pass if bindings_pass and native_pass else None,
        "open_support_ownership": (
            not exact_filter_present if bindings_pass and native_pass and replay_pass else None
        ),
        "irreducible_llm_role": None,
        "prospective_classical_control": None,
        "no_privileged_publication": True,
    }
    passed_prefix = bindings_pass and native_pass and replay_pass
    failed_at_open_support = passed_prefix and exact_filter_present
    if not failed_at_open_support:
        raise RuntimeError("frozen CausaLab snapshot did not reach the expected source gate")

    return {
        "schema_version": 1,
        "interface_version": "causalab-llm-native-source-admission-v1",
        "status": "source_failed_closed",
        "decision": "close_exact_causalab_release_as_llm_native_headline_route",
        "authorizes": "nothing",
        "failure_gate": "open_support_ownership",
        "failure_reason": "source_provides_complete_executable_candidate_filter",
        "ordering": "stopped_before_graph_records_or_episode_execution",
        "repository_binding": {
            "commit": commit,
            "tree": tree,
            "protocol_sha256": protocol_sha256,
            "source_blob_sha256": dict(sorted(source_blob_sha256.items())),
        },
        "release_metadata": {
            "declared_suite_count": manifest.get("included_file_count"),
            "declared_record_count": manifest.get("included_record_count"),
            "observed_manifest_suite_count": len(files),
            "observed_manifest_record_count": sum(
                item.get("records", 0) for item in files if isinstance(item, dict)
            ),
        },
        "source_features": dict(features),
        "access_accounting": {
            "graph_records_read": 0,
            "episode_trajectories_read": 0,
            "environment_executions": 0,
            "intervention_outcomes_opened": 0,
            "transfer_endpoints_opened": 0,
            "model_calls": 0,
            "openrouter_cost_usd": 0.0,
            "cluster_jobs": 0,
        },
        "gates": gates,
    }


def audit(*, repo: Path, protocol_path: Path) -> dict[str, Any]:
    texts = {name: git_text(repo, path) for name, path in SOURCE_PATHS.items()}
    manifest = json.loads(texts["manifest"])
    blob_hashes = {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in texts.items()
    }
    return evaluate(
        commit=git_value(repo, "HEAD"),
        tree=git_value(repo, "HEAD^{tree}"),
        protocol_sha256=sha256_file(protocol_path),
        license_text=texts["license"],
        manifest=manifest,
        features=source_features(texts),
        source_blob_sha256=blob_hashes,
    )


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
