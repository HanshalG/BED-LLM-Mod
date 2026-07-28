#!/usr/bin/env python3
"""Replicate balanced branch refreshes from a frozen initial belief."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_dark_matter_balanced_modular_replication import (
    BOOTSTRAP_SAMPLES,
    CONTINUATION_SAMPLES,
    ROOT_SAMPLES,
    balanced_refresh_messages,
    endpoint_metrics,
    parse_balanced_refresh,
    selection_metrics,
)
from scripts.discoverphysics_dark_matter_executable_support import (
    DISCOVERPHYSICS_COMMIT,
    MODEL_ID,
    REGIONS,
    REGION_PRIOR,
    support_diagnostics,
)
from scripts.discoverphysics_dark_matter_grounded_policy import (
    LOOKAHEAD_ROOT_ID,
    MIN_INTERNAL_RISK_REDUCTION,
    MYOPIC_ROOT_ID,
    POLICY_NOISE_SEED,
    ROOTS,
    action_table,
    compile_support,
    load_executor_class,
    simulate_maps,
    support_change_key,
    support_prior,
)
from scripts.discoverphysics_dark_matter_opportunity import (
    verify_discoverphysics,
)
from scripts.discoverphysics_dark_matter_structured_replication import (
    INITIAL_COMPONENT_MASS,
    REFRESH_COMPONENT_MASS,
    REFRESH_MAX_TOKENS,
    RUN_BUDGET_USD,
    compile_refresh_models,
    refresh_response_format,
)
from scripts.discoverphysics_dark_matter_structured_replication_v3 import (
    _adapter,
)
from scripts.discoverphysics_oscillator_belief_smoke import (
    SmokeExecutionError,
    checkpoint,
    sha256_file,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-fixed-initial-branch-1"
SOURCE_MODEL_SHA256 = (
    "473cf5c883929a2cf8b6d862bebf69e1bb6b6401bea8c7b0edba955e847b7e1d"
)
EXPECTED_REQUESTS = 8
MAP_SEEDS = tuple(range(24740, 24756))
NOISE_SEED = 24756
BOOTSTRAP_SEED = 24757


def load_frozen_source(
    source_model: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    actual_hash = sha256_file(source_model)
    if actual_hash != SOURCE_MODEL_SHA256:
        raise ValueError(
            "source model hash changed: "
            f"expected {SOURCE_MODEL_SHA256}, got {actual_hash}"
        )
    payload = json.loads(source_model.read_text())
    if set(payload) != {
        "schema_version",
        "protocol",
        "raw_responses_sha256",
        "initial_support",
        "branches",
        "refreshes",
    }:
        raise ValueError("source model has unexpected fields")
    return payload["initial_support"], payload["branches"]


def branch_mechanics(
    *,
    initial_support: list[dict[str, Any]],
    refreshes: dict[str, list[dict[str, Any]]],
    usage: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    initial_key = support_change_key(initial_support)
    changed_supports = sum(
        support_change_key(refresh["hypotheses"]) != initial_key
        for root_refreshes in refreshes.values()
        for refresh in root_refreshes
    )
    branch_distinct_roots = sum(
        support_change_key(refreshes[root["id"]][0]["hypotheses"])
        != support_change_key(refreshes[root["id"]][1]["hypotheses"])
        for root in ROOTS
    )
    center_continuations = [
        refresh["continuation_action"]
        for refresh in refreshes[LOOKAHEAD_ROOT_ID]
    ]
    gates = {
        "exact_8_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "zero_forced_final_requests": usage["forced_final_requests"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "all_8_refreshes_change_support": changed_supports == 8,
        "at_least_3_roots_have_branch_distinct_supports": (
            branch_distinct_roots >= 3
        ),
        "center_branches_choose_distinct_continuations": (
            len(set(center_continuations)) == 2
        ),
        "all_refreshes_have_two_maps_per_region": all(
            support_diagnostics(refresh["hypotheses"])["region_counts"]
            == {region: 2 for region in REGIONS}
            for root_refreshes in refreshes.values()
            for refresh in root_refreshes
        ),
        "all_refresh_region_masses_match_disclosed_prior": all(
            all(
                np.isclose(
                    diagnostics["region_masses"][region],
                    REGION_PRIOR[region],
                    atol=1e-12,
                )
                for region in REGIONS
            )
            for root_refreshes in refreshes.values()
            for refresh in root_refreshes
            for diagnostics in [support_diagnostics(refresh["hypotheses"])]
        ),
        "all_refreshes_have_three_geometry_types": all(
            len(support_diagnostics(refresh["hypotheses"])["geometries"])
            >= 3
            for root_refreshes in refreshes.values()
            for refresh in root_refreshes
        ),
    }
    return gates, {
        "refreshes_changed_from_initial": changed_supports,
        "roots_with_branch_distinct_supports": branch_distinct_roots,
        "center_continuations": center_continuations,
    }


def run_replication(
    *,
    discoverphysics_root: Path,
    source_model: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    commit = verify_discoverphysics(discoverphysics_root)
    if commit != DISCOVERPHYSICS_COMMIT:
        raise ValueError("DiscoverPhysics commit changed")
    initial_support, branches = load_frozen_source(source_model)
    executor_class = load_executor_class(discoverphysics_root)
    initial_prior = support_prior(initial_support)
    initial_action_means, initial_heldout = simulate_maps(
        executor_class,
        compile_support(initial_support),
        action_ids=list(action_table()),
    )
    adapter = _adapter(run_id=run_id, output_dir=output_dir)
    raw_path = output_dir / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {"refreshes": []}
    try:
        requests = [
            balanced_refresh_messages(
                initial_support,
                branches,
                root_id=root["id"],
                branch_index=branch_index,
            )
            for root in ROOTS
            for branch_index in range(2)
        ]
        responses = adapter.chat_complete_messages_batched_structured(
            requests,
            temperature=0.0,
            block_size=8,
            response_format=refresh_response_format(),
            max_new_tokens=REFRESH_MAX_TOKENS,
        )
        raw["refreshes"] = list(responses)
        checkpoint(raw_path, raw)
        parsed_flat = [
            parse_balanced_refresh(
                response,
                root_action_id=ROOTS[index // 2]["action_id"],
                label=f"refresh[{index}]",
            )
            for index, response in enumerate(responses)
        ]
    except Exception as exc:
        raise SmokeExecutionError(
            f"{type(exc).__name__}: {exc}",
            adapter.usage_snapshot(),
        ) from exc

    refreshes = {
        root["id"]: parsed_flat[index * 2 : (index + 1) * 2]
        for index, root in enumerate(ROOTS)
    }
    refresh_models = compile_refresh_models(
        executor_class=executor_class,
        refreshes=refreshes,
    )
    usage = adapter.usage_snapshot()
    mechanics_gates, mechanism = branch_mechanics(
        initial_support=initial_support,
        refreshes=refreshes,
        usage=usage,
    )
    selection = selection_metrics(
        initial_prior=initial_prior,
        initial_action_means=initial_action_means,
        initial_heldout=initial_heldout,
        branches=branches,
        refresh_models=refresh_models,
    )
    selection_gates = {
        "source_myopic_root_is_D": (
            selection["myopic_root"] == MYOPIC_ROOT_ID
        ),
        "modular_lookahead_root_is_B": (
            selection["lookahead_root"] == LOOKAHEAD_ROOT_ID
        ),
        "internal_risk_reduction_at_least_10_percent": (
            selection["internal_risk_reduction"]
            >= MIN_INTERNAL_RISK_REDUCTION
        ),
    }
    phase_a_pass = all(mechanics_gates.values()) and all(
        selection_gates.values()
    )
    protocol = {
        "interface_version": INTERFACE_VERSION,
        "discoverphysics_commit": commit,
        "source_model_sha256": SOURCE_MODEL_SHA256,
        "model": MODEL_ID,
        "reasoning_enabled": False,
        "temperature": 0.0,
        "response_format": "chat_strict_json_schema_default_routing",
        "scientific_branch_requests": EXPECTED_REQUESTS,
        "expected_accepted_requests": EXPECTED_REQUESTS,
        "run_budget_usd": RUN_BUDGET_USD,
        "initial_component_mass": INITIAL_COMPONENT_MASS,
        "refresh_component_mass": REFRESH_COMPONENT_MASS,
        "refresh_region_mass": REGION_PRIOR,
        "refresh_maps_per_region": 2,
        "policy_noise_seed": POLICY_NOISE_SEED,
        "map_seeds": list(MAP_SEEDS),
        "noise_seed": NOISE_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "root_samples": ROOT_SAMPLES,
        "continuation_samples": CONTINUATION_SAMPLES,
    }
    phase_a = {
        "status": "pass" if phase_a_pass else "null",
        "protocol": protocol,
        "selection": selection,
        "mechanism": mechanism,
        "mechanics_gates": mechanics_gates,
        "selection_gates": selection_gates,
        "usage": usage,
        "raw_responses_sha256": sha256_file(raw_path),
    }
    if not phase_a_pass:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "phase_a_null",
            "phase_a": phase_a,
            "endpoint_accessed": False,
        }

    frozen_path = output_dir / "MODEL_FROZEN.json"
    checkpoint(
        frozen_path,
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": protocol,
            "source_model_sha256": SOURCE_MODEL_SHA256,
            "raw_responses_sha256": sha256_file(raw_path),
            "initial_support": initial_support,
            "branches": branches,
            "refreshes": refreshes,
        },
    )
    policy_path = output_dir / "POLICY.json"
    checkpoint(
        policy_path,
        {
            "schema_version": SCHEMA_VERSION,
            "protocol": protocol,
            "model_frozen_sha256": sha256_file(frozen_path),
            "selection": selection,
            "mechanism": mechanism,
            "mechanics_gates": mechanics_gates,
            "selection_gates": selection_gates,
        },
    )
    phase_a["model_frozen_sha256"] = sha256_file(frozen_path)
    phase_a["policy_sha256"] = sha256_file(policy_path)

    endpoint, endpoint_gates = endpoint_metrics(
        executor_class=executor_class,
        initial_prior=initial_prior,
        initial_action_means=initial_action_means,
        initial_heldout=initial_heldout,
        branches=branches,
        refresh_models=refresh_models,
        map_seeds=MAP_SEEDS,
        noise_seed=NOISE_SEED,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    all_gates_pass = all(endpoint_gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "replication_pass" if all_gates_pass else "replication_null"
        ),
        "phase_a": phase_a,
        "fresh_hidden_endpoint": endpoint,
        "endpoint_gates": endpoint_gates,
        "all_gates_pass": all_gates_pass,
        "endpoint_accessed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discoverphysics-root", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "RESULT.json"
    failure_path = output_dir / "FAILURE.json"
    try:
        payload = run_replication(
            discoverphysics_root=args.discoverphysics_root.resolve(),
            source_model=args.source_model.resolve(),
            output_dir=output_dir,
            run_id=args.run_id,
        )
        checkpoint(result_path, payload)
    except SmokeExecutionError as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": str(exc),
            "endpoint_accessed": False,
            "usage": exc.usage,
        }
        checkpoint(failure_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
