#!/usr/bin/env python3
"""Replicate retained-support dark-matter BED with strict structured outputs."""

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

from scripts.discoverphysics_dark_matter_executable_support import (
    DISCOVERPHYSICS_COMMIT,
    GEOMETRIES,
    MODEL_ID,
    REGIONS,
    REGION_PRIOR,
    support_diagnostics,
)
from scripts.discoverphysics_dark_matter_executable_support_v2 import (
    parse_weighted_support,
    support_messages_v2,
)
from scripts.discoverphysics_dark_matter_grounded_policy import (
    BOOTSTRAP_SAMPLES,
    INTERNAL_CONTINUATION_SAMPLES,
    INTERNAL_ROOT_SAMPLES,
    LOOKAHEAD_ROOT_ID,
    MIN_HIDDEN_RISK_REDUCTION,
    MIN_INTERNAL_RISK_REDUCTION,
    MIN_RANDOM_RISK_REDUCTION,
    MYOPIC_ROOT_ID,
    POLICY_NOISE_SEED,
    RANDOM_ROOT_ID,
    ROOTS,
    _adapter,
    action_table,
    build_branches,
    compile_support,
    evaluate_fixed_support_root,
    hidden_halo_family,
    immediate_eig,
    load_executor_class,
    parse_refresh,
    refresh_messages,
    relative_reduction,
    root_by_id,
    simulate_maps,
    support_change_key,
    support_prior,
    weighted_mean,
)
from scripts.discoverphysics_dark_matter_opportunity import (
    verify_discoverphysics,
)
from scripts.discoverphysics_dark_matter_retained_support_replay import (
    evaluate_retained_root,
    retained_coverage_risk,
)
from scripts.discoverphysics_oscillator_belief_smoke import (
    SmokeExecutionError,
    checkpoint,
    sha256_file,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-structured-replication-1"
EXPECTED_REQUESTS = 10
INITIAL_MAX_TOKENS = 3500
REFRESH_MAX_TOKENS = 3200
RUN_BUDGET_USD = 0.25
INITIAL_COMPONENT_MASS = 0.95
REFRESH_COMPONENT_MASS = 0.05
REPLICATION_MAP_SEEDS = tuple(range(24700, 24716))
REPLICATION_NOISE_SEED = 24716
REPLICATION_BOOTSTRAP_SEED = 24717
REPLICATION_ROOT_SAMPLES = 8
REPLICATION_CONTINUATION_SAMPLES = 4
MIN_FIXED_SUPPORT_RISK_REDUCTION = 0.01
MIN_COVERAGE_RISK_REDUCTION = 0.05


def hypothesis_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
            "weight": {"type": "integer", "minimum": 1, "maximum": 100},
            "region": {"type": "string", "enum": list(REGIONS)},
            "center": {
                "type": "array",
                "items": {"type": "number", "minimum": -7.0, "maximum": 7.0},
                "minItems": 2,
                "maxItems": 2,
            },
            "geometry": {"type": "string", "enum": list(GEOMETRIES)},
            "major_spread": {
                "type": "number",
                "minimum": 0.15,
                "maximum": 2.5,
            },
            "minor_spread": {
                "type": "number",
                "minimum": 0.15,
                "maximum": 2.5,
            },
            "orientation_degrees": {
                "type": "number",
                "minimum": 0.0,
                "exclusiveMaximum": 180.0,
            },
        },
        "required": [
            "description",
            "weight",
            "region",
            "center",
            "geometry",
            "major_spread",
            "minor_spread",
            "orientation_degrees",
        ],
        "additionalProperties": False,
    }


def support_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "dark_matter_initial_support",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "items": hypothesis_schema(),
                        "minItems": 8,
                        "maxItems": 8,
                    }
                },
                "required": ["hypotheses"],
                "additionalProperties": False,
            },
        },
    }


def refresh_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "dark_matter_branch_refresh",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "items": hypothesis_schema(),
                        "minItems": 8,
                        "maxItems": 8,
                    },
                    "continuation_action": {
                        "type": "string",
                        "enum": sorted(action_table()),
                    },
                },
                "required": ["hypotheses", "continuation_action"],
                "additionalProperties": False,
            },
        },
    }


def replication_hidden_map_family() -> tuple[np.ndarray, list[str], np.ndarray]:
    maps = []
    regions = []
    for seed in REPLICATION_MAP_SEEDS:
        seed_maps, _ = hidden_halo_family(seed=seed)
        maps.append(seed_maps)
        regions.extend(
            region
            for region in ("NE", "NW", "SW", "SE")
            for _ in range(6)
        )
    source_maps = np.concatenate(maps, axis=0)
    prior = np.asarray(
        [
            REGION_PRIOR[region]
            / sum(candidate == region for candidate in regions)
            for region in regions
        ]
    )
    return source_maps, regions, prior


def stratified_bootstrap_interval(
    differences: np.ndarray,
    regions: list[str],
) -> tuple[float, float]:
    rng = np.random.default_rng(REPLICATION_BOOTSTRAP_SEED)
    region_array = np.asarray(regions)
    indices_by_region = {
        region: np.flatnonzero(region_array == region)
        for region in REGION_PRIOR
    }
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        estimate = 0.0
        for region, weight in REGION_PRIOR.items():
            indices = indices_by_region[region]
            sampled = rng.choice(indices, size=len(indices), replace=True)
            estimate += weight * float(np.mean(differences[sampled]))
        estimates[sample_index] = estimate
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def compile_refresh_models(
    *,
    executor_class: type,
    refreshes: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    refresh_models: dict[str, list[dict[str, Any]]] = {}
    for root in ROOTS:
        models = []
        for refresh in refreshes[root["id"]]:
            means, heldout = simulate_maps(
                executor_class,
                compile_support(refresh["hypotheses"]),
                action_ids=[
                    root["action_id"],
                    refresh["continuation_action"],
                ],
            )
            models.append(
                {
                    **refresh,
                    "prior": support_prior(refresh["hypotheses"]),
                    "root_means": means[root["action_id"]],
                    "continuation_means": means[
                        refresh["continuation_action"]
                    ],
                    "heldout": heldout,
                }
            )
        refresh_models[root["id"]] = models
    return refresh_models


def phase_a_mechanics(
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
        "exact_10_accepted_requests": (
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
        "all_refresh_supports_keep_two_regions": all(
            sum(
                count > 0
                for count in support_diagnostics(
                    refresh["hypotheses"]
                )["region_counts"].values()
            )
            >= 2
            for root_refreshes in refreshes.values()
            for refresh in root_refreshes
        ),
    }
    return gates, {
        "refreshes_changed_from_initial": changed_supports,
        "roots_with_branch_distinct_supports": branch_distinct_roots,
        "center_continuations": center_continuations,
    }


def selection_metrics(
    *,
    initial_prior: np.ndarray,
    initial_action_means: dict[str, np.ndarray],
    initial_heldout: np.ndarray,
    branches: dict[str, list[dict[str, Any]]],
    refresh_models: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    immediate_values = {
        root["id"]: immediate_eig(
            initial_action_means[root["action_id"]],
            initial_prior,
            rng=np.random.default_rng(POLICY_NOISE_SEED),
            samples_per_hypothesis=INTERNAL_ROOT_SAMPLES,
        )
        for root in ROOTS
    }
    internal_risks = {}
    for root in ROOTS:
        per_truth = evaluate_retained_root(
            root_id=root["id"],
            true_action_means=initial_action_means,
            true_heldout=initial_heldout,
            initial_action_means=initial_action_means,
            initial_heldout=initial_heldout,
            branches=branches,
            refresh_models=refresh_models,
            rng=np.random.default_rng(POLICY_NOISE_SEED),
            root_samples=INTERNAL_ROOT_SAMPLES,
            continuation_samples=INTERNAL_CONTINUATION_SAMPLES,
            initial_component_mass=INITIAL_COMPONENT_MASS,
            refresh_component_mass=REFRESH_COMPONENT_MASS,
        )
        internal_risks[root["id"]] = weighted_mean(per_truth, initial_prior)
    myopic_root = min(
        immediate_values,
        key=lambda root_id: (-immediate_values[root_id], root_id),
    )
    lookahead_root = min(
        internal_risks,
        key=lambda root_id: (internal_risks[root_id], root_id),
    )
    internal_reduction = relative_reduction(
        internal_risks[MYOPIC_ROOT_ID],
        internal_risks[LOOKAHEAD_ROOT_ID],
    )
    return {
        "immediate_eig_nats": immediate_values,
        "retained_internal_trajectory_risk": internal_risks,
        "myopic_root": myopic_root,
        "lookahead_root": lookahead_root,
        "internal_risk_reduction": internal_reduction,
    }


def phase_a_selection_gates(selection: dict[str, Any]) -> dict[str, bool]:
    return {
        "myopic_root_is_D": selection["myopic_root"] == MYOPIC_ROOT_ID,
        "lookahead_root_is_B": (
            selection["lookahead_root"] == LOOKAHEAD_ROOT_ID
        ),
        "internal_risk_reduction_at_least_10_percent": (
            selection["internal_risk_reduction"]
            >= MIN_INTERNAL_RISK_REDUCTION
        ),
    }


def run_replication(
    *,
    discoverphysics_root: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    commit = verify_discoverphysics(discoverphysics_root)
    if commit != DISCOVERPHYSICS_COMMIT:
        raise ValueError("DiscoverPhysics commit changed")
    executor_class = load_executor_class(discoverphysics_root)
    adapter = _adapter(run_id=run_id, output_dir=output_dir)
    raw_path = output_dir / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {
        "discarded_preflight": None,
        "initial_support": None,
        "refreshes": [],
    }
    try:
        preflight = adapter.chat_complete_messages_batched_structured(
            [support_messages_v2()],
            temperature=0.0,
            block_size=1,
            response_format=support_response_format(),
            max_new_tokens=INITIAL_MAX_TOKENS,
        )[0]
        raw["discarded_preflight"] = preflight
        checkpoint(raw_path, raw)
        parse_weighted_support(preflight)

        initial_response = adapter.chat_complete_messages_batched_structured(
            [support_messages_v2()],
            temperature=0.0,
            block_size=1,
            response_format=support_response_format(),
            max_new_tokens=INITIAL_MAX_TOKENS,
        )[0]
        raw["initial_support"] = initial_response
        checkpoint(raw_path, raw)
        initial_support = parse_weighted_support(initial_response)
        initial_maps = compile_support(initial_support)
        initial_prior = support_prior(initial_support)
        initial_action_means, initial_heldout = simulate_maps(
            executor_class,
            initial_maps,
            action_ids=list(action_table()),
        )
        branches = build_branches(initial_action_means, initial_prior)
        refresh_requests = [
            refresh_messages(
                initial_support,
                branches,
                root_id=root["id"],
                branch_index=branch_index,
            )
            for root in ROOTS
            for branch_index in range(2)
        ]
        refresh_responses = adapter.chat_complete_messages_batched_structured(
            refresh_requests,
            temperature=0.0,
            block_size=8,
            response_format=refresh_response_format(),
            max_new_tokens=REFRESH_MAX_TOKENS,
        )
        raw["refreshes"] = list(refresh_responses)
        checkpoint(raw_path, raw)
        parsed_flat = [
            parse_refresh(
                response,
                root_action_id=root_by_id(ROOTS[index // 2]["id"])[
                    "action_id"
                ],
                label=f"refresh[{index}]",
            )
            for index, response in enumerate(refresh_responses)
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
    mechanics_gates, mechanism = phase_a_mechanics(
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
    selection_gates = phase_a_selection_gates(selection)
    phase_a_pass = all(mechanics_gates.values()) and all(
        selection_gates.values()
    )
    protocol = {
        "interface_version": INTERFACE_VERSION,
        "discoverphysics_commit": commit,
        "model": MODEL_ID,
        "reasoning_enabled": False,
        "temperature": 0.0,
        "response_format": "chat_strict_json_schema",
        "discarded_preflight_requests": 1,
        "scientific_tree_requests": 9,
        "expected_accepted_requests": EXPECTED_REQUESTS,
        "run_budget_usd": RUN_BUDGET_USD,
        "initial_component_mass": INITIAL_COMPONENT_MASS,
        "refresh_component_mass": REFRESH_COMPONENT_MASS,
        "policy_noise_seed": POLICY_NOISE_SEED,
        "map_seeds": list(REPLICATION_MAP_SEEDS),
        "noise_seed": REPLICATION_NOISE_SEED,
        "bootstrap_seed": REPLICATION_BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "root_samples": REPLICATION_ROOT_SAMPLES,
        "continuation_samples": REPLICATION_CONTINUATION_SAMPLES,
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


def endpoint_metrics(
    *,
    executor_class: type,
    initial_prior: np.ndarray,
    initial_action_means: dict[str, np.ndarray],
    initial_heldout: np.ndarray,
    branches: dict[str, list[dict[str, Any]]],
    refresh_models: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    hidden_maps, hidden_regions, hidden_prior = replication_hidden_map_family()
    required_actions = {
        root["action_id"] for root in ROOTS
    } | {
        refresh["continuation_action"]
        for root_refreshes in refresh_models.values()
        for refresh in root_refreshes
    }
    hidden_action_means, hidden_heldout = simulate_maps(
        executor_class,
        hidden_maps,
        action_ids=sorted(required_actions),
    )
    retained_per_map = {}
    for root_id in {MYOPIC_ROOT_ID, LOOKAHEAD_ROOT_ID, RANDOM_ROOT_ID}:
        retained_per_map[root_id] = evaluate_retained_root(
            root_id=root_id,
            true_action_means=hidden_action_means,
            true_heldout=hidden_heldout,
            initial_action_means=initial_action_means,
            initial_heldout=initial_heldout,
            branches=branches,
            refresh_models=refresh_models,
            rng=np.random.default_rng(REPLICATION_NOISE_SEED),
            root_samples=REPLICATION_ROOT_SAMPLES,
            continuation_samples=REPLICATION_CONTINUATION_SAMPLES,
            initial_component_mass=INITIAL_COMPONENT_MASS,
            refresh_component_mass=REFRESH_COMPONENT_MASS,
        )
    fixed_per_map = evaluate_fixed_support_root(
        root_id=LOOKAHEAD_ROOT_ID,
        true_action_means=hidden_action_means,
        true_heldout=hidden_heldout,
        initial_action_means=initial_action_means,
        initial_heldout=initial_heldout,
        initial_prior=initial_prior,
        branches=branches,
        refresh_models=refresh_models,
        rng=np.random.default_rng(REPLICATION_NOISE_SEED),
        root_samples=REPLICATION_ROOT_SAMPLES,
        continuation_samples=REPLICATION_CONTINUATION_SAMPLES,
    )
    risks = {
        root_id: weighted_mean(values, hidden_prior)
        for root_id, values in retained_per_map.items()
    }
    fixed_risk = weighted_mean(fixed_per_map, hidden_prior)
    myopic_difference = (
        retained_per_map[MYOPIC_ROOT_ID]
        - retained_per_map[LOOKAHEAD_ROOT_ID]
    )
    fixed_difference = (
        fixed_per_map - retained_per_map[LOOKAHEAD_ROOT_ID]
    )
    myopic_ci = stratified_bootstrap_interval(
        myopic_difference,
        hidden_regions,
    )
    fixed_ci = stratified_bootstrap_interval(
        fixed_difference,
        hidden_regions,
    )
    initial_coverage, retained_coverage = retained_coverage_risk(
        root_id=LOOKAHEAD_ROOT_ID,
        hidden_action_means=hidden_action_means,
        hidden_heldout=hidden_heldout,
        hidden_prior=hidden_prior,
        initial_heldout=initial_heldout,
        branches=branches,
        refresh_models=refresh_models,
    )
    hidden_reduction = relative_reduction(
        risks[MYOPIC_ROOT_ID],
        risks[LOOKAHEAD_ROOT_ID],
    )
    random_reduction = relative_reduction(
        risks[RANDOM_ROOT_ID],
        risks[LOOKAHEAD_ROOT_ID],
    )
    fixed_reduction = relative_reduction(
        fixed_risk,
        risks[LOOKAHEAD_ROOT_ID],
    )
    coverage_reduction = relative_reduction(
        initial_coverage,
        retained_coverage,
    )
    gates = {
        "hidden_risk_reduction_at_least_10_percent": (
            hidden_reduction >= MIN_HIDDEN_RISK_REDUCTION
        ),
        "myopic_paired_bootstrap_lower_bound_positive": myopic_ci[0] > 0.0,
        "gain_vs_random_at_least_5_percent": (
            random_reduction >= MIN_RANDOM_RISK_REDUCTION
        ),
        "gain_vs_fixed_support_at_least_1_percent": (
            fixed_reduction >= MIN_FIXED_SUPPORT_RISK_REDUCTION
        ),
        "fixed_paired_bootstrap_lower_bound_positive": fixed_ci[0] > 0.0,
        "retained_coverage_gain_at_least_5_percent": (
            coverage_reduction >= MIN_COVERAGE_RISK_REDUCTION
        ),
    }
    return {
        "num_maps": len(hidden_maps),
        "retained_dynamic_trajectory_mse": risks,
        "fixed_support_center_trajectory_mse": fixed_risk,
        "lookahead_vs_myopic_risk_reduction": hidden_reduction,
        "lookahead_vs_random_risk_reduction": random_reduction,
        "lookahead_vs_fixed_support_risk_reduction": fixed_reduction,
        "myopic_paired_difference_ci95": list(myopic_ci),
        "fixed_paired_difference_ci95": list(fixed_ci),
        "initial_nearest_support_risk": initial_coverage,
        "retained_nearest_support_risk": retained_coverage,
        "coverage_risk_reduction": coverage_reduction,
        "per_map_retained_mse": {
            root_id: values.tolist()
            for root_id, values in retained_per_map.items()
        },
        "per_map_fixed_support_mse": fixed_per_map.tolist(),
        "regions": hidden_regions,
    }, gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discoverphysics-root", type=Path, required=True)
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
