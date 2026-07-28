#!/usr/bin/env python3
"""Run breadth-preserving modular support replication in DiscoverPhysics."""

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
    INTERNAL_CONTINUATION_SAMPLES,
    INTERNAL_ROOT_SAMPLES,
    LOOKAHEAD_ROOT_ID,
    MIN_HIDDEN_RISK_REDUCTION,
    MIN_INTERNAL_RISK_REDUCTION,
    MIN_RANDOM_RISK_REDUCTION,
    MYOPIC_ROOT_ID,
    OBSERVATION_NOISE_STD,
    POLICY_NOISE_SEED,
    RANDOM_ROOT_ID,
    ROOTS,
    action_table,
    build_branches,
    compile_support,
    evaluate_fixed_support_root,
    hidden_halo_family,
    immediate_eig,
    load_executor_class,
    refresh_messages,
    relative_reduction,
    root_by_id,
    simulate_maps,
    support_prior,
    weighted_mean,
)
from scripts.discoverphysics_dark_matter_opportunity import (
    verify_discoverphysics,
)
from scripts.discoverphysics_dark_matter_retained_support_replay import (
    retained_coverage_risk,
    retained_support_full_history_posterior,
)
from scripts.discoverphysics_dark_matter_structured_replication import (
    EXPECTED_REQUESTS,
    INITIAL_COMPONENT_MASS,
    INITIAL_MAX_TOKENS,
    REFRESH_COMPONENT_MASS,
    REFRESH_MAX_TOKENS,
    RUN_BUDGET_USD,
    compile_refresh_models,
    phase_a_mechanics,
    refresh_response_format,
    support_response_format,
)
from scripts.discoverphysics_dark_matter_structured_replication_v3 import (
    _adapter,
)
from scripts.discoverphysics_oscillator_belief_smoke import (
    SmokeExecutionError,
    checkpoint,
    sha256_file,
    strict_json_object,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-balanced-modular-1"
MAP_SEEDS = tuple(range(24720, 24736))
NOISE_SEED = 24736
BOOTSTRAP_SEED = 24737
ROOT_SAMPLES = 8
CONTINUATION_SAMPLES = 4
BOOTSTRAP_SAMPLES = 10_000
MIN_FIXED_SUPPORT_RISK_REDUCTION = 0.01
MIN_COVERAGE_RISK_REDUCTION = 0.05


def balanced_refresh_messages(
    initial_support: list[dict[str, Any]],
    branches: dict[str, list[dict[str, Any]]],
    *,
    root_id: str,
    branch_index: int,
) -> list[dict[str, str]]:
    messages = refresh_messages(
        initial_support,
        branches,
        root_id=root_id,
        branch_index=branch_index,
    )
    messages[-1] = {
        **messages[-1],
        "content": (
            messages[-1]["content"]
            + "\n\nFROZEN_BREADTH_CONSTRAINT: Return exactly two "
            "hypotheses for each of NE, NW, SW, and SE. Use at least "
            "three geometry types across all eight hypotheses. Integer "
            "weights express relative plausibility within each region; "
            "code will project total regional mass to the disclosed prior."
        ),
    }
    return messages


def balance_support_regions(
    hypotheses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    counts = {
        region: sum(item["region"] == region for item in hypotheses)
        for region in REGIONS
    }
    if any(counts[region] != 2 for region in REGIONS):
        raise ValueError(
            f"balanced refresh requires exactly two maps per region: {counts}"
        )
    balanced = [{**hypothesis} for hypothesis in hypotheses]
    for region in REGIONS:
        indices = [
            index
            for index, hypothesis in enumerate(balanced)
            if hypothesis["region"] == region
        ]
        total = sum(balanced[index]["probability"] for index in indices)
        if total <= 0.0:
            raise ValueError(f"balanced refresh has zero mass in {region}")
        for index in indices:
            balanced[index]["probability"] = (
                REGION_PRIOR[region]
                * balanced[index]["probability"]
                / total
            )
    if not np.isclose(
        sum(item["probability"] for item in balanced),
        1.0,
        atol=1e-12,
    ):
        raise ValueError("balanced refresh probabilities do not sum to one")
    return balanced


def parse_balanced_refresh(
    response: str,
    *,
    root_action_id: str,
    label: str,
) -> dict[str, Any]:
    value = strict_json_object(response, label=label)
    if set(value) != {"hypotheses", "continuation_action"}:
        raise ValueError(f"{label} has the wrong fields")
    continuation = value["continuation_action"]
    if continuation not in action_table():
        raise ValueError(f"{label}.continuation_action is invalid")
    if continuation == root_action_id:
        raise ValueError(f"{label} repeats the root action")
    hypotheses = parse_weighted_support(
        json.dumps(
            {"hypotheses": value["hypotheses"]},
            separators=(",", ":"),
        )
    )
    return {
        "hypotheses": balance_support_regions(hypotheses),
        "continuation_action": continuation,
    }


def modular_component_posteriors(
    *,
    initial_branch_prior: np.ndarray,
    refresh_branch_prior: np.ndarray,
    representative_observation: np.ndarray,
    actual_root_observation: np.ndarray,
    initial_root_means: np.ndarray,
    refresh_root_means: np.ndarray,
    continuation_observations: np.ndarray,
    initial_continuation_means: np.ndarray,
    refresh_continuation_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    split = len(initial_branch_prior)
    common = {
        "initial_branch_prior": initial_branch_prior,
        "refresh_branch_prior": refresh_branch_prior,
        "representative_observation": representative_observation,
        "actual_root_observation": actual_root_observation,
        "initial_root_means": initial_root_means,
        "refresh_root_means": refresh_root_means,
        "continuation_observations": continuation_observations,
        "initial_continuation_means": initial_continuation_means,
        "refresh_continuation_means": refresh_continuation_means,
    }
    initial = retained_support_full_history_posterior(
        **common,
        initial_component_mass=1.0,
        refresh_component_mass=0.0,
    )[:, :split]
    refresh = retained_support_full_history_posterior(
        **common,
        initial_component_mass=0.0,
        refresh_component_mass=1.0,
    )[:, split:]
    return initial, refresh


def evaluate_modular_root(
    *,
    root_id: str,
    true_action_means: dict[str, np.ndarray],
    true_heldout: np.ndarray,
    initial_action_means: dict[str, np.ndarray],
    initial_heldout: np.ndarray,
    branches: dict[str, list[dict[str, Any]]],
    refresh_models: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
    root_samples: int,
    continuation_samples: int,
) -> np.ndarray:
    num_truths = len(true_heldout)
    per_truth = np.empty(num_truths)
    root_action = root_by_id(root_id)["action_id"]
    centers = np.asarray(
        [
            branch["representative_final_coordinate"]
            for branch in branches[root_id]
        ]
    )
    root_noise = rng.normal(
        0.0,
        OBSERVATION_NOISE_STD,
        size=(num_truths, root_samples, 2),
    )
    continuation_noise = rng.normal(
        0.0,
        OBSERVATION_NOISE_STD,
        size=(num_truths, root_samples, continuation_samples, 2),
    )
    for truth_index in range(num_truths):
        errors = []
        for root_sample in range(root_samples):
            root_observation = (
                true_action_means[root_action][truth_index]
                + root_noise[truth_index, root_sample]
            )
            branch_index = int(
                np.argmin(
                    np.sum(
                        (centers - root_observation[None, :]) ** 2,
                        axis=-1,
                    )
                )
            )
            branch = branches[root_id][branch_index]
            model = refresh_models[root_id][branch_index]
            continuation = model["continuation_action"]
            continuation_observations = (
                true_action_means[continuation][truth_index]
                + continuation_noise[truth_index, root_sample]
            )
            initial_posterior, refresh_posterior = (
                modular_component_posteriors(
                    initial_branch_prior=np.asarray(
                        branch["posterior_probabilities"]
                    ),
                    refresh_branch_prior=model["prior"],
                    representative_observation=centers[branch_index],
                    actual_root_observation=root_observation,
                    initial_root_means=initial_action_means[root_action],
                    refresh_root_means=model["root_means"],
                    continuation_observations=continuation_observations,
                    initial_continuation_means=initial_action_means[
                        continuation
                    ],
                    refresh_continuation_means=model[
                        "continuation_means"
                    ],
                )
            )
            prediction = (
                INITIAL_COMPONENT_MASS
                * (initial_posterior @ initial_heldout)
                + REFRESH_COMPONENT_MASS
                * (refresh_posterior @ model["heldout"])
            )
            errors.extend(
                np.mean(
                    (
                        prediction
                        - true_heldout[truth_index][None, :]
                    )
                    ** 2,
                    axis=1,
                ).tolist()
            )
        per_truth[truth_index] = float(np.mean(errors))
    return per_truth


def hidden_map_family(
    *,
    map_seeds: tuple[int, ...] = MAP_SEEDS,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    maps = []
    regions = []
    for seed in map_seeds:
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
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(bootstrap_seed)
    region_array = np.asarray(regions)
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        estimate = 0.0
        for region, weight in REGION_PRIOR.items():
            indices = np.flatnonzero(region_array == region)
            sampled = rng.choice(indices, size=len(indices), replace=True)
            estimate += weight * float(np.mean(differences[sampled]))
        estimates[sample_index] = estimate
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


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
    modular_risks = {}
    for root in ROOTS:
        values = evaluate_modular_root(
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
        )
        modular_risks[root["id"]] = weighted_mean(values, initial_prior)
    myopic_root = min(
        immediate_values,
        key=lambda root_id: (-immediate_values[root_id], root_id),
    )
    lookahead_root = min(
        modular_risks,
        key=lambda root_id: (modular_risks[root_id], root_id),
    )
    reduction = relative_reduction(
        modular_risks[MYOPIC_ROOT_ID],
        modular_risks[LOOKAHEAD_ROOT_ID],
    )
    return {
        "immediate_eig_nats": immediate_values,
        "modular_internal_trajectory_risk": modular_risks,
        "myopic_root": myopic_root,
        "lookahead_root": lookahead_root,
        "internal_risk_reduction": reduction,
    }


def endpoint_metrics(
    *,
    executor_class: type,
    initial_prior: np.ndarray,
    initial_action_means: dict[str, np.ndarray],
    initial_heldout: np.ndarray,
    branches: dict[str, list[dict[str, Any]]],
    refresh_models: dict[str, list[dict[str, Any]]],
    map_seeds: tuple[int, ...] = MAP_SEEDS,
    noise_seed: int = NOISE_SEED,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], dict[str, bool]]:
    hidden_maps, hidden_regions, hidden_prior = hidden_map_family(
        map_seeds=map_seeds
    )
    required_actions = {
        root["action_id"] for root in ROOTS
    } | {
        model["continuation_action"]
        for models in refresh_models.values()
        for model in models
    }
    hidden_action_means, hidden_heldout = simulate_maps(
        executor_class,
        hidden_maps,
        action_ids=sorted(required_actions),
    )
    modular_per_map = {}
    for root_id in {MYOPIC_ROOT_ID, LOOKAHEAD_ROOT_ID, RANDOM_ROOT_ID}:
        modular_per_map[root_id] = evaluate_modular_root(
            root_id=root_id,
            true_action_means=hidden_action_means,
            true_heldout=hidden_heldout,
            initial_action_means=initial_action_means,
            initial_heldout=initial_heldout,
            branches=branches,
            refresh_models=refresh_models,
            rng=np.random.default_rng(noise_seed),
            root_samples=ROOT_SAMPLES,
            continuation_samples=CONTINUATION_SAMPLES,
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
        rng=np.random.default_rng(noise_seed),
        root_samples=ROOT_SAMPLES,
        continuation_samples=CONTINUATION_SAMPLES,
    )
    risks = {
        root_id: weighted_mean(values, hidden_prior)
        for root_id, values in modular_per_map.items()
    }
    fixed_risk = weighted_mean(fixed_per_map, hidden_prior)
    myopic_difference = (
        modular_per_map[MYOPIC_ROOT_ID]
        - modular_per_map[LOOKAHEAD_ROOT_ID]
    )
    fixed_difference = (
        fixed_per_map - modular_per_map[LOOKAHEAD_ROOT_ID]
    )
    myopic_ci = stratified_bootstrap_interval(
        myopic_difference,
        hidden_regions,
        bootstrap_seed=bootstrap_seed,
    )
    fixed_ci = stratified_bootstrap_interval(
        fixed_difference,
        hidden_regions,
        bootstrap_seed=bootstrap_seed,
    )
    initial_coverage, refreshed_coverage = retained_coverage_risk(
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
        refreshed_coverage,
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
        "modular_trajectory_mse": risks,
        "fixed_support_center_trajectory_mse": fixed_risk,
        "lookahead_vs_myopic_risk_reduction": hidden_reduction,
        "lookahead_vs_random_risk_reduction": random_reduction,
        "lookahead_vs_fixed_support_risk_reduction": fixed_reduction,
        "myopic_paired_difference_ci95": list(myopic_ci),
        "fixed_paired_difference_ci95": list(fixed_ci),
        "initial_nearest_support_risk": initial_coverage,
        "retained_nearest_support_risk": refreshed_coverage,
        "coverage_risk_reduction": coverage_reduction,
        "per_map_modular_mse": {
            root_id: values.tolist()
            for root_id, values in modular_per_map.items()
        },
        "per_map_fixed_support_mse": fixed_per_map.tolist(),
        "regions": hidden_regions,
    }, gates


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
        initial_prior = support_prior(initial_support)
        initial_action_means, initial_heldout = simulate_maps(
            executor_class,
            compile_support(initial_support),
            action_ids=list(action_table()),
        )
        branches = build_branches(initial_action_means, initial_prior)
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
    mechanics_gates, mechanism = phase_a_mechanics(
        initial_support=initial_support,
        refreshes=refreshes,
        usage=usage,
    )
    balanced_gates = {
        "all_refreshes_have_two_maps_per_region": all(
            support_diagnostics(refresh["hypotheses"])["region_counts"]
            == {region: 2 for region in REGIONS}
            for models in refreshes.values()
            for refresh in models
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
            for models in refreshes.values()
            for refresh in models
            for diagnostics in [support_diagnostics(refresh["hypotheses"])]
        ),
        "all_refreshes_have_three_geometry_types": all(
            len(support_diagnostics(refresh["hypotheses"])["geometries"])
            >= 3
            for models in refreshes.values()
            for refresh in models
        ),
    }
    selection = selection_metrics(
        initial_prior=initial_prior,
        initial_action_means=initial_action_means,
        initial_heldout=initial_heldout,
        branches=branches,
        refresh_models=refresh_models,
    )
    selection_gates = {
        "myopic_root_is_D": selection["myopic_root"] == MYOPIC_ROOT_ID,
        "modular_lookahead_root_is_B": (
            selection["lookahead_root"] == LOOKAHEAD_ROOT_ID
        ),
        "internal_risk_reduction_at_least_10_percent": (
            selection["internal_risk_reduction"]
            >= MIN_INTERNAL_RISK_REDUCTION
        ),
    }
    phase_a_pass = (
        all(mechanics_gates.values())
        and all(balanced_gates.values())
        and all(selection_gates.values())
    )
    protocol = {
        "interface_version": INTERFACE_VERSION,
        "discoverphysics_commit": commit,
        "model": MODEL_ID,
        "reasoning_enabled": False,
        "temperature": 0.0,
        "response_format": "chat_strict_json_schema_default_routing",
        "discarded_preflight_requests": 1,
        "scientific_tree_requests": 9,
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
        "balanced_support_gates": balanced_gates,
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
            "balanced_support_gates": balanced_gates,
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
