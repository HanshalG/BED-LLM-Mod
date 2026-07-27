#!/usr/bin/env python3
"""Run simulator-grounded lookahead over LLM-generated halo supports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.discoverphysics_dark_matter_executable_support import (
    DISCOVERPHYSICS_COMMIT,
    MODEL_ID,
    REGION_PRIOR,
    compile_hypothesis,
    support_diagnostics,
)
from scripts.discoverphysics_dark_matter_executable_support_v2 import (
    parse_weighted_support,
    support_messages_v2,
)
from scripts.discoverphysics_dark_matter_opportunity import (
    OBSERVATION_NOISE_STD,
    _run_to_times,
    heldout_experiments,
    hidden_halo_family,
    load_executor_class,
    posterior_batch,
    verify_discoverphysics,
)
from scripts.discoverphysics_dark_matter_semantic_smoke import (
    NonReasoningOpenRouterAdapter,
    ROOTS,
    action_table,
)
from scripts.discoverphysics_oscillator_belief_smoke import (
    SmokeExecutionError,
    canonical_text,
    checkpoint,
    entropy,
    sha256_file,
    strict_json_object,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-grounded-policy-1"
SEED = 24509
EXPECTED_REQUESTS = 9
INITIAL_MAX_TOKENS = 3500
REFRESH_MAX_TOKENS = 3200
RUN_BUDGET_USD = 0.25
PROJECTED_COST_USD = 0.14
POLICY_NOISE_SEED = 24510
HIDDEN_MAP_SEEDS = (24520, 24521, 24522, 24523)
HIDDEN_NOISE_SEED = 24524
BOOTSTRAP_SEED = 24525
INTERNAL_ROOT_SAMPLES = 16
INTERNAL_CONTINUATION_SAMPLES = 8
HIDDEN_ROOT_SAMPLES = 8
HIDDEN_CONTINUATION_SAMPLES = 4
BOOTSTRAP_SAMPLES = 10000
MYOPIC_ROOT_ID = "D"
LOOKAHEAD_ROOT_ID = "B"
RANDOM_ROOT_ID = "A"
MIN_INTERNAL_RISK_REDUCTION = 0.10
MIN_HIDDEN_RISK_REDUCTION = 0.10
MIN_RANDOM_RISK_REDUCTION = 0.05
MIN_FIXED_SUPPORT_RISK_REDUCTION = 0.05
MIN_COVERAGE_RISK_REDUCTION = 0.05


def root_by_id(root_id: str) -> dict[str, Any]:
    return next(root for root in ROOTS if root["id"] == root_id)


def compile_support(hypotheses: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [compile_hypothesis(hypothesis) for hypothesis in hypotheses]
    )


def support_prior(hypotheses: list[dict[str, Any]]) -> np.ndarray:
    prior = np.asarray(
        [hypothesis["probability"] for hypothesis in hypotheses],
        dtype=float,
    )
    return prior / prior.sum()


def simulate_maps(
    executor_class: type,
    source_maps: np.ndarray,
    *,
    action_ids: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    executor = executor_class(noise_std=0.0)
    sentinels = np.array(
        [[-20.0, -20.0], [-20.0, 20.0], [20.0, -20.0], [20.0, 20.0]]
    )
    zero_velocities = np.zeros((5, 2))
    actions = action_table()
    action_means = {
        action_id: np.empty((len(source_maps), 2)) for action_id in action_ids
    }
    heldout = []
    for map_index, source_map in enumerate(source_maps):
        executor._dark_positions_rel = np.asarray(source_map)
        for action_id in action_ids:
            probe_positions = np.vstack(
                [np.asarray(actions[action_id]), sentinels]
            )
            result = _run_to_times(
                executor,
                probe_positions=probe_positions,
                probe_velocities=zero_velocities,
                measurement_times=[0.5],
            )
            action_means[action_id][map_index] = result[0, 0]
        map_heldout = []
        for experiment in heldout_experiments():
            result = _run_to_times(
                executor,
                probe_positions=np.asarray(experiment["probe_positions"]),
                probe_velocities=np.asarray(
                    experiment["probe_velocities"]
                ),
                measurement_times=experiment["measurement_times"],
            )
            map_heldout.append(result.reshape(-1))
        heldout.append(np.concatenate(map_heldout))
    return action_means, np.asarray(heldout)


def weighted_two_means(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.sum(
        (values[:, None, :] - values[None, :, :]) ** 2,
        axis=-1,
    )
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    centers = np.asarray([values[first], values[second]], dtype=float)
    assignments = np.full(len(values), -1, dtype=int)
    for _ in range(50):
        new_assignments = np.argmin(
            np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=-1),
            axis=1,
        )
        if np.array_equal(assignments, new_assignments):
            assignments = new_assignments
            break
        assignments = new_assignments
        for cluster in range(2):
            mask = assignments == cluster
            if not np.any(mask):
                raise ValueError("two-means produced an empty branch")
            centers[cluster] = np.average(
                values[mask],
                axis=0,
                weights=weights[mask],
            )
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    inverse = np.empty(2, dtype=int)
    inverse[order] = np.arange(2)
    return centers[order], inverse[assignments]


def build_branches(
    initial_means: dict[str, np.ndarray],
    prior: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    branches = {}
    for root in ROOTS:
        means = initial_means[root["action_id"]]
        centers, assignments = weighted_two_means(means, prior)
        root_branches = []
        for branch_index in range(2):
            branch_probability = float(prior[assignments == branch_index].sum())
            representative = centers[branch_index]
            posterior = posterior_batch(
                prior[None, :],
                representative[None, :],
                means,
                OBSERVATION_NOISE_STD,
            )[0, 0]
            root_branches.append(
                {
                    "index": branch_index,
                    "probability": branch_probability,
                    "representative_final_coordinate": representative.tolist(),
                    "posterior_probabilities": posterior.tolist(),
                }
            )
        branches[root["id"]] = root_branches
    return branches


def parse_refresh(
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
    support = parse_weighted_support(
        json.dumps(
            {"hypotheses": value["hypotheses"]},
            separators=(",", ":"),
        )
    )
    return {
        "hypotheses": support,
        "continuation_action": continuation,
    }


def refresh_messages(
    initial_support: list[dict[str, Any]],
    branches: dict[str, list[dict[str, Any]]],
    *,
    root_id: str,
    branch_index: int,
) -> list[dict[str, str]]:
    root = root_by_id(root_id)
    branch = branches[root_id][branch_index]
    schema = {
        "hypotheses": [
            {
                "description": "fresh semantic hidden-halo map",
                "weight": 40,
                "region": "NE|NW|SW|SE",
                "center": [3.5, 3.5],
                "geometry": "compact|radial|tangential|elliptical",
                "major_spread": 1.2,
                "minor_spread": 0.4,
                "orientation_degrees": 45.0,
            }
        ],
        "continuation_action": "one ID from ACTION_TABLE",
    }
    return [
        {
            "role": "system",
            "content": (
                "You regenerate executable scientific hypotheses after a "
                "probe observation. Return one exact JSON object only, "
                "without markdown, comments, reasoning, or extra fields."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    "An unknown 2D field contains ten concealed positive",
                    "sources forming one compact or elongated halo.",
                    "Region prior: NE .40, NW .30, SW .20, SE .10.",
                    "A neutral probe starts at rest and reports its noisy",
                    "final coordinate at t=.5. One different probe remains.",
                    "The final goal is held-out trajectory prediction to t=5.",
                    "",
                    "INITIAL_EXECUTABLE_SUPPORT="
                    + json.dumps(initial_support, separators=(",", ":")),
                    "ROOT="
                    + json.dumps(root, separators=(",", ":")),
                    "REPRESENTATIVE_OBSERVED_FINAL_COORDINATE="
                    + json.dumps(
                        branch["representative_final_coordinate"],
                        separators=(",", ":"),
                    ),
                    "POSTERIOR_OVER_INITIAL_SUPPORT="
                    + json.dumps(
                        branch["posterior_probabilities"],
                        separators=(",", ":"),
                    ),
                    "ACTION_TABLE="
                    + json.dumps(action_table(), separators=(",", ":")),
                    "",
                    "Generate exactly eight fresh executable halo hypotheses",
                    "conditioned on this complete history. Use positive integer",
                    "weights; exact code normalizes them. Preserve plausible",
                    "alternatives rather than collapsing to one map. Cover at",
                    "least two regions and use at least two geometry types.",
                    "Choose the single best different continuation action for",
                    "reducing final held-out trajectory error.",
                    "Use the same center/spread/orientation validity ranges as",
                    "the initial support. Return exactly:",
                    json.dumps(schema, separators=(",", ":")),
                ]
            ),
        },
    ]


def _adapter(
    *,
    run_id: str,
    output_dir: Path,
) -> NonReasoningOpenRouterAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=140.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=8,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=INITIAL_MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    spec = ModelSpec(
        model=MODEL_ID,
        backend="openrouter",
        max_model_len=65536,
    )
    return NonReasoningOpenRouterAdapter(spec, config)


def immediate_eig(
    means: np.ndarray,
    prior: np.ndarray,
    *,
    rng: np.random.Generator,
    samples_per_hypothesis: int,
) -> float:
    noise = rng.normal(
        0.0,
        OBSERVATION_NOISE_STD,
        size=(len(prior), samples_per_hypothesis, means.shape[-1]),
    )
    observations = (means[:, None, :] + noise).reshape(-1, means.shape[-1])
    posteriors = posterior_batch(
        prior[None, :],
        observations,
        means,
        OBSERVATION_NOISE_STD,
    )[0]
    weights = np.repeat(prior / samples_per_hypothesis, samples_per_hypothesis)
    return float(entropy(prior) - np.sum(weights * _entropy_rows(posteriors)))


def _entropy_rows(probabilities: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(
            probabilities > 0.0,
            probabilities * np.log(probabilities),
            0.0,
        )
    return -np.sum(terms, axis=-1)


def _posterior_from_two(
    prior: np.ndarray,
    first_observation: np.ndarray,
    first_means: np.ndarray,
    second_observations: np.ndarray,
    second_means: np.ndarray,
) -> np.ndarray:
    first_ll = (
        -0.5
        * np.sum((first_means - first_observation[None, :]) ** 2, axis=-1)
        / OBSERVATION_NOISE_STD**2
    )
    second_ll = (
        -0.5
        * np.sum(
            (
                second_observations[:, None, :]
                - second_means[None, :, :]
            )
            ** 2,
            axis=-1,
        )
        / OBSERVATION_NOISE_STD**2
    )
    logits = np.log(prior[None, :] + 1e-300) + first_ll[None, :] + second_ll
    logits -= logits.max(axis=-1, keepdims=True)
    posterior = np.exp(logits)
    return posterior / posterior.sum(axis=-1, keepdims=True)


def evaluate_dynamic_root(
    *,
    root_id: str,
    true_action_means: dict[str, np.ndarray],
    true_heldout: np.ndarray,
    true_prior: np.ndarray,
    branches: dict[str, list[dict[str, Any]]],
    refresh_models: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
    root_samples: int,
    continuation_samples: int,
) -> np.ndarray:
    num_truths = len(true_prior)
    per_truth = np.zeros(num_truths)
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
            first_observation = (
                true_action_means[root_action][truth_index]
                + root_noise[truth_index, root_sample]
            )
            branch_index = int(
                np.argmin(
                    np.sum(
                        (centers - first_observation[None, :]) ** 2,
                        axis=-1,
                    )
                )
            )
            model = refresh_models[root_id][branch_index]
            continuation = model["continuation_action"]
            second_observations = (
                true_action_means[continuation][truth_index]
                + continuation_noise[truth_index, root_sample]
            )
            posteriors = posterior_batch(
                model["prior"][None, :],
                second_observations,
                model["continuation_means"],
                OBSERVATION_NOISE_STD,
            )[0]
            predictions = posteriors @ model["heldout"]
            errors.extend(
                np.mean(
                    (predictions - true_heldout[truth_index][None, :]) ** 2,
                    axis=-1,
                ).tolist()
            )
        per_truth[truth_index] = float(np.mean(errors))
    return per_truth


def evaluate_fixed_support_root(
    *,
    root_id: str,
    true_action_means: dict[str, np.ndarray],
    true_heldout: np.ndarray,
    initial_action_means: dict[str, np.ndarray],
    initial_heldout: np.ndarray,
    initial_prior: np.ndarray,
    branches: dict[str, list[dict[str, Any]]],
    refresh_models: dict[str, list[dict[str, Any]]],
    rng: np.random.Generator,
    root_samples: int,
    continuation_samples: int,
) -> np.ndarray:
    num_truths = len(true_heldout)
    per_truth = np.zeros(num_truths)
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
            first_observation = (
                true_action_means[root_action][truth_index]
                + root_noise[truth_index, root_sample]
            )
            branch_index = int(
                np.argmin(
                    np.sum(
                        (centers - first_observation[None, :]) ** 2,
                        axis=-1,
                    )
                )
            )
            continuation = refresh_models[root_id][branch_index][
                "continuation_action"
            ]
            second_observations = (
                true_action_means[continuation][truth_index]
                + continuation_noise[truth_index, root_sample]
            )
            posteriors = _posterior_from_two(
                initial_prior,
                first_observation,
                initial_action_means[root_action],
                second_observations,
                initial_action_means[continuation],
            )
            predictions = posteriors @ initial_heldout
            errors.extend(
                np.mean(
                    (predictions - true_heldout[truth_index][None, :]) ** 2,
                    axis=-1,
                ).tolist()
            )
        per_truth[truth_index] = float(np.mean(errors))
    return per_truth


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights))


def relative_reduction(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline if baseline > 0.0 else 0.0


def stratified_bootstrap_difference(
    differences: np.ndarray,
    regions: list[str],
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    region_indices = {
        region: np.flatnonzero(np.asarray(regions) == region)
        for region in REGION_PRIOR
    }
    estimates = np.empty(BOOTSTRAP_SAMPLES)
    for sample_index in range(BOOTSTRAP_SAMPLES):
        estimate = 0.0
        for region, region_weight in REGION_PRIOR.items():
            indices = region_indices[region]
            sampled = rng.choice(indices, size=len(indices), replace=True)
            estimate += region_weight * float(np.mean(differences[sampled]))
        estimates[sample_index] = estimate
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def hidden_map_family() -> tuple[np.ndarray, list[str], np.ndarray]:
    maps = []
    regions = []
    for seed in HIDDEN_MAP_SEEDS:
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


def support_change_key(hypotheses: list[dict[str, Any]]) -> frozenset[str]:
    return frozenset(
        canonical_text(hypothesis["description"])
        for hypothesis in hypotheses
    )


def coverage_risk(
    *,
    root_id: str,
    hidden_action_means: dict[str, np.ndarray],
    hidden_heldout: np.ndarray,
    hidden_prior: np.ndarray,
    initial_heldout: np.ndarray,
    branches: dict[str, list[dict[str, Any]]],
    refresh_models: dict[str, list[dict[str, Any]]],
) -> tuple[float, float]:
    root_action = root_by_id(root_id)["action_id"]
    centers = np.asarray(
        [
            branch["representative_final_coordinate"]
            for branch in branches[root_id]
        ]
    )
    initial_errors = []
    refreshed_errors = []
    for index in range(len(hidden_heldout)):
        branch_index = int(
            np.argmin(
                np.sum(
                    (
                        centers
                        - hidden_action_means[root_action][index][None, :]
                    )
                    ** 2,
                    axis=-1,
                )
            )
        )
        refreshed_heldout = refresh_models[root_id][branch_index]["heldout"]
        initial_errors.append(
            float(
                np.min(
                    np.mean(
                        (
                            initial_heldout
                            - hidden_heldout[index][None, :]
                        )
                        ** 2,
                        axis=-1,
                    )
                )
            )
        )
        refreshed_errors.append(
            float(
                np.min(
                    np.mean(
                        (
                            refreshed_heldout
                            - hidden_heldout[index][None, :]
                        )
                        ** 2,
                        axis=-1,
                    )
                )
            )
        )
    return (
        weighted_mean(np.asarray(initial_errors), hidden_prior),
        weighted_mean(np.asarray(refreshed_errors), hidden_prior),
    )


def run_policy(
    *,
    discoverphysics_root: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    commit = verify_discoverphysics(discoverphysics_root)
    if commit != DISCOVERPHYSICS_COMMIT:
        raise ValueError("DiscoverPhysics commit changed after verification")
    executor_class = load_executor_class(discoverphysics_root)
    adapter = _adapter(run_id=run_id, output_dir=output_dir)
    raw_path = output_dir / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {"initial_support": None, "refreshes": []}
    try:
        initial_response = adapter.chat_complete_messages_batched(
            [support_messages_v2()],
            temperature=0.0,
            block_size=1,
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
        refresh_responses = adapter.chat_complete_messages_batched(
            refresh_requests,
            temperature=0.0,
            block_size=8,
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
    refresh_models: dict[str, list[dict[str, Any]]] = {}
    for root in ROOTS:
        models = []
        for refresh in refreshes[root["id"]]:
            maps = compile_support(refresh["hypotheses"])
            means, heldout = simulate_maps(
                executor_class,
                maps,
                action_ids=[refresh["continuation_action"]],
            )
            models.append(
                {
                    **refresh,
                    "prior": support_prior(refresh["hypotheses"]),
                    "continuation_means": means[
                        refresh["continuation_action"]
                    ],
                    "heldout": heldout,
                }
            )
        refresh_models[root["id"]] = models

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
        per_truth = evaluate_dynamic_root(
            root_id=root["id"],
            true_action_means=initial_action_means,
            true_heldout=initial_heldout,
            true_prior=initial_prior,
            branches=branches,
            refresh_models=refresh_models,
            rng=np.random.default_rng(POLICY_NOISE_SEED),
            root_samples=INTERNAL_ROOT_SAMPLES,
            continuation_samples=INTERNAL_CONTINUATION_SAMPLES,
        )
        internal_risks[root["id"]] = weighted_mean(
            per_truth,
            initial_prior,
        )
    myopic_root = min(
        immediate_values,
        key=lambda root_id: (-immediate_values[root_id], root_id),
    )
    lookahead_root = min(
        internal_risks,
        key=lambda root_id: (internal_risks[root_id], root_id),
    )

    frozen_path = output_dir / "MODEL_FROZEN.json"
    checkpoint(
        frozen_path,
        {
            "raw_responses_sha256": sha256_file(raw_path),
            "initial_support": initial_support,
            "branches": branches,
            "refreshes": refreshes,
            "immediate_eig_nats": immediate_values,
            "internal_trajectory_risk": internal_risks,
            "myopic_root": myopic_root,
            "lookahead_root": lookahead_root,
        },
    )

    hidden_maps, hidden_regions, hidden_prior = hidden_map_family()
    required_hidden_actions = {
        root["action_id"] for root in ROOTS
    } | {
        refresh["continuation_action"]
        for root_refreshes in refreshes.values()
        for refresh in root_refreshes
    }
    hidden_action_means, hidden_heldout = simulate_maps(
        executor_class,
        hidden_maps,
        action_ids=sorted(required_hidden_actions),
    )
    dynamic_per_map = {}
    for root_id in {MYOPIC_ROOT_ID, LOOKAHEAD_ROOT_ID, RANDOM_ROOT_ID}:
        dynamic_per_map[root_id] = evaluate_dynamic_root(
            root_id=root_id,
            true_action_means=hidden_action_means,
            true_heldout=hidden_heldout,
            true_prior=hidden_prior,
            branches=branches,
            refresh_models=refresh_models,
            rng=np.random.default_rng(HIDDEN_NOISE_SEED),
            root_samples=HIDDEN_ROOT_SAMPLES,
            continuation_samples=HIDDEN_CONTINUATION_SAMPLES,
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
        rng=np.random.default_rng(HIDDEN_NOISE_SEED),
        root_samples=HIDDEN_ROOT_SAMPLES,
        continuation_samples=HIDDEN_CONTINUATION_SAMPLES,
    )
    hidden_risks = {
        root_id: weighted_mean(values, hidden_prior)
        for root_id, values in dynamic_per_map.items()
    }
    fixed_risk = weighted_mean(fixed_per_map, hidden_prior)
    difference = (
        dynamic_per_map[MYOPIC_ROOT_ID]
        - dynamic_per_map[LOOKAHEAD_ROOT_ID]
    )
    bootstrap_ci = stratified_bootstrap_difference(
        difference,
        hidden_regions,
    )
    initial_coverage_risk, refreshed_coverage_risk = coverage_risk(
        root_id=LOOKAHEAD_ROOT_ID,
        hidden_action_means=hidden_action_means,
        hidden_heldout=hidden_heldout,
        hidden_prior=hidden_prior,
        initial_heldout=initial_heldout,
        branches=branches,
        refresh_models=refresh_models,
    )

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
    usage = adapter.usage_snapshot()
    internal_reduction = relative_reduction(
        internal_risks[MYOPIC_ROOT_ID],
        internal_risks[LOOKAHEAD_ROOT_ID],
    )
    hidden_reduction = relative_reduction(
        hidden_risks[MYOPIC_ROOT_ID],
        hidden_risks[LOOKAHEAD_ROOT_ID],
    )
    random_reduction = relative_reduction(
        hidden_risks[RANDOM_ROOT_ID],
        hidden_risks[LOOKAHEAD_ROOT_ID],
    )
    fixed_reduction = relative_reduction(
        fixed_risk,
        hidden_risks[LOOKAHEAD_ROOT_ID],
    )
    coverage_reduction = relative_reduction(
        initial_coverage_risk,
        refreshed_coverage_risk,
    )
    mechanics_gates = {
        "exact_9_requests": usage["adapter_requests"] == EXPECTED_REQUESTS,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "at_least_6_refreshes_change_support": changed_supports >= 6,
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
    scientific_gates = {
        "simulator_myopic_root_is_northeast_target": (
            myopic_root == MYOPIC_ROOT_ID
        ),
        "simulator_lookahead_root_is_center_scout": (
            lookahead_root == LOOKAHEAD_ROOT_ID
        ),
        "internal_risk_reduction_at_least_10_percent": (
            internal_reduction >= MIN_INTERNAL_RISK_REDUCTION
        ),
        "hidden_risk_reduction_at_least_10_percent": (
            hidden_reduction >= MIN_HIDDEN_RISK_REDUCTION
        ),
        "hidden_paired_bootstrap_lower_bound_positive": (
            bootstrap_ci[0] > 0.0
        ),
        "hidden_gain_vs_random_at_least_5_percent": (
            random_reduction >= MIN_RANDOM_RISK_REDUCTION
        ),
        "hidden_gain_vs_fixed_support_at_least_5_percent": (
            fixed_reduction >= MIN_FIXED_SUPPORT_RISK_REDUCTION
        ),
        "refreshed_coverage_risk_reduction_at_least_5_percent": (
            coverage_reduction >= MIN_COVERAGE_RISK_REDUCTION
        ),
    }
    all_gates_pass = all(mechanics_gates.values()) and all(
        scientific_gates.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all_gates_pass else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "discoverphysics_commit": commit,
            "model": MODEL_ID,
            "reasoning_enabled": False,
            "seed": SEED,
            "expected_requests": EXPECTED_REQUESTS,
            "run_budget_usd": RUN_BUDGET_USD,
            "projected_cost_usd": PROJECTED_COST_USD,
            "policy_noise_seed": POLICY_NOISE_SEED,
            "hidden_map_seeds": list(HIDDEN_MAP_SEEDS),
            "hidden_noise_seed": HIDDEN_NOISE_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "internal_root_samples": INTERNAL_ROOT_SAMPLES,
            "internal_continuation_samples": (
                INTERNAL_CONTINUATION_SAMPLES
            ),
            "hidden_root_samples": HIDDEN_ROOT_SAMPLES,
            "hidden_continuation_samples": (
                HIDDEN_CONTINUATION_SAMPLES
            ),
            "myopic_root_id": MYOPIC_ROOT_ID,
            "lookahead_root_id": LOOKAHEAD_ROOT_ID,
            "random_root_id": RANDOM_ROOT_ID,
        },
        "selection": {
            "immediate_eig_nats": immediate_values,
            "internal_trajectory_risk": internal_risks,
            "myopic_root": myopic_root,
            "lookahead_root": lookahead_root,
            "internal_risk_reduction": internal_reduction,
        },
        "hidden_endpoint": {
            "num_maps": len(hidden_maps),
            "dynamic_trajectory_mse": hidden_risks,
            "fixed_support_center_trajectory_mse": fixed_risk,
            "lookahead_vs_myopic_risk_reduction": hidden_reduction,
            "lookahead_vs_random_risk_reduction": random_reduction,
            "lookahead_vs_fixed_support_risk_reduction": fixed_reduction,
            "paired_difference_ci95": list(bootstrap_ci),
            "initial_nearest_support_trajectory_mse": (
                initial_coverage_risk
            ),
            "refreshed_nearest_support_trajectory_mse": (
                refreshed_coverage_risk
            ),
            "coverage_risk_reduction": coverage_reduction,
            "per_map_dynamic_mse": {
                root_id: values.tolist()
                for root_id, values in dynamic_per_map.items()
            },
            "per_map_fixed_support_mse": fixed_per_map.tolist(),
            "regions": hidden_regions,
        },
        "mechanism": {
            "refreshes_changed_from_initial": changed_supports,
            "roots_with_branch_distinct_supports": branch_distinct_roots,
            "center_continuations": center_continuations,
        },
        "mechanics_gates": mechanics_gates,
        "scientific_gates": scientific_gates,
        "all_gates_pass": all_gates_pass,
        "usage": usage,
        "raw_responses_sha256": sha256_file(raw_path),
        "model_frozen_sha256": sha256_file(frozen_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discoverphysics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "POLICY.json"
    failure_path = args.output_dir / "FAILURE.json"
    try:
        payload = run_policy(
            discoverphysics_root=args.discoverphysics_root.resolve(),
            output_dir=args.output_dir.resolve(),
            run_id=args.run_id,
        )
    except SmokeExecutionError as exc:
        checkpoint(
            failure_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed",
                "interface_version": INTERFACE_VERSION,
                "error": str(exc),
                "usage": exc.usage,
                "policy_endpoint_accessed": False,
            },
        )
        raise
    checkpoint(result_path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection": payload["selection"],
                "hidden_endpoint": {
                    key: value
                    for key, value in payload["hidden_endpoint"].items()
                    if not key.startswith("per_map") and key != "regions"
                },
                "mechanism": payload["mechanism"],
                "mechanics_gates": payload["mechanics_gates"],
                "scientific_gates": payload["scientific_gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
