#!/usr/bin/env python3
"""Audit exact two-step BED opportunity in NewtonBench sound speed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
from types import ModuleType
from typing import Callable, Sequence

import numpy as np


SOURCE_REPO_COMMIT = "912a4ba5f4356ddd06acc16e44460ca30be4abc2"
ACTION_SEED = 24366
NUM_ACTIONS = 32
ACTION_BANK_SHA256 = (
    "edc95b9a080f4ab13d2dba0b6cb857e77b837b044b5940ab2b830d338d3b3c9a"
)
NOISE_LEVELS = (0.0001, 0.01, 0.1)
PRIMARY_QUADRATURE_ORDER = 15
CHECK_QUADRATURE_ORDER = 9
MIN_MARGIN_NATS = 0.01
MAX_QUADRATURE_DELTA_NATS = 0.005
ABSOLUTE_NOISE_FLOOR = 1e-9
HYPOTHESIS_IDS = tuple(
    f"{difficulty}:{version}"
    for difficulty in ("easy", "medium", "hard")
    for version in ("v0", "v1", "v2")
)
DEVELOPMENT_HYPOTHESES = (
    "medium:v2",
    "hard:v0",
    "easy:v0",
    "hard:v2",
    "easy:v2",
    "medium:v1",
)
HOLDOUT_HYPOTHESES = ("medium:v0", "hard:v1", "easy:v1")


def compact_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_action_bank(
    *,
    seed: int = ACTION_SEED,
    num_actions: int = NUM_ACTIONS,
) -> list[dict[str, float]]:
    """Reproduce the preregistered Latin-hypercube action bank."""
    rng = random.Random(seed)
    permutations: list[list[int]] = []
    for _ in range(3):
        permutation = list(range(num_actions))
        rng.shuffle(permutation)
        permutations.append(permutation)

    actions: list[dict[str, float]] = []
    for row in range(num_actions):
        unit = [
            (permutations[dimension][row] + rng.random()) / num_actions
            for dimension in range(3)
        ]
        actions.append(
            {
                "adiabatic_index": 1.3 + unit[0] * (1.7 - 1.3),
                "temperature": math.exp(
                    math.log(10.0)
                    + unit[1] * (math.log(1000.0) - math.log(10.0))
                ),
                "molar_mass": math.exp(
                    math.log(0.001)
                    + unit[2] * (math.log(0.1) - math.log(0.001))
                ),
            }
        )
    return actions


def verify_source_repo(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if commit != SOURCE_REPO_COMMIT:
        raise ValueError(
            f"NewtonBench is at {commit}, expected {SOURCE_REPO_COMMIT}"
        )
    return commit


def load_laws_module(repo: Path) -> ModuleType:
    path = repo / "modules" / "m8_sound_speed" / "laws.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(
        "_newtonbench_m8_sound_speed_laws", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ordered_laws(module: ModuleType) -> list[Callable[[float, float, float], float]]:
    registry = module.LAW_REGISTRY
    found = tuple(
        f"{difficulty}:{version}"
        for difficulty in registry
        for version in registry[difficulty]
    )
    if found != HYPOTHESIS_IDS:
        raise ValueError(
            f"sound-speed law registry changed: {found!r} != {HYPOTHESIS_IDS!r}"
        )
    return [
        registry[difficulty][version]
        for difficulty, version in (
            hypothesis.split(":", maxsplit=1) for hypothesis in HYPOTHESIS_IDS
        )
    ]


def prediction_matrix(
    laws: Sequence[Callable[[float, float, float], float]],
    actions: Sequence[dict[str, float]],
) -> np.ndarray:
    means = np.asarray(
        [
            [
                law(
                    action["adiabatic_index"],
                    action["temperature"],
                    action["molar_mass"],
                )
                for action in actions
            ]
            for law in laws
        ],
        dtype=np.float64,
    )
    if means.shape != (len(HYPOTHESIS_IDS), len(actions)):
        raise ValueError(f"unexpected prediction shape {means.shape}")
    if not np.all(np.isfinite(means)) or np.any(means <= 0.0):
        raise ValueError("official laws produced a non-positive or non-finite mean")
    return means


def entropy(probabilities: np.ndarray, *, axis: int = -1) -> np.ndarray:
    logs = np.zeros_like(probabilities, dtype=np.float64)
    np.log(probabilities, out=logs, where=probabilities > 0.0)
    terms = probabilities * logs
    return -np.sum(terms, axis=axis)


def logsumexp(values: np.ndarray, *, axis: int = -1) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    total = np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    result = maximum + np.log(total)
    return np.squeeze(result, axis=axis)


def posterior_probabilities(
    prior: np.ndarray,
    log_likelihoods: np.ndarray,
) -> np.ndarray:
    """Return posteriors for one or many observations."""
    log_prior = np.full_like(prior, -np.inf, dtype=np.float64)
    np.log(prior, out=log_prior, where=prior > 0.0)
    log_joint = log_likelihoods + log_prior
    log_normalizer = logsumexp(log_joint, axis=-1)
    return np.exp(log_joint - log_normalizer[..., None])


def quadrature_likelihood_tables(
    means: np.ndarray,
    *,
    noise_level: float,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-action observation log likelihoods and mixture indices."""
    nodes, raw_weights = np.polynomial.hermite.hermgauss(quadrature_order)
    weights = raw_weights / math.sqrt(math.pi)
    num_hypotheses, num_actions = means.shape
    source_indices = np.repeat(np.arange(num_hypotheses), quadrature_order)
    mixture_weights = np.tile(weights, num_hypotheses)
    tables = np.empty(
        (num_actions, num_hypotheses * quadrature_order, num_hypotheses),
        dtype=np.float64,
    )

    for action_index in range(num_actions):
        action_means = means[:, action_index]
        sigmas = np.maximum(
            np.abs(action_means * noise_level), ABSOLUTE_NOISE_FLOOR
        )
        observations = (
            action_means[:, None]
            + math.sqrt(2.0) * sigmas[:, None] * nodes[None, :]
        ).reshape(-1)
        residuals = (
            observations[:, None] - action_means[None, :]
        ) / sigmas[None, :]
        tables[action_index] = (
            -0.5 * residuals * residuals
            - np.log(sigmas[None, :])
            - 0.5 * math.log(2.0 * math.pi)
        )
    return tables, source_indices, mixture_weights


def expected_entropies_for_priors(
    priors: np.ndarray,
    log_likelihoods: np.ndarray,
    source_indices: np.ndarray,
    mixture_weights: np.ndarray,
) -> np.ndarray:
    """Expected posterior entropy for rows of priors under one action."""
    log_priors = np.full_like(priors, -np.inf, dtype=np.float64)
    np.log(priors, out=log_priors, where=priors > 0.0)
    log_joint = (
        log_priors[:, None, :] + log_likelihoods[None, :, :]
    )
    log_normalizers = logsumexp(log_joint, axis=-1)
    posteriors = np.exp(log_joint - log_normalizers[:, :, None])
    posterior_entropies = entropy(posteriors, axis=-1)
    predictive_weights = (
        priors[:, source_indices] * mixture_weights[None, :]
    )
    return np.sum(predictive_weights * posterior_entropies, axis=-1)


def planning_values(
    means: np.ndarray,
    *,
    noise_level: float,
    quadrature_order: int,
) -> dict[str, object]:
    num_hypotheses, num_actions = means.shape
    prior = np.full(num_hypotheses, 1.0 / num_hypotheses)
    prior_entropy = float(entropy(prior))
    tables, source_indices, mixture_weights = quadrature_likelihood_tables(
        means,
        noise_level=noise_level,
        quadrature_order=quadrature_order,
    )
    first_outcome_weights = (
        prior[source_indices] * mixture_weights
    )
    immediate_values = np.empty(num_actions, dtype=np.float64)
    depth_two_values = np.empty(num_actions, dtype=np.float64)
    continuation_counts = np.zeros((num_actions, num_actions), dtype=np.int64)

    for root_index in range(num_actions):
        first_posteriors = posterior_probabilities(
            prior, tables[root_index]
        )
        first_entropies = entropy(first_posteriors, axis=-1)
        immediate_values[root_index] = prior_entropy - float(
            np.sum(first_outcome_weights * first_entropies)
        )

        terminal_entropies = np.empty(
            (num_actions, first_posteriors.shape[0]), dtype=np.float64
        )
        for continuation_index in range(num_actions):
            terminal_entropies[continuation_index] = (
                expected_entropies_for_priors(
                    first_posteriors,
                    tables[continuation_index],
                    source_indices,
                    mixture_weights,
                )
            )
        best_continuations = np.argmin(terminal_entropies, axis=0)
        continuation_counts[root_index] = np.bincount(
            best_continuations, minlength=num_actions
        )
        expected_terminal_entropy = float(
            np.sum(
                first_outcome_weights
                * terminal_entropies[
                    best_continuations,
                    np.arange(first_posteriors.shape[0]),
                ]
            )
        )
        depth_two_values[root_index] = (
            prior_entropy - expected_terminal_entropy
        )

    greedy_root = select_root(immediate_values, depth_two_values)
    nonmyopic_root = select_root(depth_two_values, immediate_values)
    return {
        "prior_entropy_nats": prior_entropy,
        "immediate_eig_nats": immediate_values.tolist(),
        "depth_two_eig_nats": depth_two_values.tolist(),
        "greedy_root_index": greedy_root,
        "nonmyopic_root_index": nonmyopic_root,
        "immediate_margin_nats": float(
            immediate_values[greedy_root]
            - immediate_values[nonmyopic_root]
        ),
        "depth_two_margin_nats": float(
            depth_two_values[nonmyopic_root]
            - depth_two_values[greedy_root]
        ),
        "continuation_selection_counts": continuation_counts.tolist(),
    }


def select_root(
    primary_values: np.ndarray,
    secondary_values: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> int:
    """Select by primary, secondary, then frozen order with tolerant ties."""
    best_primary = float(np.max(primary_values))
    primary_candidates = np.flatnonzero(
        primary_values >= best_primary - tolerance
    )
    best_secondary = float(np.max(secondary_values[primary_candidates]))
    finalists = primary_candidates[
        secondary_values[primary_candidates] >= best_secondary - tolerance
    ]
    return int(finalists[0])


def strict_opportunity(
    primary: dict[str, object],
    check: dict[str, object],
) -> tuple[bool, list[str], dict[str, float | bool]]:
    greedy = int(primary["greedy_root_index"])
    nonmyopic = int(primary["nonmyopic_root_index"])
    check_greedy = int(check["greedy_root_index"])
    check_nonmyopic = int(check["nonmyopic_root_index"])
    primary_immediate = np.asarray(primary["immediate_eig_nats"])
    check_immediate = np.asarray(check["immediate_eig_nats"])
    primary_depth_two = np.asarray(primary["depth_two_eig_nats"])
    check_depth_two = np.asarray(check["depth_two_eig_nats"])

    diagnostics: dict[str, float | bool] = {
        "roots_differ": greedy != nonmyopic,
        "primary_immediate_margin_nats": float(
            primary["immediate_margin_nats"]
        ),
        "primary_depth_two_margin_nats": float(
            primary["depth_two_margin_nats"]
        ),
        "check_immediate_margin_nats": float(check["immediate_margin_nats"]),
        "check_depth_two_margin_nats": float(check["depth_two_margin_nats"]),
        "roots_stable": (
            greedy == check_greedy and nonmyopic == check_nonmyopic
        ),
        "max_immediate_quadrature_delta_nats": float(
            np.max(np.abs(primary_immediate - check_immediate))
        ),
        "max_depth_two_quadrature_delta_nats": float(
            np.max(np.abs(primary_depth_two - check_depth_two))
        ),
    }
    failures: list[str] = []
    if not diagnostics["roots_differ"]:
        failures.append("greedy and non-myopic roots do not differ")
    if diagnostics["primary_immediate_margin_nats"] < MIN_MARGIN_NATS:
        failures.append("primary immediate-EIG tradeoff is below 0.01 nats")
    if diagnostics["primary_depth_two_margin_nats"] < MIN_MARGIN_NATS:
        failures.append("primary depth-two advantage is below 0.01 nats")
    if diagnostics["check_immediate_margin_nats"] <= 0.0:
        failures.append("immediate-EIG tradeoff changes sign at check order")
    if diagnostics["check_depth_two_margin_nats"] <= 0.0:
        failures.append("depth-two advantage changes sign at check order")
    if not diagnostics["roots_stable"]:
        failures.append("selected roots change with quadrature order")
    if (
        diagnostics["max_immediate_quadrature_delta_nats"]
        > MAX_QUADRATURE_DELTA_NATS
    ):
        failures.append("immediate-EIG quadrature delta exceeds 0.005 nats")
    if (
        diagnostics["max_depth_two_quadrature_delta_nats"]
        > MAX_QUADRATURE_DELTA_NATS
    ):
        failures.append("depth-two-EIG quadrature delta exceeds 0.005 nats")
    return not failures, failures, diagnostics


def run_audit(repo: Path) -> dict[str, object]:
    commit = verify_source_repo(repo)
    actions = generate_action_bank()
    action_hash = compact_json_sha256(actions)
    if action_hash != ACTION_BANK_SHA256:
        raise AssertionError(
            f"action bank hash {action_hash} != {ACTION_BANK_SHA256}"
        )
    laws_module = load_laws_module(repo)
    laws = ordered_laws(laws_module)
    means = prediction_matrix(laws, actions)

    strata: list[dict[str, object]] = []
    for noise_level in NOISE_LEVELS:
        primary = planning_values(
            means,
            noise_level=noise_level,
            quadrature_order=PRIMARY_QUADRATURE_ORDER,
        )
        check = planning_values(
            means,
            noise_level=noise_level,
            quadrature_order=CHECK_QUADRATURE_ORDER,
        )
        passed, failures, diagnostics = strict_opportunity(primary, check)
        strata.append(
            {
                "noise_level": noise_level,
                "strict_opportunity": passed,
                "failures": failures,
                "diagnostics": diagnostics,
                "primary": primary,
                "quadrature_check": check,
            }
        )

    return {
        "schema_version": 1,
        "gate_name": "newtonbench_sound_speed_nonmyopic_opportunity",
        "source": {
            "repository": "HKUST-KnowComp/NewtonBench",
            "commit": commit,
            "module": "m8_sound_speed",
            "system": "vanilla_equation",
        },
        "hypotheses": list(HYPOTHESIS_IDS),
        "development_hypotheses": list(DEVELOPMENT_HYPOTHESES),
        "holdout_hypotheses": list(HOLDOUT_HYPOTHESES),
        "action_seed": ACTION_SEED,
        "action_bank_sha256": action_hash,
        "actions": actions,
        "noise_levels": list(NOISE_LEVELS),
        "primary_quadrature_order": PRIMARY_QUADRATURE_ORDER,
        "check_quadrature_order": CHECK_QUADRATURE_ORDER,
        "minimum_margin_nats": MIN_MARGIN_NATS,
        "maximum_quadrature_delta_nats": MAX_QUADRATURE_DELTA_NATS,
        "prediction_matrix": means.tolist(),
        "strata": strata,
        "gate_passed": any(
            bool(stratum["strict_opportunity"]) for stratum in strata
        ),
        "model_calls": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--newtonbench-repo",
        type=Path,
        required=True,
        help="Pinned checkout of HKUST-KnowComp/NewtonBench",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit(args.newtonbench_repo.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate_passed": result["gate_passed"],
                "model_calls": result["model_calls"],
                "output": str(args.output),
                "strata": [
                    {
                        "noise_level": stratum["noise_level"],
                        "strict_opportunity": stratum[
                            "strict_opportunity"
                        ],
                        "greedy_root_index": stratum["primary"][
                            "greedy_root_index"
                        ],
                        "nonmyopic_root_index": stratum["primary"][
                            "nonmyopic_root_index"
                        ],
                        "immediate_margin_nats": stratum["diagnostics"][
                            "primary_immediate_margin_nats"
                        ],
                        "depth_two_margin_nats": stratum["diagnostics"][
                            "primary_depth_two_margin_nats"
                        ],
                    }
                    for stratum in result["strata"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
