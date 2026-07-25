#!/usr/bin/env python3
"""Audit exact two-step BED opportunity in NewtonBench Snell's law."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from types import ModuleType
from typing import Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.newtonbench_sound_speed_opportunity_audit import (
    ABSOLUTE_NOISE_FLOOR,
    CHECK_QUADRATURE_ORDER,
    HYPOTHESIS_IDS,
    MAX_QUADRATURE_DELTA_NATS,
    MIN_MARGIN_NATS,
    NOISE_LEVELS,
    PRIMARY_QUADRATURE_ORDER,
    SOURCE_REPO_COMMIT,
    compact_json_sha256,
    planning_values,
    strict_opportunity,
    verify_source_repo,
)


ACTION_SEED = 24367
NUM_ACTIONS = 32
ACTION_BANK_SHA256 = (
    "4b38f1249b3a4753304eb569c4f43f1e2109b2d796832b64550861d20f97d440"
)
DEVELOPMENT_HYPOTHESES = (
    "medium:v0",
    "medium:v2",
    "hard:v2",
    "easy:v0",
    "hard:v0",
    "hard:v1",
)
HOLDOUT_HYPOTHESES = ("medium:v1", "easy:v2", "easy:v1")


def generate_action_bank(
    *,
    seed: int = ACTION_SEED,
    num_actions: int = NUM_ACTIONS,
) -> list[dict[str, float]]:
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
                "refractive_index_1": 1.0 + 0.5 * unit[0],
                "refractive_index_2": 1.0 + 0.5 * unit[1],
                "incidence_angle": 90.0 * unit[2],
            }
        )
    return actions


def load_laws_module(repo: Path) -> ModuleType:
    path = repo / "modules" / "m4_snell_law" / "laws.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(
        "_newtonbench_m4_snell_laws", path
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
            f"Snell law registry changed: {found!r} != {HYPOTHESIS_IDS!r}"
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
                    action["refractive_index_1"],
                    action["refractive_index_2"],
                    action["incidence_angle"],
                )
                for action in actions
            ]
            for law in laws
        ],
        dtype=np.float64,
    )
    if means.shape != (len(HYPOTHESIS_IDS), len(actions)):
        raise ValueError(f"unexpected prediction shape {means.shape}")
    if np.any(np.isinf(means)):
        raise ValueError("official laws produced an infinite mean")
    finite = np.isfinite(means)
    if np.any(means[finite] < 0.0):
        raise ValueError("official laws produced a negative finite angle")
    return means


def mixed_likelihood_tables(
    means: np.ndarray,
    *,
    noise_level: float,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate finite Gaussian angles and categorical invalid outcomes."""
    nodes, raw_weights = np.polynomial.hermite.hermgauss(quadrature_order)
    weights = raw_weights / math.sqrt(math.pi)
    num_hypotheses, num_actions = means.shape
    source_indices = np.repeat(np.arange(num_hypotheses), quadrature_order)
    mixture_weights = np.tile(weights, num_hypotheses)
    tables = np.full(
        (num_actions, num_hypotheses * quadrature_order, num_hypotheses),
        -np.inf,
        dtype=np.float64,
    )

    for action_index in range(num_actions):
        action_means = means[:, action_index]
        finite_targets = np.isfinite(action_means)
        sigmas = np.full(num_hypotheses, np.nan, dtype=np.float64)
        sigmas[finite_targets] = np.maximum(
            np.abs(action_means[finite_targets] * noise_level),
            ABSOLUTE_NOISE_FLOOR,
        )
        for source_index in range(num_hypotheses):
            rows = slice(
                source_index * quadrature_order,
                (source_index + 1) * quadrature_order,
            )
            block = tables[action_index, rows, :]
            if not finite_targets[source_index]:
                block[:, ~finite_targets] = 0.0
                continue
            observations = (
                action_means[source_index]
                + math.sqrt(2.0) * sigmas[source_index] * nodes
            )
            residuals = (
                observations[:, None] - action_means[finite_targets][None, :]
            ) / sigmas[finite_targets][None, :]
            block[:, finite_targets] = (
                -0.5 * residuals * residuals
                - np.log(sigmas[finite_targets][None, :])
                - 0.5 * math.log(2.0 * math.pi)
            )
    return tables, source_indices, mixture_weights


def run_audit(repo: Path) -> dict[str, object]:
    commit = verify_source_repo(repo)
    actions = generate_action_bank()
    action_hash = compact_json_sha256(actions)
    if action_hash != ACTION_BANK_SHA256:
        raise AssertionError(
            f"action bank hash {action_hash} != {ACTION_BANK_SHA256}"
        )
    laws = ordered_laws(load_laws_module(repo))
    means = prediction_matrix(laws, actions)

    strata: list[dict[str, object]] = []
    for noise_level in NOISE_LEVELS:
        primary = planning_values(
            means,
            noise_level=noise_level,
            quadrature_order=PRIMARY_QUADRATURE_ORDER,
            likelihood_table_builder=mixed_likelihood_tables,
        )
        check = planning_values(
            means,
            noise_level=noise_level,
            quadrature_order=CHECK_QUADRATURE_ORDER,
            likelihood_table_builder=mixed_likelihood_tables,
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
        "gate_name": "newtonbench_snell_nonmyopic_opportunity",
        "source": {
            "repository": "HKUST-KnowComp/NewtonBench",
            "commit": commit,
            "module": "m4_snell_law",
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
        "prediction_matrix": [
            [None if not math.isfinite(value) else value for value in row]
            for row in means.tolist()
        ],
        "invalid_hypothesis_counts_by_action": (
            np.sum(~np.isfinite(means), axis=0).astype(int).tolist()
        ),
        "strata": strata,
        "gate_passed": any(
            bool(stratum["strict_opportunity"]) for stratum in strata
        ),
        "model_calls": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--newtonbench-repo", type=Path, required=True)
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
                "invalid_hypothesis_counts_by_action": result[
                    "invalid_hypothesis_counts_by_action"
                ],
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
