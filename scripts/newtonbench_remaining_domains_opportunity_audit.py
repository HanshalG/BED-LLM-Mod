#!/usr/bin/env python3
"""Audit two-step BED opportunity across the remaining NewtonBench domains."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from types import ModuleType
from typing import Callable, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.newtonbench_sound_speed_opportunity_audit import (
    ABSOLUTE_NOISE_FLOOR,
    CHECK_QUADRATURE_ORDER,
    HYPOTHESIS_IDS,
    NOISE_LEVELS,
    PRIMARY_QUADRATURE_ORDER,
    compact_json_sha256,
    planning_values,
    strict_opportunity,
)


SOURCE_REPO_COMMIT = "912a4ba5f4356ddd06acc16e44460ca30be4abc2"
DEVELOPMENT_SEED_BASE = 24420
CONFIRMATION_SEED_BASE = 24520
NUM_ACTIONS = 32

DOMAIN_SPECS: dict[str, dict[str, object]] = {
    "m0_gravity": {
        "module_index": 0,
        "parameters": (
            ("mass1", 1.0, 1e3, "log"),
            ("mass2", 1.0, 1e3, "log"),
            ("distance", 1.0, 1e1, "log"),
        ),
    },
    "m1_coulomb_force": {
        "module_index": 1,
        "parameters": (
            ("q1", 1e-1, 1e1, "log"),
            ("q2", 1e-1, 1e1, "log"),
            ("distance", 1e-1, 1e1, "log"),
        ),
    },
    "m2_magnetic_force": {
        "module_index": 2,
        "parameters": (
            ("current1", 1e-3, 1e-1, "log"),
            ("current2", 1e-3, 1e-1, "log"),
            ("distance", 1e-3, 1e-1, "log"),
        ),
    },
    "m3_fourier_law": {
        "module_index": 3,
        "parameters": (
            ("k", 1e-1, 1e1, "log"),
            ("A", 1e-4, 1e-2, "log"),
            ("delta_T", 1e1, 1e3, "log"),
            ("d", 1e-2, 1.0, "log"),
        ),
    },
    "m5_radioactive_decay": {
        "module_index": 5,
        "parameters": (
            ("N0", 1.0, 1e2, "log"),
            ("lambda_constant", 1e-3, 1e-1, "log"),
            ("t", 1e-2, 1e1, "log"),
        ),
    },
    "m6_underdamped_harmonic": {
        "module_index": 6,
        "parameters": (
            ("k", 1e2, 1e4, "log"),
            ("m", 1e-1, 1e1, "log"),
            ("b", 1e-2, 1.0, "log"),
        ),
    },
    "m7_malus_law": {
        "module_index": 7,
        "parameters": (
            ("I_0", 1e2, 2e3, "log"),
            ("theta", 1e-6, math.pi / 2.0, "linear"),
        ),
    },
    "m9_hooke_law": {
        "module_index": 9,
        "parameters": (("x", 1e-3, 1.0, "log"),),
    },
    "m10_be_distribution": {
        "module_index": 10,
        "parameters": (
            ("omega", 1e8, 1e10, "log"),
            ("T", 1e1, 1e3, "log"),
        ),
    },
    "m11_heat_transfer": {
        "module_index": 11,
        "parameters": (
            ("m", 1e-3, 1e3, "log"),
            ("c", 1e2, 1e4, "log"),
            ("delta_T", 1e1, 1e3, "log"),
        ),
    },
}


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


def generate_action_bank(
    parameters: Sequence[tuple[str, float, float, str]],
    *,
    seed: int,
    num_actions: int = NUM_ACTIONS,
) -> list[dict[str, float]]:
    """Generate the frozen Latin-hypercube bank for one domain."""
    rng = random.Random(seed)
    permutations: list[list[int]] = []
    for _ in parameters:
        permutation = list(range(num_actions))
        rng.shuffle(permutation)
        permutations.append(permutation)

    actions: list[dict[str, float]] = []
    for row in range(num_actions):
        action: dict[str, float] = {}
        for dimension, (name, lower, upper, scale) in enumerate(parameters):
            unit = (
                permutations[dimension][row] + rng.random()
            ) / num_actions
            if scale == "log":
                value = math.exp(
                    math.log(lower)
                    + unit * (math.log(upper) - math.log(lower))
                )
            elif scale == "linear":
                value = lower + unit * (upper - lower)
            else:
                raise ValueError(f"unsupported parameter scale {scale!r}")
            action[name] = value
        actions.append(action)
    return actions


def generate_action_manifest() -> dict[str, object]:
    domains: dict[str, object] = {}
    for domain, raw_spec in DOMAIN_SPECS.items():
        module_index = int(raw_spec["module_index"])
        parameters = raw_spec["parameters"]
        development = generate_action_bank(
            parameters, seed=DEVELOPMENT_SEED_BASE + module_index
        )
        confirmation = generate_action_bank(
            parameters, seed=CONFIRMATION_SEED_BASE + module_index
        )
        domains[domain] = {
            "module_index": module_index,
            "parameters": [list(parameter) for parameter in parameters],
            "development_seed": DEVELOPMENT_SEED_BASE + module_index,
            "development_action_bank_sha256": compact_json_sha256(development),
            "development_actions": development,
            "confirmation_seed": CONFIRMATION_SEED_BASE + module_index,
            "confirmation_action_bank_sha256": compact_json_sha256(confirmation),
            "confirmation_actions": confirmation,
        }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_repository": "HKUST-KnowComp/NewtonBench",
        "source_commit": SOURCE_REPO_COMMIT,
        "num_actions": NUM_ACTIONS,
        "domains": domains,
    }
    manifest["manifest_sha256"] = compact_json_sha256(manifest)
    return manifest


def load_action_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = generate_action_manifest()
    if manifest != expected:
        raise ValueError("action manifest does not match the frozen generator")
    return manifest


def load_laws_module(repo: Path, domain: str) -> ModuleType:
    path = repo / "modules" / domain / "laws.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(
        f"_newtonbench_remaining_{domain}_laws", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ordered_laws(module: ModuleType) -> list[Callable[..., float]]:
    registry = module.LAW_REGISTRY
    found = tuple(
        f"{difficulty}:{version}"
        for difficulty in ("easy", "medium", "hard")
        for version in ("v0", "v1", "v2")
        if difficulty in registry and version in registry[difficulty]
    )
    if found != HYPOTHESIS_IDS:
        raise ValueError(
            f"law registry changed: {found!r} != {HYPOTHESIS_IDS!r}"
        )
    return [
        registry[difficulty][version]
        for difficulty, version in (
            hypothesis.split(":", maxsplit=1) for hypothesis in HYPOTHESIS_IDS
        )
    ]


def prediction_matrix(
    laws: Sequence[Callable[..., float]],
    actions: Sequence[dict[str, float]],
) -> np.ndarray:
    means = np.asarray(
        [
            [float(law(**action)) for action in actions]
            for law in laws
        ],
        dtype=np.float64,
    )
    if means.shape != (len(HYPOTHESIS_IDS), len(actions)):
        raise ValueError(f"unexpected prediction shape {means.shape}")
    if np.any(np.isinf(means)):
        raise ValueError("official law produced an infinite mean")
    return means


def mixed_quadrature_likelihood_tables(
    means: np.ndarray,
    *,
    noise_level: float,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build likelihood tables with an exact category for non-finite outputs."""
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
        finite = np.isfinite(action_means)
        sigmas = np.maximum(
            np.abs(action_means * noise_level), ABSOLUTE_NOISE_FLOOR
        )
        for source_index in range(num_hypotheses):
            row_slice = slice(
                source_index * quadrature_order,
                (source_index + 1) * quadrature_order,
            )
            row_block = tables[action_index, row_slice, :]
            if not finite[source_index]:
                row_block[:, ~finite] = 0.0
                continue
            observations = (
                action_means[source_index]
                + math.sqrt(2.0) * sigmas[source_index] * nodes
            )
            residuals = (
                observations[:, None] - action_means[None, finite]
            ) / sigmas[None, finite]
            row_block[:, finite] = (
                -0.5 * residuals * residuals
                - np.log(sigmas[None, finite])
                - 0.5 * math.log(2.0 * math.pi)
            )
    return tables, source_indices, mixture_weights


def evaluate_bank(
    means: np.ndarray,
) -> list[dict[str, object]]:
    strata: list[dict[str, object]] = []
    for noise_level in NOISE_LEVELS:
        primary = planning_values(
            means,
            noise_level=noise_level,
            quadrature_order=PRIMARY_QUADRATURE_ORDER,
            likelihood_table_builder=mixed_quadrature_likelihood_tables,
        )
        check = planning_values(
            means,
            noise_level=noise_level,
            quadrature_order=CHECK_QUADRATURE_ORDER,
            likelihood_table_builder=mixed_quadrature_likelihood_tables,
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
    return strata


def select_development_candidate(
    domains: Sequence[dict[str, object]],
) -> tuple[str, float] | None:
    candidates: list[tuple[float, float, int, int, str, float]] = []
    noise_order = {noise: index for index, noise in enumerate(NOISE_LEVELS)}
    for domain_result in domains:
        domain = str(domain_result["domain"])
        module_index = int(domain_result["module_index"])
        for stratum in domain_result["development_strata"]:
            if not bool(stratum["strict_opportunity"]):
                continue
            diagnostics = stratum["diagnostics"]
            noise = float(stratum["noise_level"])
            candidates.append(
                (
                    -float(diagnostics["primary_depth_two_margin_nats"]),
                    -float(diagnostics["primary_immediate_margin_nats"]),
                    module_index,
                    noise_order[noise],
                    domain,
                    noise,
                )
            )
    if not candidates:
        return None
    selected = min(candidates)
    return selected[4], selected[5]


def run_audit(
    repo: Path,
    manifest_path: Path,
) -> dict[str, object]:
    commit = verify_source_repo(repo)
    manifest = load_action_manifest(manifest_path)
    manifest_domains = manifest["domains"]
    domain_results: list[dict[str, object]] = []

    for domain, raw_spec in DOMAIN_SPECS.items():
        domain_manifest = manifest_domains[domain]
        actions = domain_manifest["development_actions"]
        laws = ordered_laws(load_laws_module(repo, domain))
        means = prediction_matrix(laws, actions)
        domain_results.append(
            {
                "domain": domain,
                "module_index": int(raw_spec["module_index"]),
                "development_action_bank_sha256": domain_manifest[
                    "development_action_bank_sha256"
                ],
                "finite_prediction_count": int(np.isfinite(means).sum()),
                "invalid_prediction_count": int((~np.isfinite(means)).sum()),
                "development_prediction_matrix": means.tolist(),
                "development_strata": evaluate_bank(means),
            }
        )

    selected = select_development_candidate(domain_results)
    confirmation: dict[str, object] | None = None
    gate_passed = False
    if selected is not None:
        selected_domain, selected_noise = selected
        domain_manifest = manifest_domains[selected_domain]
        laws = ordered_laws(load_laws_module(repo, selected_domain))
        actions = domain_manifest["confirmation_actions"]
        means = prediction_matrix(laws, actions)
        all_strata = evaluate_bank(means)
        selected_stratum = next(
            stratum
            for stratum in all_strata
            if float(stratum["noise_level"]) == selected_noise
        )
        gate_passed = bool(selected_stratum["strict_opportunity"])
        confirmation = {
            "domain": selected_domain,
            "noise_level": selected_noise,
            "confirmation_action_bank_sha256": domain_manifest[
                "confirmation_action_bank_sha256"
            ],
            "finite_prediction_count": int(np.isfinite(means).sum()),
            "invalid_prediction_count": int((~np.isfinite(means)).sum()),
            "confirmation_prediction_matrix": means.tolist(),
            "selected_stratum": selected_stratum,
        }

    return {
        "schema_version": 1,
        "gate_name": "newtonbench_remaining_domains_nonmyopic_opportunity",
        "source": {
            "repository": "HKUST-KnowComp/NewtonBench",
            "commit": commit,
            "system": "vanilla_equation",
        },
        "action_manifest_path": str(manifest_path),
        "action_manifest_sha256": manifest["manifest_sha256"],
        "hypotheses": list(HYPOTHESIS_IDS),
        "noise_levels": list(NOISE_LEVELS),
        "primary_quadrature_order": PRIMARY_QUADRATURE_ORDER,
        "check_quadrature_order": CHECK_QUADRATURE_ORDER,
        "domains": domain_results,
        "selected_development_candidate": (
            None
            if selected is None
            else {"domain": selected[0], "noise_level": selected[1]}
        ),
        "confirmation": confirmation,
        "gate_passed": gate_passed,
        "model_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write-manifest",
        type=Path,
        help="write action banks without loading any law output",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="run the frozen opportunity audit",
    )
    parser.add_argument("--newtonbench-repo", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_manifest is not None:
        write_json(args.write_manifest.resolve(), generate_action_manifest())
        return
    if (
        args.newtonbench_repo is None
        or args.manifest is None
        or args.output is None
    ):
        raise SystemExit("--run requires --newtonbench-repo, --manifest, and --output")
    result = run_audit(
        args.newtonbench_repo.resolve(),
        args.manifest.resolve(),
    )
    write_json(args.output.resolve(), result)


if __name__ == "__main__":
    main()
