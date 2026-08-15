#!/usr/bin/env python3
"""Run the frozen ChemBench posterior-sampling fidelity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.chembench_mopen.source import _query_assays
from scripts.chembench_adaptive_smc_calibration import (
    ABSOLUTE_LOG_NOISE_FLOOR,
    DIFFICULTIES,
    HISTORY_ASSAY_INDICES,
    NOISE_LEVEL,
    OBSERVATION_SEED_BASE,
    QUERY_SEEDS,
    _array_sha256,
    _evaluate_rates,
    _parameter_sha256,
    _prior_for,
    _prior_sha256,
    _sha256,
    _spearman,
    _stable_seed,
    _truth_parameters,
)
from scripts.chembench_adaptive_smc_v2 import V1_RESULT_PATH, _load_v1_cases
from scripts.chembench_local_tree_branch_fidelity import (
    PANEL_DOMAINS,
    ROOT_HISTORY_INDICES,
    V3_RESULT_PATH,
    V3_RESULT_SHA256,
    _draw_reference_outcomes,
    posterior_risks_for_observations,
)
from scripts.chembench_mopen_nonmyopic_opportunity import (
    frozen_assays,
    load_source,
    verify_source,
)
from scripts.chembench_posterior_state_branch_fidelity import (
    OUTCOME_SEED_BASE,
    PREDECESSOR_RESULT_PATH as COMPONENT_RESULT_PATH,
    PREDECESSOR_RESULT_SHA256 as COMPONENT_RESULT_SHA256,
    _fit_root_bank,
)


SCHEMA_VERSION = "chembench-posterior-sampling-fidelity-v1"
PROTOCOL_PATH = (
    "results/nonmyopic/CHEMBENCH_POSTERIOR_SAMPLING_FIDELITY_PROTOCOL_20260815.md"
)
POOLED_RESULT_PATH = (
    "results/nonmyopic/chembench_posterior_state_branch_fidelity/result.json"
)
POOLED_RESULT_SHA256 = (
    "637c031946268e8015687d5eb4af4d037bfc1399300e56c0e50bdf049b4fab07"
)
SAMPLE_COUNTS = (32, 64, 128, 256)
NUM_REPLICATES = 4
REPLICATE_BLOCK_SIZE = 256
REFERENCE_OUTCOMES = 2048
NUM_TARGETS = 128


def _action_payload(
    reference: np.ndarray,
    estimate: np.ndarray,
    action_indices: Sequence[int],
    root_risk: float,
    component_action_risks: Mapping[str, np.ndarray],
    component_root_risks: Mapping[str, float],
) -> dict[str, Any]:
    if reference.shape != estimate.shape or reference.shape != (len(action_indices),):
        raise ValueError("action risk arrays have incompatible shapes")
    selected = int(np.argmin(estimate))
    regret = max(0.0, float(reference[selected] - np.min(reference)))
    components = {}
    for bank, risks in component_action_risks.items():
        component_regret = max(0.0, float(risks[selected] - np.min(risks)))
        components[bank] = component_regret / component_root_risks[bank]
    return {
        "spearman": _spearman(reference, estimate),
        "selected_action_index": int(action_indices[selected]),
        "reference_best_action_index": int(action_indices[int(np.argmin(reference))]),
        "top_one_regret": regret,
        "normalized_top_one_regret": regret / root_risk,
        "component_bank_regret": components,
        "estimated_action_risks": estimate.tolist(),
    }


def evaluate_case_samples(
    *,
    reference_action_risks: np.ndarray,
    outcome_risks: np.ndarray,
    action_indices: Sequence[int],
    root_risk: float,
    component_action_risks: Mapping[str, np.ndarray],
    component_root_risks: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate all frozen nested sample prefixes for one case."""

    reference = np.asarray(reference_action_risks, dtype=float)
    outcomes = np.asarray(outcome_risks, dtype=float)
    expected_shape = (len(action_indices), REFERENCE_OUTCOMES)
    if outcomes.shape != expected_shape:
        raise ValueError(f"outcome risks must have shape {expected_shape}")
    if root_risk <= 0 or not math.isfinite(root_risk):
        raise ValueError("root risk must be finite and positive")
    component_arrays = {
        bank: np.asarray(values, dtype=float)
        for bank, values in component_action_risks.items()
    }
    if set(component_arrays) != {"bank_1", "bank_2"}:
        raise ValueError("both component banks are required")
    values = (reference, outcomes, *component_arrays.values())
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("sample fidelity inputs must be finite")

    estimates: dict[str, list[dict[str, Any]]] = {}
    for count in SAMPLE_COUNTS:
        estimates[str(count)] = []
        for replicate in range(NUM_REPLICATES):
            start = replicate * REPLICATE_BLOCK_SIZE
            estimate = np.mean(outcomes[:, start : start + count], axis=1)
            estimates[str(count)].append(
                _action_payload(
                    reference,
                    estimate,
                    action_indices,
                    root_risk,
                    component_arrays,
                    component_root_risks,
                )
            )
    ensemble = _action_payload(
        reference,
        np.mean(outcomes[:, : NUM_REPLICATES * REPLICATE_BLOCK_SIZE], axis=1),
        action_indices,
        root_risk,
        component_arrays,
        component_root_risks,
    )
    return {"estimates": estimates, "ensemble_1024": ensemble}


def _summary(
    cases: Sequence[Mapping[str, Any]], count: int, replicate: int | None
) -> dict[str, Any]:
    payloads = (
        [case["ensemble_1024"] for case in cases]
        if replicate is None
        else [case["estimates"][str(count)][replicate] for case in cases]
    )
    correlations = np.asarray([item["spearman"] for item in payloads])
    regrets = np.asarray([item["normalized_top_one_regret"] for item in payloads])
    component = {}
    for bank in ("bank_1", "bank_2"):
        values = np.asarray([item["component_bank_regret"][bank] for item in payloads])
        component[bank] = {
            "fraction_normalized_regret_at_most_003": float(np.mean(values <= 0.03)),
            "mean_normalized_regret": float(np.mean(values)),
        }
    return {
        "median_spearman": float(np.median(correlations)),
        "fraction_spearman_at_least_080": float(np.mean(correlations >= 0.80)),
        "fraction_normalized_regret_at_most_003": float(np.mean(regrets <= 0.03)),
        "mean_normalized_top_one_regret": float(np.mean(regrets)),
        "component_banks": component,
    }


def _summary_pass(summary: Mapping[str, Any]) -> bool:
    return (
        summary["median_spearman"] >= 0.90
        and summary["fraction_spearman_at_least_080"] >= 0.90
        and summary["fraction_normalized_regret_at_most_003"] >= 0.90
        and summary["mean_normalized_top_one_regret"] <= 0.01
        and all(
            values["fraction_normalized_regret_at_most_003"] >= 0.90
            and values["mean_normalized_regret"] <= 0.01
            for values in summary["component_banks"].values()
        )
    )


def evaluate_gates(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = {
        str(count): [
            _summary(cases, count, replicate) for replicate in range(NUM_REPLICATES)
        ]
        for count in SAMPLE_COUNTS
    }
    agreements = {}
    for count in SAMPLE_COUNTS:
        pair_values = []
        for first, second in combinations(range(NUM_REPLICATES), 2):
            pair_values.append(
                float(
                    np.mean(
                        [
                            case["estimates"][str(count)][first][
                                "selected_action_index"
                            ]
                            == case["estimates"][str(count)][second][
                                "selected_action_index"
                            ]
                            for case in cases
                        ]
                    )
                )
            )
        agreements[str(count)] = {
            "pairwise": pair_values,
            "mean": float(np.mean(pair_values)),
        }
    conditions = {
        "all_cases_finite_and_reproducible": len(cases) == 36
        and all(case["finite_and_reproducible"] for case in cases),
        "all_four_256_replicates_pass": all(
            _summary_pass(summary) for summary in summaries["256"]
        ),
    }
    return {
        "summaries": summaries,
        "replicate_selected_action_agreement": agreements,
        "ensemble_1024": _summary(cases, 1024, None),
        "conditions": conditions,
        "pass": all(conditions.values()),
    }


def _component_case_arrays(
    component_cases: Mapping[tuple[str, str, str], Mapping[str, Any]],
    difficulty: str,
    domain: str,
    action_indices: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    action_risks = {}
    root_risks = {}
    for bank in ("bank_1", "bank_2"):
        case = component_cases[(difficulty, domain, bank)]
        by_action = {
            action["assay_index"]: action["reference_expected_risk"]
            for action in case["actions"]
        }
        action_risks[bank] = np.asarray([by_action[index] for index in action_indices])
        root_risks[bank] = float(case["root_risk"])
    return action_risks, root_risks


def run(
    source_root: Path,
    v1_result_path: Path,
    v3_result_path: Path,
    pooled_result_path: Path,
    component_result_path: Path,
    *,
    batch_size: int = 128,
    progress: bool = False,
) -> dict[str, Any]:
    if _sha256(v3_result_path) != V3_RESULT_SHA256:
        raise ValueError("V3 result hash does not match the frozen binding")
    if _sha256(pooled_result_path) != POOLED_RESULT_SHA256:
        raise ValueError("pooled result hash does not match the frozen binding")
    if _sha256(component_result_path) != COMPONENT_RESULT_SHA256:
        raise ValueError("component result hash does not match the frozen binding")
    v3_result = json.loads(v3_result_path.read_text())
    if not v3_result["source_only"]["gates"]["pass"]:
        raise ValueError("V3 source-only posterior gate did not pass")
    pooled_result = json.loads(pooled_result_path.read_text())
    pooled_cases = {
        (case["difficulty"], case["domain"]): case
        for case in pooled_result["cases"]
    }
    component_result = json.loads(component_result_path.read_text())
    component_cases = {
        (case["difficulty"], case["domain"], case["bank"]): case
        for case in component_result["cases"]
    }
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    _, v1_cases = _load_v1_cases(v1_result_path)
    assays = frozen_assays()
    action_indices = tuple(
        index for index in range(len(assays)) if index not in ROOT_HISTORY_INDICES
    )
    root_designs = np.asarray([assays[index].values for index in ROOT_HISTORY_INDICES])
    action_designs = np.asarray([assays[index].values for index in action_indices])
    full_history_designs = np.asarray(
        [assays[index].values for index in HISTORY_ASSAY_INDICES]
    )
    query_designs = {
        difficulty: _query_assays(
            QUERY_SEEDS[difficulty],
            source.CHEM_INPUT_BOUNDS,
            source.CHEM_LOG_VARS,
            512,
        )
        for difficulty in DIFFICULTIES
    }
    cases = []
    started = time.perf_counter()
    for difficulty in DIFFICULTIES:
        for domain_index, domain in enumerate(PANEL_DOMAINS, start=1):
            case_started = time.perf_counter()
            prior = _prior_for(source, domain, difficulty)
            truth = _truth_parameters(source, prior, domain, difficulty, "source_only")
            truth_means = _evaluate_rates(
                source,
                domain,
                (truth,),
                full_history_designs,
                apply_secondary_effects=True,
            )[0]
            rng = np.random.default_rng(
                _stable_seed(OBSERVATION_SEED_BASE, "source_only", difficulty, domain)
            )
            observations = np.maximum(
                0.0, truth_means * (1.0 + NOISE_LEVEL * rng.normal(size=8))
            )
            compatibility = {
                "prior_sha256": _prior_sha256(prior),
                "truth_sha256": _parameter_sha256(truth),
                "history_sha256": _array_sha256(full_history_designs, observations),
                "query_sha256": _array_sha256(query_designs[difficulty]),
            }
            v1_case = v1_cases[("source_only", difficulty, domain)]
            if any(compatibility[name] != v1_case[name] for name in compatibility):
                raise ValueError(f"V1 compatibility failed for {difficulty}/{domain}")

            bank_1 = _fit_root_bank(
                source,
                domain,
                difficulty,
                prior,
                root_designs,
                observations[:4],
                2026083501,
            )
            bank_2 = _fit_root_bank(
                source,
                domain,
                difficulty,
                prior,
                root_designs,
                observations[:4],
                2026083502,
            )
            particles = np.concatenate((bank_1.particles, bank_2.particles), axis=0)
            weights = np.full(len(particles), 1.0 / len(particles))
            particle_hash = _array_sha256(particles, weights)
            predecessor_case = pooled_cases[(difficulty, domain)]
            if particle_hash != predecessor_case["pooled_particle_sha256"]:
                raise ValueError(f"pooled particle hash failed for {difficulty}/{domain}")
            parameters = prior.decode_many(particles)
            action_rates = _evaluate_rates(
                source,
                domain,
                parameters,
                action_designs,
                apply_secondary_effects=True,
            )
            action_means = np.log1p(action_rates)
            action_sigmas = np.maximum(
                NOISE_LEVEL * action_rates / (1.0 + action_rates),
                ABSOLUTE_LOG_NOISE_FLOOR,
            )
            target_values = np.log1p(
                _evaluate_rates(
                    source,
                    domain,
                    parameters,
                    query_designs[difficulty][:NUM_TARGETS],
                    apply_secondary_effects=False,
                )
            )
            saved_by_action = {
                action["assay_index"]: action["reference_expected_risk"]
                for action in predecessor_case["actions"]
            }
            reference = np.asarray([saved_by_action[index] for index in action_indices])
            outcome_risks = np.empty((len(action_indices), REFERENCE_OUTCOMES))
            outcome_digest = hashlib.sha256()
            for local_index, assay_index in enumerate(action_indices):
                outcomes = _draw_reference_outcomes(
                    action_means[:, local_index],
                    action_sigmas[:, local_index],
                    weights,
                    seed=_stable_seed(
                        OUTCOME_SEED_BASE,
                        "pooled",
                        difficulty,
                        domain,
                        str(assay_index),
                    ),
                    count=REFERENCE_OUTCOMES,
                )
                outcome_digest.update(np.asarray(outcomes, dtype=np.float64).tobytes())
                risks = posterior_risks_for_observations(
                    outcomes,
                    action_means[:, local_index],
                    action_sigmas[:, local_index],
                    target_values,
                    weights,
                    batch_size=batch_size,
                )
                if not np.isclose(
                    np.mean(risks), reference[local_index], rtol=1e-12, atol=1e-12
                ):
                    raise ValueError(
                        f"saved reference replay failed for {difficulty}/{domain}/{assay_index}"
                    )
                outcome_risks[local_index] = risks
            component_action_risks, component_root_risks = _component_case_arrays(
                component_cases, difficulty, domain, action_indices
            )
            evaluated = evaluate_case_samples(
                reference_action_risks=reference,
                outcome_risks=outcome_risks,
                action_indices=action_indices,
                root_risk=float(predecessor_case["root_risk"]),
                component_action_risks=component_action_risks,
                component_root_risks=component_root_risks,
            )
            case = {
                "domain": domain,
                "difficulty": difficulty,
                "root_risk": predecessor_case["root_risk"],
                "pooled_particle_sha256": particle_hash,
                "outcome_sha256": outcome_digest.hexdigest(),
                "v1_compatibility": compatibility,
                "finite_and_reproducible": True,
                **evaluated,
            }
            cases.append(case)
            if progress:
                rep = case["estimates"]["256"]
                print(
                    f"{difficulty} {domain_index:02d}/12 {domain} "
                    f"rho256={[round(item['spearman'], 3) for item in rep]} "
                    f"elapsed={time.perf_counter() - case_started:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
    gates = evaluate_gates(cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": {"path": PROTOCOL_PATH, "sha256": _sha256(REPO_ROOT / PROTOCOL_PATH)},
        "v1_result": {"path": V1_RESULT_PATH, "sha256": _sha256(v1_result_path)},
        "v3_result": {"path": V3_RESULT_PATH, "sha256": _sha256(v3_result_path)},
        "pooled_result": {
            "path": POOLED_RESULT_PATH,
            "sha256": _sha256(pooled_result_path),
        },
        "component_result": {
            "path": COMPONENT_RESULT_PATH,
            "sha256": _sha256(component_result_path),
        },
        "source": source_binding,
        "settings": {
            "sample_counts": SAMPLE_COUNTS,
            "num_replicates": NUM_REPLICATES,
            "replicate_block_size": REPLICATE_BLOCK_SIZE,
            "reference_outcomes": REFERENCE_OUTCOMES,
            "num_targets": NUM_TARGETS,
            "outcome_seed_base": OUTCOME_SEED_BASE,
        },
        "gates": gates,
        "cases": cases,
        "elapsed_seconds": time.perf_counter() - started,
        "model_calls": 0,
        "network_calls": 0,
        "cost_usd": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--v1-result", type=Path, default=REPO_ROOT / V1_RESULT_PATH)
    parser.add_argument("--v3-result", type=Path, default=REPO_ROOT / V3_RESULT_PATH)
    parser.add_argument(
        "--pooled-result", type=Path, default=REPO_ROOT / POOLED_RESULT_PATH
    )
    parser.add_argument(
        "--component-result", type=Path, default=REPO_ROOT / COMPONENT_RESULT_PATH
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    result = run(
        args.source_root,
        args.v1_result,
        args.v3_result,
        args.pooled_result,
        args.component_result,
        batch_size=args.batch_size,
        progress=args.progress,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
