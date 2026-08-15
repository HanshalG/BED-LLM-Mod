#!/usr/bin/env python3
"""Audit non-myopic opportunity on standard one-edit ChemBench mechanisms."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.chembench_mopen.factored import (
    FactoredModelBank,
    TypedRegistryOracleProposer,
    registry_signature,
)
from scripts.chembench_factored_mopen_oracle import evaluate_primary
from scripts.chembench_llm_proposal_atlas import build_source_context


SCHEMA_VERSION = "chembench-atomic-edit-opportunity-v1"
STANDARD_CORES = frozenset(
    {"michaelis_menten", "hill", "substrate_inhibition", "pingpong"}
)
STANDARD_MODIFIERS = frozenset(
    {
        "arrhenius",
        "competitive_inhibition",
        "uncompetitive_inhibition",
        "noncompetitive_inhibition",
        "ph_bell_curve",
        "product_inhibition",
    }
)


def is_standard_atomic_successor(bank: FactoredModelBank, candidate: int) -> bool:
    """Return whether a candidate is one standard edit from initial support."""

    if candidate in bank.initial_support:
        return False
    signature = registry_signature(bank.model_names[candidate])
    if signature.core not in STANDARD_CORES:
        return False
    if not set(signature.modifiers).issubset(STANDARD_MODIFIERS):
        return False
    for parent in bank.initial_support:
        parent_signature = registry_signature(bank.model_names[parent])
        core_changed = parent_signature.core != signature.core
        modifier_changes = len(
            set(parent_signature.modifiers).symmetric_difference(signature.modifiers)
        )
        if (not core_changed and modifier_changes == 1) or (
            core_changed and modifier_changes == 0
        ):
            return True
    return False


def reduced_atomic_bank(
    bank: FactoredModelBank,
) -> tuple[FactoredModelBank, tuple[int, ...], tuple[str, ...]]:
    """Restrict a bank to initial support and its standard atomic successors."""

    eligible = tuple(
        model
        for model in range(bank.num_models)
        if is_standard_atomic_successor(bank, model)
    )
    keep = tuple(bank.initial_support) + eligible
    old_to_new = {old: new for new, old in enumerate(keep)}
    reduced = FactoredModelBank(
        bank.likelihoods[np.asarray(keep)],
        bank.target_features[np.asarray(keep)],
        tuple(bank.model_names[index] for index in keep),
        bank.action_names,
        bank.action_groups,
        tuple(old_to_new[index] for index in bank.initial_support),
        outside_prior=bank.outside_prior,
        evidence_slots=bank.evidence_slots,
        diversity_slots=bank.diversity_slots,
        surprise_quantile=bank.surprise_quantile,
    )
    truth_indices = tuple(old_to_new[index] for index in eligible)
    truth_names = tuple(bank.model_names[index] for index in eligible)
    return reduced, truth_indices, truth_names


def compare(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    delta = left_values - right_values
    left_mean = float(np.mean(left_values))
    right_mean = float(np.mean(right_values))
    return {
        "left_mean": left_mean,
        "right_mean": right_mean,
        "absolute_reduction": left_mean - right_mean,
        "relative_reduction": (
            (left_mean - right_mean) / left_mean if left_mean > 0.0 else 0.0
        ),
        "wins": int(np.sum(delta > 1e-12)),
        "ties": int(np.sum(np.abs(delta) <= 1e-12)),
        "losses": int(np.sum(delta < -1e-12)),
    }


def run(source_root: Path) -> dict[str, Any]:
    context = build_source_context(source_root)
    slices = []
    aggregate: dict[str, list[float]] = {"d1": [], "d2": [], "d3": []}
    for slice_index, difficulty in enumerate(("easy", "medium", "hard")):
        bank, truth_indices, truth_names = reduced_atomic_bank(
            context["banks"][difficulty]
        )
        primary, _ = evaluate_primary(
            bank,
            TypedRegistryOracleProposer(bank),
            truth_indices,
            seed=2026083900 + slice_index,
        )
        levels = primary["policy_levels"]
        for level in aggregate:
            aggregate[level].extend(levels[level]["truth_losses"])
        slices.append(
            {
                "difficulty": difficulty,
                "num_truths": len(truth_indices),
                "truth_names": list(truth_names),
                "expected_terminal_mse": {
                    level: float(levels[level]["expected_terminal_mse"])
                    for level in aggregate
                },
                "root_action": {
                    level: levels[level]["root_action"] for level in aggregate
                },
                "truth_losses": {
                    level: [float(value) for value in levels[level]["truth_losses"]]
                    for level in aggregate
                },
            }
        )
    d2_vs_d1 = compare(aggregate["d1"], aggregate["d2"])
    d3_vs_d2 = compare(aggregate["d2"], aggregate["d3"])
    conditions = {
        "d2_has_material_mean_gain": d2_vs_d1["relative_reduction"] >= 0.05
        and d2_vs_d1["absolute_reduction"] > 1e-12,
        "d3_has_material_mean_gain": d3_vs_d2["relative_reduction"] >= 0.05
        and d3_vs_d2["absolute_reduction"] > 1e-12,
        "d2_has_truth_cell_win": d2_vs_d1["wins"] > 0,
        "d3_has_truth_cell_win": d3_vs_d2["wins"] > 0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all(conditions.values()) else "insufficient_horizon_gap",
        "model_calls": 0,
        "cost_usd": 0.0,
        "source": context["source_binding"],
        "eligibility": {
            "standard_cores": sorted(STANDARD_CORES),
            "standard_modifiers": sorted(STANDARD_MODIFIERS),
            "definition": (
                "one modifier add/remove with unchanged core, or one core replacement "
                "with unchanged modifiers, from frozen initial support"
            ),
        },
        "slices": slices,
        "aggregate": {
            "num_truth_cells": len(aggregate["d1"]),
            "mean_terminal_mse": {
                level: float(np.mean(values)) for level, values in aggregate.items()
            },
            "sample_sd_terminal_mse": {
                level: float(np.std(values, ddof=1)) for level, values in aggregate.items()
            },
            "d2_vs_d1": d2_vs_d1,
            "d3_vs_d2": d3_vs_d2,
        },
        "gate": {"passed": all(conditions.values()), "conditions": conditions},
        "interpretation": (
            "The standard atomic subset is saturated by depth two if the d3 truth-cell "
            "win gate fails; it is not suitable for a monotonic d1/d2/d3 efficacy test."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.source_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if not math.isfinite(result["aggregate"]["mean_terminal_mse"]["d1"]):
        raise FloatingPointError("atomic opportunity audit produced non-finite risk")


if __name__ == "__main__":
    main()
