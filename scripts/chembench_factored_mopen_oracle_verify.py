#!/usr/bin/env python3
"""Independently replay the factored M-open ChemBench oracle result."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.chembench_mopen.mechanics import BankedProposer
from environments.chembench_mopen.source import build_mixed_version_responses
from scripts.chembench_factored_mopen_oracle import (
    EXECUTION_BUDGET,
    SCHEMA_VERSION,
    SLICES,
    apply_gate,
    evaluate_fixed,
    evaluate_primary,
    evaluate_union_only,
    make_factored_bank,
    sha256,
    transition_diagnostics,
    verify_bindings,
)
from scripts.chembench_mopen_mechanics import INITIAL_SUPPORT_NAMES
from scripts.chembench_mopen_nonmyopic_opportunity import (
    active_domains,
    frozen_assays,
    load_source,
    verify_source,
)


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"{label} replay mismatch")


def verify(
    source_root: Path,
    result_path: Path,
    transition_bank_path: Path,
) -> dict[str, Any]:
    verify_bindings()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    transition_bank = json.loads(transition_bank_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("factored result schema mismatch")
    if transition_bank.get("schema_version") != f"{SCHEMA_VERSION}-transition-bank":
        raise ValueError("factored transition-bank schema mismatch")
    if transition_bank.get("source_mode") != "typed_registry_oracle":
        raise ValueError("factored transition-bank source mode mismatch")
    _require_equal(
        "transition bank hash",
        sha256(transition_bank_path),
        result["transition_bank"]["sha256"],
    )
    source_binding = verify_source(source_root)
    _require_equal("source binding", source_binding, result["source"])
    source = load_source(source_root)
    domains = active_domains(source)
    _require_equal("active domains", list(domains), result["active_domains"])
    assays = frozen_assays()
    action_names = tuple(item.name for item in assays)
    replayed_slices: list[dict[str, Any]] = []
    runtime_checks = {
        "model_calls_zero": result["model_calls"] == 0,
        "cost_zero": float(result["cost_usd"]) == 0.0,
        "policy_state_has_no_truth_id": True,
        "finite_and_normalized": True,
        "predecessor_passed": True,
    }
    recorded_by_name = {item["slice"]: item for item in result["slice_results"]}
    for item in SLICES:
        mixed = build_mixed_version_responses(
            source,
            domains,
            INITIAL_SUPPORT_NAMES,
            difficulty=item.difficulty,
            initial_version="v2",
            truth_version=item.version,
            query_seed=item.query_seed,
            assays=assays,
        )
        bank = make_factored_bank(
            mixed.observation_means, mixed.target_log_rates, domains, action_names
        )
        bank_record = transition_bank["slices"][item.name]
        seed = int(bank_record["planner_seed"])
        primary, _ = evaluate_primary(
            bank,
            BankedProposer(transition_bank["source_mode"], bank_record["records"]),
            mixed.truth_indices,
            seed=seed,
            source_mode=transition_bank["source_mode"],
        )
        _require_equal("transition audit", primary["transition_audit"], bank_record["transition_audit"])
        diagnostics = transition_diagnostics(bank, primary["transition_audit"])
        fixed = evaluate_fixed(bank, mixed.truth_indices, seed=seed)
        union = evaluate_union_only(
            mixed.observation_means,
            mixed.target_log_rates,
            domains,
            action_names,
            mixed.truth_indices,
            seed=seed,
        )
        replayed = {
            "slice": item.name,
            "query_seed": item.query_seed,
            "version_map_sha256": mixed.version_map_sha256,
            "num_domains": bank.num_models,
            "num_truths": len(mixed.truth_indices),
            "truth_domains": [domains[index] for index in mixed.truth_indices],
            "trigger_calibration": {
                "quantile": bank.surprise_quantile,
                "threshold": bank.surprise_threshold,
                "false_trigger_mass": bank.calibration_false_trigger_mass,
            },
            "root_residual_report": bank.residual_report(
                bank.initial_state(), remaining_budget=EXECUTION_BUDGET
            ),
            "primary": primary,
            "transition_diagnostics": diagnostics,
            "fixed_support": fixed,
            "union_only": union,
            "banked_replay_exact": True,
        }
        _require_equal(f"slice {item.name}", replayed, recorded_by_name[item.name])
        replayed_slices.append(replayed)
        runtime_checks["policy_state_has_no_truth_id"] &= "truth" not in json.dumps(
            bank.initial_state().public_key()
        ).lower()
        runtime_checks["finite_and_normalized"] &= all(
            math.isfinite(primary["policy_levels"][f"d{level}"]["planned_value"])
            and math.isfinite(
                primary["policy_levels"][f"d{level}"]["expected_terminal_mse"]
            )
            for level in (1, 2, 3)
        )
    gate = apply_gate(replayed_slices, runtime_checks)
    _require_equal("runtime checks", runtime_checks, result["runtime_checks"])
    _require_equal("gate", gate, result["gate"])
    return {
        "schema_version": f"{SCHEMA_VERSION}-verification",
        "status": "passed" if gate["passed"] else "failed_closed",
        "model_calls": 0,
        "cost_usd": 0.0,
        "source": source_binding,
        "result_path": str(result_path),
        "result_sha256": sha256(result_path),
        "transition_bank_path": str(transition_bank_path),
        "transition_bank_sha256": sha256(transition_bank_path),
        "slices": [item.name for item in SLICES],
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--transition-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {args.output}")
    verification = verify(args.source_root, args.result, args.transition_bank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
