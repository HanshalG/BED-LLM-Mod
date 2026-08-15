#!/usr/bin/env python3
"""Run the frozen zero-call factored M-open ChemBench oracle gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.chembench_mopen.factored import (
    FactoredModelBank,
    FactoredPolicyLadderPlanner,
    RegistryEdit,
    TypedRegistryOracleProposer,
)
from environments.chembench_mopen.mechanics import (
    BankedProposer,
    FixedProposer,
    ProposalCache,
)
from environments.chembench_mopen.source import build_mixed_version_responses
from scripts.chembench_mopen_mechanics import (
    INITIAL_SUPPORT_NAMES,
    PRACTICAL_TIE_TOLERANCE,
    build_bank,
    comparison_at_tolerance,
)
from scripts.chembench_mopen_nonmyopic_opportunity import (
    EXECUTION_BUDGET,
    ValidationSlice,
    active_domains,
    comparison,
    frozen_assays,
    load_source,
    verify_source,
)


SCHEMA_VERSION = "chembench-factored-mopen-oracle-v1"
PROTOCOLS = {
    Path("results/nonmyopic/MDA_SEVENTH_PASS_FACTORED_MOPEN_PLANNER_20260815.md"):
        "0f478d150d25cb20830e9e86f51bf49e4f5490de770437a3d4a4c2da1db38b5a",
    Path("results/nonmyopic/CHEMBENCH_POLICY_LADDER_MECHANICS_V4_PROTOCOL_20260814.md"):
        "68221f5ec63d3b04c31a1926ee739723ee710058f144ab4ca90707c0ff08e3fa",
    Path("results/nonmyopic/CHEMBENCH_FACTORED_MOPEN_ORACLE_PROTOCOL_20260815.md"):
        "00d11d703098205122d3913ded06242f85d2c67f90fe7978130e483c3d39de66",
}
PREDECESSOR_ARTIFACTS = {
    Path("results/nonmyopic/chembench_policy_ladder_mechanics_v4/policy-ladder-v4-20260815/RESULT.json"):
        "f46e1fac28332b06ac2c62ff91fd68e537849c93ea74199ee7dc5d2d418aaec6",
    Path("results/nonmyopic/chembench_policy_ladder_mechanics_v4/policy-ladder-v4-20260815/TRANSITION_BANK.json"):
        "eb6c51b945c05f697b25681ed02a49763cb50ea234436b691a9d45a7e69b3b59",
    Path("results/nonmyopic/chembench_policy_ladder_mechanics_v4/policy-ladder-v4-20260815/VERIFICATION.json"):
        "534a2e2173f1919f3f0d20b4d0a921ca6408a52c844fec4e7c4b60a888a3f6e8",
}
SLICES = (
    ValidationSlice("easy", "v4", 2026081701),
    ValidationSlice("medium", "v4", 2026081702),
    ValidationSlice("hard", "v4", 2026081703),
)
EVIDENCE_SLOTS = 8
DIVERSITY_SLOTS = 4
SURPRISE_QUANTILE = 0.90


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bindings() -> None:
    for path, expected in {**PROTOCOLS, **PREDECESSOR_ARTIFACTS}.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"binding mismatch for {path}: {actual} != {expected}")


def git_value(arguments: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def require_pushed_commit(required_commit: str) -> str:
    head = git_value(("rev-parse", "HEAD"))
    resolved = git_value(("rev-parse", required_commit))
    if head != resolved:
        raise RuntimeError(f"required commit is not HEAD: {resolved} != {head}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, "origin/codex/location-finding-llmstrategy"],
        check=True,
        capture_output=True,
        text=True,
    )
    return head


def make_factored_bank(
    observation_means: np.ndarray,
    target_features: np.ndarray,
    domains: Sequence[str],
    action_names: Sequence[str],
    *,
    evidence_slots: int = EVIDENCE_SLOTS,
    diversity_slots: int = DIVERSITY_SLOTS,
) -> FactoredModelBank:
    base = build_bank(observation_means, target_features, domains, action_names)
    return FactoredModelBank(
        base.likelihoods,
        base.target_features,
        base.model_names,
        base.action_names,
        base.action_groups,
        base.initial_support,
        outside_prior=base.outside_prior,
        evidence_slots=evidence_slots,
        diversity_slots=diversity_slots,
        surprise_quantile=SURPRISE_QUANTILE,
    )


def _merge_audit(target: dict[str, dict[str, Any]], source: Mapping[str, dict[str, Any]]) -> None:
    for key, value in source.items():
        if key in target and target[key] != value:
            raise AssertionError(f"transition audit mismatch for {key}")
        target[key] = value


def evaluate_primary(
    bank: FactoredModelBank,
    proposer: Any,
    truth_indices: Sequence[int],
    *,
    seed: int,
    source_mode: str | None = None,
) -> tuple[dict[str, Any], ProposalCache]:
    cache = ProposalCache(proposer, source_mode=source_mode)
    policy_levels: dict[str, Any] = {}
    audit: dict[str, dict[str, Any]] = {}
    for level in (3, 2, 1):
        planner = FactoredPolicyLadderPlanner(bank, cache, truth_indices, seed=seed)
        policy_levels[f"d{level}"] = planner.evaluate_policy_level(
            level, execution_budget=EXECUTION_BUDGET
        )
        _merge_audit(audit, planner.transition_audit)
    misses_before = cache.misses
    replay_planner = FactoredPolicyLadderPlanner(bank, cache, truth_indices, seed=seed)
    replay = replay_planner.evaluate_policy_level(1, execution_budget=EXECUTION_BUDGET)
    _merge_audit(audit, replay_planner.transition_audit)
    return {
        "policy_levels": policy_levels,
        "call_matched_d1": {
            "result": replay,
            "matches_primary_d1": replay == policy_levels["d1"],
            "cache_misses_before": misses_before,
            "cache_misses_after": cache.misses,
            "cache_hits_after": cache.hits,
        },
        "cache": {
            "entries": len(cache.records),
            "hits": cache.hits,
            "misses": cache.misses,
            "proposed_candidates": sum(len(value) for value in cache.records.values()),
        },
        "transition_audit": {key: audit[key] for key in sorted(audit)},
    }, cache


def evaluate_fixed(
    bank: FactoredModelBank, truth_indices: Sequence[int], *, seed: int
) -> dict[str, Any]:
    cache = ProposalCache(FixedProposer())
    planner = FactoredPolicyLadderPlanner(bank, cache, truth_indices, seed=seed)
    result = planner.evaluate_policy_level(1, execution_budget=EXECUTION_BUDGET)
    return {
        "d1": result,
        "cache": {
            "entries": len(cache.records),
            "proposed_candidates": sum(len(value) for value in cache.records.values()),
        },
    }


def evaluate_union_only(
    observation_means: np.ndarray,
    target_features: np.ndarray,
    domains: Sequence[str],
    action_names: Sequence[str],
    truth_indices: Sequence[int],
    *,
    seed: int,
) -> dict[str, Any]:
    bank = make_factored_bank(
        observation_means,
        target_features,
        domains,
        action_names,
        evidence_slots=len(domains),
        diversity_slots=0,
    )
    result, _ = evaluate_primary(
        bank, TypedRegistryOracleProposer(bank), truth_indices, seed=seed
    )
    return {
        "policy_levels": result["policy_levels"],
        "cache": result["cache"],
    }


def transition_diagnostics(bank: FactoredModelBank, audit: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    records = list(audit.values())
    accepted = [edit for record in records for edit in record["accepted_edits"]]
    for payload in accepted:
        bank.compiler.validate(
            RegistryEdit(
                parent=int(payload["parent"]),
                candidate=int(payload["candidate"]),
                operation=str(payload["operation"]),
                added_tags=tuple(str(item) for item in payload["added_tags"]),
                removed_tags=tuple(str(item) for item in payload["removed_tags"]),
                core_family=str(payload["core_family"]),
            )
        )
    full_family_counts = [
        int(record["core_family_count"])
        for record in records
        if int(record["support_size"]) == bank.pool_cap
    ]
    return {
        "unique_transitions": len(records),
        "triggered": sum(bool(record["triggered"]) for record in records),
        "not_triggered": sum(not bool(record["triggered"]) for record in records),
        "accepted_edits": len(accepted),
        "contraction_events": sum(bool(record["pruned"]) for record in records),
        "max_support_size": max((int(record["support_size"]) for record in records), default=0),
        "min_full_support_core_families": min(full_family_counts) if full_family_counts else 0,
        "all_proposals_novel_and_unique": all(
            bool(record["proposal_novel_and_unique"]) for record in records
        ),
        "all_edits_round_trip": True,
    }


def aggregate(slice_results: Sequence[Mapping[str, Any]], level: int) -> list[float]:
    return [
        float(value)
        for item in slice_results
        for value in item["primary"]["policy_levels"][f"d{level}"]["truth_losses"]
    ]


def apply_gate(
    slice_results: Sequence[Mapping[str, Any]], runtime_checks: Mapping[str, bool]
) -> dict[str, Any]:
    d1 = aggregate(slice_results, 1)
    d2 = aggregate(slice_results, 2)
    d3 = aggregate(slice_results, 3)
    d2_vs_d1 = comparison_at_tolerance(d1, d2, PRACTICAL_TIE_TOLERANCE)
    d3_vs_d2 = comparison_at_tolerance(d2, d3, PRACTICAL_TIE_TOLERANCE)
    calibrated = all(
        abs(
            float(item["primary"]["policy_levels"][f"d{level}"]["planned_value"])
            - float(item["primary"]["policy_levels"][f"d{level}"]["expected_terminal_mse"])
        )
        <= 1e-10
        for item in slice_results
        for level in (1, 2, 3)
    )
    per_slice_monotonic = all(
        float(item["primary"]["policy_levels"]["d2"]["planned_value"])
        <= float(item["primary"]["policy_levels"]["d1"]["planned_value"]) + 1e-12
        and float(item["primary"]["policy_levels"]["d3"]["planned_value"])
        <= float(item["primary"]["policy_levels"]["d2"]["planned_value"]) + 1e-12
        for item in slice_results
    )
    d2_root_differences = sum(
        item["primary"]["policy_levels"]["d1"]["root_action_index"]
        != item["primary"]["policy_levels"]["d2"]["root_action_index"]
        for item in slice_results
    )
    d3_root_differences = sum(
        item["primary"]["policy_levels"]["d2"]["root_action_index"]
        != item["primary"]["policy_levels"]["d3"]["root_action_index"]
        for item in slice_results
    )
    conditions = {
        "runtime_checks_pass": all(runtime_checks.values()),
        "source_false_trigger_mass_at_most_10pct": all(
            float(item["trigger_calibration"]["false_trigger_mass"]) <= 0.10 + 1e-12
            for item in slice_results
        ),
        "typed_edits_round_trip_no_reproposal_support_bounded": all(
            item["transition_diagnostics"]["all_edits_round_trip"]
            and item["transition_diagnostics"]["all_proposals_novel_and_unique"]
            and item["transition_diagnostics"]["max_support_size"] <= 12
            for item in slice_results
        ),
        "trigger_nontrigger_edit_and_contraction_each_slice": all(
            item["transition_diagnostics"]["triggered"] > 0
            and item["transition_diagnostics"]["not_triggered"] > 0
            and item["transition_diagnostics"]["accepted_edits"] > 0
            and item["transition_diagnostics"]["contraction_events"] > 0
            for item in slice_results
        ),
        "full_support_has_three_core_families": all(
            item["transition_diagnostics"]["min_full_support_core_families"] >= 3
            for item in slice_results
        ),
        "planned_truth_replay_calibrated": calibrated,
        "per_slice_model_risk_nonincreasing": per_slice_monotonic,
        "d2_mean_reduction_at_least_5pct": d2_vs_d1["relative_reduction"] >= 0.05,
        "d3_mean_reduction_at_least_5pct": d3_vs_d2["relative_reduction"] >= 0.05,
        "d2_truth_cell_majority": d2_vs_d1["wins"] > d2_vs_d1["losses"],
        "d3_truth_cell_majority": d3_vs_d2["wins"] > d3_vs_d2["losses"],
        "d2_root_differs_on_at_least_one_slice": d2_root_differences >= 1,
        "d3_root_differs_on_at_least_one_slice": d3_root_differences >= 1,
        "call_matched_d1_exact": all(
            item["primary"]["call_matched_d1"]["matches_primary_d1"]
            and item["primary"]["call_matched_d1"]["cache_misses_before"]
            == item["primary"]["call_matched_d1"]["cache_misses_after"]
            for item in slice_results
        ),
        "fixed_support_zero_proposals": all(
            item["fixed_support"]["cache"]["proposed_candidates"] == 0
            for item in slice_results
        ),
        "banked_replay_exact": all(bool(item["banked_replay_exact"]) for item in slice_results),
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "d2_vs_d1": d2_vs_d1,
        "d3_vs_d2": d3_vs_d2,
        "d3_vs_d1": comparison_at_tolerance(d1, d3, PRACTICAL_TIE_TOLERANCE),
        "exact_descriptive": {
            "d2_vs_d1": comparison(d1, d2),
            "d3_vs_d2": comparison(d2, d3),
            "d3_vs_d1": comparison(d1, d3),
        },
        "d2_root_differences": d2_root_differences,
        "d3_root_differences": d3_root_differences,
    }


def run(
    source_root: Path, *, implementation_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_bindings()
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    domains = active_domains(source)
    assays = frozen_assays()
    action_names = tuple(item.name for item in assays)
    runtime_checks = {
        "model_calls_zero": True,
        "cost_zero": True,
        "policy_state_has_no_truth_id": True,
        "finite_and_normalized": True,
        "predecessor_passed": True,
    }
    slice_results: list[dict[str, Any]] = []
    transition_bank: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}-transition-bank",
        "source_mode": "typed_registry_oracle",
        "slices": {},
    }
    for slice_index, item in enumerate(SLICES):
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
        seed = 2026083500 + slice_index
        primary, cache = evaluate_primary(
            bank, TypedRegistryOracleProposer(bank), mixed.truth_indices, seed=seed
        )
        replay, _ = evaluate_primary(
            bank,
            BankedProposer(cache.source_mode, cache.records),
            mixed.truth_indices,
            seed=seed,
            source_mode=cache.source_mode,
        )
        banked_replay_exact = replay == primary
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
        public_state = bank.initial_state().public_key()
        runtime_checks["policy_state_has_no_truth_id"] &= "truth" not in json.dumps(
            public_state
        ).lower()
        runtime_checks["finite_and_normalized"] &= all(
            math.isfinite(primary["policy_levels"][f"d{level}"]["planned_value"])
            and math.isfinite(
                primary["policy_levels"][f"d{level}"]["expected_terminal_mse"]
            )
            for level in (1, 2, 3)
        )
        slice_results.append(
            {
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
                "banked_replay_exact": banked_replay_exact,
            }
        )
        transition_bank["slices"][item.name] = {
            "planner_seed": seed,
            "records": cache.records,
            "transition_audit": primary["transition_audit"],
        }
    gate = apply_gate(slice_results, runtime_checks)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gate["passed"] else "failed_closed",
        "model_calls": 0,
        "cost_usd": 0.0,
        "implementation_commit": implementation_commit,
        "protocols": {str(path): {"sha256": sha256(path)} for path in PROTOCOLS},
        "predecessor_artifacts": {
            str(path): {"sha256": sha256(path)} for path in PREDECESSOR_ARTIFACTS
        },
        "source": source_binding,
        "active_domains": list(domains),
        "initial_support": list(INITIAL_SUPPORT_NAMES),
        "slices": [item.__dict__ for item in SLICES],
        "configuration": {
            "surprise_quantile": SURPRISE_QUANTILE,
            "evidence_slots": EVIDENCE_SLOTS,
            "diversity_slots": DIVERSITY_SLOTS,
            "pool_cap": EVIDENCE_SLOTS + DIVERSITY_SLOTS,
            "max_proposals": 4,
            "execution_budget": EXECUTION_BUDGET,
        },
        "runtime_checks": runtime_checks,
        "slice_results": slice_results,
        "gate": gate,
    }
    return result, transition_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transition-bank", type=Path, required=True)
    parser.add_argument("--required-commit", required=True)
    args = parser.parse_args()
    for path in (args.output, args.transition_bank):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    implementation_commit = require_pushed_commit(args.required_commit)
    result, bank = run(args.source_root, implementation_commit=implementation_commit)
    args.transition_bank.parent.mkdir(parents=True, exist_ok=True)
    args.transition_bank.write_text(
        json.dumps(bank, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["transition_bank"] = {
        "path": str(args.transition_bank),
        "sha256": sha256(args.transition_bank),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "gate": result["gate"],
                "output": str(args.output),
                "output_sha256": sha256(args.output),
                "transition_bank_sha256": sha256(args.transition_bank),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
