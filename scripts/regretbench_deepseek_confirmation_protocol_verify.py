#!/usr/bin/env python3
"""Verify the frozen RegretBench confirmation protocol without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/regretbench_deepseek_dynamic_depth2_confirmation/"
    "PROTOCOL_MANIFEST.json"
)
SOURCE_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/regretbench_llm_native_source_audit/"
    "SOURCE_PROTOCOL_MANIFEST.json"
)
SOURCE_RESULT = REPO_ROOT / (
    "results/nonmyopic/regretbench_llm_native_source_audit/RESULT.json"
)
PROTOCOL_SHA256 = (
    "7a782f02eb8c3b16d5b229cca309d02bce64df6432c5090c977d3e26d1f46498"
)
SOURCE_MANIFEST_SHA256 = (
    "8a46b40395487aae0857d1a61d6f680beef579414c4580acf09d7e0b30b38e97"
)
SOURCE_RESULT_SHA256 = (
    "d7a10f15ecf6779520fbb20712c8d43059d8b03168fa904fe72e82ffe87578de"
)
CONFIRMATION_SPLIT_SHA256 = (
    "780a0e4e172251be2781729eeb4e591592b996dc9e1cd668b26240e3076660c9"
)
DEVELOPMENT_SPLIT_SHA256 = (
    "29d33b2fda0be7cc4eea6f4d9d4fe74fe200b5c580632dcb7ba844c3b825af69"
)
MECHANICS_SPLIT_SHA256 = (
    "707be5a1d1f86d6a0dc08ee61df77da1b9597093ac557d2e7706fcad8ef3b2f6"
)
DEVELOPMENT_SEED_RANGES = {
    "initial": range(202608089000, 202608089064),
    "branch": range(202608100000, 202608101024),
    "actual_first": range(202608110000, 202608110064),
    "actual_final": range(202608120000, 202608120064),
    "truth": range(202608130000, 202608130064),
    "random": range(202608140000, 202608140064),
    "bootstrap": range(202608150000, 202608150001),
}
EXPECTED_SCIENCE_GATES = {
    "minimum_dynamic_myopic_root_disagreements": 16,
    "minimum_dynamic_history_blind_root_disagreements": 12,
    "minimum_dynamic_fixed_root_disagreements": 12,
    "minimum_predicted_brier_advantage_vs_myopic": 0.01,
    "dynamic_minus_myopic_brier_maximum": -0.02,
    "dynamic_vs_myopic_bootstrap_probability_minimum": 0.9,
    "dynamic_vs_myopic_wins_exceed_losses": True,
    "dynamic_minus_history_blind_brier_maximum": -0.015,
    "dynamic_vs_history_blind_bootstrap_probability_minimum": 0.8,
    "dynamic_vs_history_blind_wins_exceed_losses": True,
    "dynamic_minus_fixed_brier_maximum": -0.01,
    "dynamic_vs_fixed_bootstrap_probability_minimum": 0.8,
    "dynamic_vs_fixed_wins_exceed_losses": True,
    "dynamic_log_loss_no_worse_than_myopic_history_blind_and_fixed": True,
    "changed_root_spearman_minimum": 0.15,
    "changed_root_positive_spearman_bootstrap_probability_minimum": 0.8,
    "all_conjunctive": True,
    "pooled_and_secondary_metrics_can_change_status": False,
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _confirmation_seeds(protocol: Mapping[str, Any]) -> dict[str, set[int]]:
    seeds = protocol["seeds"]
    return {
        "initial": set(range(seeds["initial_start"], seeds["initial_start"] + 64)),
        "branch": {
            seeds["branch_start"] + task * 16 + hypothesis * 2 + draw
            for task in range(64)
            for hypothesis in range(8)
            for draw in range(2)
        },
        "actual_first": set(
            range(seeds["actual_first_start"], seeds["actual_first_start"] + 64)
        ),
        "actual_final": set(
            range(seeds["actual_final_start"], seeds["actual_final_start"] + 64)
        ),
        "truth": set(range(seeds["truth_start"], seeds["truth_start"] + 64)),
        "random": set(range(seeds["random_start"], seeds["random_start"] + 64)),
        "bootstrap": {seeds["bootstrap"]},
    }


def verify_protocol(
    *,
    protocol_path: Path = PROTOCOL,
    source_manifest_path: Path = SOURCE_MANIFEST,
    source_result_path: Path = SOURCE_RESULT,
    repo_root: Path = REPO_ROOT,
    require_frozen_protocol_hash: bool = True,
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    source = _load(source_manifest_path)
    split_ids = {
        name: list(source["splits"][name]["ids"])
        for name in ("mechanics", "development", "confirmation")
    }
    computed_split_hashes = {
        name: split_hash(ids) for name, ids in split_ids.items()
    }

    bindings = protocol.get("implementation_bindings_at_freeze", {})
    binding_matches = {
        relative: (repo_root / relative).is_file()
        and sha256_file(repo_root / relative) == expected
        for relative, expected in bindings.items()
    }

    confirmation_seeds = _confirmation_seeds(protocol)
    flattened_confirmation = set().union(*confirmation_seeds.values())
    flattened_development = set().union(
        *(set(values) for values in DEVELOPMENT_SEED_RANGES.values())
    )
    expected_confirmation_seed_count = 64 + 1024 + 64 * 4 + 1
    requests = protocol.get("requests", {})
    instrument = protocol.get("instrument", {})
    mechanics = protocol.get("mechanics_gates", {})
    budget = protocol.get("budget", {})
    predecessor = protocol.get("development_precondition", {})

    gates = {
        "protocol_hash_matches": (
            not require_frozen_protocol_hash
            or sha256_file(protocol_path) == PROTOCOL_SHA256
        ),
        "source_manifest_hash_matches": (
            sha256_file(source_manifest_path) == SOURCE_MANIFEST_SHA256
        ),
        "source_result_hash_matches": (
            sha256_file(source_result_path) == SOURCE_RESULT_SHA256
        ),
        "exact_source_split_hashes": computed_split_hashes
        == {
            "mechanics": MECHANICS_SPLIT_SHA256,
            "development": DEVELOPMENT_SPLIT_SHA256,
            "confirmation": CONFIRMATION_SPLIT_SHA256,
        },
        "exact_source_split_sizes": {
            name: len(ids) for name, ids in split_ids.items()
        }
        == {"mechanics": 4, "development": 64, "confirmation": 64},
        "source_splits_pairwise_disjoint": len(
            set().union(*(set(ids) for ids in split_ids.values()))
        )
        == sum(len(ids) for ids in split_ids.values()),
        "manifest_confirmation_binding_matches": (
            protocol.get("source", {}).get("confirmation_ids_sha256")
            == CONFIRMATION_SPLIT_SHA256
            and protocol.get("source", {}).get("confirmation_count") == 64
        ),
        "all_implementation_bindings_match": bool(binding_matches)
        and all(binding_matches.values()),
        "exact_confirmation_seed_schedule": (
            len(flattened_confirmation) == expected_confirmation_seed_count
            and all(
                len(values) == expected
                for values, expected in zip(
                    confirmation_seeds.values(),
                    (64, 1024, 64, 64, 64, 64, 1),
                    strict=True,
                )
            )
        ),
        "confirmation_seeds_disjoint_from_development": not (
            flattened_confirmation & flattened_development
        ),
        "exact_request_arithmetic": (
            requests.get("exact_initial") == 64
            and requests.get("exact_branch") == 8192
            and requests.get("exact_planning") == 8256
            and requests.get("maximum_actual") == 512
            and requests.get("maximum_total") == 8768
            and 64 + 64 * 4 * 8 * 2 * 2 == 8256
            and 8256 + 64 * 4 * 2 == 8768
        ),
        "exact_instrument": (
            instrument.get("model") == "deepseek/deepseek-v4-flash-0731"
            and instrument.get("reasoning") is False
            and instrument.get("temperature") == 0.7
            and instrument.get("max_output_tokens") == 2200
            and instrument.get("concurrency") == 128
            and instrument.get("hypotheses") == 8
            and instrument.get("questions") == 4
            and instrument.get("branch_draws") == 2
            and instrument.get("policies")
            == [
                "dynamic_depth2",
                "history_blind_depth2",
                "myopic_width",
                "fixed_depth2",
                "random",
            ]
            and instrument.get("optional_baselines") == []
            and instrument.get("primary_endpoint")
            == "aligned_generated_likelihood_truth_mass"
        ),
        "exact_mechanics_thresholds": (
            mechanics.get("minimum_supported_first_actions_per_policy") == 48
            and mechanics.get("minimum_supported_second_actions_per_policy") == 40
            and mechanics.get("minimum_novel_second_actions_per_policy") == 40
            and mechanics.get("minimum_exact_reply_matches_per_policy") == 40
            and mechanics.get("blind_crn_expected_groups") == 1024
            and mechanics.get("blind_crn_required_exact_groups") == 1024
            and mechanics.get("independent_result_replay_required") is True
        ),
        "science_gates_unchanged": protocol.get("science_gates")
        == EXPECTED_SCIENCE_GATES,
        "literal_verified_development_pass_required": (
            predecessor.get("required_primary_status") == "passed"
            and predecessor.get("required_mechanics_all_pass") is True
            and predecessor.get("required_science_all_pass") is True
            and predecessor.get("required_independent_replay_status") == "verified"
            and predecessor.get("null_partial_failed_or_unverified_opens_calls")
            is False
        ),
        "budget_is_hard_capped": (
            budget.get("account_wide_daily_cap_usd") == 5.0
            and budget.get("run_cap_usd") == 3.5
            and budget.get("earliest_london_date") == "2026-08-09"
            and budget.get("timezone") == "Europe/London"
        ),
        "freeze_made_zero_calls_and_cost": (
            protocol.get("model_calls_made_by_freeze") == 0
            and protocol.get("cost_usd_by_freeze") == 0.0
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "interface_version": "regretbench-confirmation-protocol-verification-1",
        "status": "verified_frozen_protocol" if gates["all_pass"] else "failed",
        "protocol_sha256": sha256_file(protocol_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_result_sha256": sha256_file(source_result_path),
        "confirmation_ids_sha256": computed_split_hashes["confirmation"],
        "confirmation_task_count": len(split_ids["confirmation"]),
        "confirmation_seed_count": len(flattened_confirmation),
        "development_seed_overlap_count": len(
            flattened_confirmation & flattened_development
        ),
        "implementation_bindings": binding_matches,
        "gates": gates,
        "model_calls_made": 0,
        "cost_usd": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_protocol()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if result["status"] == "verified_frozen_protocol" else 1


if __name__ == "__main__":
    raise SystemExit(main())
