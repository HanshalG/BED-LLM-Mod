#!/usr/bin/env python3
"""Validate a Bongard development result and freeze its allowed claim scope."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_development32_daily_execute as daily
from scripts import bongard_openworld_luna_vlm_development as development
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-claim-report-7"

SHARED_GATES = (
    "all_four_endpoint_blind_blocks_independently_replay",
    "exact_64_disjoint_development_tasks",
    "root_candidate_brier_beats_constant_half",
    "all_endpoint_metrics_are_finite",
    "confirmation_and_sealed_test_remain_unopened",
)
POLICY_GATES = (
    "at_least_24_dynamic_final_histories_differ_from_myopic",
    "at_least_24_dynamic_action_changes_clear_numerical_tie_margin",
    "dynamic_and_myopic_differ_in_every_execution_block",
    "dynamic_score_has_positive_mean_endpoint_ranking_fidelity",
    "dynamic_score_ranking_fidelity_is_not_worse_than_myopic",
    "dynamic_brier_relative_improvement_at_least_3_percent",
    "dynamic_brier_bootstrap_improvement_probability_at_least_0_80",
    "dynamic_log_loss_is_not_worse_than_myopic",
    "at_least_24_dynamic_final_histories_differ_from_compute_matched_myopic",
    "at_least_24_dynamic_action_changes_from_compute_matched_myopic_clear_numerical_tie_margin",
    "dynamic_and_compute_matched_myopic_differ_in_every_execution_block",
    "dynamic_brier_relative_improvement_vs_compute_matched_myopic_at_least_3_percent",
    "dynamic_brier_vs_compute_matched_myopic_bootstrap_probability_at_least_0_80",
    "dynamic_log_loss_is_not_worse_than_compute_matched_myopic",
    "dynamic_ranking_fidelity_is_not_worse_than_compute_matched_myopic",
    "dynamic_brier_is_not_worse_than_shuffled_control",
)
MECHANISM_GATES = (
    "at_least_24_dynamic_final_histories_differ_from_history_blind",
    "dynamic_and_history_blind_differ_in_every_execution_block",
    "dynamic_brier_relative_improvement_vs_history_blind_at_least_3_percent",
    "dynamic_brier_vs_history_blind_bootstrap_probability_at_least_0_80",
    "dynamic_log_loss_is_not_worse_than_history_blind",
    "dynamic_ranking_fidelity_is_not_worse_than_history_blind",
)
PATH_DEPENDENT_GATES = (
    "at_least_24_dynamic_final_histories_differ_from_fixed_depth2",
    "at_least_24_dynamic_action_changes_from_fixed_clear_numerical_tie_margin",
    "dynamic_and_fixed_depth2_differ_in_every_execution_block",
    "dynamic_brier_relative_improvement_vs_fixed_depth2_at_least_3_percent",
    "dynamic_brier_vs_fixed_depth2_bootstrap_probability_at_least_0_80",
    "dynamic_log_loss_is_not_worse_than_fixed_depth2",
    "dynamic_ranking_fidelity_is_not_worse_than_fixed_depth2",
    "at_least_24_dynamic_final_histories_differ_from_fixed_score_dynamic_update",
    "at_least_24_dynamic_action_changes_from_fixed_score_dynamic_update_clear_numerical_tie_margin",
    "dynamic_and_fixed_score_dynamic_update_differ_in_every_execution_block",
    "dynamic_brier_relative_improvement_vs_fixed_score_dynamic_update_at_least_3_percent",
    "dynamic_brier_vs_fixed_score_dynamic_update_bootstrap_probability_at_least_0_80",
    "dynamic_log_loss_is_not_worse_than_fixed_score_dynamic_update",
    "at_least_24_dynamic_final_histories_differ_from_history_blind_update_matched_first",
    "at_least_24_dynamic_second_action_changes_from_history_blind_update_matched_first_clear_numerical_tie_margin",
    "dynamic_and_history_blind_update_matched_first_differ_in_every_execution_block",
    "dynamic_brier_relative_improvement_vs_history_blind_update_matched_first_at_least_3_percent",
    "dynamic_brier_vs_history_blind_update_matched_first_bootstrap_probability_at_least_0_80",
    "dynamic_log_loss_is_not_worse_than_history_blind_update_matched_first",
)
EXPECTED_GATES = (
    *SHARED_GATES,
    *POLICY_GATES,
    *MECHANISM_GATES,
    *PATH_DEPENDENT_GATES,
)


CLAIM_SCOPES = {
    "full_path_dependent_llm_native_development_signal": {
        "allowed": [
            (
                "Prospective development evidence that dynamic depth-two "
                "planning improves endpoint Brier over both ordinary myopic "
                "width and a call-matched 17-belief myopic ensemble."
            ),
            (
                "Matched-control development evidence that, under a common "
                "realized answer-conditioned updater, first-query planning "
                "with answer-conditioned simulated support regeneration "
                "improves over same-seed history-blind simulation."
            ),
            (
                "Prospective development evidence that answer-conditioned "
                "dynamic support improves first-query selection over "
                "fixed-support depth-two scoring under a matched realized "
                "dynamic continuation."
            ),
            (
                "Matched-control development evidence that, after the same "
                "dynamic-selected first query and realized answer, the "
                "answer-conditioned intermediate belief selects a better "
                "second query than the same-seed history-blind intermediate "
                "belief under a common terminal updater."
            ),
        ],
        "forbidden": [
            "held-out confirmation",
            "sealed-test evidence",
            "cross-model robustness",
            "universal non-myopic benefit",
        ],
    },
    "policy_and_matched_regeneration_without_fixed_support_superiority": {
        "allowed": [
            (
                "Prospective development evidence for dynamic depth-two "
                "planning over ordinary and call-matched ensemble myopic "
                "selection."
            ),
            (
                "Matched-control development evidence for answer-conditioned "
                "versus history-blind simulated support in first-query "
                "planning under a common realized updater."
            ),
        ],
        "forbidden": [
            "superiority over every fixed-support and matched realized-updater control",
            "confirmation authorization",
            "held-out confirmation",
            "sealed-test evidence",
        ],
    },
    "policy_signal_without_matched_mechanism": {
        "allowed": [
            (
                "Prospective development evidence for the dynamic depth-two "
                "policy comparison, including the call-matched myopic "
                "ensemble, only."
            )
        ],
        "forbidden": [
            "causal attribution to the realized answer-conditioned updater",
            "held-out confirmation",
            "sealed-test evidence",
            "cross-model robustness",
        ],
    },
    "matched_mechanism_without_policy_signal": {
        "allowed": [
            (
                "Prospective development evidence for answer-conditioned "
                "simulated support in first-query planning over the matched "
                "history-blind simulation under a common realized updater."
            )
        ],
        "forbidden": [
            "a non-myopic policy win",
            "confirmation authorization",
            "sealed-test evidence",
            "cross-model robustness",
        ],
    },
    "development_null": {
        "allowed": [
            "A prospective development null with descriptive diagnostics."
        ],
        "forbidden": [
            "a non-myopic policy win",
            "a matched first-query planning-mechanism win",
            "confirmation authorization",
            "sealed-test evidence",
        ],
    },
}


def _family(gates: Mapping[str, bool], names: Sequence[str]) -> dict[str, Any]:
    failed = [name for name in names if gates[name] is not True]
    return {
        "pass": not failed,
        "required_gates": list(names),
        "failed_gates": failed,
    }


def classify_result(result: Mapping[str, Any]) -> dict[str, Any]:
    protocol = result.get("protocol") or {}
    gates = result.get("gates") or {}
    if (
        protocol.get("interface_version") != development.INTERFACE_VERSION
        or protocol.get("task_count") != development.TASKS
        or protocol.get("confirmation_accessed") is not False
        or protocol.get("sealed_test_accessed") is not False
        or set(gates) != {*EXPECTED_GATES, "all_pass"}
        or any(type(gates[name]) is not bool for name in gates)
    ):
        raise ValueError("combined result does not match the frozen claim interface")

    computed_all = all(gates[name] for name in EXPECTED_GATES)
    expected_status = "development_signal" if computed_all else "development_null"
    if (
        gates["all_pass"] is not computed_all
        or result.get("status") != expected_status
        or result.get("authorizes_confirmation_preregistration") is not computed_all
        or result.get("authorizes_confirmation_execution") is not False
    ):
        raise ValueError("combined result status disagrees with its frozen gates")

    shared = _family(gates, SHARED_GATES)
    policy = _family(gates, (*SHARED_GATES, *POLICY_GATES))
    mechanism = _family(gates, (*SHARED_GATES, *MECHANISM_GATES))
    path_dependent = _family(gates, (*SHARED_GATES, *PATH_DEPENDENT_GATES))
    if policy["pass"] and mechanism["pass"] and path_dependent["pass"]:
        tier = "full_path_dependent_llm_native_development_signal"
    elif policy["pass"] and mechanism["pass"]:
        tier = (
            "policy_and_matched_regeneration_without_"
            "fixed_support_superiority"
        )
    elif policy["pass"]:
        tier = "policy_signal_without_matched_mechanism"
    elif mechanism["pass"]:
        tier = "matched_mechanism_without_policy_signal"
    else:
        tier = "development_null"
    if computed_all != (
        tier == "full_path_dependent_llm_native_development_signal"
    ):
        raise AssertionError("claim tier and confirmation gate diverged")
    return {
        "claim_tier": tier,
        "shared_validity": shared,
        "policy_family": policy,
        "matched_mechanism_family": mechanism,
        "path_dependent_support_family": path_dependent,
        "claim_scope": CLAIM_SCOPES[tier],
        "authorizes_confirmation_preregistration": computed_all,
    }


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_finite(item) for item in value)
    return False


def build_claim_report(
    result: Mapping[str, Any],
    *,
    result_sha256: str,
    independent_verification: Mapping[str, Any],
) -> dict[str, Any]:
    classification = classify_result(result)
    if (
        independent_verification.get("verified") is not True
        or independent_verification.get("result_sha256") != result_sha256
        or independent_verification.get("status") != result.get("status")
        or independent_verification.get(
            "authorizes_confirmation_preregistration"
        )
        is not result.get("authorizes_confirmation_preregistration")
    ):
        raise ValueError("independent combined-result replay is missing or mismatched")
    metrics = {
        "pooled_policy_metrics": result["pooled_policy_metrics"],
        "comparisons_vs_myopic": result["comparisons_vs_myopic"],
        "dynamic_vs_history_blind": result["dynamic_vs_history_blind"],
        "dynamic_vs_compute_matched_myopic": result[
            "dynamic_vs_compute_matched_myopic"
        ],
        "dynamic_vs_fixed_depth2": result["dynamic_vs_fixed_depth2"],
        "dynamic_vs_fixed_score_dynamic_update": result[
            "dynamic_vs_fixed_score_dynamic_update"
        ],
        "dynamic_vs_history_blind_update_matched_first": result[
            "dynamic_vs_history_blind_update_matched_first"
        ],
        "ranking_fidelity": result["ranking_fidelity"],
        "dynamic_vs_myopic_relative_brier_improvement": result[
            "dynamic_vs_myopic_relative_brier_improvement"
        ],
        "dynamic_vs_history_blind_relative_brier_improvement": result[
            "dynamic_vs_history_blind_relative_brier_improvement"
        ],
        "dynamic_vs_compute_matched_myopic_relative_brier_improvement": result[
            "dynamic_vs_compute_matched_myopic_relative_brier_improvement"
        ],
        "dynamic_vs_fixed_depth2_relative_brier_improvement": result[
            "dynamic_vs_fixed_depth2_relative_brier_improvement"
        ],
        "dynamic_vs_fixed_score_dynamic_update_relative_brier_improvement": result[
            "dynamic_vs_fixed_score_dynamic_update_relative_brier_improvement"
        ],
        "dynamic_vs_history_blind_update_matched_first_relative_brier_improvement": result[
            "dynamic_vs_history_blind_update_matched_first_relative_brier_improvement"
        ],
        "dynamic_vs_myopic_changed_final_histories": result[
            "dynamic_vs_myopic_changed_final_histories"
        ],
        "dynamic_vs_history_blind_changed_final_histories": result[
            "dynamic_vs_history_blind_changed_final_histories"
        ],
        "dynamic_vs_compute_matched_myopic_changed_final_histories": result[
            "dynamic_vs_compute_matched_myopic_changed_final_histories"
        ],
        "dynamic_vs_fixed_depth2_changed_final_histories": result[
            "dynamic_vs_fixed_depth2_changed_final_histories"
        ],
        "dynamic_vs_fixed_score_dynamic_update_changed_final_histories": result[
            "dynamic_vs_fixed_score_dynamic_update_changed_final_histories"
        ],
        "dynamic_vs_history_blind_update_matched_first_changed_final_histories": result[
            "dynamic_vs_history_blind_update_matched_first_changed_final_histories"
        ],
        "dynamic_vs_history_blind_update_matched_first_robust_second_action_changes": result[
            "dynamic_vs_history_blind_update_matched_first_robust_second_action_changes"
        ],
    }
    if not _finite(metrics):
        raise ValueError("claim report metrics are not finite")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "claim_scope_frozen",
        "source_result_sha256": result_sha256,
        "source_interface_version": development.INTERFACE_VERSION,
        "independent_verification": dict(independent_verification),
        **classification,
        "metrics": metrics,
        "confirmation_execution_remains_unauthorized": True,
    }


def bank_claim_report(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    expected = dict(report)
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError("banked Bongard claim report changed")
        return observed
    checkpoint(path, expected)
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-result", type=Path, required=True)
    parser.add_argument("--block-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.block_result) != len(development.BLOCK_ORDER):
        raise ValueError("claim report requires exactly four block results")
    verification = daily.verify_combined_result(
        result_path=args.combined_result,
        block_results=args.block_result,
    )
    result = json.loads(args.combined_result.read_text(encoding="utf-8"))
    report = build_claim_report(
        result,
        result_sha256=development.sha256_file(args.combined_result),
        independent_verification=verification,
    )
    report = bank_claim_report(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
