#!/usr/bin/env python3
"""Render a replay-verified Bongard outcome as deterministic zero-call TeX."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import bongard_openworld_luna_claim_report as claim_report
from scripts import bongard_openworld_luna_confirmation64 as confirmation
from scripts import bongard_openworld_luna_confirmation64_daily_execute as confirmation_daily
from scripts import bongard_openworld_luna_development32_daily_execute as development_daily
from scripts import bongard_openworld_luna_vlm_development as development


REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/BONGARD_OPENWORLD_PAPER_FRAGMENT_PROTOCOL_20260808.md"
)
FRAGMENT_PROTOCOL_SHA256 = (
    "975f4097d88743cf6455d91dcc558b9219084200ca7153ea3bcc77b3f38c525c"
)
PRERESULT_MANUSCRIPT_SHA256 = (
    "b7f7f587384e15defb8ebae8bbbf5e7029db7bbbea011bbb685901ee46045f21"
)
DEFAULT_OUTPUT = REPO_ROOT / "paper/generated/bongard_openworld_result.tex"
HEADLINE_FILENAME = "bongard_openworld_headline.tex"
BOUND_FILES = {
    "claim_decision_plan": (
        "results/nonmyopic/BONGARD_OPENWORLD_LUNA_CLAIM_DECISION_PLAN.md",
        "5f19bd69ce37ab99799fa4ce34e06502d380b6cf72d0a80dcaa3afa97651545f",
    ),
    "endpoint_utility_amendment": (
        "results/nonmyopic/BONGARD_OPENWORLD_ENDPOINT_PREDICTIVE_UTILITY_AMENDMENT.md",
        "2fce4b66696d5b635f8d32c5f968a917aac20ab64305cab2728bc5f9973a55b8",
    ),
    "history_blind_estimand_clarification": (
        "results/nonmyopic/BONGARD_OPENWORLD_HISTORY_BLIND_ESTIMAND_CLARIFICATION_20260808.md",
        "65a6e901dc815d1603611e180b0abdf728e07d1a452e0c442f48fbb1812aea10",
    ),
    "matched_realized_updater_amendment": (
        "results/nonmyopic/BONGARD_OPENWORLD_LUNA_MATCHED_REALIZED_UPDATER_AMENDMENT_20260808.md",
        "dfa981153687004c8fb2c1195879d0774a281ca6c231c55d85495f2ac622178b",
    ),
    "matched_updater_integrity_amendment": (
        "results/nonmyopic/"
        "BONGARD_OPENWORLD_MATCHED_UPDATER_INTEGRITY_AMENDMENT_20260808.md",
        "1f0da098fbed4968a3594761194f661a0ebf49b12c477383d7c90bfa4989abf9",
    ),
    "llm_native_computational_role_amendment": (
        "results/nonmyopic/"
        "BONGARD_OPENWORLD_LLM_NATIVE_COMPUTATIONAL_ROLE_AMENDMENT_20260808.md",
        "e9752b0df729933579f23ec6656ca3779f70ee3d93b656c16f3901665e3e1baf",
    ),
    "novelty_and_headline_binding_amendment": (
        "results/nonmyopic/"
        "BONGARD_OPENWORLD_NOVELTY_AND_HEADLINE_BINDING_AMENDMENT_20260808.md",
        "bd98e80722d68e2702778cc2fcb6be2652b9cb9c518594009c9f67b6a3451c7d",
    ),
    "closest_prior_citation_amendment": (
        "results/nonmyopic/"
        "BONGARD_OPENWORLD_CLOSEST_PRIOR_CITATION_AMENDMENT_20260809.md",
        "44210a83ab90999a39d7cfaadf6d18b1b90dab73da2bbf94b81923bdb7e0cade",
    ),
    "2026_closest_prior_amendment": (
        "results/nonmyopic/"
        "BONGARD_OPENWORLD_2026_CLOSEST_PRIOR_AMENDMENT_20260809.md",
        "5a8f4f4c0cde5759641b4669a2d88d48fc64fd0d79c571dee15439205f073647",
    ),
    "curiositree_closest_prior_amendment": (
        "results/nonmyopic/"
        "BONGARD_OPENWORLD_CURIOSITREE_CLOSEST_PRIOR_AMENDMENT_20260809.md",
        "ddbd666abbb1a3d9437c30f6f3847f0f4275d6c48145d60a766ad2351e44cb12",
    ),
    "answer_signal_amendment": (
        "results/nonmyopic/"
        "BONGARD_OPENWORLD_ANSWER_SIGNAL_ABOVE_REGENERATION_NOISE_AMENDMENT_20260809.md",
        "f61a7ad4fb2a4cfad3011cae30b6a03be493bce08f2b2f4a79381211efbfe4f1",
    ),
    "answer_signal_audit": (
        "scripts/bongard_openworld_answer_signal_audit.py",
        "96d8fcde8c821933ca54ed9beaee37da417806d0cea951f7ceba32e5a948a35d",
    ),
    "answer_signal_label_privacy_correction": (
        "results/nonmyopic/"
        "BONGARD_OPENWORLD_ANSWER_SIGNAL_LABEL_PRIVACY_CORRECTION_20260809.md",
        "d35112d995f42a13329ab61bf1fadff9bacb04f31f550bfd55736e6f3af0bda9",
    ),
    "answer_signal_ordering_amendment": (
        "results/nonmyopic/"
        "BONGARD_OPENWORLD_ANSWER_SIGNAL_ORDERING_CORRECTION_20260809.md",
        "ab7bf1149ce1cc4044f9a5f7f0d540766974427a6385eb55965ac7ba6ebc1a1d",
    ),
    "preresult_references": (
        "paper/references.bib",
        "dea4327670d7169e32c0901e20bda4aa23868df6968c92e3855f929dc91bd93c",
    ),
    "development_manifest": (
        "results/nonmyopic/bongard_openworld_luna_vlm_development64/PROTOCOL_MANIFEST_V17.json",
        "7564ced7755f17be13f254067f013130b4beb277313fc51de16b43611a608676",
    ),
    "confirmation_manifest": (
        "results/nonmyopic/bongard_openworld_luna_confirmation64/PROTOCOL_MANIFEST_V14.json",
        "0d9c6f52ea05aa93e40bf7aa61c8ebc6323f50a6454624d6f6c49ca946d3924a",
    ),
    "development_claim_generator": (
        "scripts/bongard_openworld_luna_claim_report.py",
        "f9b828bb5dc85b2fd0ef2757c0708674f8e7489c518d3bc0fabd6536546888e3",
    ),
    "development_analyzer": (
        "scripts/bongard_openworld_luna_vlm_development.py",
        "03f06f89bc675897bf44ee82b69567f5217d7ab3d4bcd208996fe6fb623afc7a",
    ),
    "development_daily_replay": (
        "scripts/bongard_openworld_luna_development32_daily_execute.py",
        "f7caea8467c5f9d88d3634742e8e842edf9c50fd53855f1a38c81ce9601ef5fb",
    ),
    "confirmation_analyzer": (
        "scripts/bongard_openworld_luna_confirmation64.py",
        "d20aca14b8b254851caf785cac3dcf4a82a2e0b41c327506991e77ad676ad794",
    ),
    "confirmation_freeze_verifier": (
        "scripts/bongard_openworld_luna_confirmation64_verify.py",
        "1f905dcff4c2f251b45e07641c5ed7822d6a3676b8a0bcca797e1527a7c396f8",
    ),
    "confirmation_daily_executor": (
        "scripts/bongard_openworld_luna_confirmation64_daily_execute.py",
        "627eb36ee0af272c1f6f5541c9d45d5605cceb74d526e988169f9e3311296c5d",
    ),
}

DEVELOPMENT_TIERS = {
    "full_path_dependent_llm_native_development_signal": (
        "The prospective 64-task development cohort supports dynamic depth-two "
        "planning over myopic selection, answer-conditioned simulated support "
        "over same-seed history-blind simulation for first-query planning "
        "under a common realized updater, and path-dependent dynamic support "
        "over fixed-support, matched fixed-score, and dynamic-first matched "
        "history-blind intermediate-updater controls. Confirmation is pending; "
        "this development result is not a confirmed claim."
    ),
    "policy_and_matched_regeneration_without_fixed_support_superiority": (
        "The prospective development cohort supports the dynamic-depth-two "
        "policy comparison and matched answer-conditioned simulation in the "
        "first-query planning model under a common realized updater, but "
        "does not establish complete superiority over the fixed-support and "
        "matched realized-updater controls. "
        "Confirmation is unauthorized."
    ),
    "policy_signal_without_matched_mechanism": (
        "The prospective development cohort supports only dynamic depth two "
        "over myopic selection. It does not support causal attribution to the "
        "realized answer-conditioned updater, and confirmation is unauthorized."
    ),
    "matched_mechanism_without_policy_signal": (
        "The prospective development cohort supports only matched "
        "answer-conditioned over history-blind simulation in first-query "
        "planning under a common realized updater. This is not a non-myopic "
        "policy win, and confirmation is unauthorized."
    ),
    "development_null": (
        "The prospective development cohort is null under the frozen gate "
        "families. It supports neither a positive policy claim nor a matched "
        "first-query planning-mechanism claim, and confirmation is unauthorized."
    ),
}

CONFIRMATION_TIERS = {
    "full_llm_native_confirmation": (
        "The untouched 96-task confirmation cohort passes the complete frozen "
        "conjunction and confirms the registered multimodal LLM-native "
        "non-myopic result."
    ),
    "confirmation_null": (
        "The untouched 96-task confirmation cohort does not meet the complete "
        "frozen conjunction. The development result remains provisional and "
        "no Bongard headline claim is authorized."
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _verify_bound_files() -> dict[str, str]:
    if sha256_file(FRAGMENT_PROTOCOL) != FRAGMENT_PROTOCOL_SHA256:
        raise ValueError("Bongard paper-fragment protocol changed")
    observed = {}
    for name, (relative, expected) in BOUND_FILES.items():
        actual = sha256_file(REPO_ROOT / relative)
        if actual != expected:
            raise ValueError(f"bound Bongard file changed: {name}")
        observed[name] = actual
    return observed


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


def _number(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("paper metric is not finite")
    return f"{float(value):.{digits}f}"


def _percent(value: Any) -> str:
    return _number(100.0 * float(value), 2) + "\\%"


def _methods_lines(task_count: int) -> list[str]:
    return [
        "\\paragraph{Bongard-OpenWorld: multimodal LLM-native sequential BED.}",
        f"On {task_count} 14-image tasks, Luna generated ten history-conditioned predictive particles at root and counterfactual branches. Each particle contained a free-form semantic rule, a history-conditioned weight, and a predictive positive-label probability for all 14 images. The deterministic planner consumed only the weights and probability matrix; rule strings were interpretive descriptions and uniqueness checks, not numerical planner inputs. Candidate and endpoint roles were hidden. The planner saw endpoint IDs but no labels and minimized sealed endpoint-label predictive entropy. Controls were one-step PIG, fixed-support d2, same-seed history-blind planning simulation with a common realized updater, matched fixed-score/dynamic-update, dynamic-first matched history-blind intermediate updating with a common terminal updater, shuffled continuation, and random selection.",
    ]


def _comparison_summary(metrics: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name == "myopic":
        row = metrics["comparisons_vs_myopic"]["dynamic_depth2"]["mean_brier"]
    else:
        row = metrics[name]["mean_brier"]
    required = {
        "mean_difference",
        "sample_sd",
        "ci95",
        "wins",
        "ties",
        "losses",
    }
    if not isinstance(row, Mapping) or not required.issubset(row):
        raise ValueError(f"Bongard comparison summary is incomplete: {name}")
    ci = row["ci95"]
    if not isinstance(ci, Sequence) or isinstance(ci, (str, bytes)) or len(ci) != 2:
        raise ValueError(f"Bongard comparison interval is malformed: {name}")
    return row


def _metric_lines(metrics: Mapping[str, Any]) -> list[str]:
    if not _finite(metrics):
        raise ValueError("Bongard paper metrics are not finite")
    pooled = metrics["pooled_policy_metrics"]
    dynamic_brier = pooled["dynamic_depth2"]["mean_brier"]
    myopic_brier = pooled["myopic_width"]["mean_brier"]
    rows = (
        (
            "myopic",
            "myopic",
            metrics["dynamic_vs_myopic_relative_brier_improvement"],
            metrics["dynamic_vs_myopic_changed_final_histories"],
        ),
        (
            "history-blind",
            "dynamic_vs_history_blind",
            metrics["dynamic_vs_history_blind_relative_brier_improvement"],
            metrics["dynamic_vs_history_blind_changed_final_histories"],
        ),
        (
            "fixed-support",
            "dynamic_vs_fixed_depth2",
            metrics["dynamic_vs_fixed_depth2_relative_brier_improvement"],
            metrics["dynamic_vs_fixed_depth2_changed_final_histories"],
        ),
        (
            "matched fixed-score",
            "dynamic_vs_fixed_score_dynamic_update",
            metrics[
                "dynamic_vs_fixed_score_dynamic_update_relative_brier_improvement"
            ],
            metrics[
                "dynamic_vs_fixed_score_dynamic_update_changed_final_histories"
            ],
        ),
        (
            "matched realized updater",
            "dynamic_vs_history_blind_update_matched_first",
            metrics[
                "dynamic_vs_history_blind_update_matched_first_relative_brier_improvement"
            ],
            metrics[
                "dynamic_vs_history_blind_update_matched_first_changed_final_histories"
            ],
        ),
    )
    comparisons = []
    for label, key, relative, changed in rows:
        summary = _comparison_summary(metrics, key)
        comparisons.append(
            f"{label} {_percent(relative)} "
            f"($\\Delta={_number(summary['mean_difference'])}$, "
            f"95\\% CI $[{_number(summary['ci95'][0])},"
            f"{_number(summary['ci95'][1])}]$; {int(changed)} histories)"
        )
    ranking = metrics["ranking_fidelity"]
    dynamic_rank = ranking["dynamic_depth2"]["mean_spearman"]
    myopic_rank = ranking["myopic_width"]["mean_spearman"]
    return [
        "Dynamic d2 Brier was $"
        + _number(dynamic_brier)
        + "$ versus $"
        + _number(myopic_brier)
        + "$ for one-step PIG. Relative reductions (paired intervals) versus "
        + "; ".join(comparisons)
        + ".",
        "Mean first-action ranking Spearman was $"
        + _number(dynamic_rank, 3)
        + "$ for dynamic depth two and $"
        + _number(myopic_rank, 3)
        + "$ for one-step PIG; the replay-verified JSON retains all diagnostics and failed gates.",
    ]


def _base_metadata(
    *, stage: str, tier: str, bound_hashes: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "interface_version": "bongard-openworld-luna-paper-fragment-3",
        "status": "rendered",
        "stage": stage,
        "claim_tier": tier,
        "fragment_protocol_sha256": FRAGMENT_PROTOCOL_SHA256,
        "preresult_manuscript_sha256": PRERESULT_MANUSCRIPT_SHA256,
        "bound_file_sha256s": dict(bound_hashes),
        "development_and_confirmation_are_not_pooled": True,
        "manuscript_claim_is_deterministic": True,
        "model_calls": 0,
        "cost_usd": 0.0,
        "headline": {
            "authorized": False,
            "abstract_tex": "",
            "contribution_tex": "",
        },
        "llm_native_computational_role": {
            "llm_supplies_history_conditioned_particle_weights": True,
            "llm_supplies_history_conditioned_probability_matrix": True,
            "rule_strings_used_numerically": False,
            "bayesian_scoring_after_particle_generation_is_deterministic": True,
            "universal_classical_impossibility_claim": False,
        },
    }


def build_development_fragment(
    *,
    claim_report_path: Path,
    combined_result: Path,
    block_results: Sequence[Path],
) -> tuple[str, dict[str, Any]]:
    bound_hashes = _verify_bound_files()
    if len(block_results) != len(development.BLOCK_ORDER):
        raise ValueError("development rendering requires exactly four block results")
    verification = development_daily.verify_combined_result(
        result_path=combined_result, block_results=block_results
    )
    result = _load(combined_result)
    expected = claim_report.build_claim_report(
        result,
        result_sha256=development.sha256_file(combined_result),
        independent_verification=verification,
    )
    saved = _load(claim_report_path)
    if _canonical(saved) != _canonical(expected):
        raise ValueError("Bongard development claim report does not replay exactly")
    tier = str(saved.get("claim_tier"))
    if tier not in DEVELOPMENT_TIERS:
        raise ValueError("unregistered Bongard development claim tier")
    shared_valid = saved.get("shared_validity", {}).get("pass") is True
    lines = _methods_lines(development.TASKS)
    lines.extend(["", "\\textbf{Frozen development interpretation.} " + DEVELOPMENT_TIERS[tier]])
    if shared_valid:
        lines.extend(["", *_metric_lines(saved["metrics"])])
    else:
        lines.extend(
            [
                "",
                "The shared validity contract failed, so development is mechanics-inconclusive and no efficacy estimates are shown.",
            ]
        )
    lines.append("")
    metadata = _base_metadata(
        stage="development", tier=tier, bound_hashes=bound_hashes
    )
    metadata.update(
        {
            "source_claim_report_sha256": sha256_file(claim_report_path),
            "source_result_sha256": sha256_file(combined_result),
            "independent_result_verification": verification,
            "shared_validity_pass": shared_valid,
            "confirmation_authorized": False,
            "confirmation_preregistration_authorized": saved[
                "authorizes_confirmation_preregistration"
            ],
        }
    )
    return "\n".join(lines), metadata


def _confirmation_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pooled_policy_metrics": result["pooled_policy_metrics"],
        "comparisons_vs_myopic": result["comparisons_vs_myopic"],
        "dynamic_vs_history_blind": result["dynamic_vs_history_blind"],
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
        "dynamic_vs_fixed_depth2_changed_final_histories": result[
            "dynamic_vs_fixed_depth2_changed_final_histories"
        ],
        "dynamic_vs_fixed_score_dynamic_update_changed_final_histories": result[
            "dynamic_vs_fixed_score_dynamic_update_changed_final_histories"
        ],
        "dynamic_vs_history_blind_update_matched_first_changed_final_histories": result[
            "dynamic_vs_history_blind_update_matched_first_changed_final_histories"
        ],
    }


def build_confirmation_fragment(
    *, result_path: Path, block_results: Sequence[Path]
) -> tuple[str, dict[str, Any]]:
    bound_hashes = _verify_bound_files()
    if len(block_results) != len(confirmation.BLOCK_ORDER):
        raise ValueError("confirmation rendering requires exactly four block results")
    development_authorization = confirmation.verify_development_authorization()
    if development_authorization.get("verified") is not True:
        raise ValueError("full Bongard development authorization is missing")
    verification = confirmation.verify_combined_result(
        result_path=result_path, block_results=block_results
    )
    result = _load(result_path)
    tier = str(result.get("claim_tier"))
    status = result.get("status")
    expected_status = {
        "full_llm_native_confirmation": "confirmation_pass",
        "confirmation_null": "confirmation_null",
    }
    if tier not in CONFIRMATION_TIERS or status != expected_status[tier]:
        raise ValueError("unregistered or inconsistent Bongard confirmation tier")
    if (
        verification.get("verified") is not True
        or verification.get("claim_tier") != tier
        or verification.get("status") != status
        or verification.get("result_sha256") != sha256_file(result_path)
    ):
        raise ValueError("Bongard confirmation replay is missing or mismatched")
    lines = _methods_lines(confirmation.TASKS)
    lines.extend(
        [
            "",
            "\\textbf{Frozen confirmation interpretation.} "
            + CONFIRMATION_TIERS[tier],
            "",
            *_metric_lines(_confirmation_metrics(result)),
            "",
            "Development and confirmation are reported separately and are not pooled.",
            "",
        ]
    )
    metadata = _base_metadata(
        stage="confirmation", tier=tier, bound_hashes=bound_hashes
    )
    metadata.update(
        {
            "source_result_sha256": sha256_file(result_path),
            "independent_result_verification": verification,
            "development_authorization": development_authorization,
            "confirmation_authorized": tier == "full_llm_native_confirmation",
        }
    )
    if tier == "full_llm_native_confirmation":
        primary = _comparison_summary(result, "myopic")
        relative = result["dynamic_vs_myopic_relative_brier_improvement"]
        metadata["headline"] = {
            "authorized": True,
            "abstract_tex": (
                f"An untouched {confirmation.TASKS}-task visual confirmation "
                "passes the complete registered conjunction: planning over "
                "answer-conditioned VLM predictive beliefs reduces endpoint "
                f"Brier by {_percent(relative)} versus one-step PIG "
                f"(paired 95\\% CI $[{_number(primary['ci95'][0])},"
                f"{_number(primary['ci95'][1])}]$) and also beats fixed-support, "
                "history-blind, and matched-updater controls."
            ),
            "contribution_tex": (
                "\\item untouched multimodal confirmation that two-step BED over "
                "answer-conditioned VLM predictive-belief transitions outperforms "
                "one-step, fixed-support, history-blind, and matched-updater controls;"
            ),
        }
    return "\n".join(lines), metadata


def build_confirmation_mechanics_failure_fragment(
    *, failure_path: Path, combined_result: Path
) -> tuple[str, dict[str, Any]]:
    bound_hashes = _verify_bound_files()
    development_authorization = confirmation.verify_development_authorization()
    if development_authorization.get("verified") is not True:
        raise ValueError("full Bongard development authorization is missing")
    if combined_result.exists():
        raise ValueError("mechanics failure cannot coexist with a combined result")
    failure = _load(failure_path)
    if (
        failure.get("schema_version") != confirmation_daily.SCHEMA_VERSION
        or failure.get("interface_version") != confirmation_daily.INTERFACE_VERSION
        or failure.get("status") != "failed_closed"
        or failure.get("block_id") not in confirmation.BLOCK_ORDER
        or not isinstance(failure.get("error_type"), str)
        or not failure["error_type"]
        or not isinstance(failure.get("error"), str)
        or not failure["error"]
    ):
        raise ValueError("invalid frozen confirmation failed-closed record")
    tier = "confirmation_mechanics_inconclusive"
    lines = _methods_lines(confirmation.TASKS)
    lines.extend(
        [
            "",
            "\\textbf{Frozen confirmation interpretation.} The frozen confirmation executor failed closed before a replay-verified combined endpoint result existed. Confirmation is mechanics-inconclusive, no efficacy estimates are shown, and the development result remains provisional.",
            "",
            "Development and confirmation are reported separately and are not pooled.",
            "",
        ]
    )
    metadata = _base_metadata(
        stage="confirmation", tier=tier, bound_hashes=bound_hashes
    )
    metadata.update(
        {
            "source_failure_sha256": sha256_file(failure_path),
            "failed_block_id": failure["block_id"],
            "combined_result_absent": True,
            "development_authorization": development_authorization,
            "confirmation_authorized": False,
            "efficacy_metrics_rendered": False,
        }
    )
    return "\n".join(lines), metadata


def write_fragment(
    *,
    stage: str,
    output: Path = DEFAULT_OUTPUT,
    claim_report_path: Path | None = None,
    combined_result: Path | None = None,
    block_results: Sequence[Path] = (),
    failure_path: Path | None = None,
) -> dict[str, Any]:
    if stage == "development":
        if claim_report_path is None or combined_result is None:
            raise ValueError("development rendering requires claim and combined results")
        tex, metadata = build_development_fragment(
            claim_report_path=claim_report_path,
            combined_result=combined_result,
            block_results=block_results,
        )
    elif stage == "confirmation":
        if combined_result is None:
            raise ValueError("confirmation rendering requires a combined result")
        tex, metadata = build_confirmation_fragment(
            result_path=combined_result, block_results=block_results
        )
    elif stage == "confirmation-mechanics-failure":
        if failure_path is None or combined_result is None:
            raise ValueError("mechanics-failure rendering requires failure and combined paths")
        tex, metadata = build_confirmation_mechanics_failure_fragment(
            failure_path=failure_path, combined_result=combined_result
        )
    else:
        raise ValueError(f"unknown Bongard fragment stage: {stage}")
    headline = metadata.get("headline")
    if not isinstance(headline, Mapping):
        raise ValueError("Bongard headline metadata is missing")
    authorized = headline.get("authorized") is True
    abstract_tex = headline.get("abstract_tex")
    contribution_tex = headline.get("contribution_tex")
    if not isinstance(abstract_tex, str) or not isinstance(contribution_tex, str):
        raise ValueError("Bongard headline copy is malformed")
    if authorized is not bool(abstract_tex and contribution_tex):
        raise ValueError("Bongard headline authorization is inconsistent")
    if authorized is not (
        metadata["stage"] == "confirmation"
        and metadata["claim_tier"] == "full_llm_native_confirmation"
    ):
        raise ValueError("Bongard headline is not authorized by full confirmation")
    headline_tex = "\n".join(
        [
            "\\renewcommand{\\BongardAbstractResult}{%",
            abstract_tex,
            "}",
            "\\renewcommand{\\BongardContributionResult}{%",
            contribution_tex,
            "}",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(tex, encoding="utf-8")
    headline_path = output.with_name(HEADLINE_FILENAME)
    headline_path.write_text(headline_tex, encoding="utf-8")
    metadata_path = output.with_suffix(".json")
    metadata["tex_sha256"] = sha256_file(output)
    metadata["headline_tex_sha256"] = sha256_file(headline_path)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "written",
        "stage": metadata["stage"],
        "claim_tier": metadata["claim_tier"],
        "tex_path": str(output),
        "tex_sha256": metadata["tex_sha256"],
        "headline_path": str(headline_path),
        "headline_sha256": metadata["headline_tex_sha256"],
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def _default_development_blocks() -> list[Path]:
    return [
        development_daily.BLOCK_DIRS[block_id] / "RESULT.json"
        for block_id in development.BLOCK_ORDER
    ]


def _default_confirmation_blocks() -> list[Path]:
    return [
        confirmation_daily.BLOCK_DIRS[block_id] / "RESULT.json"
        for block_id in confirmation.BLOCK_ORDER
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("development", "confirmation", "confirmation-mechanics-failure"),
        required=True,
    )
    parser.add_argument("--claim-report", type=Path)
    parser.add_argument("--combined-result", type=Path)
    parser.add_argument("--block-result", type=Path, action="append")
    parser.add_argument("--failure-record", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.block_result:
        blocks = args.block_result
    elif args.stage == "development":
        blocks = _default_development_blocks()
    else:
        blocks = _default_confirmation_blocks()
    combined = args.combined_result
    claim = args.claim_report
    if args.stage == "development":
        combined = combined or development_daily.COMBINED_RESULT
        claim = claim or development_daily.ROOT / "CLAIM_REPORT.json"
    elif combined is None:
        combined = confirmation_daily.COMBINED_RESULT
    result = write_fragment(
        stage=args.stage,
        output=args.output.resolve(),
        claim_report_path=claim,
        combined_result=combined,
        block_results=blocks,
        failure_path=args.failure_record,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
