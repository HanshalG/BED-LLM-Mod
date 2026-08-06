#!/usr/bin/env python3
"""Freeze the allowed claim scope of the fully fresh Number Game result."""

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

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_qwen_fully_fresh_control_verify as verify


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-fully-fresh-claim-report-1"
SOURCE_RESULT = REPO_ROOT / (
    "results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/"
    "number-game-qwen-fully-fresh-daily-stages-20260806T000200Z/"
    "source/RESULT.json"
)
SOURCE_RESULT_SHA256 = (
    "13fd3361a8ef8f525f68733182a9bdb30151e37a4a5e14dfb35dde78540ab523"
)

SOURCE_MYOPIC_GATES = (
    "depth_three_beats_myopic_by_eight_percent",
    "depth_three_vs_myopic_ci_below_zero",
    "depth_three_wins_at_least_twenty_trees",
)
SOURCE_DYNAMIC_GATES = (
    "dynamic_and_fixed_roots_differ_on_at_least_twenty_trees",
    "dynamic_beats_fixed_by_three_percent",
    "dynamic_vs_fixed_ci_below_zero",
    "dynamic_wins_at_least_sixteen_trees",
)
CONTROL_SCIENCE_GATES = (
    "at_least_twenty_changed_root_trees",
    "prompt_benefit_contrast_to_realized_spearman_ci_above_zero",
    "second_stage_conditional_coverage_ci_not_below_history_blind",
    "second_stage_conditional_mse_ci_below_history_blind",
)
COMPOSITE_GATES = (
    "source_mechanics_passed",
    "source_myopic_policy_passed",
    "source_dynamic_support_passed",
    "control_mechanics_passed",
    "control_science_passed",
    "accepted_request_count_exact",
    "within_composite_budget",
)

CLAIM_SCOPES = {
    "full_fresh_llm_native_replication": {
        "allowed": [
            "Fresh non-myopic policy evidence versus myopic EIG.",
            "Fresh endpoint evidence that dynamic support beats fixed support.",
            "Fresh matched-control evidence for answer-conditioned belief regeneration.",
        ],
        "forbidden": [
            "universal non-myopic benefit",
            "monotonic planning-depth improvement",
            "cross-model robustness from this cohort alone",
        ],
    },
    "nonmyopic_policy_with_partial_llm_mechanism": {
        "allowed": [
            "Fresh non-myopic policy evidence versus myopic EIG.",
            "The exact mechanism families that pass, reported separately.",
        ],
        "forbidden": [
            "a full fresh causal policy-mechanism replication",
            "relabeling a failed dynamic-versus-fixed source family",
            "universal non-myopic benefit",
        ],
    },
    "nonmyopic_policy_without_fresh_mechanism": {
        "allowed": [
            "Fresh non-myopic policy evidence versus myopic EIG only.",
        ],
        "forbidden": [
            "fresh causal attribution to path-dependent support regeneration",
            "relabeling either failed mechanism family",
            "universal non-myopic benefit",
        ],
    },
    "fresh_mechanism_without_policy": {
        "allowed": [
            "The exact fresh mechanism family or families that pass.",
        ],
        "forbidden": [
            "a fresh non-myopic policy win",
            "a full fresh LLM-native replication",
            "universal non-myopic benefit",
        ],
    },
    "fresh_replication_null": {
        "allowed": [
            "A fresh null with descriptive diagnostics from valid components.",
        ],
        "forbidden": [
            "a fresh non-myopic policy win",
            "a fresh path-dependent mechanism win",
            "a full fresh LLM-native replication",
        ],
    },
}


def preflight_source_claim_ceiling(
    source_path: Path = SOURCE_RESULT,
) -> dict[str, Any]:
    if verify.sha256_file(source_path) != SOURCE_RESULT_SHA256:
        raise ValueError("fully fresh source result hash changed")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    mechanics = source.get("mechanics_gates") or {}
    if not mechanics or any(type(value) is not bool for value in mechanics.values()):
        raise ValueError("fully fresh source mechanics gates are malformed")
    policy = _family(source.get("myopic_policy_gates") or {}, SOURCE_MYOPIC_GATES)
    dynamic = _family(
        (source.get("dynamic_support") or {}).get("gates") or {},
        SOURCE_DYNAMIC_GATES,
    )
    policy["pass"] = all(mechanics.values()) and policy["pass"]
    dynamic["pass"] = all(mechanics.values()) and dynamic["pass"]
    full_reachable = policy["pass"] and dynamic["pass"]
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "claim_ceiling_frozen_without_control",
        "source_result": str(source_path),
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "nonmyopic_policy_family": policy,
        "dynamic_support_endpoint_family": dynamic,
        "full_fresh_llm_native_replication_reachable": full_reachable,
        "maximum_tier_if_control_passes": (
            "full_fresh_llm_native_replication"
            if full_reachable
            else "nonmyopic_policy_with_partial_llm_mechanism"
            if policy["pass"]
            else "fresh_mechanism_without_policy"
        ),
        "model_calls": 0,
        "files_written": 0,
        "cost_usd": 0.0,
    }


def _family(gates: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    if set(gates) != set(names) or any(type(gates[name]) is not bool for name in names):
        raise ValueError("result does not match a frozen gate family")
    failed = [name for name in names if gates[name] is not True]
    return {
        "pass": not failed,
        "required_gates": list(names),
        "failed_gates": failed,
    }


def classify_result(
    *,
    source: Mapping[str, Any],
    control: Mapping[str, Any],
    composite: Mapping[str, Any],
) -> dict[str, Any]:
    source_mechanics_gates = source.get("mechanics_gates") or {}
    control_mechanics_gates = control.get("mechanics_gates") or {}
    if (
        not source_mechanics_gates
        or not control_mechanics_gates
        or any(type(value) is not bool for value in source_mechanics_gates.values())
        or any(type(value) is not bool for value in control_mechanics_gates.values())
    ):
        raise ValueError("source or control mechanics gates are malformed")
    source_mechanics = all(source_mechanics_gates.values())
    control_mechanics = all(control_mechanics_gates.values())
    myopic = _family(source.get("myopic_policy_gates") or {}, SOURCE_MYOPIC_GATES)
    dynamic = _family(
        (source.get("dynamic_support") or {}).get("gates") or {},
        SOURCE_DYNAMIC_GATES,
    )
    matched = _family(
        (control.get("analysis") or {}).get("scientific_gates") or {},
        CONTROL_SCIENCE_GATES,
    )
    myopic_gates_pass = myopic["pass"]
    dynamic_gates_pass = dynamic["pass"]
    control_science_gates_pass = matched["pass"]

    expected_composite = {
        "source_mechanics_passed": source_mechanics,
        "source_myopic_policy_passed": myopic_gates_pass,
        "source_dynamic_support_passed": dynamic_gates_pass,
        "control_mechanics_passed": control_mechanics,
        "control_science_passed": control_science_gates_pass,
        "accepted_request_count_exact": (
            int((source.get("usage") or {}).get("adapter_requests", -1))
            == verify.EXPECTED_SOURCE_REQUESTS
            and int((control.get("usage") or {}).get("adapter_requests", -1))
            == verify.EXPECTED_CONTROL_REQUESTS
        ),
        "within_composite_budget": (
            float((source.get("usage") or {}).get("run_cost_usd", math.inf))
            + float((control.get("usage") or {}).get("run_cost_usd", math.inf))
            <= verify.DAILY_COMPOSITE_BUDGET_USD
        ),
    }
    observed_composite = composite.get("composite_gates") or {}
    if (
        set(observed_composite) != set(COMPOSITE_GATES)
        or any(type(observed_composite[name]) is not bool for name in COMPOSITE_GATES)
        or observed_composite != expected_composite
    ):
        raise ValueError("composite gates disagree with the three claim families")
    expected_status = "passed" if all(expected_composite.values()) else "gated_null"
    if not control_mechanics:
        expected_status = "mechanics_failed"
    if composite.get("status") != expected_status:
        raise ValueError("composite status disagrees with its frozen gates")

    myopic["pass"] = source_mechanics and myopic_gates_pass
    dynamic["pass"] = source_mechanics and dynamic_gates_pass
    matched["pass"] = (
        source_mechanics and control_mechanics and control_science_gates_pass
    )
    policy_pass = myopic["pass"]
    dynamic_pass = dynamic["pass"]
    matched_pass = matched["pass"]
    if policy_pass and dynamic_pass and matched_pass:
        tier = "full_fresh_llm_native_replication"
    elif policy_pass and (dynamic_pass or matched_pass):
        tier = "nonmyopic_policy_with_partial_llm_mechanism"
    elif policy_pass:
        tier = "nonmyopic_policy_without_fresh_mechanism"
    elif dynamic_pass or matched_pass:
        tier = "fresh_mechanism_without_policy"
    else:
        tier = "fresh_replication_null"
    return {
        "claim_tier": tier,
        "source_mechanics_pass": source_mechanics,
        "control_mechanics_pass": control_mechanics,
        "nonmyopic_policy_family": myopic,
        "dynamic_support_endpoint_family": dynamic,
        "matched_conditioning_family": matched,
        "claim_scope": CLAIM_SCOPES[tier],
        "authorizes_full_fresh_claim_upgrade": (
            tier == "full_fresh_llm_native_replication"
        ),
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
    *,
    source: Mapping[str, Any],
    control: Mapping[str, Any],
    composite: Mapping[str, Any],
    verification: Mapping[str, Any],
    artifact_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    classification = classify_result(
        source=source,
        control=control,
        composite=composite,
    )
    if (
        verification.get("status") != "verified"
        or verification.get("scientific_status_unchanged") is not True
        or verification.get("provider_calls") != 0
        or float(verification.get("verification_cost_usd", math.inf)) != 0.0
        or not verification.get("checks")
        or not all(verification["checks"].values())
        or verification.get("artifacts") != artifact_hashes
    ):
        raise ValueError("independent control verification is missing or mismatched")
    summary = verification.get("summary") or {}
    metrics = {
        "source_depth3_vs_myopic": summary.get("source_depth3_vs_myopic"),
        "source_dynamic_vs_fixed_depth3": summary.get(
            "source_dynamic_vs_fixed_depth3"
        ),
        "control_second_stage": summary.get("control_second_stage"),
        "control_selected_root_conditioning": summary.get(
            "control_selected_root_conditioning"
        ),
        "control_bootstrap": summary.get("control_bootstrap"),
        "control_scientific_gates": summary.get("control_scientific_gates"),
    }
    if any(value is None for value in metrics.values()) or not _finite(metrics):
        raise ValueError("claim report metrics are missing or non-finite")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "claim_scope_frozen",
        "artifacts": dict(artifact_hashes),
        "independent_verification_replayed": True,
        **classification,
        "metrics": metrics,
        "fresh_control_cannot_relabel_source": True,
        "diversity_confirmation_schedule_unchanged": True,
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    families = (
        ("Non-myopic policy", report["nonmyopic_policy_family"]),
        ("Dynamic-support endpoint", report["dynamic_support_endpoint_family"]),
        ("Matched conditioning", report["matched_conditioning_family"]),
    )
    lines = [
        "# Fully Fresh Number Game Claim Report",
        "",
        f"Claim tier: **{report['claim_tier']}**.",
        "",
        "## Gate Families",
        "",
    ]
    lines.extend(
        f"- {label}: `{'pass' if family['pass'] else 'fail'}`; "
        f"failed gates: `{family['failed_gates']}`."
        for label, family in families
    )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            *[f"- Allowed: {text}" for text in report["claim_scope"]["allowed"]],
            *[f"- Forbidden: {text}" for text in report["claim_scope"]["forbidden"]],
            "",
            "Independent public-artifact replay made zero model calls. The fresh",
            "control cannot relabel the frozen source result, and the separately",
            "preregistered diversity-confirmation schedule is unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def bank_claim_report(
    *, json_path: Path, markdown_path: Path, report: Mapping[str, Any]
) -> dict[str, Any]:
    expected = dict(report)
    markdown = render_markdown(expected)
    if json_path.exists() or markdown_path.exists():
        if not json_path.is_file() or not markdown_path.is_file():
            raise RuntimeError("banked Number Game claim report is incomplete")
        observed = json.loads(json_path.read_text(encoding="utf-8"))
        if observed != expected or markdown_path.read_text(encoding="utf-8") != markdown:
            raise RuntimeError("banked Number Game claim report changed")
        return observed
    checkpoint(json_path, expected)
    markdown_path.write_text(markdown, encoding="utf-8")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight_source_claim_ceiling(), indent=2, sort_keys=True))
        return 0
    if args.run_dir is None:
        parser.error("--run-dir is required unless --preflight is used")
    stored_path = args.run_dir / "CONTROL_VERIFICATION.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    independent = verify.verify_completed_control(
        run_dir=args.run_dir,
        write_outputs=False,
    )
    if stored != independent:
        raise ValueError("stored control verification does not replay exactly")
    source = json.loads((args.run_dir / "source/RESULT.json").read_text())
    control = json.loads((args.run_dir / "control/RESULT.json").read_text())
    composite = json.loads((args.run_dir / "RESULT.json").read_text())
    report = build_claim_report(
        source=source,
        control=control,
        composite=composite,
        verification=independent,
        artifact_hashes=independent["artifacts"],
    )
    json_path = args.output or args.run_dir / "CLAIM_REPORT.json"
    markdown_path = json_path.with_suffix(".md")
    report = bank_claim_report(
        json_path=json_path,
        markdown_path=markdown_path,
        report=report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
