#!/usr/bin/env python3
"""Independently verify a completed fully fresh Qwen control/composite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_qwen_fully_fresh_source_control32 as base
from scripts import number_game_qwen_history_blind_matched32_v3 as v3


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-fully-fresh-control-verify-1"
DAILY_COMPOSITE_BUDGET_USD = 9.25
EXPECTED_SOURCE_REQUESTS = 3680
EXPECTED_CONTROL_REQUESTS = 3072
EXPECTED_TOTAL_REQUESTS = 6752


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _artifact_hashes(run_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    source_dir = run_dir / "source"
    control_dir = run_dir / "control"
    source = {
        "result": sha256_file(source_dir / "RESULT.json"),
        "trees": sha256_file(source_dir / "TREES.json"),
        "targets": sha256_file(source_dir / "TARGETS.json"),
    }
    control = {
        "result": sha256_file(control_dir / "RESULT.json"),
        "controls": sha256_file(control_dir / "CONTROLS.json"),
    }
    return source, control


def _expected_source_summary(source_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": source_result["status"],
        "usage": source_result["usage"],
        "mechanics_gates": source_result["mechanics_gates"],
        "myopic_policy_gates": source_result["myopic_policy_gates"],
        "dynamic_support": source_result["dynamic_support"],
    }


def _expected_control_summary(control_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": control_result["status"],
        "usage": control_result["usage"],
        "mechanics_gates": control_result["mechanics_gates"],
        "analysis": control_result["analysis"],
        "second_draw_novelty": control_result["second_draw_novelty"],
    }


def _expected_composite(
    *,
    source_result: dict[str, Any],
    control_result: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, Any], str]:
    source_mechanics = all(source_result["mechanics_gates"].values())
    source_myopic = all(source_result["myopic_policy_gates"].values())
    source_dynamic = all(source_result["dynamic_support"]["gates"].values())
    control_mechanics = all(control_result["mechanics_gates"].values())
    control_science = all(
        control_result["analysis"]["scientific_gates"].values()
    )
    source_requests = int(source_result["usage"]["adapter_requests"])
    control_requests = int(control_result["usage"]["adapter_requests"])
    source_cost = float(source_result["usage"]["run_cost_usd"])
    control_cost = float(control_result["usage"]["run_cost_usd"])
    total_requests = source_requests + control_requests
    total_cost = source_cost + control_cost
    gates = {
        "source_mechanics_passed": source_mechanics,
        "source_myopic_policy_passed": source_myopic,
        "source_dynamic_support_passed": source_dynamic,
        "control_mechanics_passed": control_mechanics,
        "control_science_passed": control_science,
        "accepted_request_count_exact": (
            total_requests == EXPECTED_TOTAL_REQUESTS
        ),
        "within_composite_budget": total_cost <= DAILY_COMPOSITE_BUDGET_USD,
    }
    usage = {
        "source_requests": source_requests,
        "control_requests": control_requests,
        "total_requests": total_requests,
        "source_cost_usd": source_cost,
        "control_cost_usd": control_cost,
        "total_cost_usd": total_cost,
    }
    status = (
        "passed"
        if all(gates.values())
        else "gated_null"
    )
    if not control_mechanics:
        status = "mechanics_failed"
    return gates, usage, status


def _summary(
    *,
    source_result: dict[str, Any],
    control_result: dict[str, Any],
    composite: dict[str, Any],
) -> dict[str, Any]:
    myopic = source_result["aggregate"]["comparisons"]["myopic_eig"]
    dynamic = source_result["dynamic_support"]["comparison"]
    analysis = control_result["analysis"]
    return {
        "composite_status": composite["status"],
        "source_status": source_result["status"],
        "control_status": control_result["status"],
        "source_depth3_vs_myopic": {
            "relative_brier_reduction": myopic["relative_brier_reduction"],
            "paired_brier_difference_95pct": myopic[
                "tree_cluster_brier_difference_95pct_bootstrap"
            ],
            "wins_ties_losses": [
                myopic["brier_tree_wins"],
                myopic["brier_tree_ties"],
                myopic["brier_tree_losses"],
            ],
        },
        "source_dynamic_vs_fixed_depth3": {
            "relative_brier_reduction": dynamic[
                "relative_brier_reduction"
            ],
            "paired_brier_difference_95pct": dynamic[
                "tree_cluster_brier_difference_95pct_bootstrap"
            ],
            "wins_ties_losses": [
                dynamic["brier_tree_wins"],
                dynamic["brier_tree_ties"],
                dynamic["brier_tree_losses"],
            ],
        },
        "control_second_stage": analysis["stages"]["second"],
        "control_selected_root_conditioning": analysis[
            "selected_root_prompt_conditioning"
        ],
        "control_bootstrap": analysis["bootstrap"],
        "control_scientific_gates": analysis["scientific_gates"],
    }


def render_markdown(verification: dict[str, Any]) -> str:
    summary = verification["summary"]
    second = summary["control_second_stage"]
    selected = summary["control_selected_root_conditioning"]
    bootstrap = summary["control_bootstrap"]
    usage = verification["recomputed_composite_usage"]
    return "\n".join(
        [
            "# Fully Fresh Qwen Control Verification",
            "",
            f"Status: **{verification['status']}**.",
            "",
            "The public control trees and bootstrap analysis replay exactly,",
            "all provenance and composite checks pass, and no provider calls",
            "were made by verification.",
            "",
            "## Scientific Summary",
            "",
            f"- composite: `{summary['composite_status']}`; source: "
            f"`{summary['source_status']}`; control: "
            f"`{summary['control_status']}`;",
            "- second-stage conditional minus history-blind MSE:"
            f"  `{second['differences']['conditional_minus_history_blind_predictive_mse']:.9f}`;",
            "- second-stage conditional minus history-blind truth coverage:"
            f"  `{second['differences']['conditional_minus_history_blind_truth_coverage']:.9f}`;",
            "- selected-root conditioning-benefit correlation with realized"
            f"  advantage: `{selected['prompt_benefit_contrast_to_realized_advantage_spearman']:.6f}`;",
            "- correlation 95% interval:"
            f"  `{bootstrap['prompt_benefit_contrast_to_realized_spearman_95pct']}`;",
            f"- accepted requests/cost: `{usage['total_requests']}` /"
            f"  `${usage['total_cost_usd']:.9f}`.",
            "",
            "The source and control retain their preregistered statuses. This",
            "verification is reproducibility evidence only and cannot rescue a",
            "failed conjunction.",
            "",
            "## Provenance",
            "",
            f"- composite RESULT SHA-256: `{verification['artifacts']['composite_result']}`;",
            f"- replay payload SHA-256: `{verification['replay_payload_sha256']}`;",
            "- model calls / cost for verification: `0 / $0`.",
            "",
        ]
    )


def verify_completed_control(
    *,
    run_dir: Path,
    replay_fn: Callable[..., dict[str, Any]] = base.replay_fresh_control,
    novelty_fn: Callable[[dict[str, Any]], dict[str, Any]] = (
        v3.second_draw_novelty
    ),
    bootstrap_samples: int = base.BOOTSTRAP_SAMPLES,
    write_outputs: bool = True,
) -> dict[str, Any]:
    source_dir = run_dir / "source"
    control_dir = run_dir / "control"
    source_result = _load(source_dir / "RESULT.json")
    control_result = _load(control_dir / "RESULT.json")
    controls = _load(control_dir / "CONTROLS.json")
    composite = _load(run_dir / "RESULT.json")
    source_stage = _load(run_dir / "SOURCE_STAGE.json")
    control_stage = _load(run_dir / "CONTROL_STAGE.json")
    authorization_path = run_dir / "CONTROL_AUTHORIZATION.json"
    authorization = _load(authorization_path)
    source_hashes, control_hashes = _artifact_hashes(run_dir)
    composite_hash = sha256_file(run_dir / "RESULT.json")

    checks = {
        "source_hashes_match_source_stage": (
            source_stage["source_artifacts"] == source_hashes
        ),
        "source_hashes_match_authorization": (
            authorization["source_artifacts"] == source_hashes
        ),
        "authorization_hash_matches_source_stage": (
            sha256_file(authorization_path)
            == source_stage["control_authorization_sha256"]
        ),
        "authorization_hash_matches_composite": (
            sha256_file(authorization_path)
            == composite["protocol"]["control_authorization_sha256"]
        ),
        "authorization_structure_recomputes_exactly": (
            authorization["structural_authorization"]
            == base.source_control_authorization(source_result)
            == composite["protocol"]["control_authorization"]
        ),
        "authorization_excludes_source_science": (
            authorization["source_science_was_not_an_authorization_input"]
            is True
        ),
        "control_hashes_match_composite": (
            composite["protocol"]["control_artifacts"] == control_hashes
        ),
        "source_hashes_match_composite": (
            composite["protocol"]["source_artifacts"] == source_hashes
        ),
        "control_source_hashes_match_actual": all(
            control_result["protocol"][f"source_{name}_sha256"]
            == source_hashes[name]
            for name in ("result", "trees", "targets")
        ),
        "controls_hash_matches_control_result": (
            control_result["protocol"]["controls_sha256"]
            == control_hashes["controls"]
        ),
        "control_stage_hash_matches_composite": (
            control_stage["result_sha256"] == composite_hash
        ),
        "calendar_dates_preserve_later_control": (
            str(composite["protocol"]["control_calendar_date"])
            > str(composite["protocol"]["source_calendar_date"])
        ),
        "source_calendar_date_bindings_match": (
            str(composite["protocol"]["source_calendar_date"])
            == str(authorization["source_calendar_date"])
            == str(source_result["protocol"]["source_calendar_date"])
        ),
        "source_request_count_exact": (
            source_result["usage"]["adapter_requests"]
            == EXPECTED_SOURCE_REQUESTS
        ),
        "control_request_count_exact": (
            control_result["usage"]["adapter_requests"]
            == EXPECTED_CONTROL_REQUESTS
        ),
        "control_seed_start_exact": (
            control_result["protocol"]["control_seed_start"]
            == base.CONTROL_SEED_START
        ),
        "control_bootstrap_seed_exact": (
            control_result["protocol"]["bootstrap_seed"]
            == base.CONTROL_BOOTSTRAP_SEED
        ),
        "control_bootstrap_samples_exact": (
            control_result["protocol"]["bootstrap_samples"]
            == bootstrap_samples
        ),
        "source_mechanics_pass": all(
            source_result["mechanics_gates"].values()
        ),
        "control_mechanics_pass": all(
            control_result["mechanics_gates"].values()
        ),
        "source_summary_embedded_exactly": (
            composite["source"] == _expected_source_summary(source_result)
        ),
        "control_summary_embedded_exactly": (
            composite["control"] == _expected_control_summary(control_result)
        ),
        "control_stage_status_matches_composite": (
            control_stage["status"] == composite["status"]
            and control_stage["decision"] == composite["decision"]
        ),
        "source_science_not_used_for_authorization": (
            composite["protocol"][
                "source_science_was_not_an_authorization_input"
            ]
            is True
        ),
    }
    failed_pre_replay = [name for name, passed in checks.items() if not passed]
    if failed_pre_replay:
        raise ValueError(
            "pre-replay verification failed: " + ", ".join(failed_pre_replay)
        )

    replay = replay_fn(
        source_dir=source_dir,
        control_dir=control_dir,
        bootstrap_samples=bootstrap_samples,
    )
    checks["replayed_tree_rows_match_exactly"] = (
        replay["trees"] == control_result["trees"]
    )
    checks["replayed_analysis_matches_exactly"] = (
        replay["analysis"] == control_result["analysis"]
    )
    checks["second_draw_novelty_matches_exactly"] = (
        novelty_fn(controls) == control_result["second_draw_novelty"]
    )

    expected_gates, expected_usage, expected_status = _expected_composite(
        source_result=source_result,
        control_result=control_result,
    )
    checks["composite_gates_recompute_exactly"] = (
        composite["composite_gates"] == expected_gates
    )
    checks["composite_usage_recomputes_exactly"] = (
        composite["usage"] == expected_usage
    )
    checks["composite_status_recomputes_exactly"] = (
        composite["status"] == expected_status
    )
    checks["composite_decision_is_complete"] = (
        composite["decision"] == "complete_composite_endpoint"
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("verification failed: " + ", ".join(failed))

    verification = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "verified",
        "scientific_status_unchanged": True,
        "provider_calls": 0,
        "verification_cost_usd": 0.0,
        "artifacts": {
            "source": source_hashes,
            "control": control_hashes,
            "authorization": sha256_file(authorization_path),
            "composite_result": composite_hash,
            "control_stage": sha256_file(run_dir / "CONTROL_STAGE.json"),
        },
        "checks": checks,
        "replay_payload_sha256": _canonical_sha256(replay),
        "recomputed_composite_gates": expected_gates,
        "recomputed_composite_usage": expected_usage,
        "summary": _summary(
            source_result=source_result,
            control_result=control_result,
            composite=composite,
        ),
    }
    if write_outputs:
        checkpoint(run_dir / "CONTROL_VERIFICATION.json", verification)
        (run_dir / "CONTROL_VERIFICATION.md").write_text(
            render_markdown(verification),
            encoding="utf-8",
        )
    return verification


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    verification = verify_completed_control(run_dir=args.run_dir)
    print(
        json.dumps(
            {
                "status": verification["status"],
                "artifacts": verification["artifacts"],
                "summary": verification["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
