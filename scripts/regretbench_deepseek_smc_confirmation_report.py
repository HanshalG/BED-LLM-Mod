#!/usr/bin/env python3
"""Generate the frozen report for the RegretBench SMC confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from scripts import regretbench_deepseek_smc_confirmation as confirmation
from scripts import regretbench_deepseek_smc_confirmation_verify as verifier
from scripts import regretbench_deepseek_smc_frozen_report as shared


REPORTING_PROTOCOL = confirmation.primary.REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_SMC_CONFIRMATION_REPORTING_PROTOCOL_20260807.md"
)
REPORTING_PROTOCOL_SHA256 = (
    "696ebd9550af07d9fd77956e1ae8d39acdaabcfa03f37543545e3fe46b998489"
)
CLAIMS = {
    "mechanics_failed": (
        "smc_confirmation_mechanics_failure_no_result",
        "The untouched SMC confirmation failed its frozen mechanics contract, so no confirmation efficacy result is available.",
    ),
    "gated_null": (
        "smc_confirmation_null_no_headline_result",
        "The untouched preregistered SMC confirmation was null, so the provisional development signal is not a headline result.",
    ),
    "passed": (
        "smc_confirmed_nonmyopic_semantic_particle_result",
        "The untouched preregistered RegretBench cohort confirms the development result: depth-two planning over LLM-owned semantic particle likelihoods and retain/revise transitions outperforms the matched refresh-myopic control.",
    ),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validated(
    run_dir: Path, *, parent_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if shared.sha256_file(REPORTING_PROTOCOL) != REPORTING_PROTOCOL_SHA256:
        raise ValueError("SMC confirmation reporting protocol changed")
    result = _load(run_dir / "RESULT.json")
    stored = _load(run_dir / "VERIFICATION.json")
    replay = verifier.verify(run_dir, parent_dir=parent_dir)
    replay_artifacts = replay.get("artifact_sha256") or {}
    stored_artifacts = stored.get("artifact_sha256") or {}
    extra = set(replay_artifacts) - set(stored_artifacts)
    if extra - {"FROZEN_REPORT.json"}:
        raise ValueError("unexpected post-verification confirmation artifacts")
    replay["artifact_sha256"] = {
        key: replay_artifacts[key] for key in stored_artifacts
    }
    if replay != stored or replay.get("status") != "verified":
        raise ValueError("stored SMC confirmation verification does not replay")
    return result, stored


def build_report(run_dir: Path, *, parent_dir: Path) -> dict[str, Any]:
    result, verification = _validated(run_dir, parent_dir=parent_dir)
    tasks = result.get("tasks") or []
    if len(tasks) != 64:
        raise ValueError("SMC confirmation report requires all 64 tasks")
    tier, interpretation = CLAIMS[result["status"]]
    science = result.get("science")
    if result["status"] == "mechanics_failed":
        if science is not None:
            raise ValueError("mechanics failure cannot expose confirmation science")
        paired = disagreements = correlations = science_gates = fresh = None
    else:
        if not isinstance(science, Mapping):
            raise ValueError("mechanically valid confirmation lacks science")
        paired = {
            name: science["comparisons"][name] for name in shared.BASELINES
        }
        disagreements = {
            name: science["root_disagreements"][name]
            for name in shared.BASELINES
        }
        correlations = {
            "refresh_matched": science[
                "predicted_to_realized_dynamic_myopic_refresh_brier"
            ],
            "fixed_parent_brier": science[
                "predicted_to_realized_dynamic_myopic_brier"
            ],
            "myopic_eig": science["predicted_to_realized_dynamic_myopic"],
        }
        science_gates = science["gates"]
        if science_gates.get("all_pass") is not (result["status"] == "passed"):
            raise ValueError("confirmation status and science conjunction disagree")
        fresh = {
            "label": "secondary_descriptive",
            "comparisons": science.get(
                "fresh_regeneration_comparisons_descriptive"
            ),
            "can_change_claim_tier": False,
        }
    stability = result.get("draw_stability_diagnostic")
    if (
        not isinstance(stability, Mapping)
        or stability.get("can_change_status_authorization_or_claim_tier")
        is not False
    ):
        raise ValueError("confirmation lacks non-rescuing draw stability")
    usage = result.get("usage") or {}
    return {
        "schema_version": 1,
        "interface_version": "regretbench-smc-confirmation-report-1",
        "status": "complete",
        "stage": "confirmation",
        "result_status": result["status"],
        "claim_tier": tier,
        "interpretation": interpretation,
        "claim_tier_is_frozen_and_nonadaptive": True,
        "reporting_protocol_sha256": REPORTING_PROTOCOL_SHA256,
        "confirmation_protocol_sha256": confirmation.PROTOCOL_SHA256,
        "result_sha256": shared.sha256_file(run_dir / "RESULT.json"),
        "verification_sha256": shared.sha256_file(
            run_dir / "VERIFICATION.json"
        ),
        "independent_verification_interface": verification.get(
            "interface_version"
        ),
        "protocol": {
            "model": result["protocol"].get("model"),
            "reasoning": result["protocol"].get("reasoning"),
            "task_count": result["protocol"].get("task_count"),
            "primary_endpoint": result["protocol"].get("primary_endpoint"),
            "initial_hypotheses_regenerated": False,
            "llm_owns_reply_likelihoods": True,
            "llm_owns_retain_revise_transitions": True,
            "selection_frozen_before_truth_access": True,
        },
        "primary_policy_table": {
            "label": "aligned_generated_likelihood_with_action_and_reply_penalty",
            "policies": shared._policy_table(tasks),
        },
        "paired_primary_comparisons": paired,
        "root_disagreements": disagreements,
        "predicted_to_realized": correlations,
        "science_gates": science_gates,
        "mechanics_gates": result.get("mechanics_gates"),
        "alignment_complete_diagnostic": shared._alignment_complete(tasks),
        "draw_stability_diagnostic": stability,
        "secondary_fresh_smc_regeneration": fresh,
        "optional_naive_thinking": shared._optional_naive(result, tasks),
        "usage": {
            "deepseek_primary": usage.get("deepseek_primary"),
            "parent_bank": result.get("parent_bank", {}).get("usage"),
            "deepseek_naive_endpoint": usage.get("deepseek_naive_endpoint"),
            "naive_luna": usage.get("naive_luna"),
            "combined_requests": int(usage.get("combined_requests", 0))
            + confirmation.PARENT_REQUESTS,
            "combined_http_attempts": int(
                usage.get("combined_http_attempts", 0)
            )
            + confirmation.PARENT_REQUESTS,
            "combined_cost_usd": float(usage.get("combined_cost_usd", 0.0))
            + float(
                result.get("parent_bank", {})
                .get("usage", {})
                .get("run_cost_usd", 0.0)
            ),
        },
        "development_and_confirmation_are_not_pooled": True,
        "pooled_or_secondary_evidence_can_change_tier": False,
        "model_calls_made_by_report": 0,
        "cost_usd_by_report": 0.0,
    }


def write_report(run_dir: Path, *, parent_dir: Path) -> dict[str, Any]:
    report = build_report(run_dir, parent_dir=parent_dir)
    json_path = run_dir / "FROZEN_REPORT.json"
    markdown_path = run_dir / "FROZEN_REPORT.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rendered = shared.render_markdown(report).replace(
        "# RegretBench SMC Frozen Result Report",
        "# RegretBench SMC Confirmation Report",
        1,
    )
    markdown_path.write_text(rendered, encoding="utf-8")
    return {
        "status": "written",
        "claim_tier": report["claim_tier"],
        "json_path": str(json_path),
        "json_sha256": shared.sha256_file(json_path),
        "markdown_path": str(markdown_path),
        "markdown_sha256": shared.sha256_file(markdown_path),
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            write_report(
                args.run_dir.resolve(), parent_dir=args.parent_dir.resolve()
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
