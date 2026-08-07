#!/usr/bin/env python3
"""Generate the prospectively frozen RegretBench result report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTING_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_REPORTING_PROTOCOL_20260807.md"
)
REPORTING_PROTOCOL_SHA256 = (
    "6594c91f2ceb5897ec0b2a0d1f6e58a5e4690325503414d9655aa45332be63f2"
)
PRIMARY_POLICIES = (
    "dynamic_depth2",
    "myopic_width",
    "history_blind_depth2",
    "fixed_depth2",
    "random",
)
BASELINES = (
    "myopic_width",
    "history_blind_depth2",
    "fixed_depth2",
    "random",
)
PRIMARY_METRICS = (
    "truth_mass_final",
    "brier",
    "log_loss",
    "truth_mass_after_first",
    "valid_two_action_trajectory",
    "first_supported",
    "second_supported",
    "second_action_novel",
    "second_reply_likelihood_matched",
)
CLAIMS = {
    ("development", "mechanics_failed"): (
        "mechanics_failure_no_scientific_result",
        "The frozen mechanics contract failed, so no RegretBench policy-efficacy result is available.",
    ),
    ("confirmation", "mechanics_failed"): (
        "mechanics_failure_no_scientific_result",
        "The frozen mechanics contract failed, so no RegretBench policy-efficacy result is available.",
    ),
    ("development", "gated_null"): (
        "development_policy_null_confirmation_forbidden",
        "The preregistered RegretBench development policy test was null; confirmation is forbidden.",
    ),
    ("development", "passed"): (
        "provisional_development_signal_confirmation_required",
        "RegretBench shows a provisional development signal for LLM-native non-myopic planning; independent confirmation is required.",
    ),
    ("confirmation", "gated_null"): (
        "confirmation_null_development_not_confirmed",
        "The RegretBench development signal did not confirm on the untouched confirmation cohort.",
    ),
    ("confirmation", "passed"): (
        "confirmed_llm_native_nonmyopic_signal",
        "RegretBench independently confirms a non-myopic gain over the LLM's own path-dependent semantic belief dynamics.",
    ),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"nonfinite or nonnumeric report value: {label}")
    return float(value)


def _stats(values: Sequence[Any], label: str) -> dict[str, float | int]:
    numeric = [_finite_number(value, label) for value in values]
    if len(numeric) != 64:
        raise ValueError(f"report metric requires 64 tasks: {label}")
    return {
        "mean": statistics.fmean(numeric),
        "sample_sd": statistics.stdev(numeric),
        "n": len(numeric),
    }


def _validate_verified_result(
    run_dir: Path, *, stage: str
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    if sha256_file(REPORTING_PROTOCOL) != REPORTING_PROTOCOL_SHA256:
        raise ValueError("RegretBench reporting protocol changed")
    result_path = run_dir / "RESULT.json"
    verification_path = run_dir / "VERIFICATION.json"
    result = _load(result_path)
    verification = _load(verification_path)
    result_sha = sha256_file(result_path)
    verification_sha = sha256_file(verification_path)
    expected_interface = {
        "development": "regretbench-deepseek-dynamic-depth2-policy-1",
        "confirmation": "regretbench-deepseek-dynamic-depth2-confirmation-1",
    }[stage]
    protocol = result.get("protocol") or {}
    if (
        result.get("interface_version") != expected_interface
        or protocol.get("stage") != stage
        or result.get("status") not in {"mechanics_failed", "gated_null", "passed"}
        or verification.get("status") != "verified"
        or verification.get("result_status") != result.get("status")
        or verification.get("model_calls") != 0
        or float(verification.get("cost_usd", math.inf)) != 0.0
        or verification.get("mismatches") != []
        or verification.get("checks", {}).get("reported_result_matches_replay")
        is not True
        or verification.get("artifact_sha256", {}).get("RESULT.json")
        != result_sha
    ):
        raise ValueError("result is not an independently verified frozen artifact")
    if stage == "confirmation" and (
        result.get("confirmation_opened") is not True
        or protocol.get("source_split") != "confirmation"
    ):
        raise ValueError("confirmation report has the wrong source boundary")
    return result, verification, result_sha, verification_sha


def _policy_table(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    table = {}
    for name in PRIMARY_POLICIES:
        if not all(name in task.get("policies", {}) for task in tasks):
            raise ValueError(f"mandatory policy missing from report: {name}")
        table[name] = {
            metric: _stats(
                [task["policies"][name][metric] for task in tasks],
                f"{name}.{metric}",
            )
            for metric in PRIMARY_METRICS
        }
    return table


def _optional_naive(
    result: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]], *, stage: str
) -> dict[str, Any]:
    baseline = dict(result.get("naive_baseline") or {})
    available = len(tasks) == 64 and all(
        "naive_thinking" in task.get("policies", {}) for task in tasks
    )
    if stage == "confirmation" and available:
        raise ValueError("confirmation cannot contain the optional Luna baseline")
    if not available:
        return {
            "label": "optional_unmatched_compute_descriptive",
            "available": False,
            "banked_status": baseline.get("status", "not_reported"),
            "can_change_claim_tier": False,
        }
    metrics = {
        metric: _stats(
            [task["policies"]["naive_thinking"][metric] for task in tasks],
            f"naive_thinking.{metric}",
        )
        for metric in ("fresh_truth_mass_final", "fresh_brier", "fresh_log_loss")
    }
    return {
        "label": "optional_unmatched_compute_descriptive",
        "available": True,
        "banked_status": baseline.get("status", "available"),
        "metrics": metrics,
        "can_change_claim_tier": False,
    }


def build_report(run_dir: Path, *, stage: str) -> dict[str, Any]:
    if stage not in {"development", "confirmation"}:
        raise ValueError("report stage must be development or confirmation")
    result, verification, result_sha, verification_sha = _validate_verified_result(
        run_dir, stage=stage
    )
    tasks = result.get("tasks") or []
    if len(tasks) != 64:
        raise ValueError("RegretBench report requires all 64 tasks")
    claim_tier, interpretation = CLAIMS[(stage, result["status"])]
    science = result.get("science")
    if result["status"] == "mechanics_failed":
        if science is not None:
            raise ValueError("mechanics-failed result cannot expose science")
        paired = None
        disagreements = None
        correlation = None
        fresh = None
        science_gates = None
    else:
        if not isinstance(science, Mapping):
            raise ValueError("mechanically valid result lacks science summary")
        paired = {
            name: science["comparisons"][name] for name in BASELINES
        }
        disagreements = {
            name: science["root_disagreements"][name] for name in BASELINES
        }
        correlation = science["predicted_to_realized_dynamic_myopic"]
        fresh = {
            "label": "secondary_descriptive",
            "comparisons": science["fresh_regeneration_comparisons_descriptive"],
            "can_change_claim_tier": False,
        }
        science_gates = science["gates"]
        expected_pass = result["status"] == "passed"
        if science_gates.get("all_pass") is not expected_pass:
            raise ValueError("result status and frozen science conjunction disagree")
    protocol = result.get("protocol") or {}
    usage = result.get("usage") or {}
    report = {
        "schema_version": 1,
        "interface_version": "regretbench-frozen-report-1",
        "status": "complete",
        "stage": stage,
        "result_status": result["status"],
        "claim_tier": claim_tier,
        "interpretation": interpretation,
        "claim_tier_is_frozen_and_nonadaptive": True,
        "reporting_protocol_sha256": REPORTING_PROTOCOL_SHA256,
        "result_sha256": result_sha,
        "verification_sha256": verification_sha,
        "independent_verification_interface": verification.get(
            "interface_version"
        ),
        "protocol": {
            "model": protocol.get("model"),
            "reasoning": protocol.get("reasoning"),
            "task_count": protocol.get("task_count"),
            "source_split": protocol.get("source_split", stage),
            "primary_endpoint": protocol.get("primary_endpoint"),
            "fresh_endpoint": protocol.get("fresh_regeneration_endpoint"),
            "invalid_trajectories_are_penalized": True,
        },
        "primary_policy_table": {
            "label": "aligned_generated_likelihood_with_invalid_path_penalty",
            "policies": _policy_table(tasks),
        },
        "paired_primary_comparisons": paired,
        "root_disagreements": disagreements,
        "predicted_to_realized_dynamic_myopic": correlation,
        "science_gates": science_gates,
        "secondary_fresh_regeneration": fresh,
        "optional_naive_thinking": _optional_naive(
            result, tasks, stage=stage
        ),
        "mechanics_gates": result.get("mechanics_gates"),
        "crn_diagnostics": result.get("crn_diagnostics"),
        "usage": {
            "deepseek_primary": usage.get("deepseek_primary"),
            "deepseek_naive_endpoint": usage.get("deepseek_naive_endpoint"),
            "naive_luna": usage.get("naive_luna"),
            "combined_requests": usage.get("combined_requests"),
            "combined_http_attempts": usage.get("combined_http_attempts"),
            "combined_cost_usd": usage.get("combined_cost_usd"),
        },
        "pooled_or_secondary_evidence_can_change_tier": False,
        "model_calls_made_by_report": 0,
        "cost_usd_by_report": 0.0,
    }
    return report


def _fmt_stat(value: Mapping[str, Any]) -> str:
    return f"{float(value['mean']):.4f} ({float(value['sample_sd']):.4f})"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# RegretBench Frozen Result Report",
        "",
        f"- Stage: `{report['stage']}`",
        f"- Result status: `{report['result_status']}`",
        f"- Claim tier: `{report['claim_tier']}`",
        f"- Result SHA-256: `{report['result_sha256']}`",
        f"- Verification SHA-256: `{report['verification_sha256']}`",
        "",
        report["interpretation"],
        "",
        "## Primary Aligned Endpoint",
        "",
        "Brier and log loss use the aligned generated-likelihood endpoint; invalid trajectories receive the frozen penalty.",
        "",
        "| Policy | Truth mass | Brier | Log loss | First mass | Valid path | Q1 supported | Q2 supported | Novel Q2 | Reply match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "dynamic_depth2": "Dynamic d2",
        "myopic_width": "Myopic width",
        "history_blind_depth2": "History-blind d2",
        "fixed_depth2": "Fixed-support d2",
        "random": "Random",
    }
    table = report["primary_policy_table"]["policies"]
    for name in PRIMARY_POLICIES:
        row = table[name]
        lines.append(
            "| "
            + " | ".join(
                [
                    labels[name],
                    _fmt_stat(row["truth_mass_final"]),
                    _fmt_stat(row["brier"]),
                    _fmt_stat(row["log_loss"]),
                    _fmt_stat(row["truth_mass_after_first"]),
                    _fmt_stat(row["valid_two_action_trajectory"]),
                    _fmt_stat(row["first_supported"]),
                    _fmt_stat(row["second_supported"]),
                    _fmt_stat(row["second_action_novel"]),
                    _fmt_stat(row["second_reply_likelihood_matched"]),
                ]
            )
            + " |"
        )
    paired = report.get("paired_primary_comparisons")
    lines.extend(["", "## Paired Comparisons", ""])
    if paired is None:
        lines.append("Not interpreted because the mechanics contract failed.")
    else:
        lines.extend(
            [
                "| Control | Brier diff | 95% CI | P(improve) | W/T/L | Log-loss diff | Log 95% CI | Root changes |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in BASELINES:
            row = paired[name]
            brier = row["brier_dynamic_minus_baseline"]
            log_loss = row["log_loss_dynamic_minus_baseline"]
            wtl = row["wins_ties_losses"]
            lines.append(
                f"| {labels[name]} | {brier['mean']:.4f} ({brier['sample_sd']:.4f}) | "
                f"[{brier['ci95'][0]:.4f}, {brier['ci95'][1]:.4f}] | "
                f"{brier['probability_improvement']:.3f} | "
                f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
                f"{log_loss['mean']:.4f} ({log_loss['sample_sd']:.4f}) | "
                f"[{log_loss['ci95'][0]:.4f}, {log_loss['ci95'][1]:.4f}] | "
                f"{report['root_disagreements'][name]} |"
            )
        correlation = report["predicted_to_realized_dynamic_myopic"]
        lines.extend(
            [
                "",
                "### Ranking Fidelity",
                "",
                (
                    "Changed-root predicted-to-realized Spearman: "
                    f"`{correlation['spearman']}`; 95% CI "
                    f"`{correlation['ci95']}`; P(positive) "
                    f"`{correlation['probability_positive']}`; "
                    f"n=`{correlation['n']}`."
                ),
                "",
                "### Science Gates",
                "",
            ]
        )
        for name, passed in report["science_gates"].items():
            lines.append(f"- `{name}`: `{passed}`")

        lines.extend(
            [
                "",
                "## Secondary Fresh Regeneration",
                "",
                "These comparisons are `secondary_descriptive` and cannot change the claim tier.",
                "",
                "| Control | Fresh Brier diff | 95% CI | P(improve) | W/T/L |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        fresh = report["secondary_fresh_regeneration"]["comparisons"]
        for name, row in fresh.items():
            brier = row["brier_dynamic_minus_baseline"]
            wtl = row["wins_ties_losses"]
            label = labels.get(name, "Naive thinking" if name == "naive_thinking" else name)
            lines.append(
                f"| {label} | {brier['mean']:.4f} ({brier['sample_sd']:.4f}) | "
                f"[{brier['ci95'][0]:.4f}, {brier['ci95'][1]:.4f}] | "
                f"{brier['probability_improvement']:.3f} | "
                f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} |"
            )

    naive = report["optional_naive_thinking"]
    lines.extend(["", "## Optional Naive-Thinking Baseline", ""])
    if not naive["available"]:
        lines.append(
            "Unavailable; banked status: "
            f"`{naive['banked_status']}`. This cannot change the claim tier."
        )
    else:
        lines.append(
            "Label: `optional_unmatched_compute_descriptive`; this cannot change the claim tier."
        )
        for name, value in naive["metrics"].items():
            lines.append(f"- `{name}`: {_fmt_stat(value)}")

    lines.extend(["", "## Mechanics And Usage", ""])
    for name, passed in (report.get("mechanics_gates") or {}).items():
        lines.append(f"- `{name}`: `{passed}`")
    usage = report["usage"]
    primary_usage = usage.get("deepseek_primary") or {}
    lines.extend(
        [
            "",
            f"Combined requests: `{usage['combined_requests']}`; HTTP attempts: `{usage['combined_http_attempts']}`; cost: `${float(usage['combined_cost_usd']):.6f}`.",
            (
                "Primary DeepSeek transport: "
                f"accepted=`{primary_usage.get('adapter_requests')}`, "
                f"attempts=`{primary_usage.get('http_attempts')}`, "
                f"retries=`{primary_usage.get('retry_count')}`, "
                f"provider retries=`{primary_usage.get('provider_error_retries')}`, "
                f"reasoning tokens=`{primary_usage.get('adapter_reasoning_tokens')}`, "
                f"forced exits=`{primary_usage.get('forced_exits')}`."
            ),
            "",
            "Fresh regeneration is `secondary_descriptive`; the optional Luna baseline is `optional_unmatched_compute_descriptive`. Neither can change the claim tier.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(run_dir: Path, *, stage: str) -> dict[str, Any]:
    report = build_report(run_dir, stage=stage)
    json_path = run_dir / "FROZEN_REPORT.json"
    markdown_path = run_dir / "FROZEN_REPORT.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return {
        "status": "written",
        "claim_tier": report["claim_tier"],
        "json_path": str(json_path),
        "json_sha256": sha256_file(json_path),
        "markdown_path": str(markdown_path),
        "markdown_sha256": sha256_file(markdown_path),
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("development", "confirmation"), required=True
    )
    args = parser.parse_args()
    result = write_report(args.run_dir.resolve(), stage=args.stage)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
