#!/usr/bin/env python3
"""Generate the prospectively frozen RegretBench SMC policy report."""

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
    "results/nonmyopic/REGRETBENCH_SMC_REPORTING_PROTOCOL_20260807.md"
)
REPORTING_PROTOCOL_SHA256 = (
    "09239a432e825e0e1e3132243eabb75308b98fb5f5f09a1d603932101242b9c5"
)
SMC_POLICY_PROTOCOL_SHA256 = (
    "97e9e7a582d38ae989042352755ee500ae4a42726dd6a431309a8e66102dd3cd"
)
VERIFIER_SHA256 = (
    "a76e61149f1537a048d6338d19f0a51b2aa6d396df06c5c45928763616a70afa"
)
CLAIM_GATE_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_SMC_PRIMARY_CLAIM_GATE_AMENDMENT_20260807.md"
)
CLAIM_GATE_AMENDMENT_SHA256 = (
    "7f7140418bd08e207bf1f52e9838c93234acf66d5eec48cf9c9879302426df62"
)
PRIMARY_POLICIES = (
    "smc_dynamic_depth2",
    "smc_myopic_refresh_brier",
    "smc_myopic_brier",
    "smc_myopic_width",
    "smc_history_blind_depth2",
    "smc_fixed_depth2",
    "random",
)
BASELINES = PRIMARY_POLICIES[1:]
PRIMARY_METRICS = (
    "truth_mass_final",
    "brier",
    "log_loss",
    "truth_mass_after_first",
    "valid_two_action_trajectory",
    "truth_consistent_first_reply_likelihood_matched",
    "likelihood_aligned_two_action_trajectory",
    "first_supported",
    "second_supported",
    "second_action_novel",
    "second_reply_likelihood_matched",
    "posterior_parent_update_applied",
)
CLAIMS = {
    "mechanics_failed": (
        "smc_mechanics_failure_no_scientific_result",
        "The frozen SMC mechanics contract failed, so no RegretBench policy-efficacy result is available.",
    ),
    "gated_null": (
        "smc_development_policy_null_confirmation_forbidden",
        "The preregistered RegretBench SMC development policy test was null; confirmation is forbidden.",
    ),
    "passed": (
        "smc_provisional_development_signal_confirmation_required",
        "RegretBench shows a provisional development signal for non-myopic planning over retained and revised LLM semantic particles; independent confirmation is required.",
    ),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"report metric is nonnumeric or nonfinite: {label}")
    return float(value)


def _stats(values: Sequence[Any], label: str) -> dict[str, float | int]:
    numeric = [_number(value, label) for value in values]
    if len(numeric) != 64:
        raise ValueError(f"SMC report metric requires 64 tasks: {label}")
    return {
        "mean": statistics.fmean(numeric),
        "sample_sd": statistics.stdev(numeric),
        "n": len(numeric),
    }


def _validate_verified_result(
    run_dir: Path, *, primary_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    from scripts import regretbench_deepseek_smc_dynamic_depth2_verify as verifier

    if sha256_file(REPORTING_PROTOCOL) != REPORTING_PROTOCOL_SHA256:
        raise ValueError("RegretBench SMC reporting protocol changed")
    if sha256_file(CLAIM_GATE_AMENDMENT) != CLAIM_GATE_AMENDMENT_SHA256:
        raise ValueError("RegretBench SMC claim-gate amendment changed")
    if sha256_file(Path(verifier.__file__).resolve()) != VERIFIER_SHA256:
        raise ValueError("RegretBench SMC independent verifier changed")
    result_path = run_dir / "RESULT.json"
    verification_path = run_dir / "VERIFICATION.json"
    result = _load(result_path)
    stored_verification = _load(verification_path)
    replay = verifier.verify(run_dir, primary_dir=primary_dir)
    replay_artifacts = replay.get("artifact_sha256") or {}
    stored_artifacts = stored_verification.get("artifact_sha256") or {}
    extra_artifacts = set(replay_artifacts) - set(stored_artifacts)
    if extra_artifacts - {"FROZEN_REPORT.json"}:
        raise ValueError("unexpected post-verification SMC artifacts are present")
    replay["artifact_sha256"] = {
        key: replay_artifacts[key] for key in stored_artifacts
    }
    result_sha = sha256_file(result_path)
    verification_sha = sha256_file(verification_path)
    protocol = result.get("protocol") or {}
    if (
        _canonical(replay) != _canonical(stored_verification)
        or replay.get("status") != "verified"
        or replay.get("result_status") != result.get("status")
        or replay.get("model_calls") != 0
        or float(replay.get("cost_usd", math.inf)) != 0.0
        or replay.get("mismatches") != []
        or replay.get("checks", {}).get("reported_result_matches_replay")
        is not True
        or replay.get("artifact_sha256", {}).get("RESULT.json") != result_sha
        or result.get("interface_version")
        != "regretbench-deepseek-smc-dynamic-depth2-experiment-1"
        or result.get("status") not in CLAIMS
        or protocol.get("stage") != "development"
        or protocol.get("task_count") != 64
        or protocol.get("protocol_sha256") != SMC_POLICY_PROTOCOL_SHA256
        or protocol.get("selection_frozen_before_truth_access") is not True
        or protocol.get("initial_hypotheses_regenerated") is not False
        or protocol.get("conditioned_blind_same_seed") is not True
        or protocol.get("hidden_cig_exposed_to_model") is not False
        or protocol.get("confirmation_opened") is not False
    ):
        raise ValueError("SMC result is not an independently verified frozen artifact")
    return result, replay, result_sha, verification_sha


def _policy_table(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    table = {}
    for name in PRIMARY_POLICIES:
        if not all(name in task.get("policies", {}) for task in tasks):
            raise ValueError(f"mandatory SMC policy is missing: {name}")
        table[name] = {
            metric: _stats(
                [task["policies"][name][metric] for task in tasks],
                f"{name}.{metric}",
            )
            for metric in PRIMARY_METRICS
        }
    return table


def _optional_naive(
    result: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    baseline = dict(result.get("naive_baseline") or {})
    available = len(tasks) == 64 and all(
        "naive_thinking" in task.get("policies", {}) for task in tasks
    )
    if not available:
        return {
            "label": "optional_unmatched_compute_descriptive",
            "available": False,
            "banked_status": baseline.get("status", "not_reported"),
            "can_change_claim_tier": False,
        }
    return {
        "label": "optional_unmatched_compute_descriptive",
        "available": True,
        "banked_status": baseline.get("status", "available"),
        "metrics": {
            metric: _stats(
                [task["policies"]["naive_thinking"][metric] for task in tasks],
                f"naive_thinking.{metric}",
            )
            for metric in ("fresh_truth_mass_final", "fresh_brier", "fresh_log_loss")
        },
        "can_change_claim_tier": False,
    }


def _alignment_complete(tasks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    controls = {}
    for baseline in BASELINES:
        eligible = [
            task
            for task in tasks
            if task["policies"]["smc_dynamic_depth2"][
                "likelihood_aligned_two_action_trajectory"
            ]
            and task["policies"][baseline][
                "likelihood_aligned_two_action_trajectory"
            ]
        ]
        differences = [
            task["policies"]["smc_dynamic_depth2"]["brier"]
            - task["policies"][baseline]["brier"]
            for task in eligible
        ]
        controls[baseline] = {
            "eligible_task_count": len(eligible),
            "mean_dynamic_minus_control_brier": (
                statistics.fmean(differences) if differences else None
            ),
            "wins_ties_losses": {
                "wins": sum(value < -1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
                "losses": sum(value > 1e-12 for value in differences),
            },
        }
    return {
        "label": "alignment_complete_non_rescuing_diagnostic",
        "controls": controls,
        "can_change_result_status_or_claim_tier": False,
    }


def build_report(run_dir: Path, *, primary_dir: Path) -> dict[str, Any]:
    result, verification, result_sha, verification_sha = _validate_verified_result(
        run_dir, primary_dir=primary_dir
    )
    tasks = result.get("tasks") or []
    if len(tasks) != 64:
        raise ValueError("RegretBench SMC report requires all 64 tasks")
    tier, interpretation = CLAIMS[result["status"]]
    science = result.get("science")
    if result["status"] == "mechanics_failed":
        if science is not None:
            raise ValueError("mechanics-failed SMC result cannot expose science")
        paired = disagreements = science_gates = correlations = fresh = None
        primary_claim_gates = primary_claim_all_pass = all_diagnostics_pass = None
    else:
        if not isinstance(science, Mapping):
            raise ValueError("mechanically valid SMC result lacks science")
        paired = {name: science["comparisons"][name] for name in BASELINES}
        disagreements = {
            name: science["root_disagreements"][name] for name in BASELINES
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
            raise ValueError("SMC status and frozen science conjunction disagree")
        fresh = {
            "label": "secondary_descriptive",
            "comparisons": science.get("fresh_regeneration_comparisons_descriptive"),
            "can_change_claim_tier": False,
        }
        primary_claim_gates = science["primary_claim_gates"]
        primary_claim_all_pass = science["primary_claim_all_pass"]
        all_diagnostics_pass = science["all_34_diagnostic_gates_pass"]
    stability = result.get("draw_stability_diagnostic")
    if (
        not isinstance(stability, Mapping)
        or stability.get("label")
        != "non_gating_non_rescuing_draw_stability_diagnostic"
        or stability.get("can_change_status_authorization_or_claim_tier") is not False
    ):
        raise ValueError("SMC result lacks its non-rescuing draw diagnostic")
    protocol = result["protocol"]
    if protocol.get("claim_gate_amendment_sha256") != CLAIM_GATE_AMENDMENT_SHA256:
        raise ValueError("SMC result lacks the frozen claim-gate amendment")
    usage = result.get("usage") or {}
    return {
        "schema_version": 1,
        "interface_version": "regretbench-smc-frozen-report-1",
        "status": "complete",
        "stage": "development",
        "result_status": result["status"],
        "claim_tier": tier,
        "interpretation": interpretation,
        "claim_tier_is_frozen_and_nonadaptive": True,
        "reporting_protocol_sha256": REPORTING_PROTOCOL_SHA256,
        "smc_policy_protocol_sha256": SMC_POLICY_PROTOCOL_SHA256,
        "claim_gate_amendment_sha256": CLAIM_GATE_AMENDMENT_SHA256,
        "result_sha256": result_sha,
        "verification_sha256": verification_sha,
        "independent_verification_interface": verification.get("interface_version"),
        "protocol": {
            "model": protocol.get("model"),
            "reasoning": protocol.get("reasoning"),
            "task_count": protocol.get("task_count"),
            "primary_endpoint": protocol.get("primary_endpoint"),
            "fresh_endpoint": protocol.get("fresh_smc_regeneration_endpoint"),
            "initial_hypotheses_regenerated": False,
            "llm_owns_reply_likelihoods": True,
            "llm_owns_retain_revise_transitions": True,
            "selection_frozen_before_truth_access": True,
            "claim_gate_amendment_sha256": protocol.get(
                "claim_gate_amendment_sha256"
            ),
        },
        "primary_policy_table": {
            "label": "aligned_generated_likelihood_with_action_and_reply_penalty",
            "policies": _policy_table(tasks),
        },
        "paired_primary_comparisons": paired,
        "root_disagreements": disagreements,
        "predicted_to_realized": correlations,
        "science_gates": science_gates,
        "primary_claim_gates": primary_claim_gates,
        "primary_claim_all_pass": primary_claim_all_pass,
        "all_34_diagnostic_gates_pass": all_diagnostics_pass,
        "mechanics_gates": result.get("mechanics_gates"),
        "alignment_complete_diagnostic": _alignment_complete(tasks),
        "draw_stability_diagnostic": stability,
        "secondary_fresh_smc_regeneration": fresh,
        "optional_naive_thinking": _optional_naive(result, tasks),
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


def _fmt(value: Mapping[str, Any]) -> str:
    return f"{float(value['mean']):.4f} ({float(value['sample_sd']):.4f})"


def render_markdown(report: Mapping[str, Any]) -> str:
    labels = {
        "smc_dynamic_depth2": "SMC dynamic d2",
        "smc_myopic_refresh_brier": "SMC refresh-matched myopic",
        "smc_myopic_brier": "SMC fixed-parent Brier myopic",
        "smc_myopic_width": "SMC myopic EIG",
        "smc_history_blind_depth2": "SMC history-blind d2",
        "smc_fixed_depth2": "Fixed-parent d2",
        "random": "Random",
    }
    lines = [
        "# RegretBench SMC Frozen Result Report",
        "",
        f"- Result status: `{report['result_status']}`",
        f"- Claim tier: `{report['claim_tier']}`",
        f"- Result SHA-256: `{report['result_sha256']}`",
        f"- Verification SHA-256: `{report['verification_sha256']}`",
        "",
        str(report["interpretation"]),
        "",
        "The LLM supplies semantic reply likelihoods and path-dependent retain/revise particle transitions. Initial hypotheses and questions are exact banked parent slots.",
        "",
        "## Primary Aligned Endpoint",
        "",
        "| Policy | Truth mass | Brier | Log loss | First mass | Valid path | Q2 reply match | Parent update |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table = report["primary_policy_table"]["policies"]
    for name in PRIMARY_POLICIES:
        row = table[name]
        lines.append(
            f"| {labels[name]} | {_fmt(row['truth_mass_final'])} | "
            f"{_fmt(row['brier'])} | {_fmt(row['log_loss'])} | "
            f"{_fmt(row['truth_mass_after_first'])} | "
            f"{_fmt(row['valid_two_action_trajectory'])} | "
            f"{_fmt(row['second_reply_likelihood_matched'])} | "
            f"{_fmt(row['posterior_parent_update_applied'])} |"
        )
    lines.extend(["", "## Paired Primary Comparisons", ""])
    paired = report["paired_primary_comparisons"]
    if paired is None:
        lines.append("No efficacy comparison is interpreted because mechanics failed.")
    else:
        lines.extend(
            [
                "| Control | Root changes | Brier diff (SD) | 95% CI | P(improve) | W/T/L | Log-loss diff |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in BASELINES:
            row = paired[name]
            brier = row["brier_dynamic_minus_baseline"]
            log_loss = row["log_loss_dynamic_minus_baseline"]
            wtl = row["wins_ties_losses"]
            lines.append(
                f"| {labels[name]} | {report['root_disagreements'][name]} | "
                f"{brier['mean']:.4f} ({brier['sample_sd']:.4f}) | "
                f"[{brier['ci95'][0]:.4f}, {brier['ci95'][1]:.4f}] | "
                f"{brier['probability_improvement']:.3f} | "
                f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} | "
                f"{log_loss['mean']:.4f} ({log_loss['sample_sd']:.4f}) |"
            )
        lines.extend(["", "## Ranking And Gates", ""])
        for name, value in report["predicted_to_realized"].items():
            lines.append(
                f"- `{name}` Spearman: `{value.get('spearman')}`; "
                f"95% CI `{value.get('ci95')}`; P(positive) "
                f"`{value.get('probability_positive')}`; n=`{value.get('n')}`."
            )
        for name, passed in report["science_gates"].items():
            lines.append(f"- `{name}`: `{passed}`")
        lines.extend(
            [
                f"- Frozen 13-gate primary conjunction: `{report['primary_claim_all_pass']}`",
                f"- Legacy all-34 diagnostic conjunction: `{report['all_34_diagnostic_gates_pass']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Non-Rescuing Diagnostics",
            "",
            "Alignment-complete subsets, the two-draw stability diagnostic, fresh final SMC regeneration, and optional Luna thinking are descriptive only and cannot alter this tier.",
            "",
            "## Mechanics And Usage",
            "",
        ]
    )
    for name, passed in (report.get("mechanics_gates") or {}).items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.extend(
        [
            f"- Combined requests: `{report['usage']['combined_requests']}`",
            f"- Combined cost USD: `{report['usage']['combined_cost_usd']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(run_dir: Path, *, primary_dir: Path) -> dict[str, Any]:
    report = build_report(run_dir, primary_dir=primary_dir)
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
    parser.add_argument("--primary-dir", type=Path, required=True)
    args = parser.parse_args()
    result = write_report(
        args.run_dir.resolve(), primary_dir=args.primary_dir.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
