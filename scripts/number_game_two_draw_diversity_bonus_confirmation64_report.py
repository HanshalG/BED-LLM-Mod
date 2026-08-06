#!/usr/bin/env python3
"""Render the verified staged-64 diversity result without selective fields."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_two_draw_diversity_bonus_audit as audit
from scripts import number_game_two_draw_diversity_bonus_confirmation64_daily_execute as daily
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_two_draw_diversity_bonus_confirmation64_verify import (
    verify_completed_confirmation,
)


REPORT_PATH = REPO_ROOT / (
    "results/nonmyopic/"
    "NUMBER_GAME_TWO_DRAW_DIVERSITY_BONUS_CONFIRMATION64_RESULT.md"
)
CLAIM_PLAN = REPO_ROOT / (
    "results/nonmyopic/"
    "NUMBER_GAME_TWO_DRAW_DIVERSITY_BONUS_CONFIRMATION64_CLAIM_PLAN.md"
)
CLAIM_PLAN_SHA256 = (
    "27f494f4fa85412a4f0b35c37d40215ea4287cd70aee273a90017d85cbf982e3"
)
CLAIM_REPORT_NAME = "CLAIM_REPORT.json"
SCHEMA_VERSION = 1
INTERFACE_VERSION = (
    "number-game-two-draw-diversity-bonus-confirmation64-claim-report-1"
)
DEPTH_GATES = (
    "depth_three_reduction_vs_depth_two_at_least_three_percent",
    "depth_three_vs_depth_two_interval_below_zero",
    "depth_three_wins_exceed_losses_vs_depth_two",
)
VIABILITY_GATES = (
    "bonus_changes_at_least_sixteen_original_roots",
    "bonus_mean_brier_not_worse_than_original",
)
CLAIM_SCOPES = {
    "full_llm_native_dynamic_nonmyopic_confirmation": {
        "allowed": [
            "Prospective depth-three benefit over dynamic depth two.",
            "Prospective diversity-selector superiority over unadjusted depth three.",
            "Prospective dynamic-support endpoint benefit over fixed-support depth three.",
        ],
        "forbidden": [
            "universal non-myopic benefit",
            "monotonicity beyond depth two versus three",
            "cross-model robustness from this cohort alone",
        ],
    },
    "nonmyopic_with_diversity_selector_gain": {
        "allowed": [
            "Prospective depth-three benefit over dynamic depth two.",
            "Prospective diversity-selector superiority over unadjusted depth three.",
        ],
        "forbidden": [
            "a confirmed dynamic-versus-fixed support benefit",
            "universal non-myopic benefit",
        ],
    },
    "nonmyopic_with_viable_diversity_selector": {
        "allowed": [
            "Prospective depth-three benefit over dynamic depth two.",
            "A behaviorally active diversity selector that is non-worse in mean Brier.",
        ],
        "forbidden": [
            "causal diversity-selector superiority",
            "a full LLM-native dynamic-belief confirmation",
        ],
    },
    "nonmyopic_depth_only": {
        "allowed": [
            "The depth-three versus depth-two family, reported descriptively.",
        ],
        "forbidden": [
            "a passed registered combined confirmation",
            "a useful diversity-selector claim",
        ],
    },
    "mechanism_only_without_nonmyopic_depth": {
        "allowed": [
            "Only the exact selector or dynamic-support family that passes.",
        ],
        "forbidden": [
            "a prospective non-myopic planning win",
            "a full LLM-native dynamic-belief confirmation",
        ],
    },
    "prospective_null": {
        "allowed": ["A prospective null with verified diagnostics."],
        "forbidden": [
            "a prospective non-myopic planning win",
            "a useful diversity-selector claim",
        ],
    },
    "mechanics_failed": {
        "allowed": ["A mechanics failure without efficacy interpretation."],
        "forbidden": ["any scientific efficacy claim"],
    },
}
COMPARISON_ORDER = (
    ("crossfit_depth_two", "Cross-fitted dynamic depth two"),
    ("original_depth_three", "Unadjusted dynamic depth three"),
    ("myopic_eig", "Myopic EIG"),
    ("fixed_support_depth_three", "Fixed-support depth three"),
    ("positive_test_strategy", "Positive-test strategy"),
    ("uniform_random_candidate_root", "Uniform random roots"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float) -> str:
    return f"{float(value):.6f}"


def _status_text(status: str) -> str:
    if status == "passed":
        return (
            "The prospective monotonic-depth confirmation passed every "
            "frozen mechanics and scientific gate."
        )
    if status == "gated_null":
        return (
            "The prospective selector is a gated null. The frozen "
            "coefficient route closes without tuning on this cohort."
        )
    if status == "mechanics_failed":
        return (
            "The experiment is mechanics-failed; no scientific efficacy "
            "claim is made."
        )
    raise ValueError(f"unexpected staged result status: {status}")


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return False


def classify_claim_scope(result: Mapping[str, Any]) -> dict[str, Any]:
    """Separate registered depth evidence from stricter mechanism evidence."""
    status = str(result.get("status"))
    if status not in {"passed", "gated_null", "mechanics_failed"}:
        raise ValueError("unexpected staged result status")
    gates = result.get("scientific_gates") or {}
    if set(gates) != set(DEPTH_GATES + VIABILITY_GATES) or any(
        type(gates[name]) is not bool for name in gates
    ):
        raise ValueError("staged result has an invalid frozen gate family")
    comparisons = result.get("comparisons") or {}
    for name in (
        "crossfit_depth_two",
        "original_depth_three",
        "fixed_support_depth_three",
    ):
        if name not in comparisons or not _finite(comparisons[name]):
            raise ValueError(f"staged result has invalid comparison: {name}")
    rank = result.get("rank_metrics") or {}
    required_rank = {
        "original_mean_candidate_root_spearman",
        "bonus_mean_candidate_root_spearman",
        "original_mean_candidate_set_oracle_regret",
        "bonus_mean_candidate_set_oracle_regret",
    }
    if not required_rank.issubset(rank) or not _finite(rank):
        raise ValueError("staged result has invalid ranking diagnostics")

    registered_depth = all(gates[name] for name in DEPTH_GATES)
    registered_viability = all(gates[name] for name in VIABILITY_GATES)
    expected_status = (
        "passed" if registered_depth and registered_viability else "gated_null"
    )
    if status != "mechanics_failed" and status != expected_status:
        raise ValueError("staged status disagrees with frozen scientific gates")

    original = comparisons["original_depth_three"]
    selector_gates = {
        "bonus_mean_brier_strictly_below_unadjusted": (
            float(original["mean_candidate_minus_baseline_brier"]) < 0.0
        ),
        "bonus_vs_unadjusted_interval_below_zero": (
            float(original["tree_bootstrap_95pct"][1]) < 0.0
        ),
        "bonus_wins_exceed_losses_vs_unadjusted": (
            int(original["wins"]) > int(original["losses"])
        ),
    }
    fixed = comparisons["fixed_support_depth_three"]
    dynamic_gates = {
        "dynamic_reduction_vs_fixed_at_least_three_percent": (
            float(fixed["relative_brier_reduction"]) >= 0.03
        ),
        "dynamic_vs_fixed_interval_below_zero": (
            float(fixed["tree_bootstrap_95pct"][1]) < 0.0
        ),
        "dynamic_wins_exceed_losses_vs_fixed": (
            int(fixed["wins"]) > int(fixed["losses"])
        ),
    }
    ranking_gates = {
        "bonus_candidate_ranking_spearman_higher": (
            float(rank["bonus_mean_candidate_root_spearman"])
            > float(rank["original_mean_candidate_root_spearman"])
        ),
        "bonus_candidate_set_oracle_regret_lower": (
            float(rank["bonus_mean_candidate_set_oracle_regret"])
            < float(rank["original_mean_candidate_set_oracle_regret"])
        ),
    }
    selector_superiority = all(selector_gates.values())
    dynamic_endpoint = all(dynamic_gates.values())
    ranking_mechanism = all(ranking_gates.values())

    if status == "mechanics_failed":
        tier = "mechanics_failed"
    elif (
        registered_depth
        and registered_viability
        and selector_superiority
        and dynamic_endpoint
    ):
        tier = "full_llm_native_dynamic_nonmyopic_confirmation"
    elif registered_depth and registered_viability and selector_superiority:
        tier = "nonmyopic_with_diversity_selector_gain"
    elif registered_depth and registered_viability:
        tier = "nonmyopic_with_viable_diversity_selector"
    elif registered_depth:
        tier = "nonmyopic_depth_only"
    elif selector_superiority or dynamic_endpoint:
        tier = "mechanism_only_without_nonmyopic_depth"
    else:
        tier = "prospective_null"
    return {
        "claim_tier": tier,
        "registered_nonmyopic_depth_family": {
            "pass": registered_depth,
            "gates": {name: gates[name] for name in DEPTH_GATES},
        },
        "registered_selector_viability_family": {
            "pass": registered_viability,
            "gates": {name: gates[name] for name in VIABILITY_GATES},
        },
        "diversity_selector_superiority_family": {
            "pass": selector_superiority,
            "gates": selector_gates,
            "registered_scientific_gate": False,
        },
        "dynamic_support_endpoint_family": {
            "pass": dynamic_endpoint,
            "gates": dynamic_gates,
            "registered_scientific_gate": False,
        },
        "ranking_mechanism_family": {
            "pass": ranking_mechanism,
            "gates": ranking_gates,
            "diagnostic_only": True,
        },
        "claim_scope": CLAIM_SCOPES[tier],
        "authorizes_full_llm_native_dynamic_claim": (
            tier == "full_llm_native_dynamic_nonmyopic_confirmation"
        ),
    }


def build_claim_report(
    *,
    run_dir: Path,
    verifier: Callable[..., dict[str, Any]] = verify_completed_confirmation,
) -> dict[str, Any]:
    if audit.sha256_file(CLAIM_PLAN) != CLAIM_PLAN_SHA256:
        raise RuntimeError("diversity confirmation claim plan changed")
    verification = verifier(run_dir=run_dir)
    if verification.get("status") != "verified" or not all(
        (verification.get("checks") or {}).values()
    ):
        raise RuntimeError("staged result is not independently verified")
    result_path = run_dir / "RESULT.json"
    verification_path = run_dir / "VERIFICATION.json"
    if _load(verification_path) != verification:
        raise RuntimeError("stored verification does not match independent replay")
    result = _load(result_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "claim_scope_frozen",
        "registered_result_status": result["status"],
        "artifacts": {
            "result_sha256": audit.sha256_file(result_path),
            "verification_sha256": audit.sha256_file(verification_path),
            "claim_plan_sha256": CLAIM_PLAN_SHA256,
        },
        "independent_verification_replayed": True,
        **classify_claim_scope(result),
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def render_report(
    *,
    run_dir: Path,
    verifier: Callable[..., dict[str, Any]] = verify_completed_confirmation,
    claim_report: Mapping[str, Any] | None = None,
) -> str:
    claim_report = claim_report or build_claim_report(
        run_dir=run_dir, verifier=verifier
    )
    result = _load(run_dir / "RESULT.json")
    comparisons = result.get("comparisons") or {}
    missing = [name for name, _ in COMPARISON_ORDER if name not in comparisons]
    if missing:
        raise ValueError("verified result omitted comparisons: " + ", ".join(missing))
    gates = result.get("scientific_gates") or {}
    if not gates:
        raise ValueError("verified result omitted scientific gates")

    lines = [
        "# Number Game Diversity-Bonus Confirmation-64 Result",
        "",
        f"Status: **{result['status']}**.",
        "",
        _status_text(str(result["status"])),
        "",
        "## Paired Brier Comparisons",
        "",
        (
            "| Baseline | Bonus d3 mean (SD) | Baseline mean (SD) | "
            "Relative reduction | Paired difference 95% CI | W/T/L | "
            "Changed roots |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in COMPARISON_ORDER:
        item = comparisons[name]
        interval = item["tree_bootstrap_95pct"]
        lines.append(
            "| "
            + label
            + " | "
            + f"{_fmt(item['candidate_mean_brier'])} "
            + f"({_fmt(item['candidate_brier_sample_sd'])})"
            + " | "
            + f"{_fmt(item['baseline_mean_brier'])} "
            + f"({_fmt(item['baseline_brier_sample_sd'])})"
            + " | "
            + f"{100.0 * float(item['relative_brier_reduction']):.2f}%"
            + " | "
            + f"[{_fmt(interval[0])}, {_fmt(interval[1])}]"
            + " | "
            + f"{item['wins']}/{item['ties']}/{item['losses']}"
            + " | "
            + str(item["changed_roots"])
            + " |"
        )

    lines.extend(
        [
            "",
            "The PTS and uniform-random baselines are paired within tree and "
            "average their two frozen roots before comparison. They are "
            "descriptive controls, not scientific gates.",
            "",
            "## Frozen Gates",
            "",
        ]
    )
    for name, passed in gates.items():
        lines.append(f"- `{name}`: **{'pass' if passed else 'fail'}**")

    rank = result.get("rank_metrics") or {}
    families = (
        (
            "Registered non-myopic depth",
            claim_report["registered_nonmyopic_depth_family"],
        ),
        (
            "Registered selector viability",
            claim_report["registered_selector_viability_family"],
        ),
        (
            "Diversity-selector superiority",
            claim_report["diversity_selector_superiority_family"],
        ),
        (
            "Dynamic-support endpoint",
            claim_report["dynamic_support_endpoint_family"],
        ),
        ("Ranking mechanism", claim_report["ranking_mechanism_family"]),
    )
    lines.extend(
        [
            "",
            "## Ranking Diagnostics",
            "",
            (
                "- Mean candidate-root Spearman: "
                f"original `{_fmt(rank['original_mean_candidate_root_spearman'])}`, "
                f"bonus `{_fmt(rank['bonus_mean_candidate_root_spearman'])}`."
            ),
            (
                "- Mean candidate-set oracle regret: "
                f"original `{_fmt(rank['original_mean_candidate_set_oracle_regret'])}`, "
                f"bonus `{_fmt(rank['bonus_mean_candidate_set_oracle_regret'])}`."
            ),
            "",
            "## Claim Scope",
            "",
            f"Claim tier: **{claim_report['claim_tier']}**.",
            "",
            *[
                f"- {label}: **{'pass' if family['pass'] else 'fail'}**."
                for label, family in families
            ],
            "",
            *[
                f"- Allowed: {text}"
                for text in claim_report["claim_scope"]["allowed"]
            ],
            *[
                f"- Forbidden: {text}"
                for text in claim_report["claim_scope"]["forbidden"]
            ],
            "",
            "## Mechanics And Provenance",
            "",
            f"- trees / accepted requests: `64 / {result['usage']['adapter_requests']}`;",
            f"- total measured cost: `${float(result['usage']['run_cost_usd']):.8f}`;",
            "- model calls made by this reporter: `0`;",
            f"- result SHA-256: `{audit.sha256_file(run_dir / 'RESULT.json')}`;",
            f"- verification SHA-256: `{audit.sha256_file(run_dir / 'VERIFICATION.json')}`.",
            f"- claim-plan SHA-256: `{CLAIM_PLAN_SHA256}`.",
            "",
            "This report preserves the source result status and does not "
            "reclassify any earlier cohort.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(
    *,
    run_dir: Path,
    output_path: Path,
    claim_path: Path | None = None,
    verifier: Callable[..., dict[str, Any]] = verify_completed_confirmation,
) -> str:
    claim_report = build_claim_report(run_dir=run_dir, verifier=verifier)
    rendered = render_report(run_dir=run_dir, claim_report=claim_report)
    claim_path = claim_path or run_dir / CLAIM_REPORT_NAME
    if output_path.exists() or claim_path.exists():
        if not output_path.is_file() or not claim_path.is_file():
            raise RuntimeError("banked diversity result report is incomplete")
        observed_claim = _load(claim_path)
        if (
            output_path.read_text(encoding="utf-8") != rendered
            or observed_claim != claim_report
        ):
            raise RuntimeError("banked diversity result report changed")
        return rendered
    checkpoint(claim_path, claim_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=daily.RUN_DIR)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = write_report(run_dir=args.run_dir, output_path=args.output)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
