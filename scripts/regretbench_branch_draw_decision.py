#!/usr/bin/env python3
"""Interpret the frozen RegretBench branch-draw fidelity diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/REGRETBENCH_BRANCH_DRAW_DECISION_PROTOCOL_20260807.md"
)
FIDELITY_BINDING = REPO_ROOT / (
    "results/nonmyopic/regretbench_branch_draw_fidelity/EXECUTION_BINDING.json"
)
BINDING = REPO_ROOT / (
    "results/nonmyopic/regretbench_branch_draw_decision/EXECUTION_BINDING.json"
)

NEXT_ACTIONS = {
    "unavailable_mechanics_failed": (
        "Bank the mechanics failure; do not interpret branch fidelity."
    ),
    "insufficient_changed_roots": (
        "Bank the diagnostic and make no model, prompt, or draw-count selection from it."
    ),
    "averaging_reduces_draw_noise": (
        "Preregister a same-model four-draw comparison on an untouched development cohort; preserve the first two draws and all common random numbers."
    ),
    "shared_ranking_error": (
        "Do not add same-model draws; diagnose transition prompt, support, likelihood, and model bias on completed development histories before freezing a new holdout repair."
    ),
    "draw_sensitive": (
        "Run a schema-clean four-draw smoke, then use an untouched cohort for any draw-count comparison without selecting favorable seeds."
    ),
    "fidelity_adequate_no_draw_escalation": (
        "Do not optimize draw count; preserve the primary result and its already frozen confirmation rule."
    ),
    "inconclusive_no_adaptive_repair": (
        "Bank the result and make no adaptive model, prompt, subset, or confirmation choice from it."
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"nonnumeric decision value: {label}")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"nonfinite decision value: {label}")
    return output


def validate_binding() -> dict[str, Any]:
    binding = _load(BINDING)
    if binding.get("status") != "frozen_before_any_regretbench_policy_response":
        raise ValueError("branch-draw decision binding status changed")
    expected = {
        "protocol": PROTOCOL,
        "script": Path(__file__).resolve(),
        "fidelity_binding": FIDELITY_BINDING,
    }
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)):
            raise ValueError(f"branch-draw decision {name} path changed")
        if row.get("sha256") != sha256_file(path):
            raise ValueError(f"branch-draw decision {name} hash changed")
    return binding


def _unavailable(audit: Mapping[str, Any], *, binding_sha256: str) -> dict[str, Any]:
    region = "unavailable_mechanics_failed"
    return {
        "schema_version": 1,
        "interface_version": "regretbench-branch-draw-decision-1",
        "status": "complete_descriptive_non_gating",
        "region": region,
        "gates": None,
        "next_action": NEXT_ACTIONS[region],
        "fidelity_result_sha256": audit.get("result_sha256"),
        "fidelity_verification_sha256": audit.get("verification_sha256"),
        "decision_binding_sha256": binding_sha256,
        "can_change_status_authorization_or_claim_tier": False,
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def classify(audit: Mapping[str, Any], *, binding_sha256: str) -> dict[str, Any]:
    if audit.get("interface_version") != "regretbench-branch-draw-fidelity-1":
        raise ValueError("unsupported branch-draw fidelity interface")
    if (
        audit.get("can_change_status_authorization_or_claim_tier") is not False
        or audit.get("model_calls") != 0
        or float(audit.get("cost_usd", math.inf)) != 0.0
    ):
        raise ValueError("fidelity artifact violates its non-gating boundary")
    if audit.get("status") == "unavailable_mechanics_failed":
        return _unavailable(audit, binding_sha256=binding_sha256)
    if audit.get("status") != "complete_descriptive_non_gating":
        raise ValueError("incomplete branch-draw fidelity artifact")

    metrics = audit.get("metrics") or {}
    predicted = metrics.get("predicted_to_realized") or {}
    draw0 = predicted.get("draw0") or {}
    draw1 = predicted.get("draw1") or {}
    ensemble = predicted.get("ensemble") or {}
    bootstrap = metrics.get("bootstrap") or {}
    agreement = metrics.get("draw_prediction_agreement") or {}

    n = int(metrics.get("changed_root_task_count", 0))
    requested = int(bootstrap.get("requested_samples", 0))
    retained = int(bootstrap.get("retained_samples", 0))
    draw0_ci = draw0.get("ci95") or [None, None]
    draw1_ci = draw1.get("ci95") or [None, None]
    ensemble_ci = ensemble.get("ci95") or [None, None]
    required_values = {
        "draw0_spearman": draw0.get("spearman"),
        "draw1_spearman": draw1.get("spearman"),
        "ensemble_spearman": ensemble.get("spearman"),
        "draw0_ci_low": draw0_ci[0] if len(draw0_ci) == 2 else None,
        "draw0_ci_high": draw0_ci[1] if len(draw0_ci) == 2 else None,
        "draw1_ci_low": draw1_ci[0] if len(draw1_ci) == 2 else None,
        "draw1_ci_high": draw1_ci[1] if len(draw1_ci) == 2 else None,
        "ensemble_ci_low": ensemble_ci[0] if len(ensemble_ci) == 2 else None,
        "ensemble_ci_high": ensemble_ci[1] if len(ensemble_ci) == 2 else None,
        "draw0_probability_positive": draw0.get("probability_positive"),
        "draw1_probability_positive": draw1.get("probability_positive"),
        "ensemble_probability_positive": ensemble.get("probability_positive"),
        "draw_pearson": agreement.get("pearson"),
        "ensemble_exceeds_both_probability": bootstrap.get(
            "ensemble_exceeds_both_probability"
        ),
    }
    available = all(value is not None for value in required_values.values())
    enough_bootstrap = requested > 0 and retained >= 0.9 * requested
    insufficient = n < 16 or not enough_bootstrap or not available

    numeric = (
        {name: _number(value, name) for name, value in required_values.items()}
        if available
        else {}
    )
    if available:
        numeric.update(
            {
                "draw0_rmse": _number(draw0.get("rmse"), "draw0_rmse"),
                "draw1_rmse": _number(draw1.get("rmse"), "draw1_rmse"),
                "ensemble_rmse": _number(ensemble.get("rmse"), "ensemble_rmse"),
                "draw_rms_gap": _number(
                    agreement.get("root_mean_square_gap"), "draw_rms_gap"
                ),
            }
        )

    gates = {
        "at_least_16_changed_roots": n >= 16,
        "at_least_90pct_bootstrap_retained": enough_bootstrap,
        "all_correlations_and_probabilities_available": available,
    }
    if available:
        gates.update(
            {
                "ensemble_spearman_at_least_015": numeric["ensemble_spearman"] >= 0.15,
                "ensemble_probability_positive_at_least_080": numeric[
                    "ensemble_probability_positive"
                ]
                >= 0.80,
                "ensemble_exceeds_both_probability_at_least_080": numeric[
                    "ensemble_exceeds_both_probability"
                ]
                >= 0.80,
                "ensemble_rmse_at_most_90pct_best_draw": numeric["ensemble_rmse"]
                <= 0.9 * min(numeric["draw0_rmse"], numeric["draw1_rmse"]),
                "draw_rms_gap_at_least_002": numeric["draw_rms_gap"] >= 0.02,
                "draw_pearson_at_least_080": numeric["draw_pearson"] >= 0.80,
                "draw_rms_gap_at_most_002": numeric["draw_rms_gap"] <= 0.02,
                "all_draw_and_ensemble_spearman_below_015": max(
                    numeric["draw0_spearman"],
                    numeric["draw1_spearman"],
                    numeric["ensemble_spearman"],
                )
                < 0.15,
                "ensemble_probability_positive_below_080": numeric[
                    "ensemble_probability_positive"
                ]
                < 0.80,
                "absolute_draw_spearman_gap_at_least_030": abs(
                    numeric["draw0_spearman"] - numeric["draw1_spearman"]
                )
                >= 0.30,
                "stronger_draw_probability_positive_at_least_080": max(
                    numeric["draw0_probability_positive"],
                    numeric["draw1_probability_positive"],
                )
                >= 0.80,
                "weaker_draw_probability_positive_at_most_050": min(
                    numeric["draw0_probability_positive"],
                    numeric["draw1_probability_positive"],
                )
                <= 0.50,
            }
        )

    draw_variance = available and all(
        gates[name]
        for name in (
            "ensemble_spearman_at_least_015",
            "ensemble_probability_positive_at_least_080",
            "ensemble_exceeds_both_probability_at_least_080",
            "ensemble_rmse_at_most_90pct_best_draw",
            "draw_rms_gap_at_least_002",
        )
    )
    shared_error = available and all(
        gates[name]
        for name in (
            "draw_pearson_at_least_080",
            "draw_rms_gap_at_most_002",
            "all_draw_and_ensemble_spearman_below_015",
            "ensemble_probability_positive_below_080",
        )
    )
    draw_sensitive = available and all(
        gates[name]
        for name in (
            "absolute_draw_spearman_gap_at_least_030",
            "stronger_draw_probability_positive_at_least_080",
            "weaker_draw_probability_positive_at_most_050",
        )
    )
    adequate = available and all(
        gates[name]
        for name in (
            "ensemble_spearman_at_least_015",
            "ensemble_probability_positive_at_least_080",
        )
    )

    if insufficient:
        region = "insufficient_changed_roots"
    elif draw_variance:
        region = "averaging_reduces_draw_noise"
    elif shared_error:
        region = "shared_ranking_error"
    elif draw_sensitive:
        region = "draw_sensitive"
    elif adequate:
        region = "fidelity_adequate_no_draw_escalation"
    else:
        region = "inconclusive_no_adaptive_repair"
    gates.update(
        {
            "ordered_region_draw_variance": bool(draw_variance),
            "ordered_region_shared_error": bool(shared_error),
            "ordered_region_draw_sensitive": bool(draw_sensitive),
            "ordered_region_fidelity_adequate": bool(adequate),
        }
    )
    return {
        "schema_version": 1,
        "interface_version": "regretbench-branch-draw-decision-1",
        "status": "complete_descriptive_non_gating",
        "region": region,
        "gates": gates,
        "next_action": NEXT_ACTIONS[region],
        "fidelity_result_sha256": audit.get("result_sha256"),
        "fidelity_verification_sha256": audit.get("verification_sha256"),
        "decision_binding_sha256": binding_sha256,
        "can_change_status_authorization_or_claim_tier": False,
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# RegretBench Branch-Draw Decision",
        "",
        f"- Region: `{result['region']}`",
        f"- Next action: {result['next_action']}",
        "- This decision is descriptive, non-gating, and non-rescuing.",
        "- Model calls: `0`; cost: `$0.00`.",
    ]
    gates = result.get("gates")
    if isinstance(gates, Mapping):
        lines.extend(["", "## Literal Gates", ""])
        for name, passed in gates.items():
            lines.append(f"- `{name}`: `{passed}`")
    lines.append("")
    return "\n".join(lines)


def run(fidelity_path: Path) -> dict[str, Any]:
    validate_binding()
    audit = _load(fidelity_path)
    if audit.get("execution_binding_sha256") != sha256_file(FIDELITY_BINDING):
        raise ValueError("fidelity artifact binding changed")
    return classify(audit, binding_sha256=sha256_file(BINDING))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    result = run(args.fidelity.resolve())
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.markdown:
        args.markdown.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.markdown.resolve().write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
