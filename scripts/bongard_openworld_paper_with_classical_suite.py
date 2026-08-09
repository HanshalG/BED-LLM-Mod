#!/usr/bin/env python3
"""Render Bongard results with mandatory classical, compute, and random controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_classical_suite_outcome as suite_outcome
from scripts import bongard_openworld_compute_matched_control as compute_control
from scripts import bongard_openworld_luna_confirmation64 as confirmation
from scripts import bongard_openworld_luna_confirmation64_daily_execute as confirmation_daily
from scripts import bongard_openworld_luna_development32_daily_execute as development_daily
from scripts import bongard_openworld_luna_paper_fragment as luna_fragment
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_path_mediation as path_mediation
from scripts import bongard_openworld_random_strategy_control as random_control


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-paper-with-classical-suite-5"
LUNA_RENDERER_SHA256 = (
    "9cf6dc0e187330de7592a5d72ec2abeb465dc0a195a2d865ce69f8ca66497c53"
)
CLASSICAL_SUITE_OUTCOME_SHA256 = (
    "8c2bc93b3d416a47c2d1e19112a670f270b79712c22e4ba3d2181308e3d0e032"
)
COMPUTE_MATCHED_CONTROL_SHA256 = (
    "929eda107f8cb60caf4cd7363f07856e135adaa89f16e946edb710c1a93dfbba"
)
RANDOM_STRATEGY_CONTROL_SHA256 = (
    "f99b68adb9b0db9d066ac2aa36a11351330ff476e6df430361431d07191f7441"
)
PATH_MEDIATION_SHA256 = (
    "1aef1c9eb90757bd31fec4beb077ddf79965e1a42b2715b4f7a6788e57e8b912"
)
PATH_MEDIATION_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/BONGARD_OPENWORLD_PATH_MEDIATION_PROTOCOL_20260809.md"
)
PATH_MEDIATION_PROTOCOL_SHA256 = (
    "db9e4d53d2fb36856625b0e6d12b152a0a58d4fe460c0290903b70cbbc594d1d"
)
COMPUTE_PAPER_HANDOFF_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_COMPUTE_MATCHED_PAPER_HANDOFF_AMENDMENT_20260809.md"
)
COMPUTE_PAPER_HANDOFF_AMENDMENT_SHA256 = (
    "a1a08f899266dc8a0fbab40c741e83306eaf69307be34b210a517d16d2521f79"
)
RANDOM_PAPER_HANDOFF_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_RANDOM_STRATEGY_PAPER_HANDOFF_AMENDMENT_20260809.md"
)
RANDOM_PAPER_HANDOFF_AMENDMENT_SHA256 = (
    "56d310e8e19a22e9613f57618c6bcaf8ebdc6c1862d25dd4c9b49ad5d3b70961"
)
MEDIATION_PAPER_HANDOFF_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_PATH_MEDIATION_PAPER_HANDOFF_AMENDMENT_20260809.md"
)
MEDIATION_PAPER_HANDOFF_AMENDMENT_SHA256 = (
    "1902eff1a655bb1a8456e9c9d14e3d1bdf7adbcf76865686bef1263b1188d36b"
)
DEFAULT_OUTPUT = luna_fragment.DEFAULT_OUTPUT


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _number(value: Any, digits: int = 4) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("classical-suite paper metric is not finite")
    return f"{float(value):.{digits}f}"


def control_addendum_tex_lines(lines: Sequence[str]) -> list[str]:
    if not lines:
        raise ValueError("Bongard paper control addendum is empty")
    return ["{\\fontsize{8}{8.3}\\selectfont", *lines, "}"]


def verify_bound_implementations() -> dict[str, str]:
    observed = {
        "luna_renderer": suite_outcome.siglip.sha256_file(
            REPO_ROOT / "scripts/bongard_openworld_luna_paper_fragment.py"
        ),
        "classical_suite_outcome": suite_outcome.siglip.sha256_file(
            REPO_ROOT / "scripts/bongard_openworld_classical_suite_outcome.py"
        ),
        "compute_matched_control": suite_outcome.siglip.sha256_file(
            REPO_ROOT / "scripts/bongard_openworld_compute_matched_control.py"
        ),
        "random_strategy_control": suite_outcome.siglip.sha256_file(
            REPO_ROOT / "scripts/bongard_openworld_random_strategy_control.py"
        ),
        "path_mediation": suite_outcome.siglip.sha256_file(
            REPO_ROOT / "scripts/bongard_openworld_path_mediation.py"
        ),
        "path_mediation_protocol": suite_outcome.siglip.sha256_file(
            PATH_MEDIATION_PROTOCOL
        ),
        "compute_paper_handoff_amendment": suite_outcome.siglip.sha256_file(
            COMPUTE_PAPER_HANDOFF_AMENDMENT
        ),
        "random_paper_handoff_amendment": suite_outcome.siglip.sha256_file(
            RANDOM_PAPER_HANDOFF_AMENDMENT
        ),
        "mediation_paper_handoff_amendment": suite_outcome.siglip.sha256_file(
            MEDIATION_PAPER_HANDOFF_AMENDMENT
        ),
    }
    expected = {
        "luna_renderer": LUNA_RENDERER_SHA256,
        "classical_suite_outcome": CLASSICAL_SUITE_OUTCOME_SHA256,
        "compute_matched_control": COMPUTE_MATCHED_CONTROL_SHA256,
        "random_strategy_control": RANDOM_STRATEGY_CONTROL_SHA256,
        "path_mediation": PATH_MEDIATION_SHA256,
        "path_mediation_protocol": PATH_MEDIATION_PROTOCOL_SHA256,
        "compute_paper_handoff_amendment": (
            COMPUTE_PAPER_HANDOFF_AMENDMENT_SHA256
        ),
        "random_paper_handoff_amendment": RANDOM_PAPER_HANDOFF_AMENDMENT_SHA256,
        "mediation_paper_handoff_amendment": (
            MEDIATION_PAPER_HANDOFF_AMENDMENT_SHA256
        ),
    }
    if observed != expected:
        raise ValueError(
            "bound Bongard classical-suite renderers changed: expected "
            f"{expected}, observed {observed}"
        )
    suite_outcome.verify_frozen_inputs()
    return observed


def _encoder_tex(
    *,
    subject: str,
    comparison_name: str,
    pooled: Mapping[str, Any],
    myopic_policy: str,
    depth2_policy: str,
    horizon: Mapping[str, Any],
    luna_comparison: Mapping[str, Any],
) -> str:
    horizon_ci = horizon["bootstrap_95pct_ci"]
    luna_ci = luna_comparison["bootstrap_95pct_ci"]
    values = (
        pooled[myopic_policy]["mean_brier"],
        pooled[depth2_policy]["mean_brier"],
        horizon["mean"],
        *horizon_ci,
        luna_comparison["mean"],
        *luna_ci,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("classical-suite paper values are non-finite")
    return (
        f"{subject} obtained endpoint Brier "
        f"{_number(pooled[myopic_policy]['mean_brier'])} with one-step PIG and "
        f"{_number(pooled[depth2_policy]['mean_brier'])} with exact depth-two "
        "lookahead. Its paired depth-two minus myopic difference was "
        f"{_number(horizon['mean'])} (95\\% bootstrap CI "
        f"$[{_number(horizon_ci[0])},{_number(horizon_ci[1])}]$). Luna dynamic "
        f"depth two minus {comparison_name} depth two was "
        f"{_number(luna_comparison['mean'])} in Brier (95\\% bootstrap CI "
        f"$[{_number(luna_ci[0])},{_number(luna_ci[1])}]$)."
    )


def classical_tex_lines(result: Mapping[str, Any]) -> list[str]:
    if (
        result.get("status") != "classical_suite_complete"
        or result.get("all_gates_pass") is not True
        or result.get("authorizes_paid_calls") is not False
    ):
        raise ValueError("paper input is not a complete frozen classical suite")
    dino = result["dino"]
    siglip = result["siglip"]
    dino_comparison = result["paired_luna_minus_dino"][
        "luna_dynamic_minus_dinov2_depth2"
    ]["mean_brier"]
    siglip_comparison = result["paired_luna_minus_siglip"][
        "luna_dynamic_minus_siglip_depth2"
    ]["mean_brier"]
    return [
        "\\paragraph{Frozen classical vision comparators.}",
        _encoder_tex(
            subject="The fixed DINOv2-small prototype model",
            comparison_name="DINOv2",
            pooled=dino["pooled"],
            myopic_policy="dinov2_myopic",
            depth2_policy="dinov2_depth2",
            horizon=dino["paired_depth2_minus_myopic"]["mean_brier"],
            luna_comparison=dino_comparison,
        ),
        _encoder_tex(
            subject="The fixed SigLIP2-So400m prototype model",
            comparison_name="SigLIP2",
            pooled=siglip["pooled"],
            myopic_policy="siglip_myopic",
            depth2_policy="siglip_depth2",
            horizon=siglip["paired_depth2_minus_myopic"]["mean_brier"],
            luna_comparison=siglip_comparison,
        ),
        (
            "For Luna-minus-classical Brier, negative values favor Luna. Both "
            "comparators are reported regardless of direction and do not test "
            "universal classical impossibility."
        ),
    ]


def _validated_compute_summary(
    comparison: Mapping[str, Any], metric: str, *, task_count: int
) -> dict[str, Any]:
    summary = comparison.get(metric)
    if not isinstance(summary, Mapping):
        raise ValueError(f"compute-matched paper input lacks {metric}")
    values = {
        name: summary.get(name)
        for name in ("mean_difference", "sample_sd", "standard_error")
    }
    interval = summary.get("ci95")
    counts = [summary.get(name) for name in ("wins", "ties", "losses")]
    if (
        summary.get("n") != task_count
        or summary.get("bootstrap_draws") != 20_000
        or summary.get("negative_favors") != "dynamic_depth2"
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values.values()
        )
        or not isinstance(interval, Sequence)
        or isinstance(interval, (str, bytes))
        or len(interval) != 2
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in interval
        )
        or float(interval[0]) > float(interval[1])
        or not all(isinstance(value, int) and value >= 0 for value in counts)
        or sum(counts) != task_count
        or not math.isclose(
            float(values["standard_error"]),
            float(values["sample_sd"]) / math.sqrt(task_count),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"compute-matched paper input has invalid {metric}")
    return {
        "n": task_count,
        "mean_difference": float(values["mean_difference"]),
        "sample_sd": float(values["sample_sd"]),
        "standard_error": float(values["standard_error"]),
        "ci95": [float(interval[0]), float(interval[1])],
        "bootstrap_draws": 20_000,
        "wins": int(counts[0]),
        "ties": int(counts[1]),
        "losses": int(counts[2]),
        "negative_favors": "dynamic_depth2",
    }


def compute_matched_tex_lines(
    result: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    task_count = result.get("task_count")
    if (
        result.get("status") != "compute_matched_control_audit_complete"
        or result.get("compute_contract_exact") is not True
        or result.get("strict_compute_matched_control")
        != "shuffled_dynamic_depth2"
        or result.get("matched_request_count_control")
        != "history_blind_depth2"
        or result.get("online_regeneration_greedy_control") != "myopic_width"
        or not isinstance(task_count, int)
        or task_count <= 0
        or result.get("model_calls") != 0
        or result.get("cost_usd") != 0.0
        or result.get("authorizes_paid_calls") is not False
        or result.get("changes_claim_tier") is not False
    ):
        raise ValueError("paper input is not the complete compute-matched audit")
    comparisons = result.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise ValueError("compute-matched paper comparisons are missing")
    shuffled = comparisons.get("shuffled_dynamic_depth2")
    if not isinstance(shuffled, Mapping):
        raise ValueError("strict compute-matched comparison is missing")
    brier = _validated_compute_summary(
        shuffled, "mean_brier", task_count=task_count
    )
    log_loss = _validated_compute_summary(
        shuffled, "mean_log_loss", task_count=task_count
    )
    first_changes = shuffled.get("first_query_changes")
    history_changes = shuffled.get("final_history_changes")
    if (
        not isinstance(first_changes, int)
        or not 0 <= first_changes <= task_count
        or not isinstance(history_changes, int)
        or not 0 <= history_changes <= task_count
    ):
        raise ValueError("compute-matched action-change counts are invalid")
    lines = [
        "\\paragraph{Compute-matched shuffled continuation.}",
        (
            "The strict branch-bank-compute-matched shuffle preserved root scores "
            "and the continuation-value multiset. Dynamic minus shuffled was "
            f"{_number(brier['mean_difference'])} Brier (95\\% CI "
            f"$[{_number(brier['ci95'][0])},{_number(brier['ci95'][1])}]$) and "
            f"{_number(log_loss['mean_difference'])} log loss (95\\% CI "
            f"$[{_number(log_loss['ci95'][0])},{_number(log_loss['ci95'][1])}]$); "
            f"first query/final history changed on {first_changes}/{task_count} "
            f"and {history_changes}/{task_count}. Negative favors dynamic; this "
            "all-task audit is descriptive and non-gating."
        ),
    ]
    metadata = {
        "strict_compute_matched_control": "shuffled_dynamic_depth2",
        "task_count": task_count,
        "mean_brier": brier,
        "mean_log_loss": log_loss,
        "first_query_changes": first_changes,
        "final_history_changes": history_changes,
        "changes_claim_tier": False,
        "authorizes_paid_calls": False,
    }
    return lines, metadata


def _validated_random_summary(
    comparisons: Mapping[str, Any], metric: str, *, task_count: int
) -> dict[str, Any]:
    summary = comparisons.get(metric)
    if not isinstance(summary, Mapping):
        raise ValueError(f"random-strategy paper input lacks {metric}")
    values = {
        name: summary.get(name)
        for name in ("mean_difference", "sample_sd", "standard_error")
    }
    interval = summary.get("ci95")
    counts = [summary.get(name) for name in ("wins", "ties", "losses")]
    if (
        summary.get("n") != task_count
        or summary.get("bootstrap_draws") != 20_000
        or summary.get("negative_favors") != "dynamic_depth2"
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values.values()
        )
        or not isinstance(interval, Sequence)
        or isinstance(interval, (str, bytes))
        or len(interval) != 2
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in interval
        )
        or float(interval[0]) > float(interval[1])
        or not all(isinstance(value, int) and value >= 0 for value in counts)
        or sum(counts) != task_count
        or not math.isclose(
            float(values["standard_error"]),
            float(values["sample_sd"]) / math.sqrt(task_count),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"random-strategy paper input has invalid {metric}")
    return {
        "n": task_count,
        "mean_difference": float(values["mean_difference"]),
        "sample_sd": float(values["sample_sd"]),
        "standard_error": float(values["standard_error"]),
        "ci95": [float(interval[0]), float(interval[1])],
        "bootstrap_draws": 20_000,
        "wins": int(counts[0]),
        "ties": int(counts[1]),
        "losses": int(counts[2]),
        "negative_favors": "dynamic_depth2",
    }


def random_strategy_tex_lines(
    result: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    task_count = result.get("task_count")
    if (
        result.get("status") != "random_strategy_control_audit_complete"
        or result.get("random_policy_draws_without_replacement") is not True
        or result.get("random_draws_replayed_exactly") is not True
        or not isinstance(task_count, int)
        or task_count <= 0
        or result.get("model_calls") != 0
        or result.get("cost_usd") != 0.0
        or result.get("authorizes_paid_calls") is not False
        or result.get("changes_claim_tier") is not False
    ):
        raise ValueError("paper input is not the complete random-strategy audit")
    comparisons = result.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise ValueError("random-strategy paper comparisons are missing")
    brier = _validated_random_summary(
        comparisons, "mean_brier", task_count=task_count
    )
    log_loss = _validated_random_summary(
        comparisons, "mean_log_loss", task_count=task_count
    )
    first_changes = result.get("first_query_changes")
    history_changes = result.get("final_history_changes")
    if (
        not isinstance(first_changes, int)
        or not 0 <= first_changes <= task_count
        or not isinstance(history_changes, int)
        or not 0 <= history_changes <= task_count
    ):
        raise ValueError("random-strategy action-change counts are invalid")
    lines = [
        "\\paragraph{Frozen random-strategy sanity baseline.}",
        (
            "Dynamic minus task-hashed random was "
            f"{_number(brier['mean_difference'])} Brier (95\\% CI "
            f"$[{_number(brier['ci95'][0])},{_number(brier['ci95'][1])}]$; "
            f"{brier['wins']}/{brier['ties']}/{brier['losses']} W/T/L) and "
            f"{_number(log_loss['mean_difference'])} log loss (95\\% CI "
            f"$[{_number(log_loss['ci95'][0])},{_number(log_loss['ci95'][1])}]$); "
            f"first query/final history changed on {first_changes}/{task_count} "
            f"and {history_changes}/{task_count}. Negative favors dynamic. Random "
            "is descriptive, non-gating, and not compute matched."
        ),
    ]
    metadata = {
        "task_count": task_count,
        "mean_brier": brier,
        "mean_log_loss": log_loss,
        "first_query_changes": first_changes,
        "final_history_changes": history_changes,
        "changes_claim_tier": False,
        "authorizes_paid_calls": False,
        "compute_matched": False,
    }
    return lines, metadata


def _validated_mediation_effect(
    effect: Mapping[str, Any], *, task_count: int
) -> dict[str, Any]:
    values = {
        name: effect.get(name)
        for name in (
            "mean_difference",
            "sample_sd",
            "bootstrap_probability_improvement",
        )
    }
    interval = effect.get("ci95")
    counts = [effect.get(name) for name in ("wins", "ties", "losses")]
    if (
        effect.get("n") != task_count
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values.values()
        )
        or not 0.0 <= float(values["bootstrap_probability_improvement"]) <= 1.0
        or not isinstance(interval, Sequence)
        or isinstance(interval, (str, bytes))
        or len(interval) != 2
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in interval
        )
        or float(interval[0]) > float(interval[1])
        or not all(isinstance(value, int) and value >= 0 for value in counts)
        or sum(counts) != task_count
    ):
        raise ValueError("path-mediation paper input has invalid all-task effect")
    return {
        "n": task_count,
        "mean_difference": float(values["mean_difference"]),
        "sample_sd": float(values["sample_sd"]),
        "ci95": [float(interval[0]), float(interval[1])],
        "bootstrap_probability_improvement": float(
            values["bootstrap_probability_improvement"]
        ),
        "wins": int(counts[0]),
        "ties": int(counts[1]),
        "losses": int(counts[2]),
    }


def path_mediation_tex_lines(
    result: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    task_count = result.get("task_count")
    if (
        result.get("status") != "path_mediation_complete"
        or not isinstance(task_count, int)
        or task_count <= 0
        or result.get("model_calls") != 0
        or result.get("cost_usd") != 0.0
        or result.get("authorizes_paid_calls") is not False
        or result.get("changes_claim_tier") is not False
    ):
        raise ValueError("paper input is not the complete path-mediation audit")
    summary = result.get("summary")
    endpoint_effects = result.get("endpoint_effects")
    if not isinstance(summary, Mapping) or not isinstance(
        endpoint_effects, Mapping
    ):
        raise ValueError("path-mediation paper summaries are missing")
    count_names = (
        "second_action_changed",
        "robust_second_action_changed",
        "both_supports_robustly_prefer_own_action",
    )
    counts = {name: summary.get(name) for name in count_names}
    metric_names = (
        "mean_rule_jaccard",
        "mean_candidate_predictive_probability_mae",
        "mean_endpoint_predictive_probability_mae",
        "mean_second_query_score_spearman",
        "endpoint_shift_vs_realized_brier_benefit_spearman",
        "dynamic_action_gap_vs_realized_brier_benefit_spearman",
    )
    metrics = {name: summary.get(name) for name in metric_names}
    if (
        not all(
            isinstance(value, int) and 0 <= value <= task_count
            for value in counts.values()
        )
        or counts["robust_second_action_changed"]
        > counts["second_action_changed"]
        or counts["both_supports_robustly_prefer_own_action"]
        > counts["second_action_changed"]
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in metrics.values()
        )
        or not 0.0 <= float(metrics["mean_rule_jaccard"]) <= 1.0
        or float(metrics["mean_candidate_predictive_probability_mae"]) < 0.0
        or float(metrics["mean_endpoint_predictive_probability_mae"]) < 0.0
        or not -1.0 <= float(metrics["mean_second_query_score_spearman"]) <= 1.0
        or not -1.0
        <= float(metrics["endpoint_shift_vs_realized_brier_benefit_spearman"])
        <= 1.0
        or not -1.0
        <= float(metrics["dynamic_action_gap_vs_realized_brier_benefit_spearman"])
        <= 1.0
    ):
        raise ValueError("path-mediation paper summary is invalid")
    all_tasks = endpoint_effects.get("all_tasks")
    if not isinstance(all_tasks, Mapping):
        raise ValueError("path-mediation all-task endpoint effects are missing")
    brier_raw = all_tasks.get("dynamic_minus_history_blind_brier")
    if not isinstance(brier_raw, Mapping):
        raise ValueError("path-mediation all-task Brier effect is missing")
    brier = _validated_mediation_effect(brier_raw, task_count=task_count)
    lines = [
        "\\paragraph{Replayed belief-to-action mediation.}",
        (
            "After the same first query/answer, dynamic versus same-seed blind "
            f"beliefs had rule Jaccard {_number(metrics['mean_rule_jaccard'])}, "
            f"candidate/endpoint predictive MAE "
            f"{_number(metrics['mean_candidate_predictive_probability_mae'])}/"
            f"{_number(metrics['mean_endpoint_predictive_probability_mae'])}, and "
            f"score Spearman {_number(metrics['mean_second_query_score_spearman'], 3)}. "
            f"Second actions changed/robustly changed/mutually robust on "
            f"{counts['second_action_changed']}/{counts['robust_second_action_changed']}/"
            f"{counts['both_supports_robustly_prefer_own_action']} of {task_count}. "
            f"Dynamic minus matched-blind Brier was {_number(brier['mean_difference'])} "
            f"(95\\% CI $[{_number(brier['ci95'][0])},{_number(brier['ci95'][1])}]$); "
            "endpoint-shift/action-gap correlations with realized benefit were "
            f"{_number(metrics['endpoint_shift_vs_realized_brier_benefit_spearman'], 3)}/"
            f"{_number(metrics['dynamic_action_gap_vs_realized_brier_benefit_spearman'], 3)}. "
            "Associations are descriptive and non-gating."
        ),
    ]
    metadata = {
        "task_count": task_count,
        **{name: int(value) for name, value in counts.items()},
        **{name: float(value) for name, value in metrics.items()},
        "all_task_dynamic_minus_history_blind_brier": brier,
        "changes_claim_tier": False,
        "authorizes_paid_calls": False,
    }
    return lines, metadata


def _validated_paired_summary(
    comparison: Mapping[str, Any], metric: str
) -> dict[str, Any]:
    summary = comparison.get(metric)
    if not isinstance(summary, Mapping):
        raise ValueError(f"classical challenge lacks {metric}")
    mean = summary.get("mean")
    interval = summary.get("bootstrap_95pct_ci")
    if (
        not isinstance(mean, (int, float))
        or not math.isfinite(float(mean))
        or not isinstance(interval, Sequence)
        or isinstance(interval, (str, bytes))
        or len(interval) != 2
        or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in interval
        )
        or float(interval[0]) > float(interval[1])
        or summary.get("bootstrap_draws") != 20_000
    ):
        raise ValueError(f"classical challenge has invalid {metric}")
    return {
        "mean": float(mean),
        "bootstrap_95pct_ci": [float(interval[0]), float(interval[1])],
        "bootstrap_draws": 20_000,
    }


def classical_claim_scope(result: Mapping[str, Any]) -> dict[str, Any]:
    """Classify task-level scope without changing the within-Luna claim tier."""
    if (
        result.get("status") != "classical_suite_complete"
        or result.get("all_gates_pass") is not True
        or result.get("authorizes_paid_calls") is not False
    ):
        raise ValueError("classical claim scope requires the complete frozen suite")
    comparisons = {
        "dinov2_depth2": result.get("paired_luna_minus_dino", {}).get(
            "luna_dynamic_minus_dinov2_depth2"
        ),
        "siglip_depth2": result.get("paired_luna_minus_siglip", {}).get(
            "luna_dynamic_minus_siglip_depth2"
        ),
    }
    details: dict[str, Any] = {}
    for name, comparison in comparisons.items():
        if not isinstance(comparison, Mapping):
            raise ValueError(f"classical challenge lacks {name}")
        brier = _validated_paired_summary(comparison, "mean_brier")
        log_loss = _validated_paired_summary(comparison, "mean_log_loss")
        details[name] = {
            "luna_minus_classical_brier": brier,
            "luna_minus_classical_log_loss": log_loss,
            "brier_interval_strictly_below_zero": (
                brier["bootstrap_95pct_ci"][1] < 0.0
            ),
            "log_loss_nonworse": log_loss["mean"] <= 0.0,
        }
        details[name]["clears_challenge"] = (
            brier["mean"] < 0.0
            and details[name]["brier_interval_strictly_below_zero"]
            and details[name]["log_loss_nonworse"]
        )
    clears_both = all(row["clears_challenge"] for row in details.values())
    return {
        "status": (
            "luna_clears_both_fixed_classical_challengers"
            if clears_both
            else "within_luna_only_no_task_level_necessity"
        ),
        "clears_both_fixed_classical_challengers": clears_both,
        "criterion": (
            "For both DINOv2 depth two and SigLIP2 depth two, Luna dynamic minus "
            "classical Brier must have mean below zero and paired two-sided 95% "
            "bootstrap upper endpoint below zero, with mean log loss nonworse."
        ),
        "changes_within_luna_claim_tier": False,
        "authorizes_paid_calls": False,
        "comparisons": details,
    }


def _headline_with_classical_scope(
    *,
    original_headline: str,
    original_metadata: Mapping[str, Any],
    scope: Mapping[str, Any] | None,
) -> str:
    headline = original_metadata.get("headline")
    if not isinstance(headline, Mapping):
        raise ValueError("original Luna headline metadata is missing")
    authorized = headline.get("authorized") is True
    abstract_tex = headline.get("abstract_tex")
    contribution_tex = headline.get("contribution_tex")
    if not isinstance(abstract_tex, str) or not isinstance(contribution_tex, str):
        raise ValueError("original Luna headline copy is malformed")
    if authorized is not bool(abstract_tex and contribution_tex):
        raise ValueError("original Luna headline authorization is inconsistent")
    if not authorized:
        return original_headline
    if (
        original_metadata.get("stage") != "confirmation"
        or original_metadata.get("claim_tier") != "full_llm_native_confirmation"
        or scope is None
    ):
        raise ValueError("classical headline qualification lacks full confirmation")
    if scope.get("clears_both_fixed_classical_challengers") is True:
        abstract_qualifier = (
            " Luna dynamic depth two also clears both frozen fixed semantic-vision "
            "challengers: its paired endpoint-Brier intervals versus DINOv2 and "
            "SigLIP2 are strictly below zero with nonworse mean log loss."
        )
        contribution_qualifier = (
            "\\item a prospectively scoped comparison showing lower endpoint Brier "
            "than both frozen DINOv2 and SigLIP2 depth-two planners, with paired "
            "intervals below zero and nonworse mean log loss;"
        )
    elif scope.get("status") == "within_luna_only_no_task_level_necessity":
        abstract_qualifier = (
            " This is a within-Luna path-dependent-belief result; it does not clear "
            "both frozen classical challengers and therefore does not establish "
            "task-level LLM necessity."
        )
        contribution_qualifier = (
            "\\item a prospectively scoped classical comparison that limits the "
            "Bongard finding to within-Luna path-dependent belief dynamics rather "
            "than task-level LLM necessity;"
        )
    else:
        raise ValueError("unknown classical headline scope")
    return "\n".join(
        [
            "\\renewcommand{\\BongardAbstractResult}{%",
            abstract_tex + abstract_qualifier,
            "}",
            "\\renewcommand{\\BongardContributionResult}{%",
            contribution_tex,
            contribution_qualifier,
            "}",
            "",
        ]
    )


def replay_classical_suite(
    *,
    stage: str,
    saved_path: Path,
    result_path: Path,
    block_results: Sequence[Path],
    wrapper_result: Path | None,
    output_path: Path,
) -> dict[str, Any]:
    replay = suite_outcome.run_outcome(
        stage=stage,
        result_path=result_path,
        output_path=output_path,
        block_results=block_results,
        wrapper_result=wrapper_result,
    )
    saved = _load(saved_path)
    if _canonical(saved) != _canonical(replay):
        raise ValueError("saved classical-suite outcome does not independently replay")
    return replay


def replay_compute_matched_audit(
    *,
    stage: str,
    saved_path: Path,
    result_path: Path,
    block_results: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    replay = compute_control.run_report(
        stage=stage,
        result_path=result_path,
        output_path=output_path,
        block_results=block_results,
        wrapper_result=None,
    )
    saved = _load(saved_path)
    if _canonical(saved) != _canonical(replay):
        raise ValueError("saved compute-matched audit does not independently replay")
    return replay


def replay_random_strategy_audit(
    *,
    stage: str,
    saved_path: Path,
    result_path: Path,
    block_results: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    replay = random_control.run_report(
        stage=stage,
        result_path=result_path,
        output_path=output_path,
        block_results=block_results,
        wrapper_result=None,
    )
    saved = _load(saved_path)
    if _canonical(saved) != _canonical(replay):
        raise ValueError("saved random-strategy audit does not independently replay")
    return replay


def replay_path_mediation(
    *,
    stage: str,
    saved_path: Path,
    result_path: Path,
    block_results: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    replay = path_mediation.run_report(
        stage=stage,
        result_path=result_path,
        output_path=output_path,
        block_results=block_results,
        wrapper_result=None,
    )
    saved = _load(saved_path)
    if _canonical(saved) != _canonical(replay):
        raise ValueError("saved path-mediation report does not independently replay")
    return replay


def write_combined_fragment(
    *,
    stage: str,
    output: Path,
    classical_suite_path: Path | None,
    compute_audit_path: Path | None,
    random_audit_path: Path | None,
    mediation_path: Path | None,
    claim_report_path: Path | None = None,
    combined_result: Path | None = None,
    block_results: Sequence[Path] = (),
    failure_path: Path | None = None,
) -> dict[str, Any]:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    bound = verify_bound_implementations()
    with tempfile.TemporaryDirectory(prefix="bongard-paper-classical-") as tmp:
        temporary = Path(tmp)
        original_output = temporary / "bongard_openworld_result.tex"
        original = luna_fragment.write_fragment(
            stage=stage,
            output=original_output,
            claim_report_path=claim_report_path,
            combined_result=combined_result,
            block_results=block_results,
            failure_path=failure_path,
        )
        original_tex = original_output.read_text(encoding="utf-8")
        original_headline = original_output.with_name(
            luna_fragment.HEADLINE_FILENAME
        ).read_text(encoding="utf-8")
        original_metadata = _load(original_output.with_suffix(".json"))

        if stage == "confirmation-mechanics-failure":
            if classical_suite_path is not None:
                raise ValueError(
                    "mechanics failure cannot have a classical endpoint result"
                )
            if compute_audit_path is not None:
                raise ValueError(
                    "mechanics failure cannot have a compute-matched endpoint audit"
                )
            if random_audit_path is not None:
                raise ValueError(
                    "mechanics failure cannot have a random-strategy endpoint audit"
                )
            if mediation_path is not None:
                raise ValueError(
                    "mechanics failure cannot have a path-mediation endpoint report"
                )
            addendum = [
                "\\paragraph{Frozen classical vision comparators.}",
                "No DINO or SigLIP endpoint comparison is rendered because no replay-verified combined endpoint result exists.",
            ]
            suite_metadata = None
            compute_metadata = None
            random_metadata = None
            mediation_metadata = None
            claim_scope = None
        else:
            if (
                classical_suite_path is None
                or compute_audit_path is None
                or random_audit_path is None
                or mediation_path is None
                or combined_result is None
            ):
                raise ValueError(
                    "endpoint result rendering requires the frozen classical, "
                    "compute-matched, random-strategy, and path-mediation suites"
                )
            replay = replay_classical_suite(
                stage=stage,
                saved_path=classical_suite_path,
                result_path=combined_result,
                block_results=block_results,
                wrapper_result=None,
                output_path=temporary / "CLASSICAL_SUITE_REPLAY.json",
            )
            addendum = classical_tex_lines(replay)
            claim_scope = classical_claim_scope(replay)
            suite_metadata = {
                "outcome_sha256": suite_outcome.siglip.sha256_file(
                    classical_suite_path
                ),
                "stage_result_sha256": replay["stage_result_sha256"],
                "status": replay["status"],
                "claim_scope": claim_scope,
            }
            compute_replay = replay_compute_matched_audit(
                stage=stage,
                saved_path=compute_audit_path,
                result_path=combined_result,
                block_results=block_results,
                output_path=temporary / "COMPUTE_MATCHED_REPLAY.json",
            )
            if (
                compute_replay.get("stage") != stage
                or compute_replay.get("stage_result_sha256")
                != replay.get("stage_result_sha256")
            ):
                raise ValueError(
                    "compute-matched audit does not match the rendered stage result"
                )
            compute_lines, compute_summary = compute_matched_tex_lines(
                compute_replay
            )
            addendum.extend(compute_lines)
            compute_metadata = {
                "outcome_sha256": suite_outcome.siglip.sha256_file(
                    compute_audit_path
                ),
                "stage_result_sha256": compute_replay["stage_result_sha256"],
                "status": compute_replay["status"],
                "summary": compute_summary,
            }
            random_replay = replay_random_strategy_audit(
                stage=stage,
                saved_path=random_audit_path,
                result_path=combined_result,
                block_results=block_results,
                output_path=temporary / "RANDOM_STRATEGY_REPLAY.json",
            )
            if (
                random_replay.get("stage") != stage
                or random_replay.get("stage_result_sha256")
                != replay.get("stage_result_sha256")
            ):
                raise ValueError(
                    "random-strategy audit does not match the rendered stage result"
                )
            random_lines, random_summary = random_strategy_tex_lines(
                random_replay
            )
            addendum.extend(random_lines)
            random_metadata = {
                "outcome_sha256": suite_outcome.siglip.sha256_file(
                    random_audit_path
                ),
                "stage_result_sha256": random_replay["stage_result_sha256"],
                "status": random_replay["status"],
                "summary": random_summary,
            }
            mediation_replay = replay_path_mediation(
                stage=stage,
                saved_path=mediation_path,
                result_path=combined_result,
                block_results=block_results,
                output_path=temporary / "PATH_MEDIATION_REPLAY.json",
            )
            if (
                mediation_replay.get("stage") != stage
                or mediation_replay.get("stage_result_sha256")
                != replay.get("stage_result_sha256")
            ):
                raise ValueError(
                    "path-mediation report does not match the rendered stage result"
                )
            mediation_lines, mediation_summary = path_mediation_tex_lines(
                mediation_replay
            )
            addendum.extend(mediation_lines)
            mediation_metadata = {
                "outcome_sha256": suite_outcome.siglip.sha256_file(
                    mediation_path
                ),
                "stage_result_sha256": mediation_replay["stage_result_sha256"],
                "status": mediation_replay["status"],
                "summary": mediation_summary,
            }

        formatted_addendum = control_addendum_tex_lines(addendum)
        combined_tex = (
            original_tex.rstrip()
            + "\n\n"
            + "\n".join(formatted_addendum)
            + "\n"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(combined_tex, encoding="utf-8")
        headline_path = output.with_name(luna_fragment.HEADLINE_FILENAME)
        qualified_headline = _headline_with_classical_scope(
            original_headline=original_headline,
            original_metadata=original_metadata,
            scope=claim_scope,
        )
        headline_path.write_text(qualified_headline, encoding="utf-8")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "stage": stage,
            "claim_tier": original_metadata["claim_tier"],
            "bound_implementations": bound,
            "original_renderer": original,
            "original_metadata": original_metadata,
            "classical_suite": suite_metadata,
            "classical_claim_scope": claim_scope,
            "compute_matched_audit": compute_metadata,
            "random_strategy_audit": random_metadata,
            "path_mediation": mediation_metadata,
            "tex_sha256": suite_outcome.siglip.sha256_file(output),
            "headline_tex_sha256": suite_outcome.siglip.sha256_file(
                headline_path
            ),
            "model_calls": 0,
            "cost_usd": 0.0,
        }
        metadata_path = output.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "status": (
            "written_with_mandatory_classical_compute_random_and_mediation_suites"
        ),
        "stage": stage,
        "claim_tier": metadata["claim_tier"],
        "tex_path": str(output),
        "tex_sha256": metadata["tex_sha256"],
        "headline_path": str(headline_path),
        "headline_sha256": metadata["headline_tex_sha256"],
        "metadata_path": str(metadata_path),
        "metadata_sha256": suite_outcome.siglip.sha256_file(metadata_path),
        "model_calls": 0,
        "cost_usd": 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("development", "confirmation", "confirmation-mechanics-failure"),
        required=True,
    )
    parser.add_argument("--classical-suite", type=Path)
    parser.add_argument("--compute-matched-audit", type=Path)
    parser.add_argument("--random-strategy-audit", type=Path)
    parser.add_argument("--path-mediation", type=Path)
    parser.add_argument("--claim-report", type=Path)
    parser.add_argument("--combined-result", type=Path)
    parser.add_argument("--block-result", type=Path, action="append", default=[])
    parser.add_argument("--failure-record", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocks = args.block_result
    if not blocks:
        blocks = (
            [
                development_daily.BLOCK_DIRS[block_id] / "RESULT.json"
                for block_id in development.BLOCK_ORDER
            ]
            if args.stage == "development"
            else [
                confirmation_daily.BLOCK_DIRS[block_id] / "RESULT.json"
                for block_id in confirmation.BLOCK_ORDER
            ]
        )
    combined = args.combined_result
    claim = args.claim_report
    if args.stage == "development":
        combined = combined or development_daily.COMBINED_RESULT
        claim = claim or development_daily.ROOT / "CLAIM_REPORT.json"
    elif combined is None:
        combined = confirmation_daily.COMBINED_RESULT
    result = write_combined_fragment(
        stage=args.stage,
        output=args.output.resolve(),
        classical_suite_path=args.classical_suite,
        compute_audit_path=args.compute_matched_audit,
        random_audit_path=args.random_strategy_audit,
        mediation_path=args.path_mediation,
        claim_report_path=claim,
        combined_result=combined,
        block_results=blocks,
        failure_path=args.failure_record,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
