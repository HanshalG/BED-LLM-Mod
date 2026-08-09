#!/usr/bin/env python3
"""Audit answer-conditioned predictive shifts against regeneration noise."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_vlm_bed as bed


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-answer-signal-audit-1"
PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_ANSWER_SIGNAL_ABOVE_REGENERATION_NOISE_AMENDMENT_20260809.md"
)
PROTOCOL_SHA256 = "f61a7ad4fb2a4cfad3011cae30b6a03be493bce08f2b2f4a79381211efbfe4f1"
EXPECTED_TASKS = 4
EXPECTED_PAIRS = 32
MIN_DYNAMIC_PREDICTION_MAE = 0.05
MIN_POOLED_ADVANTAGE = 0.01
DEFAULT_MECHANICS_RESULT = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_mechanics_tree/"
    "bongard-openworld-luna-vlm-mechanics-tree-20260810/RESULT.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_mechanics_tree/"
    "bongard-openworld-luna-vlm-mechanics-tree-20260810/"
    "ANSWER_SIGNAL_AUDIT_RESULT.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _belief_maps(
    *,
    cases: Sequence[mechanics.BeliefCase],
    responses: Sequence[str],
) -> tuple[
    dict[str, dict[tuple[str, bool], bed.SemanticBelief]],
    dict[str, dict[tuple[str, bool], bed.SemanticBelief]],
]:
    if len(cases) != len(responses):
        raise ValueError("mechanics first-stage response count changed")
    dynamic: dict[str, dict[tuple[str, bool], bed.SemanticBelief]] = {}
    blind: dict[str, dict[tuple[str, bool], bed.SemanticBelief]] = {}
    for case, response in zip(cases, responses, strict=True):
        belief = bed.parse_belief_response(
            response,
            image_ids=case.task.image_ids,
            history=case.history,
        )
        if case.kind not in {"branch", "history_blind"}:
            continue
        key = (str(case.candidate_id), bool(case.simulated_label))
        target = dynamic if case.kind == "branch" else blind
        target.setdefault(case.task.task_id, {})[key] = belief
    return dynamic, blind


def answer_signal_metrics(
    *,
    tasks: Sequence[bed.VisualTask],
    dynamic_by_task: Mapping[
        str, Mapping[tuple[str, bool], bed.SemanticBelief]
    ],
    blind_by_task: Mapping[
        str, Mapping[tuple[str, bool], bed.SemanticBelief]
    ],
) -> dict[str, Any]:
    rows = []
    task_rows = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        dynamic = dynamic_by_task.get(task.task_id) or {}
        blind = blind_by_task.get(task.task_id) or {}
        expected = {
            (candidate_id, label)
            for candidate_id in task.candidate_ids
            for label in (False, True)
        }
        if set(dynamic) != expected or set(blind) != expected:
            raise ValueError("answer-signal branch maps are incomplete or extra")
        current = []
        for candidate_id in task.candidate_ids:
            answer_shift = serving.branch_sensitivity(
                dynamic[(candidate_id, False)],
                dynamic[(candidate_id, True)],
                candidate_id=candidate_id,
            )
            noise_shift = serving.branch_sensitivity(
                blind[(candidate_id, False)],
                blind[(candidate_id, True)],
                candidate_id=candidate_id,
            )
            dynamic_mae = float(answer_shift["unobserved_prediction_mae"])
            blind_mae = float(noise_shift["unobserved_prediction_mae"])
            row = {
                "task_id": task.task_id,
                "candidate_id": candidate_id,
                "unobserved_image_count": answer_shift["unobserved_image_count"],
                "dynamic_prediction_mae": dynamic_mae,
                "history_blind_prediction_mae": blind_mae,
                "prediction_mae_advantage": dynamic_mae - blind_mae,
                "dynamic_rule_jaccard": answer_shift["rule_jaccard"],
                "history_blind_rule_jaccard": noise_shift["rule_jaccard"],
            }
            current.append(row)
            rows.append(row)
        task_rows.append(
            {
                "task_id": task.task_id,
                "pair_count": len(current),
                "mean_dynamic_prediction_mae": sum(
                    row["dynamic_prediction_mae"] for row in current
                )
                / len(current),
                "mean_history_blind_prediction_mae": sum(
                    row["history_blind_prediction_mae"] for row in current
                )
                / len(current),
                "mean_prediction_mae_advantage": sum(
                    row["prediction_mae_advantage"] for row in current
                )
                / len(current),
            }
        )
    pooled_dynamic = sum(row["dynamic_prediction_mae"] for row in rows) / len(rows)
    pooled_blind = sum(
        row["history_blind_prediction_mae"] for row in rows
    ) / len(rows)
    pooled_advantage = pooled_dynamic - pooled_blind
    finite = all(
        math.isfinite(value)
        for row in rows
        for key, value in row.items()
        if key.endswith("mae") or key.endswith("advantage") or key.endswith("jaccard")
    )
    gates = {
        "exact_four_tasks_and_32_candidate_pairs": (
            len(tasks) == len(task_rows) == EXPECTED_TASKS
            and len(rows) == EXPECTED_PAIRS
            and all(row["pair_count"] == 8 for row in task_rows)
        ),
        "all_answer_signal_metrics_are_finite": finite,
        "pooled_answer_conditioned_prediction_mae_at_least_0_05": (
            pooled_dynamic >= MIN_DYNAMIC_PREDICTION_MAE
        ),
        "pooled_answer_conditioned_minus_history_blind_mae_at_least_0_01": (
            pooled_advantage >= MIN_POOLED_ADVANTAGE
        ),
        "answer_conditioned_mae_advantage_is_positive_in_every_task": all(
            row["mean_prediction_mae_advantage"] > 0.0 for row in task_rows
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "pair_count": len(rows),
        "task_count": len(task_rows),
        "pooled_dynamic_prediction_mae": pooled_dynamic,
        "pooled_history_blind_prediction_mae": pooled_blind,
        "pooled_prediction_mae_advantage": pooled_advantage,
        "task_summaries": task_rows,
        "candidate_rows": rows,
        "gates": gates,
    }


def build_report(
    *,
    mechanics_result: Path,
    tasks: Sequence[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("answer-signal protocol changed")
    result = _load(mechanics_result)
    raw_path = mechanics_result.parent / "private/RAW_RESPONSES.json"
    if (
        result.get("status") != "mechanics_pass"
        or not raw_path.is_file()
        or result.get("raw_responses_sha256") != sha256_file(raw_path)
    ):
        raise ValueError("answer-signal audit requires an exact mechanics pass")
    raw = _load(raw_path)
    selected_tasks = sorted(
        tasks or bed.load_mechanics_tasks(), key=lambda task: task.task_id
    )
    cases = mechanics.first_stage_cases(selected_tasks)
    expected_ids = [case.case_id for case in cases]
    responses = raw.get("first_stage_responses")
    if raw.get("first_stage_case_ids") != expected_ids or not isinstance(
        responses, list
    ):
        raise ValueError("mechanics first-stage raw response map changed")
    dynamic, blind = _belief_maps(cases=cases, responses=responses)
    metrics = answer_signal_metrics(
        tasks=selected_tasks,
        dynamic_by_task=dynamic,
        blind_by_task=blind,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "answer_signal_valid" if metrics["gates"]["all_pass"] else "gated_null"
        ),
        "protocol": {"path": str(PROTOCOL), "sha256": PROTOCOL_SHA256},
        "mechanics_result": {
            "path": str(mechanics_result.resolve()),
            "sha256": sha256_file(mechanics_result),
            "raw_responses_sha256": sha256_file(raw_path),
        },
        "thresholds": {
            "minimum_pooled_dynamic_prediction_mae": MIN_DYNAMIC_PREDICTION_MAE,
            "minimum_pooled_prediction_mae_advantage": MIN_POOLED_ADVANTAGE,
            "positive_advantage_required_in_every_task": True,
        },
        "metrics": metrics,
        "model_calls_made": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "authorizes_rerun": False,
        "authorizes_development": metrics["gates"]["all_pass"],
    }


def run_report(
    *,
    mechanics_result: Path = DEFAULT_MECHANICS_RESULT,
    output_path: Path = DEFAULT_OUTPUT,
    tasks: Sequence[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    expected = build_report(mechanics_result=mechanics_result, tasks=tasks)
    if output_path.exists():
        observed = _load(output_path)
        if observed != expected:
            raise ValueError("banked answer-signal audit changed")
        return observed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(expected, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return expected


def verify_report(
    *,
    report_path: Path,
    mechanics_result: Path = DEFAULT_MECHANICS_RESULT,
    tasks: Sequence[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    observed = _load(report_path)
    expected = build_report(mechanics_result=mechanics_result, tasks=tasks)
    if observed != expected or observed.get("status") != "answer_signal_valid":
        raise ValueError("answer-signal audit is not an exact passing report")
    return {
        "verified": True,
        "report_sha256": sha256_file(report_path),
        "mechanics_result_sha256": sha256_file(mechanics_result),
        "pooled_prediction_mae_advantage": observed["metrics"][
            "pooled_prediction_mae_advantage"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanics-result", type=Path, default=DEFAULT_MECHANICS_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_report(
        mechanics_result=args.mechanics_result,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "answer_signal_valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
