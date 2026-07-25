#!/usr/bin/env python3
"""Audit whether full-minus-myopic scores rank exact future retrieval gain."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tau_knowledge_first_link_scorer import pairwise_ranking_points
from scripts.tau_knowledge_retrieval_opportunity import analyze_record


FIRST_LINK_SHA256 = (
    "dfbf597f8405b109d61d90206606962b5fbda439c8b81f086a9592c07fa247d1"
)
RECEDING_SHA256 = (
    "f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _argmax(values: Sequence[int]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def exact_half_step_sign_flip_pvalue(values: Sequence[float]) -> float:
    scaled_values = []
    for value in values:
        scaled = round(abs(value) * 2)
        if abs(abs(value) * 2 - scaled) > 1e-9:
            raise ValueError("sign-flip input is not a half-step value")
        if scaled:
            scaled_values.append(scaled)
    if not scaled_values:
        return 1.0
    observed = round(sum(values) * 2)
    distribution: Counter[int] = Counter({0: 1})
    for value in scaled_values:
        updated: Counter[int] = Counter()
        for subtotal, count in distribution.items():
            updated[subtotal - value] += count
            updated[subtotal + value] += count
        distribution = updated
    extreme = sum(
        count for subtotal, count in distribution.items() if subtotal >= observed
    )
    return extreme / (2 ** len(scaled_values))


def _task_row(
    record: dict[str, Any],
    myopic: dict[str, Any],
    full_tree: dict[str, Any],
) -> dict[str, Any]:
    endpoint = analyze_record(record)
    immediate = endpoint["one_step_counts"]
    total = [max(values) for values in endpoint["pair_counts"]]
    future_gain = [
        total_value - immediate_value
        for total_value, immediate_value in zip(
            total,
            immediate,
            strict=True,
        )
    ]
    myopic_scores = myopic["scores"]
    full_scores = full_tree["scores"]
    if not (
        len(immediate)
        == len(myopic_scores)
        == len(full_scores)
        == len(future_gain)
    ):
        raise ValueError("tau root arrays do not align")
    uplift_scores = [
        full_score - myopic_score
        for full_score, myopic_score in zip(
            full_scores,
            myopic_scores,
            strict=True,
        )
    ]
    uplift_points, comparable = pairwise_ranking_points(
        uplift_scores,
        future_gain,
    )
    full_gain_points, full_gain_comparable = pairwise_ranking_points(
        full_scores,
        future_gain,
    )
    myopic_gain_points, myopic_gain_comparable = pairwise_ranking_points(
        myopic_scores,
        future_gain,
    )
    if not comparable == full_gain_comparable == myopic_gain_comparable:
        raise ValueError("future-gain comparable counts differ")
    myopic_immediate_points, immediate_comparable = pairwise_ranking_points(
        myopic_scores,
        immediate,
    )
    full_total_points, total_comparable = pairwise_ranking_points(
        full_scores,
        total,
    )
    uplift_root = _argmax(uplift_scores)
    myopic_root = _argmax(myopic_scores)
    full_root = _argmax(full_scores)
    return {
        "task_id": record["task_id"],
        "immediate_values": immediate,
        "total_values": total,
        "future_gain_values": future_gain,
        "uplift_scores": uplift_scores,
        "future_gain_comparable": comparable,
        "uplift_future_gain_points": uplift_points,
        "full_future_gain_points": full_gain_points,
        "myopic_future_gain_points": myopic_gain_points,
        "myopic_immediate_comparable": immediate_comparable,
        "myopic_immediate_points": myopic_immediate_points,
        "full_total_comparable": total_comparable,
        "full_total_points": full_total_points,
        "uplift_root_index": uplift_root,
        "myopic_root_index": myopic_root,
        "full_root_index": full_root,
        "uplift_selected_future_gain": future_gain[uplift_root],
        "myopic_selected_future_gain": future_gain[myopic_root],
        "full_selected_future_gain": future_gain[full_root],
        "uplift_agrees_with_full": uplift_root == full_root,
    }


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    comparable = sum(row["future_gain_comparable"] for row in rows)
    uplift_points = sum(row["uplift_future_gain_points"] for row in rows)
    full_points = sum(row["full_future_gain_points"] for row in rows)
    myopic_points = sum(row["myopic_future_gain_points"] for row in rows)
    immediate_comparable = sum(
        row["myopic_immediate_comparable"] for row in rows
    )
    total_comparable = sum(row["full_total_comparable"] for row in rows)
    task_above_chance = [
        row["uplift_future_gain_points"]
        - 0.5 * row["future_gain_comparable"]
        for row in rows
    ]
    return {
        "task_count": len(rows),
        "tasks_with_comparable_future_gain": sum(
            row["future_gain_comparable"] > 0 for row in rows
        ),
        "future_gain_comparable_pairs": comparable,
        "uplift_future_gain_pairwise_accuracy": (
            uplift_points / comparable if comparable else 0.0
        ),
        "uplift_future_gain_points": uplift_points,
        "uplift_above_chance_task_sign_flip_p": (
            exact_half_step_sign_flip_pvalue(task_above_chance)
        ),
        "full_score_future_gain_pairwise_accuracy": (
            full_points / comparable if comparable else 0.0
        ),
        "myopic_score_future_gain_pairwise_accuracy": (
            myopic_points / comparable if comparable else 0.0
        ),
        "myopic_score_immediate_pairwise_accuracy": (
            sum(row["myopic_immediate_points"] for row in rows)
            / immediate_comparable
            if immediate_comparable
            else 0.0
        ),
        "myopic_immediate_comparable_pairs": immediate_comparable,
        "full_score_total_pairwise_accuracy": (
            sum(row["full_total_points"] for row in rows)
            / total_comparable
            if total_comparable
            else 0.0
        ),
        "full_total_comparable_pairs": total_comparable,
        "uplift_selected_future_gain_total": sum(
            row["uplift_selected_future_gain"] for row in rows
        ),
        "myopic_selected_future_gain_total": sum(
            row["myopic_selected_future_gain"] for row in rows
        ),
        "full_selected_future_gain_total": sum(
            row["full_selected_future_gain"] for row in rows
        ),
        "uplift_full_root_agreement_count": sum(
            row["uplift_agrees_with_full"] for row in rows
        ),
        "task_rows": list(rows),
    }


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _task_row(record, myopic, full_tree)
        for record, myopic, full_tree in zip(
            payload["records"],
            payload["myopic_scores"],
            payload["nonmyopic_scores"],
            strict=True,
        )
    ]


def analyze(
    first_link: dict[str, Any],
    receding: dict[str, Any],
) -> dict[str, Any]:
    first_rows = _rows(first_link)
    receding_rows = _rows(receding)
    first_ids = {row["task_id"] for row in first_rows}
    receding_ids = {row["task_id"] for row in receding_rows}
    if first_ids & receding_ids:
        raise ValueError("tau future-uplift task splits overlap")
    first_summary = _summarize_rows(first_rows)
    receding_summary = _summarize_rows(receding_rows)
    pooled_summary = _summarize_rows([*first_rows, *receding_rows])
    strong_checks = {
        "first_split_at_least_30_comparable_pairs": (
            first_summary["future_gain_comparable_pairs"] >= 30
        ),
        "receding_split_at_least_30_comparable_pairs": (
            receding_summary["future_gain_comparable_pairs"] >= 30
        ),
        "first_split_uplift_accuracy_above_point_50": (
            first_summary["uplift_future_gain_pairwise_accuracy"] > 0.50
        ),
        "receding_split_uplift_accuracy_above_point_50": (
            receding_summary["uplift_future_gain_pairwise_accuracy"] > 0.50
        ),
        "pooled_uplift_accuracy_at_least_point_60": (
            pooled_summary["uplift_future_gain_pairwise_accuracy"] >= 0.60
        ),
        "pooled_task_sign_flip_p_at_most_point_05": (
            pooled_summary["uplift_above_chance_task_sign_flip_p"] <= 0.05
        ),
    }
    if all(strong_checks.values()):
        classification = "strong_future_value_signal"
    elif (
        first_summary["uplift_future_gain_pairwise_accuracy"] >= 0.50
        and receding_summary["uplift_future_gain_pairwise_accuracy"] >= 0.50
        and pooled_summary["uplift_future_gain_pairwise_accuracy"] >= 0.55
    ):
        classification = "directional_future_value_signal"
    else:
        classification = "null_or_adverse"
    return {
        "status": "posthoc_future_uplift_audit",
        "classification": classification,
        "task_splits_disjoint": True,
        "splits": {
            "first_link_v2": first_summary,
            "receding_v3_1": receding_summary,
        },
        "pooled": pooled_summary,
        "strong_future_value_checks": strong_checks,
        "limitations": [
            "post hoc analysis on already-open task splits",
            "score subtraction assumes comparable within-task 0-100 scales",
            "future trees combine regenerated beliefs, queries, and documents",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-link-confirmation", type=Path, required=True)
    parser.add_argument("--receding-confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.first_link_confirmation) != FIRST_LINK_SHA256:
        raise ValueError("first-link confirmation hash mismatch")
    if _sha256(args.receding_confirmation) != RECEDING_SHA256:
        raise ValueError("receding confirmation hash mismatch")
    first_link = json.loads(
        args.first_link_confirmation.read_text(encoding="utf-8")
    )
    receding = json.loads(
        args.receding_confirmation.read_text(encoding="utf-8")
    )
    result = {
        "first_link_confirmation_sha256": FIRST_LINK_SHA256,
        "receding_confirmation_sha256": RECEDING_SHA256,
        **analyze(first_link, receding),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
