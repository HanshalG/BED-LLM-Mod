from __future__ import annotations

from scripts.longvid_bridge_path_opportunity_audit import _id_hash
from scripts.longvid_four_hop_tradeoff_confirmation_v2 import (
    apply_scorable_eligibility,
    summarize,
)
from scripts.longvid_four_hop_tradeoff_opportunity import (
    CONFIRMATION_VIDEO_HASH,
)


def test_frozen_confirmation_video_hash_is_stable() -> None:
    video_ids = [
        "6OXt-Yywz7s",
        "Ytv-9RM4e0o",
        "zK8mqwjhqrs",
        "ZBD5ToU5HBI",
        "5qi6AS-5zKk",
        "V9NZWMFydH4",
        "XL8RwvLNKCc",
        "YN4wicCk2b0",
        "LAbtlJJhUlY",
        "O3Hwh0uv8Mg",
        "YQw3ZOq6Zfs",
        "vS8eQ5nVrRQ",
        "NYYxIlCsFr0",
        "WTT7XZko3qk",
        "g66-6uFbsf4",
        "g-FH4-kKJbE",
        "s5edwp0PEqk",
        "DB3bprN1yM8",
        "NMHmqgO04rU",
        "Uvzq-vDhVmA",
        "z2hgtFBBbxk",
        "CW8l_VPgEgI",
        "TRoxAWpi4MI",
        "zdKX7Xo3Cb8",
        "3M4Saf-yiKY",
        "R_ZGwbK0aXE",
        "wCQC8ZIIc1o",
        "GH14cdbU2Jc",
        "eugVGJkOjsI",
        "ao56rH0gnVY",
        "q7QP_lfqnQM",
        "uFig609YWjU",
        "o9sDPX2FXeQ",
        "7TljSpTBS9c",
        "2u3ujFhxEug",
        "Ys_ENxeNctk",
        "O6UedmnRJc0",
        "GZW6OjARMGU",
        "6vLG6Eo1qTI",
        "tKwuwLChP2g",
    ]
    assert _id_hash(video_ids) == CONFIRMATION_VIDEO_HASH


def test_unscorable_task_cannot_enter_strict_endpoint() -> None:
    record = {
        "num_answer_terms": 0,
        "strict_tradeoff": True,
        "strict_gap_count": 2,
        "strict_answer_sacrifice": 0.5,
    }
    enriched = apply_scorable_eligibility(record)
    assert not enriched["answer_scorable"]
    assert not enriched["eligible_strict_tradeoff"]
    assert enriched["eligible_strict_gap_count"] == 0
    assert enriched["eligible_strict_answer_sacrifice"] == 0.0


def _passing_records() -> list[dict[str, object]]:
    records = []
    for index in range(40):
        strict = index < 6
        scorable = index < 38
        records.append(
            {
                "num_captions": 80,
                "num_roots": 8,
                "answer_scorable": scorable,
                "distinct_root_top1": 3 if index < 30 else 2,
                "depth_four_gain_count": 1 if index < 24 else 0,
                "oracle_final_coverage": 0.75 if index < 24 else 0.25,
                "coverage_gain": 0.5 if index < 24 else 0.0,
                "eligible_strict_tradeoff": strict,
                "eligible_strict_gap_count": 1 if strict else 0,
                "eligible_strict_answer_sacrifice": (
                    0.2 if strict else 0.0
                ),
            }
        )
    return records


def test_v2_summary_passes_exact_thresholds() -> None:
    summary = summarize(_passing_records())
    assert summary["answer_scorable_task_count"] == 38
    assert summary["eligible_strict_tradeoff_count"] == 6
    assert summary["eligible_strict_total_gap"] == 6
    assert summary["gates"]["all_pass"]


def test_v2_summary_rejects_too_few_scorable_tasks() -> None:
    records = _passing_records()
    records[37]["answer_scorable"] = False
    summary = summarize(records)
    assert summary["answer_scorable_task_count"] == 37
    assert not summary["gates"]["answer_scorable_tasks_at_least_38"]
    assert not summary["gates"]["all_pass"]
