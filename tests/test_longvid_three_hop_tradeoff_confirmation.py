from __future__ import annotations

import pytest

from scripts.longvid_bridge_path_opportunity_audit import _id_hash
from scripts.longvid_three_hop_tradeoff_confirmation import (
    RESERVE_VIDEO_HASH,
    add_semantic_tradeoff,
    summarize,
)


def test_frozen_reserve_video_hash_is_stable() -> None:
    video_ids = [
        "NT8Qkc7mzX4",
        "T38zTlAQm3E",
        "KbFziURyfhY",
        "OAcbasjxljY",
        "LlqsCCa6y58",
        "TanLRBKlGeE",
        "pcgVe9Oo-N0",
        "vBG621XEegk",
        "uEIQo3UP2iw",
        "nr_pFDbvz6A",
        "gY0yACShcnU",
        "6gxA9veS_3I",
        "xIDAlQAHc3s",
        "LtZs4FfwiSs",
        "C7dbCrsf0l4",
        "wGYXpRj3-Es",
        "fZBC3nmvJb8",
        "SIYWxZZ9igc",
        "i3EntVc8dUI",
        "78JvysqpNMc",
        "8o8EzSwIZ6Q",
        "RXWyuO3v4vM",
        "FZfmPQMJ6rw",
        "GTIjylkB-TI",
        "h5rdgtyx844",
        "HPxh7kk_hE4",
        "KFweVYFztbk",
        "QG6-B_EFmxI",
        "JJjPIS7Ko-U",
        "kswaAPPRmWY",
        "pJI5ZU6wxqg",
        "m0vIzYjLw5Q",
        "_uL3a3aMdMQ",
        "xHvvMGHmwuU",
        "y6c6jz4NRWY",
        "pMJYCFdVLt8",
        "U__Us5KPRrA",
        "Q6xlMp1yyng",
        "YX2jnHv6R-w",
        "J456aSUrN8o",
    ]
    assert _id_hash(video_ids) == RESERVE_VIDEO_HASH


def test_semantic_tradeoff_requires_sacrifice_and_final_gain() -> None:
    base = {
        "greedy_root_index": 0,
        "oracle_root_index": 1,
        "greedy_direct_answer": 0.6,
        "oracle_direct_answer": 0.2,
        "greedy_triple_count": 2,
        "oracle_triple_count": 3,
    }
    strict = add_semantic_tradeoff(base)
    assert strict["semantic_tradeoff"]
    assert strict["semantic_tradeoff_gap_count"] == 1
    assert strict["semantic_tradeoff_answer_sacrifice"] == pytest.approx(0.4)

    no_sacrifice = add_semantic_tradeoff(
        {**base, "oracle_direct_answer": 0.6}
    )
    assert not no_sacrifice["semantic_tradeoff"]
    no_gain = add_semantic_tradeoff({**base, "oracle_triple_count": 2})
    assert not no_gain["semantic_tradeoff"]


def _passing_records() -> list[dict[str, object]]:
    records = []
    for index in range(40):
        strict = index < 5
        records.append(
            {
                "num_captions": 80,
                "num_answer_terms": 3,
                "num_roots": 8,
                "distinct_root_top1": 3 if index < 30 else 2,
                "depth_three_gain_count": 1 if index < 20 else 0,
                "oracle_triple_coverage": 1.0 if index < 20 else 1 / 3,
                "coverage_gain": 2 / 3 if index < 20 else 0.0,
                "semantic_tradeoff": strict,
                "semantic_tradeoff_gap_count": 1 if strict else 0,
                "semantic_tradeoff_answer_sacrifice": (
                    0.2 if strict else 0.0
                ),
            }
        )
    return records


def test_confirmation_summary_passes_exact_thresholds() -> None:
    summary = summarize(_passing_records())
    assert summary["semantic_tradeoff_count"] == 5
    assert summary["semantic_tradeoff_total_gap"] == 5
    assert summary["gates"]["all_pass"]


def test_confirmation_summary_fails_without_replication() -> None:
    records = _passing_records()
    records[4]["semantic_tradeoff"] = False
    records[4]["semantic_tradeoff_gap_count"] = 0
    records[4]["semantic_tradeoff_answer_sacrifice"] = 0.0
    summary = summarize(records)
    assert summary["semantic_tradeoff_count"] == 4
    assert not summary["gates"]["semantic_tradeoffs_at_least_5"]
    assert not summary["gates"]["all_pass"]
