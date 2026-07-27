from __future__ import annotations

from scripts.longvid_bridge_path_opportunity_audit import _id_hash
from scripts.longvid_three_hop_opportunity_audit import (
    DEVELOPMENT_ID_HASH,
    OPPORTUNITY_ID_HASH,
    OPPORTUNITY_VIDEO_HASH,
    RESERVE_ID_HASH,
    _best_trajectory,
    summarize,
)


def test_frozen_three_hop_split_hashes_are_stable() -> None:
    opportunity_ids = [
        1432, 2882, 2531, 1673, 1465, 2864, 2135, 2609, 2202, 2525,
        990, 933, 1939, 836, 328, 114, 2697, 2332, 2488, 102,
        2756, 2599, 1131, 1392, 2411, 2521, 2578, 1542, 1842, 412,
        1610, 1945, 877, 1565, 2887, 1394, 2777, 1706, 2375, 741,
    ]
    development_ids = [
        1551, 2968, 650, 1253, 419, 844, 711, 2231, 676, 1834,
        840, 587, 893, 2442, 2450, 1716, 1967, 2269, 2071, 2826,
    ]
    reserve_ids = [
        867, 1168, 728, 925, 794, 1214, 2407, 2673, 2632, 2321,
        1931, 262, 2807, 805, 444, 2740, 1848, 1139, 1997, 309,
        382, 1086, 532, 557, 1949, 578, 707, 1030, 660, 2121,
        2393, 2190, 1581, 2796, 2854, 2396, 1273, 1019, 1447, 631,
    ]
    assert _id_hash(opportunity_ids) == OPPORTUNITY_ID_HASH
    assert _id_hash(development_ids) == DEVELOPMENT_ID_HASH
    assert _id_hash(reserve_ids) == RESERVE_ID_HASH
    assert len(set(opportunity_ids + development_ids + reserve_ids)) == 100


def test_frozen_three_hop_opportunity_video_hash_is_stable() -> None:
    video_ids = [
        "YWhMCXo4lkQ",
        "yV8QeblH2gA",
        "sgeTrVfx2rc",
        "bMxSJAzZWMw",
        "YebUIUOCo94",
        "yDNj0aZ7oFY",
        "lD2tB1riyl4",
        "tjcpZoyOlzg",
        "m7ubevtp7Zg",
        "sU5YG5Fdizg",
        "Pc96BmhEpP4",
        "OAzSxWukMzE",
        "glYMjtt0ayQ",
        "MvgJAD6tZXo",
        "7b_1ZCV4Pwc",
        "2kmb1ARg9m0",
        "vLkKSLgjLMY",
        "o2F-N42Ufo4",
        "rP7sQe784k8",
        "2W2hpfKrtu4",
        "wcJfWoLwe6g",
        "tP68QwVvAZk",
        "SC4nwtBAYPA",
        "WzWtxOufgbM",
        "pr8-agDG8sI",
        "sEiyR7-0FOA",
        "t6vzJ6ceJqs",
        "ZR046eA9kEE",
        "fPLjjr8w6DU",
        "BdTWh6dfye4",
        "aX_HgA5SNLQ",
        "gusNxEOdo-o",
        "NWMUKBUkK1o",
        "_qMcEMjLYhw",
        "yWOsyFAUS4g",
        "X2sKt-WEIeU",
        "ww9wYPmkRGA",
        "cOaODWZ48IE",
        "ojsaQcmtlj0",
        "KjIRSbMJ8zY",
    ]
    assert _id_hash(video_ids) == OPPORTUNITY_VIDEO_HASH


def test_best_trajectory_uses_frozen_candidate_order_for_ties() -> None:
    trajectories = [
        {
            "first_followup_index": 1,
            "second_followup_index": 0,
            "second_id": "2",
            "third_id": "3",
            "triple_count": 3,
            "first_observation_term_count": 1,
            "second_observation_term_count": 1,
        },
        {
            "first_followup_index": 0,
            "second_followup_index": 4,
            "second_id": "3",
            "third_id": "2",
            "triple_count": 3,
            "first_observation_term_count": 1,
            "second_observation_term_count": 1,
        },
    ]
    best = _best_trajectory(trajectories, first_gold=1)
    assert best["first_followup_index"] == 0
    assert best["second_followup_index"] == 4


def _passing_records() -> list[dict[str, object]]:
    records = []
    for index in range(40):
        strict = index < 5
        records.append(
            {
                "row_index": index,
                "num_captions": 80,
                "num_answer_terms": 3,
                "num_roots": 8,
                "distinct_root_top1": 3 if index < 30 else 2,
                "depth_three_gain_count": 1 if index < 20 else 0,
                "ordered_chain_recovered": index < 15,
                "oracle_triple_coverage": 1.0 if index < 20 else 1 / 3,
                "coverage_gain": 2 / 3 if index < 20 else 0.0,
                "nonmyopic_gap_count": 1 if strict else 0,
                "answer_sacrifice": 0.2 if strict else 0.0,
                "strict_opportunity": strict,
            }
        )
    return records


def test_summary_passes_only_with_strict_three_hop_tradeoffs() -> None:
    summary = summarize(_passing_records())
    assert summary["strict_opportunity_count"] == 5
    assert summary["strict_total_gap"] == 5
    assert summary["gates"]["all_pass"]


def test_summary_rejects_order_commutative_retrieval() -> None:
    records = _passing_records()
    for record in records:
        record["strict_opportunity"] = False
        record["nonmyopic_gap_count"] = 0
        record["answer_sacrifice"] = 0.0
    summary = summarize(records)
    assert summary["depth_three_gain_task_count"] == 20
    assert summary["ordered_chain_task_count"] == 15
    assert not summary["gates"]["strict_opportunities_at_least_5"]
    assert not summary["gates"]["all_pass"]
