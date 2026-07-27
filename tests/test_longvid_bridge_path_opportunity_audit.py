from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.bright_biology_unlock_audit import BM25Corpus
from scripts.longvid_bridge_path_opportunity_audit import (
    FRESH_VIDEO_HASH,
    FRESH_VIDEO_IDS,
    OPPORTUNITY_ID_HASH,
    _id_hash,
    followup_queries,
    load_selected_captions,
    root_queries,
    stream_json_array,
    summarize,
)


def _tiny_corpus() -> BM25Corpus:
    return BM25Corpus(
        [
            {
                "id": "1",
                "content": "A courier takes a brass key from the blue cabinet.",
                "raw_source": (
                    "A courier takes a brass key from the blue cabinet."
                ),
                "title": "Clip 1",
            },
            {
                "id": "2",
                "content": "The brass key later opens the observatory door.",
                "raw_source": (
                    "The brass key later opens the observatory door."
                ),
                "title": "Clip 2",
            },
            {
                "id": "3",
                "content": "A musician performs beside a red curtain.",
                "raw_source": (
                    "A musician performs beside a red curtain."
                ),
                "title": "Clip 3",
            },
            {
                "id": "4",
                "content": "Coffee is poured into a white cup.",
                "raw_source": "Coffee is poured into a white cup.",
                "title": "Clip 4",
            },
        ]
    )


def test_stream_json_array_handles_chunk_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    rows = [
        {"question": "first", "nested": [1, 2, 3]},
        {"question": "second", "value": {"ok": True}},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    assert list(stream_json_array(path, chunk_size=7)) == list(enumerate(rows))


def test_frozen_opportunity_and_fresh_video_hashes_are_stable() -> None:
    opportunity_ids = [
        2007,
        2602,
        1506,
        819,
        829,
        607,
        1586,
        1263,
        2990,
        139,
        242,
        758,
        1173,
        822,
        401,
        2251,
        491,
        1009,
        2008,
        1386,
        2645,
        1972,
        1874,
        177,
        618,
        1781,
        1679,
        2063,
        2034,
        1917,
        2175,
        2160,
        2937,
        1552,
        2020,
        1885,
        910,
        2783,
        2919,
        424,
    ]
    assert _id_hash(opportunity_ids) == OPPORTUNITY_ID_HASH
    assert _id_hash(sorted(FRESH_VIDEO_IDS)) == FRESH_VIDEO_HASH


def test_caption_filter_does_not_return_fresh_video_rows(
    tmp_path: Path,
) -> None:
    allowed = "development-video"
    fresh = sorted(FRESH_VIDEO_IDS)[0]
    path = tmp_path / "captions.parquet"
    pq.write_table(
        pa.table(
            {
                "vid": [allowed, allowed, fresh],
                "slice_num": [1, 2, 1],
                "cap": ["visible one", "visible two", "sealed caption"],
            }
        ),
        path,
    )
    result = load_selected_captions(path, {allowed})
    assert set(result) == {allowed}
    assert [row["raw_source"] for row in result[allowed]] == [
        "visible one",
        "visible two",
    ]


def test_roots_and_followups_use_visible_text_only() -> None:
    corpus = _tiny_corpus()
    question = (
        "What object eventually opens the observatory door, and where was it "
        "first obtained?"
    )
    roots = root_queries(question, corpus)
    assert len(roots) >= 5
    first = corpus.search(roots[0], top_k=1)
    followups = followup_queries(question, roots[0], first[0], corpus)
    assert followups
    question_terms = set(question.lower().split())
    for followup, visible_terms in followups:
        assert visible_terms
        assert set(visible_terms).isdisjoint(question_terms)
        assert all(term in first[0]["raw_source"].lower() for term in visible_terms)
        assert all(term in followup.lower() for term in visible_terms)


def test_summary_passes_only_with_strict_bridge_tradeoffs() -> None:
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
                "pair_gain_count": 1 if index < 20 else 0,
                "ordered_chain_recovered": index < 15,
                "oracle_pair_coverage": 1.0 if index < 20 else 0.5,
                "pair_coverage_gain": 0.5 if index < 20 else 0.0,
                "nonmyopic_gap_count": 1 if strict else 0,
                "answer_sacrifice": 0.2 if strict else 0.0,
                "strict_opportunity": strict,
            }
        )
    summary = summarize(records)
    assert summary["strict_opportunity_count"] == 5
    assert summary["strict_total_gap"] == 5
    assert summary["gates"]["all_pass"]


def test_summary_rejects_order_commutative_retrieval() -> None:
    records = [
        {
            "row_index": index,
            "num_captions": 80,
            "num_answer_terms": 3,
            "num_roots": 8,
            "distinct_root_top1": 4,
            "pair_gain_count": 1,
            "ordered_chain_recovered": True,
            "oracle_pair_coverage": 1.0,
            "pair_coverage_gain": 0.5,
            "nonmyopic_gap_count": 0,
            "answer_sacrifice": 0.0,
            "strict_opportunity": False,
        }
        for index in range(40)
    ]
    summary = summarize(records)
    assert summary["pair_gain_task_count"] == 40
    assert summary["ordered_chain_task_count"] == 40
    assert not summary["gates"]["strict_opportunities_at_least_5"]
    assert not summary["gates"]["all_pass"]


def test_frozen_two_hop_artifact_closes_backtracking_route() -> None:
    artifact = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "nonmyopic"
        / "longvid_bridge_path_opportunity"
        / "AUDIT.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert payload["status"] == "gate_failed"
    assert summary["num_records"] == 40
    assert summary["pair_gain_task_count"] == 27
    assert summary["ordered_chain_task_count"] == 21
    assert summary["strict_opportunity_count"] == 1
    assert summary["strict_total_gap"] == 1
    assert (
        sum(record["greedy_pair_count"] == 2 for record in payload["records"])
        == 23
    )
    assert not summary["gates"]["all_pass"]
