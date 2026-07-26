from __future__ import annotations

import copy
import json

import pytest

from scripts import infoquest_cached_trajectory_opportunity as v1
from scripts import infoquest_cached_trajectory_opportunity_v2 as v2


def _manifest() -> dict:
    original = v1.source_manifest.split_record_ids()
    return {
        "selection": {
            "splits": {
                name: {
                    "record_ids": record_ids,
                    "ordered_sha256": v1.source_manifest.canonical_sha256(
                        record_ids
                    ),
                }
                for name, record_ids in original.items()
            }
        }
    }


def _empty_later_row() -> dict:
    evaluations = [
        {
            "done": False,
            "generation_time": 1.0,
            "invalid_responses": 0,
            "questions": {"criterion": {}},
            "total_reward": 0,
        },
        {
            "done": False,
            "generation_time": 1.0,
            "invalid_responses": 0,
            "questions": {"criterion": {}},
            "total_reward": 2,
        },
        {
            "done": True,
            "generation_time": 1.0,
            "invalid_responses": 0,
            "questions": {"criterion": {}},
            "total_reward": 5,
        },
    ]
    history = [
        {"role": "system", "content": "hidden context"},
        {"role": "user", "content": "Initial response"},
        {"role": "assistant", "content": "Budget is limited"},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "Schedule changes weekly"},
        {"role": "user", "content": "Tell me about the weekly schedule"},
    ]
    return {
        "id": 0,
        "user_history1": copy.deepcopy(history),
        "evaluations1": copy.deepcopy(evaluations),
        "generation_time1": 1.0,
        "user_history2": copy.deepcopy(history),
        "evaluations2": copy.deepcopy(evaluations),
        "generation_time2": 1.0,
    }


def test_v2_split_is_fresh_reproducible_and_excludes_quarantine():
    splits = v2.split_record_ids_v2(_manifest())
    assert {name: len(ids) for name, ids in splits.items()} == {
        "opportunity": 80,
        "development": 30,
        "holdout": 277,
    }
    assert {
        name: v1.source_manifest.canonical_sha256(ids)
        for name, ids in splits.items()
    } == v2.EXPECTED_SPLIT_HASHES
    assert (
        v1.source_manifest.canonical_sha256(splits)
        == v2.EXPECTED_COMBINED_SPLITS_HASH
    )
    assert 4 not in {record_id for ids in splits.values() for record_id in ids}
    assert not (
        set(splits["opportunity"])
        & set(_manifest()["selection"]["splits"]["opportunity"]["record_ids"])
    )


def test_v2_retains_empty_later_message_as_zero_information():
    row = _empty_later_row()
    with pytest.raises(ValueError, match="content is empty"):
        v1.trajectory_metrics(
            row,
            seed_message="request",
            run=0,
            record_id=0,
            world=1,
        )
    result = v1.trajectory_metrics(
        row,
        seed_message="request",
        run=0,
        record_id=0,
        world=1,
        allow_empty_later=True,
    )
    assert result["empty_later_message_count"] == 1
    assert result["later_message_count"] == 4
    assert result["reward_trace"] == [0, 2, 5]
    assert "content" not in json.dumps(result)


def test_v2_still_rejects_empty_first_policy_message():
    row = _empty_later_row()
    row["user_history1"][1]["content"] = ""
    with pytest.raises(ValueError, match="content is empty"):
        v1.trajectory_metrics(
            row,
            seed_message="request",
            run=0,
            record_id=0,
            world=1,
            allow_empty_later=True,
        )
