from __future__ import annotations

import random

from scripts.discoveryworld_deterministic_replay_audit import (
    build_action_schedule,
    compare_trace_records,
    patch_object_rng,
    stable_object_seed,
)


class _FakeWorld:
    def __init__(self, seed: int):
        self.randomSeed = seed


class _FakeObject:
    def __init__(self, world: _FakeWorld, uuid: int):
        self.world = world
        self.uuid = uuid
        self.rng = random.Random()


def test_stable_object_seed_is_reproducible_and_object_specific():
    assert stable_object_seed(7, 11) == stable_object_seed(7, 11)
    assert stable_object_seed(7, 11) != stable_object_seed(7, 12)
    assert stable_object_seed(7, 11) != stable_object_seed(8, 11)


def test_patch_object_rng_replays_object_streams():
    patch_object_rng(_FakeObject)

    left = _FakeObject(_FakeWorld(3), 19)
    right = _FakeObject(_FakeWorld(3), 19)
    other = _FakeObject(_FakeWorld(3), 20)

    left_values = [left.rng.random() for _ in range(4)]
    right_values = [right.rng.random() for _ in range(4)]
    other_values = [other.rng.random() for _ in range(4)]
    assert left_values == right_values
    assert left_values != other_values


def test_build_action_schedule_prefers_sorted_teleports():
    assert build_action_schedule(["a", "b"], num_steps=3) == [
        {"action": "TELEPORT_TO_LOCATION", "arg1": "a"},
        {"action": "TELEPORT_TO_LOCATION", "arg1": "b"},
        {"action": "TELEPORT_TO_LOCATION", "arg1": "a"},
    ]


def test_build_action_schedule_has_deterministic_movement_fallback():
    assert build_action_schedule([], num_steps=3) == [
        {"action": "MOVE_DIRECTION", "arg1": "north"},
        {"action": "MOVE_DIRECTION", "arg1": "east"},
        {"action": "MOVE_DIRECTION", "arg1": "south"},
    ]


def test_compare_trace_records_reports_exact_and_mismatched_fields():
    trace = {
        "process_seed": 1,
        "actions": [],
        "action_results": [],
        "observation_hashes": ["a"],
        "scorecard_hashes": ["b"],
        "final_score_normalized": 0.0,
        "completed": False,
        "completed_successfully": False,
    }
    exact = compare_trace_records(trace, dict(trace))
    changed = dict(trace)
    changed["observation_hashes"] = ["c"]
    mismatch = compare_trace_records(trace, changed)

    assert exact["exact_match"] is True
    assert exact["mismatched_fields"] == []
    assert mismatch["exact_match"] is False
    assert mismatch["mismatched_fields"] == ["observation_hashes"]
