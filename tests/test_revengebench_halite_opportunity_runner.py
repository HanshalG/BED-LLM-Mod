from __future__ import annotations

from scripts import revengebench_halite_opportunity_runner as runner


def test_frame_round_trip_shapes_and_target_actions() -> None:
    data = {"width": 2, "height": 1, "productions": [[3, 4]],
            "frames": [[[[2, 10], [0, 5]]], [[[2, 11], [0, 5]]]],
            "moves": [[[[1], [0]]]]}
    # Use the actual replay move-grid scalar shape.
    data["moves"] = [[[1, 0]]]
    assert runner.encode_productions(data) == "3 4"
    assert runner.encode_frame(data["frames"][0], 2, 1) == "1 2 1 0 10 5"
    assert runner.target_actions(data, 2) == [[[0, 0, 1]]]


def test_action_distance_matches_native_per_cell_fraction() -> None:
    left = [[0, 0, 1], [0, 1, 0]]
    right = [[0, 0, 2], [0, 1, 0]]
    assert runner.action_distance(left, right) == 0.5
    assert runner.action_distance(left, left) == 0.0


def test_decode_moves_converts_xy_to_row_column() -> None:
    assert runner.decode_moves("5 2 1 3 4 0") == [[2, 5, 1]]


def test_decision_frames_repeat_init_frame_for_turn_zero() -> None:
    frames = ["frame-zero", "frame-one", "terminal-frame"]
    data = {"frames": frames, "moves": ["move-zero", "move-one"]}

    assert runner.decision_frames(data) == ["frame-zero", "frame-one"]
