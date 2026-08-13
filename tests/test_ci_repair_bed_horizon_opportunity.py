from __future__ import annotations

import math

from scripts import ci_repair_bed_horizon_opportunity as audit


def test_canonicalization_removes_run_specific_values_but_keeps_semantics():
    left = audit.canonicalize_response(
        "\x1b[31mERROR /home/alice/project/tests/test_api.py:123 at "
        "2026-08-11T01:02:03Z https://example.test/a deadbeef\x1b[0m"
    )
    right = audit.canonicalize_response(
        "ERROR /home/bob/project/tests/test_api.py:456 at "
        "2025-01-01T09:08:07Z https://other.test/b cafebabe"
    )
    assert left == right
    assert "error" in left
    assert "test_api.py" in left
    assert "<number>" in left


def test_action_responses_are_hash_only_and_signal_absence_is_deterministic():
    row = {
        "workflow": "name: CI\nsteps:\n  - run: pytest",
        "logs": [{"name": "unit", "step_name": "test", "log": "all good"}],
    }
    responses = audit.action_responses(row)
    assert tuple(responses) == audit.ACTION_IDS
    assert all(len(value) == 64 for value in responses.values())
    assert responses["first_error"] == responses["traceback"]


def test_exact_information_matches_binary_partition():
    rows = [
        {action: ("left" if index < 2 else "right") for action in audit.ACTION_IDS}
        for index in range(4)
    ]
    assert math.isclose(
        audit.partition_information(rows, audit.ACTION_IDS[0], tuple(range(4))),
        math.log(2),
        abs_tol=1e-12,
    )


def test_depth_two_can_prefer_adaptive_first_action_over_greedy():
    # Action a has the best root split, but c has better adaptive continuations.
    patterns = {
        "a": [1, 2, 0, 3, 1, 0, 1, 0],
        "b": [0, 1, 1, 0, 0, 0, 1, 1],
        "c": [1, 3, 2, 0, 0, 2, 0, 0],
        "d": [0, 1, 1, 0, 0, 1, 1, 1],
    }
    original_actions = audit.ACTION_IDS
    audit.ACTION_IDS = tuple(patterns)
    try:
        rows = [
            {action: str(values[index]) for action, values in patterns.items()}
            for index in range(8)
        ]
        root = {
            action: audit.partition_information(rows, action, tuple(range(8)))
            for action in audit.ACTION_IDS
        }
        greedy = max(root, key=root.get)
        scores = {action: audit.two_step_score(rows, action)[0] for action in audit.ACTION_IDS}
        depth_two = max(scores, key=scores.get)
        assert greedy == "a"
        assert depth_two == "c"
        assert scores[depth_two] > scores[greedy]
    finally:
        audit.ACTION_IDS = original_actions


def test_content_projection_excludes_every_repair_outcome():
    assert audit.CONTENT_COLUMNS == ("id", "repo_name", "workflow", "logs")
    assert not {"diff", "changed_files", "error_type", "sha_success"}.intersection(
        audit.CONTENT_COLUMNS
    )
