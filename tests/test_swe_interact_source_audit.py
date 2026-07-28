from __future__ import annotations

import pytest

from scripts.swe_interact_source_audit import (
    common_prefix_length,
    count_positive_rubrics,
    count_requirement_bullets,
)


def test_common_prefix_length_stops_at_private_task_block() -> None:
    assert common_prefix_length(
        [
            ["shared one", "shared two", "task A"],
            ["shared one", "shared two", "task B"],
            ["shared one", "shared two", "task C"],
        ]
    ) == 2


def test_requirement_bullets_are_counted_by_released_structure() -> None:
    assert count_requirement_bullets(
        "- first externally checked behavior\n"
        "- second externally checked behavior\n"
        "  continuation text\n"
        "- third externally checked behavior\n"
    ) == 3


def test_positive_rubrics_exclude_headings_and_negative_checks() -> None:
    rubrics = [
        {"annotations": {"type": "high level intent"}},
        {"annotations": {"type": "positive hli verifier"}},
        {"annotations": {"type": "negative hli verifier"}},
        {"annotations": {"type": "positive hli verifier"}},
    ]
    assert count_positive_rubrics(rubrics) == 2


def test_positive_rubrics_require_list() -> None:
    with pytest.raises(ValueError, match="Expected a list"):
        count_positive_rubrics({"rubric": "not a list"})
