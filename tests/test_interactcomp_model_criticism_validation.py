import math

import pytest

from scripts.interactcomp_first_link_opportunity import Hypothesis
from scripts.interactcomp_model_criticism_validation import (
    SCREEN_INDICES,
    enroll_collapsed_tasks,
    parse_semantic_distinctness,
    root_scores,
    unique_entity_count,
)


def test_semantic_distinctness_parser_is_strict():
    assert parse_semantic_distinctness("D") is True
    assert parse_semantic_distinctness("s") is False
    with pytest.raises(ValueError):
        parse_semantic_distinctness("distinct")


def test_collapsed_support_enrollment_preserves_manifest_order():
    populations = {}
    for offset, index in enumerate(SCREEN_INDICES):
        width = 2 if offset % 2 == 0 else 8
        populations[index] = [
            Hypothesis(f"Candidate {sample % width}", "profile")
            for sample in range(8)
        ]
    enrolled = enroll_collapsed_tasks(populations)
    assert enrolled == list(SCREEN_INDICES[0:11:2])
    assert all(unique_entity_count(populations[index]) <= 4 for index in enrolled)


def test_root_scores_separate_model_criticism_from_augmented_eig():
    current = ["YYYY"] * 8
    auxiliary = ["NYNN"] * 8
    scores = root_scores(current, auxiliary)
    assert scores["current_eig"] == pytest.approx([0.0] * 4)
    assert scores["model_criticism"][0] == pytest.approx(math.log(2.0))
    assert scores["model_criticism"][1] == pytest.approx(0.0)
    assert scores["augmented_eig"][0] == pytest.approx(math.log(2.0))
    assert scores["augmented_eig"][1] == pytest.approx(0.0)
