from scripts.voi_medical_global_likelihood_tree_mechanics import (
    BASE_REQUESTS,
    alias_global_maps,
    unique_new_questions,
)
from scripts.voi_medical_future_tree_mechanics import OUTCOMES


def test_unique_questions_reuse_roots_and_prior_followups() -> None:
    roots = ["Do you cough?", "Do you have pain?"]
    followups = {
        (root_index, outcome): [
            "Do you cough?" if (root_index, outcome) == (0, "Yes") else "New one?",
            f"R{root_index} {outcome}?",
        ]
        for root_index in range(2)
        for outcome in OUTCOMES
    }
    unique = unique_new_questions(roots, followups)
    assert unique[:3] == ["R0 Yes?", "New one?", "R0 No?"]
    assert unique.count("New one?") == 1
    assert "Do you cough?" not in unique
    assert len(unique) == 7
    assert BASE_REQUESTS + len(unique) == 24


def test_global_maps_alias_every_repeated_occurrence() -> None:
    roots = ["Root one?", "Root two?"]
    followups = {
        (root_index, outcome): ["Root one?", "New question?"]
        for root_index in range(2)
        for outcome in OUTCOMES
    }
    root_maps = {
        "Root one?": {"a": "Yes"},
        "Root two?": {"a": "No"},
    }
    aliases = alias_global_maps(
        roots,
        root_maps,
        ["New question?"],
        {"New question?": {"a": "Maybe"}},
        followups,
    )
    assert aliases["Root one?"] == {"a": "Yes"}
    assert aliases["New question?"] == {"a": "Maybe"}
