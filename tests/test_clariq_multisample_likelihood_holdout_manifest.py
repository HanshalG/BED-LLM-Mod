from scripts.clariq_multisample_likelihood_holdout_manifest import (
    MAX_SELECTED_TOPICS,
    MIN_SELECTED_TOPICS,
)


def test_holdout_manifest_bounds_are_preregistered() -> None:
    assert MIN_SELECTED_TOPICS == 8
    assert MAX_SELECTED_TOPICS == 12
