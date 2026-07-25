from pathlib import Path

from scripts.clariq_multisample_likelihood_development_v2 import (
    EXPECTED_REQUESTS,
    SELECTED_TOPIC_IDS,
    VALID_ROOT_COUNTS,
    load_manifest,
)


MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "results/nonmyopic/clariq_multisample_likelihood_v2_manifest/MANIFEST.json"
)


def test_frozen_v2_manifest_reproduces() -> None:
    tasks = load_manifest(MANIFEST)
    assert tuple(tasks) == SELECTED_TOPIC_IDS
    assert tuple(len(task["questions"]) for task in tasks.values()) == (
        VALID_ROOT_COUNTS
    )
    assert sum(len(task["questions"]) for task in tasks.values()) * 5 == (
        EXPECTED_REQUESTS
    )
