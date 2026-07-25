from pathlib import Path

from helpers import load_config
import scripts.clariq_likelihood_stability_smoke as clariq_smoke
from scripts.clariq_likelihood_stability_smoke import (
    DeterministicFixtureModel,
    EXPECTED_FACET_IDS,
    EXPECTED_REQUESTS,
    QUESTION_IDS,
    run_smoke,
)


def test_frozen_clariq_stability_shape() -> None:
    assert len(EXPECTED_FACET_IDS) == 5
    assert len(QUESTION_IDS) == 4
    assert EXPECTED_REQUESTS == 10


def test_deterministic_fixture_passes_all_stability_gates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    task = {
        "initial_request": "all men are created equal",
        "facets": [
            {"facet_id": facet_id, "description": f"facet {index}"}
            for index, facet_id in enumerate(EXPECTED_FACET_IDS)
        ],
        "questions": [
            {
                "question_id": QUESTION_IDS[0],
                "question": (
                    "are you looking for declaration of independence at the "
                    "national archives"
                ),
            },
            {
                "question_id": QUESTION_IDS[1],
                "question": "are you trying to look up a speech",
            },
            {
                "question_id": QUESTION_IDS[2],
                "question": (
                    "would you like to know its interpretation by the us "
                    "supreme court"
                ),
            },
            {
                "question_id": QUESTION_IDS[3],
                "question": "would you like to learn about the author of that quote",
            },
        ],
    }
    monkeypatch.setattr(clariq_smoke, "verify_source", lambda _: task)
    config = load_config(
        str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "config_clariq_likelihood_stability_openrouter.yaml"
        )
    )

    payload = run_smoke(
        config,
        source_root=tmp_path,
        raw_path=tmp_path / "raw.json",
        model=DeterministicFixtureModel(),
    )

    assert payload["status"] == "passed"
    assert payload["gates"]["all_pass"] is True
    assert payload["usage"]["physical_requests"] == EXPECTED_REQUESTS
    assert payload["metrics"]["unique_base_map_count"] == 4
    assert payload["metrics"]["base_eig_range"] > 0.17
    assert set(payload["metrics"]["repeat_unique_counts"].values()) == {1}
