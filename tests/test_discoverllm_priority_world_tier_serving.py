from __future__ import annotations

from pathlib import Path

from scripts import discoverllm_priority_world_mechanics as cardinal
from scripts import discoverllm_priority_world_ordinal_serving as ordinal
from scripts import discoverllm_priority_world_tier_serving as tier


def _fixture_task() -> cardinal.MechanicsTask:
    worlds = tuple(
        {
            "criterion": f"Criterion {label}",
            "subcriteria": [],
        }
        for label in cardinal.WORLD_LABELS
    )
    return cardinal.MechanicsTask(
        key=tier.SERVING_TASK_ID,
        conversation=({"role": "user", "content": "Help revise this."},),
        actions=("Action A", "Action B"),
        worlds=worlds,
    )


def _tier_text() -> str:
    return "\n".join(
        f"{key}|W1:H,W2:M,W3:L,W4:L"
        for key in sorted(cardinal._branch_keys())
    )


def test_tier_parser_requires_exact_world_order_and_labels():
    expected = cardinal._branch_keys()
    parsed = tier._parse_tiers(_tier_text(), expected)
    assert set(parsed) == expected
    assert all(value["W1"] == "H" for value in parsed.values())

    invalid = (
        _tier_text().replace("W1:H", "W1:X", 1),
        _tier_text().replace("W1:H,W2:M", "W2:M,W1:H", 1),
        _tier_text().replace("A_O1|", "A_O2|", 1),
        "\n".join(_tier_text().splitlines()[:-1]),
    )
    for text in invalid:
        try:
            tier._parse_tiers(text, expected)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("invalid tier response parsed")


def test_tiers_allow_ties_and_produce_different_entropy_profiles():
    expected = {"A_O1"}
    sharp = tier._parse_tiers(
        "A_O1|W1:H,W2:L,W3:L,W4:L",
        expected,
    )
    ambiguous = tier._parse_tiers(
        "A_O1|W1:H,W2:H,W3:L,W4:L",
        expected,
    )
    sharp_weights = [
        tier.TIER_WEIGHTS[sharp["A_O1"][world]]
        for world in cardinal.WORLD_LABELS
    ]
    ambiguous_weights = [
        tier.TIER_WEIGHTS[ambiguous["A_O1"][world]]
        for world in cardinal.WORLD_LABELS
    ]
    sharp_entropy = cardinal._entropy(cardinal._normalize(sharp_weights))
    ambiguous_entropy = cardinal._entropy(
        cardinal._normalize(ambiguous_weights)
    )
    assert sharp_entropy != ambiguous_entropy


def test_fixture_tier_serving_passes(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(tier, "_load_task", lambda paths, manifest: _fixture_task())
    result = tier.run_serving(
        object(),
        paths={},
        manifest_path=tmp_path / "manifest.json",
        raw_path=tmp_path / "raw.json",
        model=tier.DeterministicFixtureModel(),
    )
    assert result["status"] == "passed"
    assert result["parse_counts"]["root_tiers"] == 8
    assert result["parse_counts"]["followup_tiers"] == 8
    assert result["protocol"]["tier_weights"] == tier.TIER_WEIGHTS
    assert result["protocol"]["semantic_content_emitted"] is False


def test_shared_ordinal_runner_remains_compatible(monkeypatch, tmp_path: Path):
    task = _fixture_task()
    monkeypatch.setattr(ordinal, "_load_task", lambda paths, manifest: task)
    result = ordinal.run_serving(
        object(),
        paths={},
        manifest_path=tmp_path / "manifest.json",
        raw_path=tmp_path / "raw.json",
        model=ordinal.DeterministicFixtureModel(),
    )
    assert result["status"] == "passed"
    assert result["parse_counts"]["root_rankings"] == 8
