from __future__ import annotations

from pathlib import Path

from scripts import discoverllm_priority_world_mechanics as cardinal
from scripts import discoverllm_priority_world_ordinal_serving as serving


def _fixture_task() -> cardinal.MechanicsTask:
    worlds = tuple(
        {
            "criterion": f"Criterion {label}",
            "subcriteria": [
                {
                    "criterion": f"Detail {label}",
                    "subcriteria": [],
                }
            ],
        }
        for label in cardinal.WORLD_LABELS
    )
    return cardinal.MechanicsTask(
        key=serving.SERVING_TASK_ID,
        conversation=({"role": "user", "content": "Help me revise this."},),
        actions=("Action A", "Action B"),
        worlds=worlds,
    )


def _ranking_text() -> str:
    return "\n".join(
        f"{key}|W1>W2>W3>W4"
        for key in sorted(cardinal._branch_keys())
    )


def test_ranking_parser_requires_exact_permutations():
    expected = cardinal._branch_keys()
    parsed = serving._parse_rankings(_ranking_text(), expected)
    assert set(parsed) == expected
    assert all(value == ("W1", "W2", "W3", "W4") for value in parsed.values())

    invalid = (
        _ranking_text().replace("W1>W2>W3>W4", "W1>W1>W3>W4", 1),
        _ranking_text().replace("A_O1|", "A_O2|", 1),
        "\n".join(_ranking_text().splitlines()[:-1]),
        _ranking_text() + "\nextra",
    )
    for text in invalid:
        try:
            serving._parse_rankings(text, expected)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("invalid ranking response parsed")


def test_world_presentation_is_reproducible_and_stage_shuffled():
    task = _fixture_task()
    first = serving._world_presentation(task, "ROOT_RANKINGS")
    second = serving._world_presentation(task, "ROOT_RANKINGS")
    followup = serving._world_presentation(task, "FOLLOWUP_RANKINGS")
    assert first == second
    assert set(first) == set(cardinal.WORLD_LABELS)
    assert list(first) != list(followup)


def test_transport_retries_are_bounded_and_accounted():
    base = {
        "adapter_requests": 5,
        "http_attempts": 7,
        "retry_count": 2,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "adapter_cost_usd": 0.14,
    }
    assert serving._serving_gates(base)["all_pass"] is True
    assert serving._serving_gates(
        {**base, "http_attempts": 8, "retry_count": 3}
    )["all_pass"] is False


def test_fixture_serving_passes_without_emitting_semantic_content(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(serving, "_load_task", lambda paths, manifest: _fixture_task())
    model = serving.DeterministicFixtureModel()
    result = serving.run_serving(
        object(),
        paths={},
        manifest_path=tmp_path / "manifest.json",
        raw_path=tmp_path / "raw.json",
        model=model,
    )
    assert result["status"] == "passed"
    assert result["gates"]["all_pass"] is True
    assert result["parse_counts"] == {
        "root_observations": 8,
        "root_rankings": 8,
        "followups": 8,
        "followup_observations": 8,
        "followup_rankings": 8,
    }
    assert result["protocol"]["semantic_content_emitted"] is False
    assert "tasks" not in result
