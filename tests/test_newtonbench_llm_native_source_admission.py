from __future__ import annotations

from pathlib import Path

from scripts import newtonbench_llm_native_source_admission as source


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_literal_registry_shape() -> None:
    shape = source.registry_shape(
        "LAW_REGISTRY = {'easy': {'v0': 1, 'v1': 2, 'v2': 3}, "
        "'medium': {'v0': 1, 'v1': 2, 'v2': 3}, "
        "'hard': {'v0': 1, 'v1': 2, 'v2': 3}}"
    )
    assert shape["complete_three_by_three_registry"] is True
    assert shape["versions_per_difficulty"] == {
        "easy": 3,
        "hard": 3,
        "medium": 3,
    }


def test_frozen_source_fails_before_task_execution() -> None:
    result = source.audit(
        repo=Path("/tmp/bed-source-audits/newtonbench-20260813"),
        protocol_path=REPO_ROOT
        / "results/nonmyopic/NEWTONBENCH_LLM_NATIVE_SOURCE_ADMISSION_PROTOCOL_20260813.md",
    )
    assert result["status"] == "source_failed_closed"
    assert result["failure_gate"] == "finite_budget_horizon"
    assert result["gates"]["bindings_and_population"] is True
    assert result["gates"]["native_sequential_experiment"] is True
    assert result["gates"]["finite_budget_horizon"] is False
    assert result["gates"]["open_support_ownership"] is None
    assert result["aggregate_source"]["complete_source_law_registries_observed"] is True
    assert not any(result["access_accounting"].values())
