from pathlib import Path

from scripts import number_game_cross_planner_replication as cross


def test_cross_planner_uses_fresh_seeds_and_swapped_models(monkeypatch):
    captured = {}

    def fake_run_powered_replication(**kwargs):
        captured.update(kwargs)
        return {"status": "test"}

    monkeypatch.setattr(
        cross,
        "run_powered_replication",
        fake_run_powered_replication,
    )

    result = cross.run_cross_planner_replication(
        output_dir=Path("/tmp/cross-planner"),
        run_id="test-run",
    )

    assert result == {"status": "test"}
    assert captured["planning_model"] == "openai/gpt-5.4-mini"
    assert captured["target_model"] == "google/gemini-2.5-flash"
    assert captured["tree_seeds"] == tuple(range(27000, 27032))
    assert captured["target_seeds"] == tuple(range(27100, 27132))
    assert captured["run_budget_usd"] == 3.0


def test_cross_planner_seeds_do_not_overlap_prior_powered_run():
    assert not set(cross.TREE_SEEDS) & set(range(26400, 26432))
    assert not set(cross.TARGET_SEEDS) & set(range(26500, 26532))
