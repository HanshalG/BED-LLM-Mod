from __future__ import annotations

import json
from pathlib import Path

from scripts import hiddenbench_dynamic_belief_v3_endpoint_score as scorer
from tests.test_hiddenbench_dynamic_belief_v3_serving import FakeAdapter, synthetic_views
from scripts import hiddenbench_dynamic_belief_v3_serving as serving


def test_endpoint_scorer_is_finite_on_synthetic_transaction(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(serving, "load_views", lambda source_path: synthetic_views())
    run_dir = tmp_path / "run"
    label_free = serving.run_serving(source_path=tmp_path / "unused", output_dir=run_dir, adapter=FakeAdapter())
    raw = json.loads((run_dir / "private/RAW_RESPONSES.json").read_text())
    endpoints = {"endpoints": [{"slot": f"T{i + 1}", "correct_option_id": "O3"} for i in range(4)]}
    result = scorer.score(raw=raw, label_free=label_free, endpoints=endpoints)
    assert len(result["tasks"]) == 4
    assert all(0 < row["policies"][policy]["correct_probability"] < 1 for row in result["tasks"] for policy in row["policies"])
    assert set(result["gates"]) == {
        "dynamic_changes_first_query",
        "dynamic_brier_wins_three_of_four",
        "dynamic_mean_brier_gain_at_least_001",
        "dynamic_log_loss_better_than_myopic",
        "dynamic_nonworse_than_fixed",
        "dynamic_better_than_random",
        "all_rows_finite_nonsaturated",
    }
