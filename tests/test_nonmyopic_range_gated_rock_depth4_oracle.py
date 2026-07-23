from __future__ import annotations

from scripts.audit_nonmyopic_range_gated_rock_depth4_oracle import audit
from scripts.nonmyopic_range_gated_rock_depth4_oracle import (
    RangeGatedDepth4Config,
    _model,
    run_qualification,
)


def test_corner_start_makes_depth_four_first_move_load_bearing() -> None:
    config = RangeGatedDepth4Config()
    config.validate()
    model = _model(config)

    assert model.map_spec.start_position == (6, 6)
    assert model.map_spec.rock_positions[4] == (6, 3)


def test_depth_four_qualification_passes_small_monkeypatched_protocol(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        RangeGatedDepth4Config,
        "validate",
        lambda self: None,
    )
    config = RangeGatedDepth4Config(
        num_trials=4,
        bootstrap_replicates=100,
        trial_concurrency=1,
    )
    result = run_qualification(config)

    assert result["primary_gate_passed"]
    assert result["truth_log_corroboration_passed"]
    assert all(result["mechanics"].values())
    assert result["comparison"]["entropy_auc_wins_ties_losses"] == [4, 0, 0]
    replay = audit(result)
    assert replay["passed"]
    assert all(replay["mechanics"].values())
