from __future__ import annotations

from scripts import number_game_truth_coverage_alignment_power_audit as audit


def test_coverage_gate_requires_mean_interval_and_win_direction() -> None:
    assert audit._coverage_gate([0.2, 0.2, 0.1, 0.1])
    assert not audit._coverage_gate([0.2, 0.2, -0.2, -0.2])
    assert not audit._coverage_gate([-0.1, -0.1, -0.2, -0.2])


def test_simulation_rejects_empty_rows() -> None:
    try:
        audit.simulate_power([], draws=2, sample_sizes=(4,))
    except ValueError as exc:
        assert "historical rows" in str(exc)
    else:
        raise AssertionError("empty power source should fail")
