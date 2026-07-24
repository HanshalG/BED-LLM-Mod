import pytest

from scripts.clindiag_multisample_atomic_opportunity import (
    ACTIONS,
    EXPECTED_FORMAL_REQUESTS,
    EXPECTED_SMOKE_REQUESTS,
    FORMAL_IDS,
    SAMPLES_PER_STATE,
    SEQUENCES,
    SMOKE_IDS,
    aggregate_measurements,
    analyze_record,
    state_id,
)


def _value(soft, coverage):
    return {
        "soft_value": soft,
        "coverage_probability": coverage,
        "sample_scores": [soft] * SAMPLES_PER_STATE,
    }


def test_protocol_request_counts_are_exact():
    assert len(ACTIONS) == 6
    assert len(SEQUENCES) == 30
    assert len(SMOKE_IDS) == 2
    assert len(FORMAL_IDS) == 4
    assert EXPECTED_SMOKE_REQUESTS == 26
    assert EXPECTED_FORMAL_REQUESTS == 608


def test_state_id_preserves_evidence_order():
    assert state_id(()) == "initial"
    assert state_id(("lab_1", "exam_1")) == "lab_1>exam_1"
    assert state_id(("exam_1", "lab_1")) == "exam_1>lab_1"


def test_aggregate_measurements_reports_soft_and_hard_recall():
    result = aggregate_measurements(
        [
            {"best_match_score": 0.9},
            {"best_match_score": 0.8},
            {"best_match_score": 0.1},
        ]
    )
    assert result["soft_value"] == pytest.approx(0.6)
    assert result["coverage_probability"] == pytest.approx(2 / 3)


def test_analysis_recovers_nonmyopic_pair_over_greedy_continuation():
    one = {action: _value(0.2, 0.0) for action in ACTIONS}
    one[ACTIONS[0]] = _value(0.7, 2 / 3)
    sequences = {
        state_id(sequence): _value(0.3, 0.0) for sequence in SEQUENCES
    }
    for second in ACTIONS[1:]:
        sequences[f"{ACTIONS[0]}>{second}"] = _value(0.72, 2 / 3)
    oracle = f"{ACTIONS[1]}>{ACTIONS[2]}"
    sequences[oracle] = _value(0.95, 1.0)
    record = {
        "source_id": "test",
        "initial": _value(0.0, 0.0),
        "one_step": one,
        "sequences": sequences,
        "replay_sequence": state_id(SEQUENCES[0]),
        "replay": sequences[state_id(SEQUENCES[0])],
    }
    result = analyze_record(record)
    assert result["greedy_action"] == ACTIONS[0]
    assert result["oracle_sequence"] == oracle
    assert result["nonmyopic_soft_gap"] == pytest.approx(0.23)
    assert result["pair_gain_over_best_one_step"] == pytest.approx(0.25)
    assert result["oracle_first_differs_from_greedy"]
