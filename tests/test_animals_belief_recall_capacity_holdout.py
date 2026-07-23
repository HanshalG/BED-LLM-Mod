from scripts.animals_belief_recall_capacity_holdout import capacity_gated_records


def _record(support_size):
    return {
        "belief_support_size": support_size,
        "belief_recall_scores": [0.1, 0.9],
        "candidate_dynamics": [
            {"immediate_eig": 0.8},
            {"immediate_eig": 0.2},
        ],
    }


def test_capacity_gate_uses_ranker_at_or_below_generation_capacity():
    below, equal, above = capacity_gated_records(
        [_record(15), _record(16), _record(17)],
        support_capacity=16,
    )

    assert below["capacity_gate_active"]
    assert equal["capacity_gate_active"]
    assert below["belief_recall_scores"] == [0.1, 0.9]
    assert equal["belief_recall_scores"] == [0.1, 0.9]
    assert not above["capacity_gate_active"]
    assert above["belief_recall_scores"] == [0.8, 0.2]
