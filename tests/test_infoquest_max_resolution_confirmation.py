from __future__ import annotations

from scripts import infoquest_max_resolution_confirmation as gate
from scripts import infoquest_support_causal_link_gate as base


def _fixtures() -> list[base.WorldFixture]:
    return [
        base.WorldFixture(
            fixture_id=f"D{record_id}W{world}",
            record_id=record_id,
            world=world,
            seed_message=f"Request {record_id}",
            simulator_system="Answer briefly.",
            truth_packet={},
            checklist=tuple(f"Need {index}" for index in range(5)),
        )
        for record_id in gate.CONFIRMATION_IDS
        for world in (1, 2)
    ]


def _initial(record_id: int) -> base.InitialPolicy:
    return base.InitialPolicy(
        hypotheses=tuple(f"Context {record_id}-{index}" for index in range(8)),
        roots=tuple(f"What is detail {index}?" for index in range(5)),
    )


def test_serving_gate_covers_all_four_model_stages(tmp_path):
    result = gate.run_serving_gate(
        gate._dry_models(serving=True),
        raw_path=tmp_path / "raw.json",
    )
    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == 10
    assert result["protocol"]["scientific_endpoint_evaluated"] is False


def test_fresh_metrics_max_resolution_beats_linear_and_random():
    fixtures = _fixtures()
    initials = {
        record_id: _initial(record_id) for record_id in gate.CONFIRMATION_IDS
    }
    bits = tuple(
        tuple(int(bit) for bit in value)
        for value in gate.DeterministicFreshChecklist.BITS
    )
    judgments = [base.ChecklistJudgment(bits, bits, bits) for _ in fixtures]
    selected = gate._dry_selected_actions()
    generator = gate.DeterministicFreshGenerator(
        "generator",
        selected_actions=selected,
    )
    requests = []
    for fixture in fixtures:
        initial = initials[fixture.record_id]
        for root_index in range(5):
            requests.append(
                gate.needs.information_need_messages(
                    fixture.seed_message,
                    initial,
                    root_index,
                    "Answer.",
                )
            )
    responses = generator.chat_complete_messages_batched(
        requests,
        temperature=0.0,
        block_size=50,
    )
    flat = []
    for response_index, response in enumerate(responses):
        fixture_index, root_index = divmod(response_index, 5)
        flat.append(
            gate.needs.parse_information_need_belief(
                response,
                initials[fixtures[fixture_index].record_id],
                root_index,
            )
        )
    beliefs = [flat[index : index + 5] for index in range(0, 50, 5)]
    metrics, _, gates = gate._score_metrics(
        fixtures,
        initials,
        beliefs,
        judgments,
    )
    assert all(gates.values())
    assert metrics["mean_max_selected_target_gain"] == 2.0
    assert metrics["max_target_optimal_cells"] == 50


def test_pairwise_counts_wins_ties_and_losses():
    assert gate._pairwise([2, 1, 0], [1, 1, 1]) == [1, 1, 1]
