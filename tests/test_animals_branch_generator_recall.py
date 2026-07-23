import json

from scripts.animals_branch_generator_recall import branch_histories


def test_branch_histories_include_only_observed_and_counterfactual_qa():
    record = {
        "target_measurement_only": "Hidden Echidna",
        "history": [{"question": "Does it fly?", "answer": "No"}],
        "belief_support": ["Cat", "Dog"],
        "candidate_dynamics": [
            {"question": "Is it found in Australia?"},
            {"question": "Is it a mammal?"},
        ],
    }

    histories = branch_histories(record)
    encoded = json.dumps(histories)

    assert len(histories) == 4
    assert histories[0][-2:] == [
        {"role": "assistant", "content": "Is it found in Australia?"},
        {"role": "user", "content": "Yes"},
    ]
    assert histories[1][-1]["content"] == "No"
    assert "Hidden Echidna" not in encoded
    assert "target_measurement_only" not in encoded
