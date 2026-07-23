import json

from scripts.animals_branch_support_ranker import build_messages, prompt_payload


def _record():
    return {
        "target_measurement_only": "Hidden Echidna",
        "truth_covered_before_counterfactuals": False,
        "history": [{"question": "Does it fly?", "answer": "No"}],
        "belief_support": ["Cat", "Dog"],
        "candidate_dynamics": [
            {
                "question": "Is it found in Australia?",
                "p_yes": 0.2,
                "p_no": 0.8,
                "support_if_yes": ["Echidna", "Wombat"],
                "support_if_no": ["Cat", "Dog"],
                "immediate_eig": 0.4,
                "expected_truth_coverage": 0.2,
                "truth_covered_if_yes": True,
            }
        ],
    }


def test_branch_support_payload_includes_supports_but_excludes_target_fields():
    payload = prompt_payload(_record())
    encoded = json.dumps(payload)

    assert payload["current_support"] == ["Cat", "Dog"]
    assert payload["candidates"][0]["regenerated_support_if_yes"] == [
        "Echidna",
        "Wombat",
    ]
    assert "Hidden Echidna" not in encoded
    assert "target_measurement_only" not in encoded
    assert "expected_truth_coverage" not in encoded
    assert "truth_covered" not in encoded
    assert "immediate_eig" not in encoded
    assert "unknown target is not provided" in build_messages(_record())[1]["content"]
