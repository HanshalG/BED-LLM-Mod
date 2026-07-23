import json

import pytest

from scripts.animals_belief_recall_ranker import (
    build_messages,
    parse_scores,
    prompt_payload,
    summarize_rankings,
)


def _record():
    return {
        "target_measurement_only": "Hidden Otter",
        "truth_covered_before_counterfactuals": False,
        "history": [{"question": "Does it fly?", "answer": "No"}],
        "candidate_dynamics": [
            {
                "question": "Does it live in water?",
                "p_yes": 0.4,
                "p_no": 0.6,
                "support_size_if_yes": 12,
                "support_size_if_no": 18,
                "immediate_eig": 0.3,
                "expected_truth_coverage": 0.8,
            },
            {
                "question": "Is it nocturnal?",
                "p_yes": 0.5,
                "p_no": 0.5,
                "support_size_if_yes": 15,
                "support_size_if_no": 14,
                "immediate_eig": 0.7,
                "expected_truth_coverage": 0.1,
            },
        ],
    }


def test_prompt_payload_excludes_all_target_measurement_fields():
    payload = prompt_payload(_record())
    encoded = json.dumps(payload)

    assert "target_measurement_only" not in encoded
    assert "truth_covered" not in encoded
    assert "Hidden Otter" not in encoded
    assert [item["index"] for item in payload["candidates"]] == [0, 1]
    assert "unknown target is not provided" in build_messages(_record())[1]["content"]


def test_parse_scores_requires_one_finite_score_per_candidate():
    assert parse_scores('{"scores":[0.25,3]}', 2) == [0.25, 3.0]
    assert parse_scores('```json\n{"scores":[0.25,3]}\n```', 2) == [0.25, 3.0]

    with pytest.raises(ValueError, match="bare JSON"):
        parse_scores('Scores: {"scores":[1,2]}', 2)
    with pytest.raises(ValueError, match="exactly 2"):
        parse_scores('{"scores":[1]}', 2)
    with pytest.raises(ValueError, match="only a scores"):
        parse_scores('{"scores":[1,2],"target":"leak"}', 2)


def test_summarize_rankings_reports_paired_coverage_gain():
    first = _record()
    first["belief_recall_scores"] = [0.9, 0.1]
    second = _record()
    second["belief_recall_scores"] = [0.2, 0.8]
    second["candidate_dynamics"][0]["expected_truth_coverage"] = 0.2
    second["candidate_dynamics"][1]["expected_truth_coverage"] = 0.2

    summary = summarize_rankings([first, second])

    assert summary["num_states"] == 2
    assert summary["num_active_states"] == 1
    assert summary["mean_selected_expected_truth_coverage_belief_recall"] == pytest.approx(0.5)
    assert summary["mean_selected_expected_truth_coverage_immediate_eig"] == pytest.approx(0.15)
    assert summary["mean_paired_selected_coverage_gain"] == pytest.approx(0.35)
    assert summary["ranker_immediate_wins_ties_losses"] == [1, 1, 0]
