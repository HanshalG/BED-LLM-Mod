from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_mediq_smoke import analyze


def _write_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    item_dir = run_dir / "items" / "000_EIG"
    item_dir.mkdir(parents=True)
    prior = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    likelihoods = {
        "A": [0.8, 0.1, 0.1],
        "B": [0.1, 0.8, 0.1],
        "C": [0.1, 0.8, 0.1],
        "D": [0.1, 0.8, 0.1],
    }
    predictive_values = [
        sum(prior[label] * likelihoods[label][index] for label in prior)
        for index in range(3)
    ]
    predictive = dict(
        zip(
            ["Yes", "No", "Information unavailable / not in record"],
            predictive_values,
            strict=True,
        )
    )
    score = 0.298950342680141
    trials = []
    for task_index in range(5):
        facts = ["Initial fact.", "A directly relevant hidden fact."]
        turns = []
        for round_index in range(2):
            query = f"Is relevant finding number {round_index + 1} present?"
            outcomes = list(predictive)
            turns.append(
                {
                    "query": query,
                    "outcomes": outcomes,
                    "candidate_queries": [query],
                    "candidate_details": [
                        {
                            "query": query,
                            "outcomes": outcomes,
                            "semantic_validation": {
                                "valid": True,
                                "reason": "valid atomic partition",
                            },
                            "score": score,
                            "prior": prior,
                            "likelihoods": likelihoods,
                            "predictive_outcome_probabilities": predictive,
                        }
                    ],
                    "selected_score": score,
                    "reply": facts[1],
                    "selected_fact_indices": [1],
                    "mapped_outcome": outcomes[0],
                    "mapped_cleanly": True,
                    "grounded": True,
                    "relevant": True,
                    "cannot_answer": False,
                    "metrics": {
                        "realized_entropy_drop": 0.1,
                        "realized_truth_log_probability_gain": 0.2,
                        "observed_outcome_predictive_probability": predictive_values[0],
                    },
                }
            )
        trials.append(
            {
                "task_id": f"mediq:imedqa:{task_index}",
                "source_id": str(task_index),
                "dataset": "imedqa",
                "question": "Which option is correct?",
                "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "answer_idx": "A",
                "initial_info": facts[0],
                "facts": facts,
                "turns": turns,
            }
        )
    (item_dir / "mediq_interactions.json").write_text(json.dumps(trials))
    (item_dir / "mediq_data_manifest.json").write_text(
        json.dumps(
            {
                "repository": "https://github.com/stellali7/MediQ.git",
                "commit": "faa2ce62fef0423e35af4c31d7537aad973173eb",
                "dataset": "imedqa",
                "expected_sha256": "3bfc7090d060dd8d11e4237344ed78846707faab433a84d078191627ad3c9526",
                "raw_row_count": 1272,
                "usable_row_count": 1269,
                "skip_unusable_tasks": True,
                "excluded_source_ids": ["224", "298", "779"],
                "task_offset_within_usable_rows": 0,
                "selected_source_ids": [str(index) for index in range(5)],
            }
        )
    )
    metrics = {
        "accuracy": [0.4, 0.6],
        "structured_parse_retries": [0.0, 1.0],
        "structured_parse_failures": [0.0, 0.0],
        "candidate_validation_checks": [5.0, 10.0],
        "candidate_validation_retries": [0.0, 0.0],
        "candidate_validation_failures": [0.0, 0.0],
        "patient_relevance_failures": [0.0, 0.0],
        "backend_requests": [235],
        "backend_prompt_tokens": [1000],
        "backend_completion_tokens": [200],
        "backend_reasoning_tokens": [0],
        "backend_forced_exits": [0],
        "backend_cost_usd": [0.05],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps({"items": [{"method": "EIG", "metrics": metrics}]})
    )
    (run_dir / "run.log").write_text("")
    return run_dir


def test_mediq_smoke_analyzer_passes_mechanics_and_requires_manual_review(
    tmp_path: Path,
) -> None:
    report = analyze(_write_run(tmp_path))
    assert report["automated_pass"] is True
    assert report["status"] == "automated_pass_manual_review_pending"
    assert report["manual_transcript_review_required"] is True
    assert report["answer_set_coverage"] == 1.0
    assert report["patient_grounding_rate"] == 1.0
    assert report["num_candidate_scores"] == 10
    assert report["positive_candidate_eig_rate"] == 1.0
    assert report["backend_cost_usd"] == 0.05


def test_mediq_smoke_analyzer_rejects_nonverbatim_patient_reply(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    path = next(run_dir.rglob("mediq_interactions.json"))
    trials = json.loads(path.read_text())
    trials[0]["turns"][0]["reply"] = "An inferred fact not in the record."
    path.write_text(json.dumps(trials))
    report = analyze(run_dir)
    assert report["automated_pass"] is False
    assert report["checks"]["verbatim_patient_grounding"] is False


def test_mediq_smoke_analyzer_rejects_bad_mapping_or_eig_table(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    path = next(run_dir.rglob("mediq_interactions.json"))
    trials = json.loads(path.read_text())
    trials[0]["turns"][0]["mapped_cleanly"] = False
    trials[1]["turns"][0]["candidate_details"][0]["score"] = 99.0
    path.write_text(json.dumps(trials))
    report = analyze(run_dir)
    assert report["automated_pass"] is False
    assert report["checks"]["valid_mapping_contract"] is False
    assert report["checks"]["valid_finite_target_eig_tables"] is False


def test_mediq_smoke_analyzer_allows_clean_mapping_abstention_above_threshold(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(tmp_path)
    path = next(run_dir.rglob("mediq_interactions.json"))
    trials = json.loads(path.read_text())
    trials[0]["turns"][0]["mapped_cleanly"] = False
    trials[0]["turns"][0]["mapped_outcome"] = None
    path.write_text(json.dumps(trials))
    report = analyze(run_dir)
    assert report["automated_pass"] is True
    assert report["answer_set_coverage"] == 0.9
    assert len(report["unmapped_turns"]) == 1
