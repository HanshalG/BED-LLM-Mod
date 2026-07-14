from __future__ import annotations

import json
import re
from pathlib import Path

from environments.mediq.env import UNAVAILABLE_OUTCOME, MediQEnvironment
from helpers import Config
from scripts.replay_mediq_likelihood_calibration import replay_records


FIXTURE = Path(__file__).parent / "fixtures" / "mediq_tiny.jsonl"


class CalibrationModel:
    def chat_complete_messages_batched(
        self,
        batch_messages,
        temperature,
        block_size,
        max_new_tokens=None,
    ):
        del temperature, block_size, max_new_tokens
        return [self._route(messages) for messages in batch_messages]

    @staticmethod
    def _route(messages):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "estimate record coverage for the MediQ benchmark" in system:
            return json.dumps(
                {
                    "probabilities": {
                        "Answerable from record": 0.6,
                        "Not answerable from record": 0.4,
                    }
                }
            )
        if "counterfactual clinical-record model" in system:
            label = re.search(r"correct answer is ([A-Z]):", user).group(1)
            values = (0.9, 0.1) if label == "A" else (0.1, 0.9)
            return json.dumps(
                {"probabilities": {"Yes": values[0], "No": values[1]}}
            )
        raise AssertionError(system)


def _source_record(task, *, observed: str) -> dict:
    turns = []
    for round_index, outcome in enumerate((observed, UNAVAILABLE_OUTCOME), 1):
        query = f"Is diagnostic finding {round_index} present?"
        turns.append(
            {
                "query": query,
                "outcomes": ["Yes", "No", UNAVAILABLE_OUTCOME],
                "reply": (
                    f"Frozen reply {round_index}"
                    if outcome != UNAVAILABLE_OUTCOME
                    else "The patient cannot answer this question from the supplied record."
                ),
                "mapped_outcome": outcome,
                "mapped_cleanly": True,
                "candidate_details": [
                    {
                        "query": query,
                        "prior": {label: 0.25 for label in task.option_labels},
                    }
                ],
            }
        )
    return {"task_id": task.task_id, "turns": turns}


def test_frozen_mediq_likelihood_replay_passes_factored_contract() -> None:
    config = Config(
        task="mediq",
        method_names=["EIG"],
        mediq_data_path=str(FIXTURE),
        mediq_verify_official_hash=False,
        mediq_num_trials=2,
        mediq_num_rounds=2,
        mediq_trial_batch_size=2,
        mediq_num_candidates=1,
        mediq_likelihood_mode="factored_record",
    )
    model = CalibrationModel()
    environment = MediQEnvironment(config, model).configure_for_run(config)
    environment.set_questioner(model)
    first, second = environment.tasks
    records = [
        _source_record(first, observed="Yes"),
        _source_record(second, observed="Yes"),
        _source_record(first, observed="Yes"),
        _source_record(second, observed="Yes"),
        _source_record(first, observed="Yes"),
    ]

    report = replay_records(records, environment)

    assert report["status"] == "pass"
    assert all(report["checks"].values())
    assert report["summary"]["num_turns"] == 10
    assert report["summary"]["maximum_unavailable_likelihood_span"] == 0.0
    assert report["summary"]["maximum_unavailable_posterior_linf_change"] < 1e-12
    assert report["summary"]["available_true_label_favored_rate"] == 1.0
