from __future__ import annotations

import json
import math

import pytest

from scripts import regretbench_typed_action_smoke as smoke
from scripts import regretbench_typed_action_smoke_verify as verifier


class _Adapter:
    def __init__(self, *, codec_mode: str = "pass"):
        self.tasks = {row["task_id"]: row for row in smoke.load_public_tasks()}
        self.requests = 0
        self.attempts = 0
        self.cost = 0.0
        self.batches = []
        self.codec_mode = codec_mode

    @staticmethod
    def _options():
        return [
            {"option_id": 0, "label": "First plausible meaning"},
            {"option_id": 1, "label": "Second plausible meaning"},
            {"option_id": 2, "label": "Third plausible meaning"},
            {"option_id": 3, "label": smoke.OTHER_LABEL},
        ]

    def _proposal(self, payload):
        return json.dumps(
            {
                "hypotheses": [
                    {
                        "interpretation": f"semantic interpretation {index}",
                        "final_answer": f"candidate factual answer {index}",
                    }
                    for index in range(smoke.HYPOTHESES)
                ],
                "actions": [
                    {"action_id": row["action_id"], "options": self._options()}
                    for row in payload["allowed_actions"]
                ],
            }
        )

    @staticmethod
    def _evaluation(payload):
        particles = []
        for row in payload["particles"]:
            index = row["particle_index"]
            likelihoods = []
            for action_index in range(smoke.ACTIONS):
                likelihoods.append(
                    [1.0, 0.0, 0.0, 0.0]
                    if action_index == 0 and index % 2 == 0
                    else [0.0, 1.0, 0.0, 0.0]
                    if action_index == 0
                    else [1.0, 1.0, 1.0, 1.0]
                )
            particles.append(
                {
                    "particle_index": index,
                    "prior_weight": 1.0,
                    "option_likelihoods": likelihoods,
                }
            )
        return json.dumps({"particles": particles})

    def _codec(self, payload, seed):
        mappings = [
            {"value_index": row["value_index"], "option_id": row["value_index"] % 2}
            for row in payload["values"]
        ]
        if self.codec_mode == "disagree" and seed % 10 == 1:
            mappings[0]["option_id"] = 2
        elif self.codec_mode == "all_other":
            for row in mappings:
                row["option_id"] = 3
        elif self.codec_mode == "top_unrealized":
            for row in mappings:
                row["option_id"] = 0 if row["value_index"] % 2 == 0 else 2
        return json.dumps({"mappings": mappings})

    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages,
        seeds,
        *,
        temperature,
        response_format,
        max_new_tokens=None,
    ):
        name = response_format["json_schema"]["name"]
        self.batches.append(
            {
                "name": name,
                "messages": batch_messages,
                "seeds": list(seeds),
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            }
        )
        responses = []
        for messages, seed in zip(batch_messages, seeds, strict=True):
            payload = json.loads(messages[1]["content"])
            if name.endswith("_proposal"):
                responses.append(self._proposal(payload))
            elif name.endswith("_likelihood_evaluator"):
                responses.append(self._evaluation(payload))
            else:
                responses.append(self._codec(payload, seed))
        self.requests += len(responses)
        self.attempts += len(responses)
        self.cost += 0.0001 * len(responses)
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.attempts,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": self.cost,
            "adapter_prompt_tokens": 100 * self.requests,
            "adapter_completion_tokens": 100 * self.requests,
        }


def test_exact_eight_typed_rehearsal_passes_and_loads_values_after_roots(
    monkeypatch, tmp_path
):
    adapter = _Adapter()
    original = smoke.load_selected_cig

    def guarded_load(task):
        assert adapter.requests == 4
        return original(task)

    monkeypatch.setattr(smoke, "load_selected_cig", guarded_load)
    result = smoke.run_smoke(
        output_dir=tmp_path,
        adapter=adapter,
        daily_budget_status={"status": "synthetic"},
    )
    assert result["status"] == "passed"
    assert result["gates"]["all_pass"] is True
    assert result["usage"]["adapter_requests"] == 8
    assert [len(batch["messages"]) for batch in adapter.batches] == [2, 2, 2, 2]
    assert result["gates"]["source_values_loaded_only_after_four_root_calls"] is True
    assert result["source_values_publicly_reported"] is False
    assert result["codec_mappings_publicly_reported"] is False

    ordering = json.loads((tmp_path / "private/ORDERING.json").read_text())
    assert ordering["root_requests_completed"] == 4
    assert ordering["source_values_loaded_after_root_requests"] is True
    assert len(ordering["root_payload_sha256"]) == 4
    replay = verifier.replay(tmp_path)
    assert replay["status"] == "verified"
    assert replay["mismatches"] == []
    assert replay["model_calls_made"] == 0


def test_root_payloads_include_only_public_actions_and_no_source_values(tmp_path):
    adapter = _Adapter()
    smoke.run_smoke(output_dir=tmp_path, adapter=adapter)
    for batch in adapter.batches[:2]:
        for messages in batch["messages"]:
            payload = json.loads(messages[1]["content"])
            text = smoke.canonical_json(payload).lower()
            assert "allowed_actions" in payload
            assert all(
                set(row) == {"action_id", "question", "public_order"}
                for row in payload["allowed_actions"]
            )
            assert "source_values" not in payload
            assert "intent_descriptions" not in payload
            assert "endpoint" not in text


def test_categorical_mutual_information_matches_manual_value():
    support = {
        "hypotheses": [
            {
                "probability": 0.5,
                "option_likelihoods": [[1.0, 0.0, 0.0, 0.0]],
            },
            {
                "probability": 0.5,
                "option_likelihoods": [[0.0, 1.0, 0.0, 0.0]],
            },
        ]
    }
    assert smoke.categorical_mutual_information(support, 0) == pytest.approx(
        math.log(2.0)
    )


@pytest.mark.parametrize(
    ("mode", "failed_gate"),
    [
        ("disagree", "all_codec_replicates_agree"),
        ("all_other", "all_codec_mappings_use_at_least_two_non_other_options"),
        ("top_unrealized", "all_top_predictive_options_are_realized"),
    ],
)
def test_codec_semantic_failures_authorize_nothing(tmp_path, mode, failed_gate):
    result = smoke.run_smoke(output_dir=tmp_path, adapter=_Adapter(codec_mode=mode))
    assert result["status"] == "calibration_failed"
    assert result["authorizes"] == "nothing"
    assert result["gates"][failed_gate] is False
    assert result["mechanics_opened"] is False


def test_parsers_reject_wrong_other_label_and_zero_likelihood():
    task = smoke.load_public_tasks()[0]
    payload = {"task_id": task["task_id"], "allowed_actions": task["actions"]}
    proposal = json.loads(_Adapter()._proposal(payload))
    proposal["actions"][0]["options"][3]["label"] = "Other"
    with pytest.raises(ValueError, match="exact other"):
        smoke.parse_proposal(json.dumps(proposal), task)

    clean = smoke.parse_proposal(_Adapter()._proposal(payload), task)
    evaluation = json.loads(
        _Adapter._evaluation(
            {
                "particles": [
                    {"particle_index": index}
                    for index in range(smoke.HYPOTHESES)
                ]
            }
        )
    )
    evaluation["particles"][0]["option_likelihoods"][0] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="zero mass"):
        smoke.parse_evaluation(json.dumps(evaluation), clean)


def test_parser_rejects_missing_or_duplicate_action_id():
    task = smoke.load_public_tasks()[0]
    payload = {"task_id": task["task_id"], "allowed_actions": task["actions"]}
    proposal = json.loads(_Adapter()._proposal(payload))
    proposal["actions"][0]["action_id"] = proposal["actions"][1]["action_id"]
    with pytest.raises(ValueError, match="action ID"):
        smoke.parse_proposal(json.dumps(proposal), task)


def test_independent_verifier_rejects_result_and_privacy_tamper(tmp_path):
    smoke.run_smoke(output_dir=tmp_path, adapter=_Adapter())
    result_path = tmp_path / "RESULT.json"
    result = json.loads(result_path.read_text())
    result["task_diagnostics"][0]["selected_mutual_information_nats"] = -1.0
    result_path.write_text(json.dumps(result))
    replay = verifier.replay(tmp_path)
    assert replay["status"] == "rejected"
    assert "task_diagnostics" in replay["mismatches"]

    smoke.run_smoke(output_dir=tmp_path, adapter=_Adapter())
    privacy_path = tmp_path / "private/PRIVACY.json"
    privacy = json.loads(privacy_path.read_text())
    privacy["audits"][0]["payload_sha256"] = "0" * 64
    privacy_path.write_text(json.dumps(privacy))
    with pytest.raises(ValueError, match="privacy replay"):
        verifier.replay(tmp_path)


def test_independent_verifier_rejects_ordering_tamper(tmp_path):
    smoke.run_smoke(output_dir=tmp_path, adapter=_Adapter())
    path = tmp_path / "private/ORDERING.json"
    ordering = json.loads(path.read_text())
    ordering["root_requests_completed"] = 3
    path.write_text(json.dumps(ordering))
    with pytest.raises(ValueError, match="ordering replay"):
        verifier.replay(tmp_path)
