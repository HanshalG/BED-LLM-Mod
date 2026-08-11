from __future__ import annotations

import json
import re

import pytest

from scripts import regretbench_proposal_evaluator_v1_smoke as smoke
from scripts import regretbench_proposal_evaluator_v1_smoke_verify as verifier


def _root_questions(cig):
    references = [row.text for row in cig.reference_questions]
    if cig.cig_id == "ambigdocs_test_60701":
        return [
            references[0],
            references[1],
            "Are you asking about Lord Norton as a title or as a person?",
            "Which historical era is relevant to the person you mean?",
        ]
    return references


def _child_questions(cig, *, conditioned: bool, branch: int):
    references = [row.text for row in cig.reference_questions]
    if cig.cig_id == "ambigdocs_test_60701":
        return [
            references[1],
            "Which historical era is relevant to the person you mean?",
            "What time period is relevant?",
            "Is the relevant period historical or modern?",
        ]
    if not conditioned:
        return [references[3], references[1], references[2], references[0]]
    if branch == 0:
        return [references[1], references[2], references[3], references[0]]
    return [references[2], references[1], references[3], references[0]]


class _Adapter:
    def __init__(self, *, honor_focus: bool = True):
        self.cigs = {cig.cig_id: cig for cig in smoke.load_smoke_cigs()}
        self.requests = 0
        self.attempts = 0
        self.cost = 0.0
        self.batches = []
        self.honor_focus = honor_focus

    def _proposal(self, payload, seed, *, root):
        cig = self.cigs[payload["task_id"]]
        dialogue = payload["dialogue"]
        conditioned = bool(dialogue)
        branch = int(seed % 10)
        focus = dialogue[-1]["content"] if conditioned else None
        hypotheses = []
        for index in range(8):
            marker = f"focus={focus}|" if focus is not None else "blind|"
            hypotheses.append(
                {
                    "interpretation": f"{marker}seed={seed}|intent_index={index % len(cig.intents)}|particle={index}",
                    "final_answer": f"candidate-answer-{index}",
                }
            )
        questions = (
            _root_questions(cig)
            if root
            else _child_questions(cig, conditioned=conditioned, branch=branch)
        )
        return json.dumps({"hypotheses": hypotheses, "questions": questions})

    def _evaluation(self, payload):
        cig = self.cigs[payload["task_id"]]
        questions = payload["questions"]
        child = len(questions) == 5
        particles = []
        for item in payload["particles"]:
            index = item["particle_index"]
            interpretation = item["interpretation"]
            intent_match = re.search(r"intent_index=(\d+)", interpretation)
            assert intent_match
            intent = cig.intents[int(intent_match.group(1))]
            replies = []
            for question_index, question in enumerate(questions):
                if child and question_index == 0:
                    focus = re.search(r"^focus=(.*?)\|seed=", interpretation)
                    if focus and self.honor_focus:
                        reply = focus.group(1)
                    else:
                        action = smoke.mapped_action(cig, question)
                        reply = str(intent.slots[action["facet"]])
                elif child:
                    reply = f"option-{(index // 2 + question_index) % 2}"
                elif question_index == 0:
                    action = smoke.mapped_action(cig, question)
                    reply = str(intent.slots[action["facet"]])
                else:
                    reply = "constant"
                replies.append(reply)
            particles.append(
                {
                    "particle_index": index,
                    "prior_weight": 1.0,
                    "predicted_replies": replies,
                }
            )
        return json.dumps({"particles": particles})

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
        root = name.endswith("_proposal") and len(batch_messages) == 2
        responses = []
        for messages, seed in zip(batch_messages, seeds, strict=True):
            payload = json.loads(messages[1]["content"])
            if name.endswith("_proposal"):
                responses.append(self._proposal(payload, seed, root=root))
            else:
                responses.append(self._evaluation(payload))
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


def test_exact_twenty_full_rehearsal_passes(tmp_path):
    adapter = _Adapter()
    result = smoke.run_smoke(
        output_dir=tmp_path,
        adapter=adapter,
        daily_budget_status={"status": "synthetic"},
    )
    assert result["status"] == "passed"
    assert result["gates"]["all_pass"] is True
    assert result["usage"]["adapter_requests"] == 20
    assert adapter.requests == 20
    assert [len(batch["messages"]) for batch in adapter.batches] == [2, 2, 8, 8]
    assert result["signal_summary"]["own_minus_opposite_mean"] == pytest.approx(1.0)
    assert result["signal_summary"]["own_minus_answer_free_mean"] > 0.0
    assert result["signal_summary"]["selected_child_facet_change_count"] >= 1
    assert result["endpoint_outcomes_opened"] is False
    assert result["development_opened"] is False

    raw = json.loads((tmp_path / "private/RAW_RESPONSES.json").read_text())
    assert len(raw["root_proposals"]) == 2
    assert len(raw["root_evaluations"]) == 2
    assert len(raw["branch_proposals"]) == 8
    assert len(raw["branch_evaluations"]) == 8
    assert raw["branch_proposal_seeds"] == [
        smoke.BRANCH_PROPOSAL_SEED_START,
        smoke.BRANCH_PROPOSAL_SEED_START,
        smoke.BRANCH_PROPOSAL_SEED_START + 1,
        smoke.BRANCH_PROPOSAL_SEED_START + 1,
        smoke.BRANCH_PROPOSAL_SEED_START + 10,
        smoke.BRANCH_PROPOSAL_SEED_START + 10,
        smoke.BRANCH_PROPOSAL_SEED_START + 11,
        smoke.BRANCH_PROPOSAL_SEED_START + 11,
    ]
    replay = verifier.replay(tmp_path)
    assert replay["status"] == "verified"
    assert replay["mismatches"] == []
    assert replay["model_calls_made"] == 0


def test_evaluator_payloads_have_no_dedicated_answer_or_dialogue(tmp_path):
    adapter = _Adapter()
    smoke.run_smoke(output_dir=tmp_path, adapter=adapter)
    evaluator_batches = [
        batch
        for batch in adapter.batches
        if batch["name"].endswith(("_evaluator_4", "_evaluator_5"))
    ]
    assert len(evaluator_batches) == 2
    for batch in evaluator_batches:
        for messages in batch["messages"]:
            payload = json.loads(messages[1]["content"])
            assert set(payload) == {
                "task_id",
                "prompt",
                "particles",
                "questions",
                "proposal_sha256",
                "source",
            }
            assert "dialogue" not in payload
            assert "answer" not in payload
            assert "reply" not in payload
            assert "probability" not in payload
            assert "lineage" not in payload


def test_exact_conditioning_matches_manual_partition():
    support = {
        "hypotheses": [
            {"probability": 0.1, "predicted_replies": ["A"]},
            {"probability": 0.2, "predicted_replies": ["B"]},
            {"probability": 0.3, "predicted_replies": ["A"]},
            {"probability": 0.4, "predicted_replies": ["B"]},
        ]
    }
    updated, diagnostic = smoke.condition_on_reply(support, 0, "A")
    assert diagnostic["prior_predictive_mass"] == pytest.approx(0.4)
    assert [row["probability"] for row in updated["hypotheses"]] == pytest.approx(
        [0.25, 0.0, 0.75, 0.0]
    )
    assert diagnostic["posterior_predictive_matched_reply"] == pytest.approx(1.0)


def test_answer_signal_gate_rejects_answer_ignoring_proposals(tmp_path):
    result = smoke.run_smoke(output_dir=tmp_path, adapter=_Adapter(honor_focus=False))
    assert result["status"] == "mechanics_failed"
    assert result["authorizes"] == "nothing"
    assert result["gates"]["mean_own_minus_opposite_at_least_010"] is False
    assert result["gates"]["mean_own_minus_answer_free_at_least_005"] is False


def test_parsers_reject_probability_in_proposal_and_duplicate_evaluator_index():
    proposal = {
        "hypotheses": [
            {
                "interpretation": f"interpretation-{index}",
                "final_answer": f"answer-{index}",
                "prior_weight": 1.0,
            }
            for index in range(8)
        ],
        "questions": [f"Question {index}?" for index in range(4)],
    }
    with pytest.raises(ValueError, match="wrong fields"):
        smoke.parse_proposal(json.dumps(proposal))

    clean = smoke.parse_proposal(
        json.dumps(
            {
                "hypotheses": [
                    {
                        "interpretation": f"interpretation-{index}",
                        "final_answer": f"answer-{index}",
                    }
                    for index in range(8)
                ],
                "questions": [f"Question {index}?" for index in range(4)],
            }
        )
    )
    evaluation = {
        "particles": [
            {
                "particle_index": 0 if index == 7 else index,
                "prior_weight": 1.0,
                "predicted_replies": ["reply"] * 4,
            }
            for index in range(8)
        ]
    }
    with pytest.raises(ValueError, match="invalid values"):
        smoke.parse_evaluation(
            json.dumps(evaluation), clean, clean["questions"]
        )


def test_independent_verifier_rejects_public_result_tamper(tmp_path):
    smoke.run_smoke(output_dir=tmp_path, adapter=_Adapter())
    result_path = tmp_path / "RESULT.json"
    result = json.loads(result_path.read_text())
    result["signal_summary"]["own_minus_opposite_mean"] = -1.0
    result_path.write_text(json.dumps(result))
    replay = verifier.replay(tmp_path)
    assert replay["status"] == "rejected"
    assert "signal_summary" in replay["mismatches"]


def test_independent_verifier_rejects_privacy_hash_tamper(tmp_path):
    smoke.run_smoke(output_dir=tmp_path, adapter=_Adapter())
    privacy_path = tmp_path / "private/PRIVACY.json"
    privacy = json.loads(privacy_path.read_text())
    privacy["audits"][-1]["payload_sha256"] = "0" * 64
    privacy_path.write_text(json.dumps(privacy))
    with pytest.raises(ValueError, match="privacy replay changed"):
        verifier.replay(tmp_path)
