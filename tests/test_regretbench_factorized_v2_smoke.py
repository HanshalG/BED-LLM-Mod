from __future__ import annotations

import json

from scripts import regretbench_factorized_v2_smoke as smoke
from scripts import regretbench_factorized_v2_smoke_verify as verify
from scripts import regretbench_deepseek_support_recovery as source


def _question(text: str) -> str:
    return text.strip().rstrip("?") + "?"


def _task_info():
    result = {}
    for index, cig in enumerate(smoke.load_smoke_cigs()):
        _, truth = source.sample_truth(cig, smoke.ROOT_SEED_START + 10_000 + index)
        supported = []
        for row in cig.reference_questions:
            question = _question(row.text)
            mapping = source.map_and_answer(cig, question, truth)
            if mapping["supported"] and all(
                mapping["facet"] != previous[1]["facet"] for previous in supported
            ):
                supported.append((question, mapping))
        assert len(supported) >= 2
        result[cig.cig_id] = {
            "first_question": supported[0][0],
            "first_reply": supported[0][1]["answer"],
            "second_question": supported[1][0],
            "second_reply": supported[1][1]["answer"],
        }
    return result


class _Adapter:
    def __init__(self):
        self.info = _task_info()
        self.requests = 0
        self.seeds = []
        self.schemas = []

    def _root(self, payload):
        info = self.info[payload["task_id"]]
        questions = [
            info["first_question"],
            info["second_question"],
            "Which alternate category is intended?",
            "Which alternate comparison is intended?",
        ]
        return json.dumps(
            {
                "hypotheses": [
                    {
                        "interpretation": f"root interpretation {index}",
                        "final_answer": f"root answer {index}",
                        "prior_weight": index + 1,
                        "predicted_replies": [
                            info["first_reply"] if index < 2 else f"first {index}",
                            info["second_reply"] if index == 0 else f"second {index}",
                            f"category {index % 2}",
                            f"comparison {index % 3}",
                        ],
                    }
                    for index in range(8)
                ],
                "questions": questions,
            }
        )

    def _transition(self, payload):
        info = self.info[payload["task_id"]]
        return json.dumps(
            {
                "hypotheses": [
                    {
                        "parent_index": row["parent_index"],
                        "revision_type": (
                            "retained" if row["parent_index"] < 2 else "revised"
                        ),
                        "interpretation": (
                            row["interpretation"]
                            if row["parent_index"] < 2
                            else row["interpretation"] + " revised"
                        ),
                        "final_answer": row["final_answer"],
                        "prior_weight": row["parent_index"] + 1,
                        "predicted_replies": [
                            f"discarded {row['parent_index']} {question}"
                            for question in range(4)
                        ],
                    }
                    for row in payload["parent_particles"]
                ],
                "questions": [
                    info["second_question"],
                    "Which follow-up category is intended?",
                    "Which follow-up period is intended?",
                    "Which follow-up comparison is intended?",
                ],
            }
        )

    def _static(self, payload):
        info = self.info[payload["task_id"]]
        return json.dumps(
            {
                "particles": [
                    {
                        "particle_index": row["particle_index"],
                        "predicted_replies": [
                            (
                                info["first_reply"]
                                if row["particle_index"] < 2
                                else f"condition {row['particle_index']}"
                            ),
                            (
                                info["second_reply"]
                                if row["particle_index"] == 0
                                else f"followup {row['particle_index']}"
                            ),
                            f"category {row['particle_index'] % 2}",
                            f"period {row['particle_index'] % 3}",
                            f"comparison {row['particle_index'] % 4}",
                        ],
                    }
                    for row in payload["child_particles"]
                ]
            }
        )

    def chat_complete_seeded_messages_batched_structured(
        self,
        messages,
        seeds,
        *,
        temperature,
        response_format,
        max_new_tokens,
    ):
        name = response_format["json_schema"]["name"]
        self.requests += len(messages)
        self.seeds.extend(seeds)
        self.schemas.extend([name] * len(messages))
        responses = []
        for request in messages:
            payload = json.loads(request[1]["content"])
            if name == "regretbench_enriched_support":
                responses.append(self._root(payload))
            elif name == "regretbench_smc_enriched_transition":
                responses.append(self._transition(payload))
            else:
                assert name == "regretbench_factorized_v2_static_likelihood"
                responses.append(self._static(payload))
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
            "adapter_prompt_tokens": self.requests * 10,
            "adapter_completion_tokens": self.requests * 10,
        }


def test_factorized_v2_exact10_smoke_passes(tmp_path):
    adapter = _Adapter()
    output = tmp_path / "smoke"
    result = smoke.run_smoke(
        output_dir=output,
        adapter=adapter,
        daily_budget_status={"authorized": True},
    )
    assert result["status"] == "passed"
    assert result["gates"]["all_pass"] is True
    assert result["authorizes"] == (
        "separate_factorized_v2_policy_preregistration_only"
    )
    assert adapter.requests == 10
    assert adapter.schemas.count("regretbench_enriched_support") == 2
    assert adapter.schemas.count("regretbench_smc_enriched_transition") == 4
    assert adapter.schemas.count("regretbench_factorized_v2_static_likelihood") == 4
    assert adapter.seeds == [
        smoke.ROOT_SEED_START,
        smoke.ROOT_SEED_START + 1,
        smoke.TRANSITION_SEED_START,
        smoke.TRANSITION_SEED_START,
        smoke.TRANSITION_SEED_START + 1,
        smoke.TRANSITION_SEED_START + 1,
        smoke.STATIC_SEED_START,
        smoke.STATIC_SEED_START,
        smoke.STATIC_SEED_START + 1,
        smoke.STATIC_SEED_START + 1,
    ]
    privacy = json.loads((output / "private/PRIVACY.json").read_text())["audits"]
    assert len(privacy) == 10
    assert all(row["passed"] for row in privacy)
    replay = verify.verify_smoke(output)
    assert replay["status"] == "verified"
    assert replay["mismatches"] == []
    assert all(replay["checks"].values())


def test_exact_conditioning_rejects_unrepresented_answer():
    support = {
        "hypotheses": [
            {
                "interpretation": f"i{index}",
                "final_answer": f"a{index}",
                "probability": 0.125,
                "conditioning_reply": f"reply {index}",
                "predicted_replies": ["x", "y", "z", "w"],
            }
            for index in range(8)
        ],
        "questions": ["Q1?", "Q2?", "Q3?", "Q4?"],
    }
    try:
        smoke.condition_on_observed_reply(support, "absent")
    except ValueError as error:
        assert "no static child likelihood" in str(error)
    else:
        raise AssertionError("unrepresented observation was accepted")


def test_static_payload_excludes_answer_weights_and_lineage():
    cig = smoke.load_smoke_cigs()[0]
    child = {
        "hypotheses": [
            {
                "interpretation": f"interpretation {index}",
                "final_answer": f"final {index}",
                "probability": 0.125,
                "parent_index": index,
                "revision_type": "retained" if index < 2 else "revised",
                "predicted_replies": ["old"] * 4,
            }
            for index in range(8)
        ],
        "questions": ["One?", "Two?", "Three?", "Four?"],
    }
    messages, audit = smoke.static_messages_for(cig, child, "Conditioning?")
    payload = json.loads(messages[1]["content"])
    assert audit["passed"] is True
    assert payload["dialogue"] == []
    serialized = json.dumps(payload)
    for forbidden in (
        '"probability"',
        '"parent_index"',
        '"revision_type"',
        '"predicted_replies"',
        '"answer"',
    ):
        assert forbidden not in serialized


def test_independent_verifier_rejects_result_tamper(tmp_path):
    output = tmp_path / "smoke"
    smoke.run_smoke(
        output_dir=output,
        adapter=_Adapter(),
        daily_budget_status={"authorized": True},
    )
    result_path = output / "RESULT.json"
    result = json.loads(result_path.read_text())
    result["status"] = "mechanics_failed"
    result_path.write_text(json.dumps(result))
    replay = verify.verify_smoke(output)
    assert replay["status"] == "verification_failed"
    assert "$.status" in replay["mismatches"]


def test_independent_verifier_rejects_raw_static_tamper(tmp_path):
    output = tmp_path / "smoke"
    smoke.run_smoke(
        output_dir=output,
        adapter=_Adapter(),
        daily_budget_status={"authorized": True},
    )
    raw_path = output / "private/RAW_RESPONSES.json"
    raw = json.loads(raw_path.read_text())
    static = json.loads(raw["static"][0])
    static["particles"][0]["predicted_replies"] = ["tampered"] * 5
    raw["static"][0] = json.dumps(static)
    raw_path.write_text(json.dumps(raw))
    replay = verify.verify_smoke(output)
    assert replay["status"] == "verification_failed"
    assert replay["mismatches"]
