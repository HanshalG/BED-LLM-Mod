from __future__ import annotations

import json

import pytest

from scripts import regretbench_deepseek_support_recovery as run
from scripts import regretbench_deepseek_result_verify as verify


class _FakeAdapter:
    def __init__(self, truth_answers: dict[str, str]) -> None:
        self.truth_answers = truth_answers
        self.requests = 0
        self.calls: list[tuple[dict, int]] = []
        cigs = [*run.load_stage_cigs("smoke"), *run.load_stage_cigs("development")]
        self.first_facets = {
            cig.cig_id: cig.semantic_facets[0].replace("_", " ") for cig in cigs
        }

    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages,
        seeds,
        **kwargs,
    ):
        assert kwargs["temperature"] == run.TEMPERATURE
        assert kwargs["max_new_tokens"] == run.MAX_TOKENS
        assert kwargs["response_format"] == run.support_response_format()
        responses = []
        for messages, seed in zip(batch_messages, seeds, strict=True):
            payload = json.loads(messages[-1]["content"])
            self.calls.append((payload, seed))
            self.requests += 1
            task_id = payload["task_id"]
            conditioned = bool(payload["dialogue"])
            answers = [f"zzfixturedecoy000{index}" for index in range(8)]
            if conditioned:
                answers[0] = self.truth_answers[task_id]
            responses.append(
                json.dumps(
                    {
                        "hypotheses": [
                            {
                                "interpretation": f"distinct interpretation {index}",
                                "final_answer": answer,
                                "prior_weight": index + 1,
                            }
                            for index, answer in enumerate(answers)
                        ],
                        "questions": [
                            f"Which {self.first_facets[task_id]} do you mean?",
                            "Could you clarify the relevant context?",
                            "Are you referring to a person or an object?",
                            "Which version is relevant?",
                        ],
                    }
                )
            )
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
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
        }


def _truth_answers(stage: str) -> dict[str, str]:
    result = {}
    protocol = run.STAGES[stage]
    for index, cig in enumerate(run.load_stage_cigs(stage)):
        _, truth = run.sample_truth(cig, protocol["truth_seed_start"] + index)
        result[cig.cig_id] = str((truth.slots or {})["answer_aliases"]).split("|")[0]
    return result


def test_source_bindings_and_frozen_splits_are_exact() -> None:
    manifest = run.validate_source_bindings()

    assert manifest["splits"]["mechanics"]["ids"] == [
        "ambigdocs_test_880",
        "ambigdocs_test_39692",
        "ambigdocs_test_40148",
        "ambigdocs_test_46637",
    ]
    assert len(manifest["splits"]["development"]["ids"]) == 64
    assert run.STAGES["smoke"]["expected_requests"] == 10
    assert run.STAGES["development"]["expected_requests"] == 192


def test_parser_is_strict_and_deduplicates_without_truth() -> None:
    raw = json.dumps(
        {
            "hypotheses": [
                {
                    "interpretation": f"meaning {index}",
                    "final_answer": f"answer {index}",
                    "prior_weight": index + 1,
                }
                for index in range(8)
            ],
            "questions": [
                "Which person do you mean?",
                "Which place do you mean?",
                "Which version do you mean?",
                "Which date do you mean?",
            ],
        }
    )

    support = run.parse_support(raw)

    assert support["diagnostic"] == {
        "codec_mode": "strict_json",
        "raw_hypothesis_count": 8,
        "valid_unique_count": 8,
        "question_count": 4,
    }
    assert sum(row["probability"] for row in support["hypotheses"]) == pytest.approx(1.0)

    malformed = json.loads(raw)
    malformed["questions"][0] = "Not a question"
    with pytest.raises(ValueError, match="not a question"):
        run.parse_support(json.dumps(malformed))


def test_lexical_match_is_conservative() -> None:
    assert run.lexical_alias_match("Los Lagos Region of Chile.", "Los Lagos Region of Chile")
    assert run.lexical_alias_match(
        "It is in Chubut Province in the Patagonian region of Argentina",
        "Chubut Province in the Patagonian region of Argentina",
    )
    assert not run.lexical_alias_match("Argentina", "in argentina")
    assert not run.lexical_alias_match("artist", "artist and explorer")
    assert not run.lexical_alias_match("binary 1", "binary 0")
    assert not run.lexical_alias_match("New York", "Newark X")


def test_payload_contract_rejects_extra_hidden_fields() -> None:
    cig = run.load_stage_cigs("smoke")[0]
    payload = run.public_payload(cig, [])

    assert run.privacy_audit(cig, payload)["passed"] is True
    with pytest.raises(ValueError, match="forbidden fields"):
        run.privacy_audit(cig, {**payload, "intents": []})


def test_exact_ten_smoke_full_path_has_no_efficacy_gate(tmp_path) -> None:
    answers = _truth_answers("smoke")
    adapter = _FakeAdapter(answers)

    result = run.run_stage(
        stage="smoke",
        output_dir=tmp_path / "smoke",
        run_id="fixture-smoke",
        adapter=adapter,
    )

    assert result["status"] == "passed"
    assert result["authorizes"] == "development_support_recovery_only"
    assert result["science"] is None
    assert result["protocol"]["support_recovery_endpoint_accessed"] is False
    assert all(result["mechanics_gates"].values())
    assert adapter.requests == 10
    assert len(adapter.calls[:4]) == 4
    for index in range(3):
        conditioned = adapter.calls[4 + 2 * index]
        blind = adapter.calls[5 + 2 * index]
        assert conditioned[1] == blind[1]
        assert conditioned[0]["dialogue"]
        assert blind[0]["dialogue"] == []
    serialized = (tmp_path / "smoke" / "RESULT.json").read_text()
    assert all(answer not in serialized for answer in answers.values())
    replay = verify.verify_support(tmp_path / "smoke", stage="smoke")
    assert replay["status"] == "verified"
    assert replay["model_calls"] == 0

    result_path = tmp_path / "smoke" / "RESULT.json"
    tampered = json.loads(result_path.read_text())
    tampered["tasks"][0]["root_covered"] = not tampered["tasks"][0][
        "root_covered"
    ]
    result_path.write_text(json.dumps(tampered))
    failed = verify.verify_support(tmp_path / "smoke", stage="smoke")
    assert failed["status"] == "verification_failed"
    assert "$.tasks[0].root_covered" in failed["mismatches"]


def test_full_development_path_passes_matched_recovery_gate(tmp_path) -> None:
    answers = _truth_answers("development")
    adapter = _FakeAdapter(answers)
    smoke = run.run_stage(
        stage="smoke",
        output_dir=tmp_path / "smoke",
        run_id="fixture-smoke-predecessor",
        adapter=_FakeAdapter(_truth_answers("smoke")),
    )
    assert smoke["status"] == "passed"

    result = run.run_stage(
        stage="development",
        output_dir=tmp_path / "development",
        run_id="fixture-development",
        adapter=adapter,
        bootstrap_samples=500,
        smoke_result_path=tmp_path / "smoke" / "RESULT.json",
    )

    assert result["status"] == "passed"
    assert result["authorizes"] == "separately_preregistered_development_policy_only"
    assert result["protocol"]["support_recovery_endpoint_accessed"] is True
    assert result["protocol"]["policy_endpoint_opened"] is False
    assert result["protocol"]["smoke_predecessor"]["sha256"] == run.sha256_file(
        tmp_path / "smoke" / "RESULT.json"
    )
    assert all(result["mechanics_gates"].values())
    assert all(result["science"]["gates"].values())
    assert result["science"]["population"] == {
        "all_tasks": 64,
        "supported_tasks": 64,
        "root_missing_supported_tasks": 64,
    }
    assert result["science"]["coverage"]["conditioned"] == 1.0
    assert result["science"]["coverage"]["history_blind"] == 0.0
    assert adapter.requests == 192
    assert len(result["tasks"]) == 64
    assert "question" not in result["tasks"][0]
    assert "aliases" not in result["tasks"][0]
    for index in range(64):
        conditioned = adapter.calls[64 + 2 * index]
        blind = adapter.calls[65 + 2 * index]
        assert conditioned[1] == blind[1]
        assert conditioned[0]["task_id"] == blind[0]["task_id"]
        assert conditioned[0]["dialogue"]
        assert blind[0]["dialogue"] == []
    replay = verify.verify_support(tmp_path / "development", stage="development")
    assert replay["status"] == "verified"


def test_scientific_gate_fails_equal_matched_arms() -> None:
    rows = [
        {
            "supported": True,
            "root_covered": False,
            "conditioned_covered": bool(index % 2),
            "blind_covered": bool(index % 2),
        }
        for index in range(64)
    ]

    summary = run.scientific_summary(rows, samples=100)

    assert summary["gates"]["all_pass"] is False
    assert summary["coverage"]["conditioned_minus_history_blind"]["mean"] == 0.0
    assert summary["paired_outcomes"]["conditioned_recoveries"] == 0


def test_development_requires_passing_smoke(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires a passing smoke"):
        run.run_stage(
            stage="development",
            output_dir=tmp_path / "development",
            run_id="missing-predecessor",
            adapter=_FakeAdapter(_truth_answers("development")),
            bootstrap_samples=10,
        )
