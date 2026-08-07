from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import regretbench_deepseek_smc_dynamic_depth2_policy as policy


QUESTIONS = [
    "Which scope is intended?",
    "Which period is intended?",
    "Which category is intended?",
    "Which comparison is intended?",
]


def _parent_population():
    return {
        "particles": [
            {
                "parent_index": index,
                "interpretation": f"interpretation {index}",
                "final_answer": f"answer {index}",
                "probability": index + 1,
            }
            for index in range(8)
        ],
        "raw_parent_sha256": "a" * 64,
    }


def _annotation_raw() -> str:
    return json.dumps(
        {
            "particles": [
                {
                    "parent_index": index,
                    "predicted_replies": [
                        f"scope {index}",
                        f"period {index}",
                        f"category {index}",
                        f"comparison {index}",
                    ],
                }
                for index in range(8)
            ]
        }
    )


def _initial_support():
    return policy.parse_parent_annotation(
        _annotation_raw(), _parent_population(), QUESTIONS
    )


def _transition_raw(*, retained: int = 2) -> str:
    parent = _initial_support()
    hypotheses = []
    for index, row in enumerate(parent["hypotheses"]):
        keep = index < retained
        hypotheses.append(
            {
                "parent_index": index,
                "revision_type": "retained" if keep else "revised",
                "interpretation": row["interpretation"]
                if keep
                else row["interpretation"] + " revised",
                "final_answer": row["final_answer"],
                "prior_weight": index + 1,
                "predicted_replies": [
                    f"new scope {index}",
                    f"new period {index}",
                    f"new category {index}",
                    f"new comparison {index}",
                ],
            }
        )
    return json.dumps({"hypotheses": hypotheses, "questions": QUESTIONS})


def _cig():
    return SimpleNamespace(
        cig_id="task-1",
        prompt="What does the ambiguous prompt mean?",
        intents=[SimpleNamespace(description="hidden intent", slots={})],
        latent_variables=[],
        reference_questions=[],
    )


def test_protocol_binding_is_frozen() -> None:
    policy.validate_protocol_binding()


def test_annotation_schema_cannot_regenerate_parent_content() -> None:
    item = policy.annotation_response_format()["json_schema"]["schema"][
        "properties"
    ]["particles"]["items"]

    assert item["required"] == ["parent_index", "predicted_replies"]
    assert "interpretation" not in item["properties"]
    assert "final_answer" not in item["properties"]
    assert "prior_weight" not in item["properties"]


def test_parent_annotation_preserves_all_slots_and_weights() -> None:
    support = _initial_support()

    assert len(support["hypotheses"]) == 8
    assert [row["interpretation"] for row in support["hypotheses"]] == [
        f"interpretation {index}" for index in range(8)
    ]
    assert sum(row["probability"] for row in support["hypotheses"]) == pytest.approx(
        1.0
    )
    assert support["diagnostic"]["initial_hypotheses_regenerated"] is False
    assert support["diagnostic"]["initial_questions_regenerated"] is False


def test_parent_annotation_preserves_duplicate_raw_slots_as_particles() -> None:
    parents = _parent_population()
    parents["particles"][1]["interpretation"] = parents["particles"][0][
        "interpretation"
    ]
    parents["particles"][1]["final_answer"] = parents["particles"][0][
        "final_answer"
    ]

    support = policy.parse_parent_annotation(_annotation_raw(), parents, QUESTIONS)

    assert support["hypotheses"][0]["interpretation"] == support["hypotheses"][1][
        "interpretation"
    ]
    assert support["hypotheses"][0]["parent_index"] == 0
    assert support["hypotheses"][1]["parent_index"] == 1


def test_annotation_requires_exact_parent_permutation() -> None:
    value = json.loads(_annotation_raw())
    value["particles"][-1]["parent_index"] = 0

    with pytest.raises(ValueError, match="exact permutation"):
        policy.parse_parent_annotation(
            json.dumps(value), _parent_population(), QUESTIONS
        )


def test_transition_has_strict_lineage_and_reply_vectors() -> None:
    parent = _initial_support()
    child = policy.parse_enriched_transition(_transition_raw(), parent)

    assert len(child["hypotheses"]) == 8
    assert child["diagnostic"]["parent_index_permutation_exact"] is True
    assert child["diagnostic"]["retained_count"] == 2
    assert child["diagnostic"]["revised_count"] == 6
    assert child["diagnostic"]["parent_support_sha256"] == parent["support_sha256"]
    assert all(len(row["predicted_replies"]) == 4 for row in child["hypotheses"])


def test_transition_rejects_false_retention_and_unchanged_revision() -> None:
    parent = _initial_support()
    false_retained = json.loads(_transition_raw())
    false_retained["hypotheses"][0]["interpretation"] += " changed"
    with pytest.raises(ValueError, match="revision label"):
        policy.parse_enriched_transition(json.dumps(false_retained), parent)

    unchanged_revision = json.loads(_transition_raw())
    unchanged_revision["hypotheses"][2]["interpretation"] = parent["hypotheses"][2][
        "interpretation"
    ]
    with pytest.raises(ValueError, match="revision label"):
        policy.parse_enriched_transition(json.dumps(unchanged_revision), parent)


def test_transition_rejects_retention_outside_frozen_range() -> None:
    with pytest.raises(ValueError, match="between two and six"):
        policy.parse_enriched_transition(_transition_raw(retained=1), _initial_support())

    with pytest.raises(ValueError, match="between two and six"):
        policy.parse_enriched_transition(_transition_raw(retained=7), _initial_support())


def test_transition_payload_contains_only_public_history_and_parent_provenance() -> None:
    messages, audit = policy.transition_messages_for(
        _cig(),
        [
            {"role": "assistant", "content": "Which scope is intended?"},
            {"role": "user", "content": "The broad scope."},
        ],
        _initial_support(),
    )
    payload = json.loads(messages[1]["content"])

    assert audit["passed"] is True
    assert set(payload) == {
        "task_id",
        "prompt",
        "dialogue",
        "parent_particles",
        "parent_questions",
        "parent_support_sha256",
        "parent_source",
    }
    assert "hidden intent" not in messages[1]["content"]
    assert payload["parent_support_sha256"] == _initial_support()["support_sha256"]


def test_annotation_payload_exposes_only_banked_model_objects() -> None:
    messages, audit = policy.annotation_messages_for(
        _cig(), _parent_population(), QUESTIONS
    )
    payload = json.loads(messages[1]["content"])

    assert audit["passed"] is True
    assert payload["dialogue"] == []
    assert payload["parent_source"] == "verified_primary_raw_root"
    assert payload["parent_population_sha256"] == "a" * 64
    assert payload["questions"] == QUESTIONS
    assert "hidden intent" not in messages[1]["content"]


def test_exact_reply_updates_weights_and_unmodelled_reply_preserves_them() -> None:
    support = _initial_support()
    updated, represented = policy.posterior_parent_after_reply(
        support, 0, "scope 3"
    )

    assert represented is True
    assert updated["hypotheses"][3]["probability"] == pytest.approx(1.0)
    assert sum(row["probability"] for row in updated["hypotheses"]) == pytest.approx(
        1.0
    )

    unchanged, represented = policy.posterior_parent_after_reply(
        support, 0, "an unmodelled answer"
    )
    assert represented is False
    assert [row["probability"] for row in unchanged["hypotheses"]] == pytest.approx(
        [row["probability"] for row in support["hypotheses"]]
    )
    assert unchanged["diagnostic"]["unmodelled_reply_preserves_parent_weights"]


def test_only_verified_smc_support_pass_authorizes_policy(tmp_path) -> None:
    result_path = tmp_path / "RESULT.json"
    verification_path = tmp_path / "VERIFICATION.json"
    ledger_path = tmp_path / "LEDGER.json"
    daily_path = tmp_path / "DAILY_RESULT.json"
    result = {
        "interface_version": "regretbench-deepseek-smc-support-recovery-daily-1",
        "status": "passed",
        "authorizes": "separately_preregistered_smc_policy_only",
        "protocol": {
            "protocol_sha256": policy.smc_support.PROTOCOL_SHA256,
            "smc_policy_endpoint_opened": False,
            "primary_policy_endpoint_opened": False,
            "primary_confirmation_opened": False,
        },
        "mechanics_gates": {"all_pass": True},
        "science": {"gates": {"all_pass": True}},
    }
    result_path.write_text(json.dumps(result))
    verification = {
        "status": "verified",
        "result_status": "passed",
        "mismatches": [],
        "model_calls": 0,
        "cost_usd": 0.0,
        "artifact_sha256": {"RESULT.json": policy.sha256_file(result_path)},
    }
    verification_path.write_text(json.dumps(verification))
    ledger = {
        "date": "2026-08-09",
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
        "account_wide_usage_counts_against_cap": True,
        "stage": {"status": "passed"},
    }
    ledger_path.write_text(json.dumps(ledger))
    daily = {
        "status": "complete_reconciled",
        "development_status": "passed",
        "independent_replay_passed": True,
        "smc_policy_endpoint_opened": False,
        "primary_policy_endpoint_opened": False,
        "primary_confirmation_opened": False,
        "authorizes": "separately_preregistered_smc_policy_only",
        "result_sha256": policy.sha256_file(result_path),
        "verification_sha256": policy.sha256_file(verification_path),
        "ledger_sha256": policy.sha256_file(ledger_path),
    }
    daily_path.write_text(json.dumps(daily))

    authorization = policy.validate_smc_support_predecessor(
        result_path=result_path,
        verification_path=verification_path,
        daily_result_path=daily_path,
        ledger_path=ledger_path,
    )
    assert authorization["status"] == "authorized_smc_policy_development_only"

    result["status"] = "gated_null"
    result["authorizes"] = "nothing"
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="does not authorize"):
        policy.validate_smc_support_predecessor(
            result_path=result_path,
            verification_path=verification_path,
            daily_result_path=daily_path,
            ledger_path=ledger_path,
        )
