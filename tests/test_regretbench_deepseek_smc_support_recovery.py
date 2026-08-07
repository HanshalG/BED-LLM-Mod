from __future__ import annotations

import json

import pytest

from scripts import regretbench_deepseek_smc_support_recovery as smc


def _parent_raw() -> str:
    return json.dumps(
        {
            "hypotheses": [
                {
                    "interpretation": f"parent interpretation {index}",
                    "final_answer": f"parent answer {index}",
                    "prior_weight": index + 1,
                }
                for index in range(8)
            ],
            "questions": [
                "Which person do you mean?",
                "Which place do you mean?",
                "Which period do you mean?",
                "Which version do you mean?",
            ],
        }
    )


def _child_payload(*, retained: int = 4) -> dict:
    hypotheses = []
    for index in range(8):
        keep = index < retained
        hypotheses.append(
            {
                "parent_index": index,
                "revision_type": "retained" if keep else "revised",
                "interpretation": (
                    f"parent interpretation {index}"
                    if keep
                    else f"revised interpretation {index}"
                ),
                "final_answer": (
                    f"parent answer {index}"
                    if keep
                    else f"revised answer {index}"
                ),
                "prior_weight": index + 1,
            }
        )
    return {
        "hypotheses": hypotheses,
        "questions": [
            "Which person do you mean?",
            "Which place do you mean?",
            "Which period do you mean?",
            "Which version do you mean?",
        ],
    }


def test_protocol_and_schema_are_frozen() -> None:
    smc.validate_protocol_binding()

    schema = smc.child_response_format()["json_schema"]["schema"]

    assert schema["properties"]["hypotheses"]["minItems"] == 8
    assert schema["properties"]["hypotheses"]["maxItems"] == 8
    assert schema["properties"]["hypotheses"]["items"]["properties"][
        "revision_type"
    ]["enum"] == ["retained", "revised"]
    assert smc.BRANCH_SEED_START == 202608270000
    assert smc.RUN_BUDGET_USD == 0.50


def test_parent_population_preserves_exact_raw_slots_and_provenance() -> None:
    raw = _parent_raw()

    parent = smc.parse_parent_population(raw)

    assert [row["parent_index"] for row in parent["particles"]] == list(range(8))
    assert sum(row["probability"] for row in parent["particles"]) == pytest.approx(
        1.0
    )
    assert parent["raw_parent_sha256"] == __import__("hashlib").sha256(
        raw.encode()
    ).hexdigest()
    assert all("prior_weight" not in row for row in parent["particles"])


def test_payload_contains_only_public_history_and_model_generated_parents() -> None:
    cig = smc.primary.load_stage_cigs("smoke")[0]
    parent = smc.parse_parent_population(_parent_raw())
    dialogue = [
        {"role": "assistant", "content": "Which person do you mean?"},
        {"role": "user", "content": "The composer."},
    ]

    messages, audit = smc.messages_for(cig, dialogue, parent)
    payload = json.loads(messages[-1]["content"])

    assert audit["passed"] is True
    assert payload["task_id"] == cig.cig_id
    assert payload["dialogue"] == dialogue
    assert payload["parent_particles"] == parent["particles"]
    assert payload["parent_source"] == "verified_primary_raw_root"
    assert payload["parent_population_sha256"] == parent["raw_parent_sha256"]
    assert set(payload) == {
        "task_id",
        "prompt",
        "dialogue",
        "parent_particles",
        "parent_population_sha256",
        "parent_source",
    }


def test_child_parser_enforces_real_retention_and_revision() -> None:
    parent = smc.parse_parent_population(_parent_raw())

    child = smc.parse_child_support(json.dumps(_child_payload()), parent)

    assert child["diagnostic"] == {
        "codec_mode": "strict_json",
        "valid_unique_count": 8,
        "parent_index_permutation_exact": True,
        "retained_count": 4,
        "revised_count": 4,
        "question_count": 4,
        "parent_population_sha256": parent["raw_parent_sha256"],
    }
    assert sum(row["probability"] for row in child["hypotheses"]) == pytest.approx(
        1.0
    )

    false_retention = _child_payload()
    false_retention["hypotheses"][0]["interpretation"] = "changed"
    with pytest.raises(ValueError, match="retained child differs"):
        smc.parse_child_support(json.dumps(false_retention), parent)

    false_revision = _child_payload()
    false_revision["hypotheses"][4]["interpretation"] = "parent interpretation 4"
    false_revision["hypotheses"][4]["final_answer"] = "parent answer 4"
    with pytest.raises(ValueError, match="revised child is unchanged"):
        smc.parse_child_support(json.dumps(false_revision), parent)


@pytest.mark.parametrize("retained", [0, 1, 7, 8])
def test_child_parser_rejects_cosmetic_or_absent_revision(retained: int) -> None:
    parent = smc.parse_parent_population(_parent_raw())

    with pytest.raises(ValueError, match="retain between two and six"):
        smc.parse_child_support(
            json.dumps(_child_payload(retained=retained)), parent
        )


def test_child_parser_rejects_duplicate_or_missing_parent_lineage() -> None:
    parent = smc.parse_parent_population(_parent_raw())
    payload = _child_payload()
    payload["hypotheses"][7]["parent_index"] = 6
    payload["hypotheses"][7]["interpretation"] = "different child"
    payload["hypotheses"][7]["final_answer"] = "different answer"

    with pytest.raises(ValueError, match="exact permutation"):
        smc.parse_child_support(json.dumps(payload), parent)


def test_primary_predecessor_requires_verified_scientific_null(tmp_path) -> None:
    result = {
        "interface_version": smc.primary.INTERFACE_VERSION,
        "status": "gated_null",
        "authorizes": "nothing",
        "protocol": {
            "stage": "development",
            "model": smc.MODEL_ID,
            "support_recovery_endpoint_accessed": True,
            "policy_endpoint_opened": False,
        },
        "mechanics_gates": {"all_pass": True},
    }
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    verification = {
        "status": "verified",
        "result_status": "gated_null",
        "mismatches": [],
        "model_calls": 0,
        "cost_usd": 0.0,
        "artifact_sha256": {"RESULT.json": smc.sha256_file(result_path)},
    }
    verification_path = tmp_path / "VERIFICATION.json"
    verification_path.write_text(json.dumps(verification), encoding="utf-8")

    authorization = smc.validate_primary_null_predecessor(
        result_path, verification_path
    )

    assert authorization["status"] == "authorized_primary_scientific_null"

    result["status"] = "passed"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="does not authorize"):
        smc.validate_primary_null_predecessor(result_path, verification_path)


def test_mechanics_requires_exact_paired_lineage_schedule_and_accounting() -> None:
    parent = smc.parse_parent_population(_parent_raw())
    conditioned = smc.parse_child_support(json.dumps(_child_payload()), parent)
    blind = smc.parse_child_support(json.dumps(_child_payload()), parent)
    rows = [
        {
            "task_index": index,
            "task_id": f"task-{index}",
            "refresh_seed": smc.branch_seed(index),
            "conditioned_dispatch_index": 2 * index,
            "blind_dispatch_index": 2 * index + 1,
            "parent_population_sha256": parent["raw_parent_sha256"],
            "supported": index < 60,
            "conditioned_support": conditioned,
            "blind_support": blind,
        }
        for index in range(64)
    ]
    usage = {
        "adapter_requests": 128,
        "http_attempts": 128,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.20,
    }
    privacy = [{"passed": True}] * 128

    gates = smc.mechanics_gates(rows=rows, privacy=privacy, usage=usage)

    assert gates["all_pass"] is True

    rows[4] = {**rows[4], "blind_dispatch_index": 99}
    failed = smc.mechanics_gates(rows=rows, privacy=privacy, usage=usage)
    assert failed[
        "conditioned_blind_seed_formula_and_adjacency_exact"
    ] is False
    assert failed["all_pass"] is False


def test_science_requires_recovery_and_root_coverage_retention() -> None:
    passing = []
    for index in range(64):
        root_covered = index >= 32
        passing.append(
            {
                "supported": True,
                "root_covered": root_covered,
                "conditioned_covered": True,
                "blind_covered": root_covered,
            }
        )

    summary = smc.scientific_summary(passing, samples=500)

    assert summary["gates"]["all_pass"] is True
    assert summary["coverage"]["conditioned_minus_history_blind"][
        "mean"
    ] == pytest.approx(0.5)
    assert summary["coverage"]["conditioned_root_covered_retention"] == 1.0

    losing = [dict(row) for row in passing]
    for index in range(4):
        losing[32 + index]["conditioned_covered"] = False
    failed = smc.scientific_summary(losing, samples=500)

    assert failed["gates"][
        "conditioned_root_covered_retention_at_least_090"
    ] is False
    assert failed["gates"][
        "conditioned_root_covered_losses_no_more_than_blind"
    ] is False
    assert failed["gates"]["all_pass"] is False
