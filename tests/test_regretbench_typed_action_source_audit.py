from __future__ import annotations

import json

from scripts import regretbench_typed_action_source_audit as audit


def test_typed_action_source_audit_freezes_fresh_executable_cohort(tmp_path):
    result = audit.run_audit(output_dir=tmp_path)
    assert result["status"] == "source_protocol_pass"
    assert result["authorizes"] == "typed_action_exact8_only"
    assert result["counts"] == {
        "available_test_files": 6286,
        "eligible_cigs": 2419,
        "prior_excluded_tasks": 396,
        "fresh_typed_eligible_cigs": 891,
        "fresh_selected_tasks": 132,
    }
    assert result["model_calls_made"] == 0
    assert result["cost_usd"] == 0.0
    assert result["policy_source_values_opened"] is False
    assert result["policy_endpoint_opened"] is False

    manifest = json.loads((tmp_path / "SOURCE_PROTOCOL_MANIFEST.json").read_text())
    assert {name: row["size"] for name, row in manifest["splits"].items()} == {
        "typed_calibration": 2,
        "mechanics": 2,
        "development": 64,
        "confirmation": 64,
    }
    selected = [
        task_id
        for split in manifest["splits"].values()
        for task_id in split["ids"]
    ]
    prior, counts = audit.prior_ids()
    assert counts == {
        "original": 132,
        "factorized_v2": 132,
        "option_id": 132,
        "union": 396,
    }
    assert len(selected) == len(set(selected)) == 132
    assert not set(selected) & prior
    assert manifest["source_values_serialized_in_public_artifacts"] is False

    public = json.loads((tmp_path / "TYPED_PUBLIC_TASKS.json").read_text())
    assert len(public["tasks"]) == 2
    assert public["source_values_included"] is False
    assert public["intent_descriptions_included"] is False
    assert public["canonical_questions_included"] is True
    assert public["action_metadata_in_model_prompts"] is True
    assert public["endpoint_outcomes_included"] is False
    for task_index, task in enumerate(public["tasks"]):
        assert set(task) == {
            "task_index",
            "task_id",
            "prompt",
            "task_file_sha256",
            "actions",
        }
        assert task["task_index"] == task_index
        assert len(task["actions"]) == 4
        assert [row["public_order"] for row in task["actions"]] == list(range(4))
        assert len({row["action_id"] for row in task["actions"]}) == 4
        assert all(row["question"].endswith("?") for row in task["actions"])
        assert all(set(row) == {"action_id", "question", "public_order"} for row in task["actions"])


def test_executable_actions_uses_canonical_question_and_private_value_gate():
    cig = {
        "semantic_action_schema": {"ask_facets": ["kind", "place"]},
        "reference_questions": [
            {"semantic_action": "ask:kind", "text": "Which kind?"},
            {"semantic_action": "ask:kind", "text": "  What kind?  "},
            {"semantic_action": "ask:place", "text": "Which place?"},
        ],
        "intents": [
            {"slots": {"kind": "A", "place": "same"}},
            {"slots": {"kind": "B", "place": "same"}},
        ],
    }
    actions = audit.executable_actions(cig)
    assert actions == [
        {
            "action_id": "kind",
            "question": "What kind?",
            "public_order": 0,
            "private_distinct_value_count": 2,
        }
    ]


def test_every_frozen_public_action_replays_from_its_bound_cig():
    public = audit.load_object(audit.OUTPUT_DIR / "TYPED_PUBLIC_TASKS.json")
    for task in public["tasks"]:
        path = audit.original.TEST_ROOT / f"{task['task_id']}.json"
        assert audit.sha256_file(path) == task["task_file_sha256"]
        cig = audit.load_object(path)
        replay = [
            {
                "action_id": row["action_id"],
                "question": row["question"],
                "public_order": index,
            }
            for index, row in enumerate(audit.executable_actions(cig))
        ]
        assert replay == task["actions"]
