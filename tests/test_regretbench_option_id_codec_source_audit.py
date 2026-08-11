from __future__ import annotations

import json

from scripts import regretbench_option_id_codec_source_audit as audit


def test_option_id_source_audit_freezes_disjoint_four_stage_cohort(tmp_path):
    result = audit.run_audit(output_dir=tmp_path)
    assert result["status"] == "source_protocol_pass"
    assert result["authorizes"] == "option_id_codec_exact8_only"
    assert result["counts"]["prior_excluded_tasks"] == 264
    assert result["counts"]["fresh_selected_tasks"] == 132
    assert result["model_calls_made"] == 0
    assert result["source_values_opened"] is False

    manifest = json.loads((tmp_path / "SOURCE_PROTOCOL_MANIFEST.json").read_text())
    assert {name: row["size"] for name, row in manifest["splits"].items()} == {
        "codec_calibration": 2,
        "mechanics": 2,
        "development": 64,
        "confirmation": 64,
    }
    selected = [
        cig_id
        for split in manifest["splits"].values()
        for cig_id in split["ids"]
    ]
    prior, counts = audit.prior_ids()
    assert counts == {"original": 132, "factorized_v2": 132, "union": 264}
    assert len(selected) == len(set(selected)) == 132
    assert not (set(selected) & prior)

    public = json.loads((tmp_path / "CODEC_PUBLIC_TASKS.json").read_text())
    assert len(public["tasks"]) == 2
    assert set(public["tasks"][0]) == {
        "task_index",
        "task_id",
        "prompt",
        "task_file_sha256",
        "semantic_facets",
        "reference_questions",
    }
    assert public["source_values_included"] is False
    assert public["intent_descriptions_included"] is False
    assert public["reference_questions_included_for_code_only"] is True
    assert public["action_metadata_in_model_prompts"] is False
    assert public["endpoint_outcomes_included"] is False
