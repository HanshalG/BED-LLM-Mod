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
