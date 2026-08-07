from __future__ import annotations

import json

from scripts import regretbench_llm_native_source_audit as audit


def test_published_source_audit_replays_exactly(tmp_path) -> None:
    result = audit.run_audit(output_dir=tmp_path)

    assert result["status"] == "source_protocol_pass"
    assert result["counts"]["available_test_files"] == 6_286
    assert result["counts"]["eligible_cigs"] == 2_419
    assert result["counts"]["missing_manifest_train_files"] == 21_252
    assert result["counts"]["selected_strict_positive_fixed_support_depth_gains"] == 0
    assert all(result["gates"].values())
    manifest = json.loads((tmp_path / "SOURCE_PROTOCOL_MANIFEST.json").read_text())
    assert manifest["splits"]["mechanics"]["ids_sha256"] == audit.EXPECTED_SPLIT_HASHES[
        "mechanics"
    ]
    assert manifest["splits"]["development"]["ids_sha256"] == audit.EXPECTED_SPLIT_HASHES[
        "development"
    ]
    assert manifest["splits"]["confirmation"]["ids_sha256"] == audit.EXPECTED_SPLIT_HASHES[
        "confirmation"
    ]


def test_eligibility_rejects_incomplete_facet_values() -> None:
    path = next(audit.TEST_ROOT.glob("ambigdocs_*.json"))
    cig = json.loads(path.read_text())
    assert isinstance(audit.eligible(cig), bool)
    if audit.eligible(cig):
        facet = audit._facets(cig)[0]
        cig["intents"][0]["slots"][facet] = ""
        assert audit.eligible(cig) is False
