from scripts import regretbench_factorized_v2_source_audit as audit


def test_fresh_source_audit_passes_and_is_disjoint(tmp_path):
    result = audit.run_audit(output_dir=tmp_path)
    assert result["status"] == "source_protocol_pass"
    assert result["authorizes"] == "factorized_v2_exact10_smoke_only"
    assert result["gates"]["all_pass"] is True
    assert result["counts"]["fresh_selected_tasks"] == 132
    assert result["hidden_truth_opened"] is False
    assert result["policy_endpoint_opened"] is False


def test_frozen_source_audit_replays_byte_identically(tmp_path):
    first = audit.run_audit(output_dir=tmp_path / "first")
    second = audit.run_audit(output_dir=tmp_path / "second")
    assert first == second
    assert (
        tmp_path / "first" / "SOURCE_PROTOCOL_MANIFEST.json"
    ).read_bytes() == (
        tmp_path / "second" / "SOURCE_PROTOCOL_MANIFEST.json"
    ).read_bytes()
