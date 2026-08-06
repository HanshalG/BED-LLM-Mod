from __future__ import annotations

import json

import pytest

from scripts import bongard_openworld_luna_confirmation64_freeze as freeze


def test_confirmation_block_economics_fit_daily_cap() -> None:
    assert freeze.MAX_REQUESTS_PER_BLOCK == 688
    assert freeze.MAX_PRECHARGED_EXPOSURE_PER_BLOCK_USD == pytest.approx(2.752)
    assert freeze.MAX_PRECHARGED_EXPOSURE_PER_BLOCK_USD < freeze.DAILY_CAP_USD


def test_confirmation_rows_are_fixed_opaque_and_disjoint() -> None:
    rows = freeze._confirmation_rows()
    assert len(rows) == len({row["task_id"] for row in rows}) == 64
    assert {row["block_id"] for row in rows} == set(freeze.BLOCK_ORDER)
    assert all(
        set(row) == {"task_id", "source_row_sha256", "block_id"}
        for row in rows
    )
    assert all("uid" not in json.dumps(row).casefold() for row in rows)


def test_science_gate_families_bind_policy_and_matched_mechanism() -> None:
    gates = freeze._science_gates()
    assert set(gates) == {"shared", "policy", "matched_mechanism"}
    assert any("myopic" in gate for gate in gates["policy"])
    assert any("history_blind" in gate for gate in gates["matched_mechanism"])
    assert all("bootstrap_95pct_upper_below_zero" in gate for gate in (
        gates["policy"][6],
        gates["matched_mechanism"][3],
    ))


def test_build_manifest_is_fail_closed_and_truth_free(tmp_path) -> None:
    output = tmp_path / "MANIFEST.json"
    result = freeze.build_manifest(output_path=output)
    assert result["status"] == "frozen"
    assert result["gates"]["all_pass"] is True
    assert result["development_precondition"]["required_claim_tier"] == (
        "full_llm_native_development_signal"
    )
    assert result["development_precondition"][
        "authorization_amendment_sha256"
    ] == freeze.AUTHORIZATION_AMENDMENT_SHA256
    rendered = json.dumps(result["tasks"], sort_keys=True).casefold()
    for forbidden in ("concept", "caption", "imagefiles", "positive", "negative"):
        assert forbidden not in rendered
    with pytest.raises(FileExistsError):
        freeze.build_manifest(output_path=output)
