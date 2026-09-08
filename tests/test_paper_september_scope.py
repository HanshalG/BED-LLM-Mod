import json
from pathlib import Path


def test_draft_distinguishes_horizon_and_calibration_estimands():
    text = " ".join(Path("paper/main.tex").read_text().split())
    assert (
        "root selection with greedy future queries, not full receding-horizon optimization"
        in text
    )
    assert "not full predictive calibration of regenerated beliefs" in text
    assert (
        "not fresh confirmation, an online-updater test, or a non-myopic efficacy result"
        in text
    )
    assert "finite symbolic controls do not establish" in text


def test_new_claims_are_bound_as_qualified_diagnostics():
    claims = {
        c["id"]: c
        for c in json.loads(Path("paper/claim_manifest.json").read_text())["claims"]
    }
    for name in (
        "september_initial_full_horizon_reference_null",
        "september_first_refresh_retrospective_diagnosis",
        "september_validation_weight_transfer_retrospective",
    ):
        claim = claims[name]
        assert len(claim["artifact"]["sha256"]) == 64
        assert "not" in claim["paper_scope"].lower()
        assert any(c["pointer"] == "/status" for c in claim["checks"])
