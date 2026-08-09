from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import bongard_openworld_aug10_postprocess as postprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ADDENDUM = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_AUG10_V2_READINESS_ADDENDUM_20260809.json"
)
HISTORICAL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_AUG10_FINAL_CURRENT_HEAD_READINESS_20260809.json"
)
CLOSEST_PRIOR_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_2026_CLOSEST_PRIOR_AMENDMENT_20260809.md"
)
CLOSEST_PRIOR_AMENDMENT_SHA256 = (
    "5a8f4f4c0cde5759641b4669a2d88d48fc64fd0d79c571dee15439205f073647"
)
PAPER_ONLY_BINDINGS = {"mandatory_paper_wrapper_v3"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_aug10_v2_readiness_binds_the_exact_current_handoff() -> None:
    report = _load(ADDENDUM)

    assert report["status"] == "ready_without_paid_calls"
    assert report["execution_date"] == "2026-08-10"
    assert postprocess.INTERFACE_VERSION == (
        "bongard-openworld-aug10-postprocess-2"
    )
    for name, record in report["bindings"].items():
        path = REPO_ROOT / record["path"]
        assert path.is_file()
        if name not in PAPER_ONLY_BINDINGS:
            assert _sha256(path) == record["sha256"]
    assert _sha256(CLOSEST_PRIOR_AMENDMENT) == CLOSEST_PRIOR_AMENDMENT_SHA256


def test_aug10_v2_readiness_supersedes_only_stale_downstream_fields() -> None:
    report = _load(ADDENDUM)
    historical = _load(HISTORICAL)
    supersession = report["historical_readiness"]

    assert _sha256(HISTORICAL) == supersession["json_sha256"]
    assert supersession["remains_immutable"] is True
    assert supersession["paid_chain_fields_remain_in_force"] is True
    assert supersession["superseded_fields"] == [
        "mechanics_analysis_bindings",
        "mechanics_terminal_handoff_instructions",
    ]
    old_hash = historical["mechanics_analysis_bindings"][
        "postprocess_implementation_sha256"
    ]
    new_hash = report["bindings"]["postprocess_v2"]["sha256"]
    assert old_hash != new_hash
    assert historical["production_bindings"]["aug10_wrapper_sha256"] == (
        report["bindings"]["paid_aug10_wrapper"]["sha256"]
    )


def test_aug10_v2_readiness_never_authorizes_paid_or_duplicate_work() -> None:
    report = _load(ADDENDUM)
    authorization = report["authorization"]
    preflight = report["preflight"]

    assert authorization["authorizes_early_execution"] is False
    assert authorization["requires_fresh_aug10_same_day_preflight"] is True
    assert authorization["postprocess_creates_development_authorization"] is False
    assert authorization["postprocess_authorizes_rerun"] is False
    assert authorization["mechanics_compute_audit_is_owned_by_postprocess_v2"] is True
    assert (
        authorization["separate_mechanics_compute_audit_after_success_is_forbidden"]
        is True
    )
    assert set(preflight["execution_paths"].values()) == {"absent"}
    assert preflight["model_calls_made"] == 0
    assert preflight["files_written"] == 0
    assert report["verification"]["paid_model_calls"] == 0
    assert report["verification"]["cost_usd"] == 0.0
