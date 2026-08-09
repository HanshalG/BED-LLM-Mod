from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts import bongard_openworld_aug10_final_handoff as final_handoff
from scripts import bongard_openworld_aug10_postprocess as postprocess
from scripts import bongard_openworld_development_final_handoff as development_final
from scripts import bongard_openworld_paper_with_classical_suite as paper_wrapper


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
RANDOM_PAPER_HANDOFF_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_RANDOM_STRATEGY_PAPER_HANDOFF_AMENDMENT_20260809.md"
)
RANDOM_PAPER_HANDOFF_AMENDMENT_SHA256 = (
    "56d310e8e19a22e9613f57618c6bcaf8ebdc6c1862d25dd4c9b49ad5d3b70961"
)
MEDIATION_PAPER_HANDOFF_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_PATH_MEDIATION_PAPER_HANDOFF_AMENDMENT_20260809.md"
)
MEDIATION_PAPER_HANDOFF_AMENDMENT_SHA256 = (
    "1902eff1a655bb1a8456e9c9d14e3d1bdf7adbcf76865686bef1263b1188d36b"
)
FINAL_HANDOFF_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_AUG10_FINAL_HANDOFF_PROTOCOL_20260809.md"
)
FINAL_HANDOFF_PROTOCOL_SHA256 = (
    "76858ea571f478572f8067fa85084e0ea54798017b452d30caca33d9a0cc7e5a"
)
FINAL_HANDOFF_IMPLEMENTATION = (
    REPO_ROOT / "scripts/bongard_openworld_aug10_final_handoff.py"
)
FINAL_HANDOFF_IMPLEMENTATION_SHA256 = (
    "016e535e6f8e53f80fd815baf387a5aba0de770bc40e7284e92aa67688596084"
)
ATOMIC_REHEARSAL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_AUG10_ATOMIC_HANDOFF_REHEARSAL_20260809.md"
)
ATOMIC_REHEARSAL_SHA256 = (
    "8fa393c0c91cc9baedc45f3dea781634b44cb3eb34586fc7b80836c7b0664d3a"
)
ATOMIC_REHEARSAL_TEST = (
    REPO_ROOT / "tests/test_bongard_openworld_aug10_atomic_rehearsal.py"
)
ATOMIC_REHEARSAL_TEST_SHA256 = (
    "04186fb837250f1fe9d203a5acfa369beb1dc4f6634acd78ca15856333a557ce"
)
DEVELOPMENT_DAILY_HANDOFF_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_DEVELOPMENT_DAILY_HANDOFF_PROTOCOL_20260809.md"
)
DEVELOPMENT_DAILY_HANDOFF_PROTOCOL_SHA256 = (
    "6453a42e866f025d461f72888d41b958aa0d00f6750e434f7200627580a3a2a4"
)
DEVELOPMENT_DAILY_HANDOFF_IMPLEMENTATION = (
    REPO_ROOT / "scripts/bongard_openworld_development_daily_handoff.py"
)
DEVELOPMENT_DAILY_HANDOFF_IMPLEMENTATION_SHA256 = (
    "0c94b16aaf12a4f509b051b49ecd256260b2b254c9c5b62b014baac7a6265027"
)
DEVELOPMENT_DAILY_HANDOFF_TEST = (
    REPO_ROOT / "tests/test_bongard_openworld_development_daily_handoff.py"
)
DEVELOPMENT_DAILY_HANDOFF_TEST_SHA256 = (
    "8c15a41212b45ec6e0f5b76d832de6acaaefb9f9bba37240c94462a6550e4504"
)
DETERMINISTIC_PAPER_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_DETERMINISTIC_PAPER_METADATA_AMENDMENT_20260809.md"
)
DETERMINISTIC_PAPER_AMENDMENT_SHA256 = (
    "b39ef5be597b3527d082469369b0a097b4ab476837d1997eb477dd6d9e84bbe0"
)
DEVELOPMENT_FINAL_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_DEVELOPMENT_FINAL_HANDOFF_PROTOCOL_20260809.md"
)
DEVELOPMENT_FINAL_PROTOCOL_SHA256 = (
    "474dc8efbf175abf59da8e3f6f12889405e2de135a6d92dcf024d1534eb2fbac"
)
DEVELOPMENT_FINAL_IMPLEMENTATION = (
    REPO_ROOT / "scripts/bongard_openworld_development_final_handoff.py"
)
DEVELOPMENT_FINAL_IMPLEMENTATION_SHA256 = (
    "77730d16f67f47ba07fbc1857f3d03bb658a5164b580108d0566606a90223a2c"
)
DEVELOPMENT_FINAL_TEST = (
    REPO_ROOT / "tests/test_bongard_openworld_development_final_handoff.py"
)
DEVELOPMENT_FINAL_TEST_SHA256 = (
    "923efe01c8f7bb82bd58bd7b66effcaa3cf83916a37599e2952c6442799d5534"
)
PAPER_WRAPPER = REPO_ROOT / "scripts/bongard_openworld_paper_with_classical_suite.py"
PAPER_WRAPPER_SHA256 = (
    "57fa47391f85cf66ec194a93998f2bf50a68da674709a84f58115f1b7bd5b151"
)
PAPER_WRAPPER_TEST = (
    REPO_ROOT / "tests/test_bongard_openworld_paper_with_classical_suite.py"
)
PAPER_WRAPPER_TEST_SHA256 = (
    "daf56b5d8b2b05819d598540404c8006fa4fd3d2be68a25caa389fb5e8644e2a"
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
    assert _sha256(RANDOM_PAPER_HANDOFF_AMENDMENT) == (
        RANDOM_PAPER_HANDOFF_AMENDMENT_SHA256
    )
    assert _sha256(MEDIATION_PAPER_HANDOFF_AMENDMENT) == (
        MEDIATION_PAPER_HANDOFF_AMENDMENT_SHA256
    )


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


def test_aug10_final_handoff_binds_immutable_paid_and_zero_call_components() -> None:
    assert _sha256(FINAL_HANDOFF_PROTOCOL) == FINAL_HANDOFF_PROTOCOL_SHA256
    assert _sha256(FINAL_HANDOFF_IMPLEMENTATION) == (
        FINAL_HANDOFF_IMPLEMENTATION_SHA256
    )
    assert _sha256(ATOMIC_REHEARSAL) == ATOMIC_REHEARSAL_SHA256
    assert _sha256(ATOMIC_REHEARSAL_TEST) == ATOMIC_REHEARSAL_TEST_SHA256
    assert _sha256(DEVELOPMENT_DAILY_HANDOFF_PROTOCOL) == (
        DEVELOPMENT_DAILY_HANDOFF_PROTOCOL_SHA256
    )
    assert _sha256(DEVELOPMENT_DAILY_HANDOFF_IMPLEMENTATION) == (
        DEVELOPMENT_DAILY_HANDOFF_IMPLEMENTATION_SHA256
    )
    assert _sha256(DEVELOPMENT_DAILY_HANDOFF_TEST) == (
        DEVELOPMENT_DAILY_HANDOFF_TEST_SHA256
    )
    assert final_handoff.PROTOCOL_SHA256 == FINAL_HANDOFF_PROTOCOL_SHA256
    assert final_handoff.BOUND_IMPLEMENTATIONS == {
        "aug10_wrapper": (
            "scripts/bongard_openworld_luna_aug10_execute.py",
            "adf0cede0c14e1ac96206461371f2f53f434f5b748327f9cf93ae0e7f521f9a5",
        ),
        "postprocess_v2": (
            "scripts/bongard_openworld_aug10_postprocess.py",
            "883707739185f91fc7d60fe12661896e3a62c690b406fc993e5ccbdeffd69ce0",
        ),
        "random_strategy_control": (
            "scripts/bongard_openworld_random_strategy_control.py",
            "f99b68adb9b0db9d066ac2aa36a11351330ff476e6df430361431d07191f7441",
        ),
    }
    assert final_handoff.verify_bindings()["implementations"]


def test_development_terminal_handoff_binds_deterministic_v6_paper_path() -> None:
    assert paper_wrapper.INTERFACE_VERSION == (
        "bongard-openworld-paper-with-classical-suite-6"
    )
    assert _sha256(DETERMINISTIC_PAPER_AMENDMENT) == (
        DETERMINISTIC_PAPER_AMENDMENT_SHA256
    )
    assert _sha256(DEVELOPMENT_FINAL_PROTOCOL) == DEVELOPMENT_FINAL_PROTOCOL_SHA256
    assert _sha256(DEVELOPMENT_FINAL_IMPLEMENTATION) == (
        DEVELOPMENT_FINAL_IMPLEMENTATION_SHA256
    )
    assert _sha256(DEVELOPMENT_FINAL_TEST) == DEVELOPMENT_FINAL_TEST_SHA256
    assert _sha256(PAPER_WRAPPER) == PAPER_WRAPPER_SHA256
    assert _sha256(PAPER_WRAPPER_TEST) == PAPER_WRAPPER_TEST_SHA256
    assert development_final.PROTOCOL_SHA256 == DEVELOPMENT_FINAL_PROTOCOL_SHA256
    assert development_final.BOUND_IMPLEMENTATIONS["paper_wrapper_v6"] == (
        "scripts/bongard_openworld_paper_with_classical_suite.py",
        PAPER_WRAPPER_SHA256,
    )
    assert development_final.verify_bindings()["implementations"]
