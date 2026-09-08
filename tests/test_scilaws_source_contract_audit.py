from pathlib import Path

import pytest

from scripts.scilaws_source_contract_audit import audit, fixture_bundle, source_bytes

REPO = Path(__file__).resolve().parents[1] / "external/SciLaws-source-audit"


@pytest.fixture
def source():
    if not (REPO / ".git").exists():
        pytest.skip("pinned external source audit checkout is not installed")
    return source_bytes(REPO)


def test_full_runtime_contract(source):
    report = audit(REPO)
    assert report["benchmark_outcomes_opened"] == 0
    assert not report["paid_authorization"]
    for case in report["fixtures"].values():
        assert case["same_seed_same_point_replays"]
        assert case["clipped_coordinate"] == 1
        assert case["clipped_count"] == 1
        assert case["explicit_point_rejected_after_budget"]
        assert case["target_filtered_values"] == [11.0]
    assert report["fixtures"]["I"]["where_rows_after_budget"] == 0
    assert report["fixtures"]["II"]["where_rows_after_budget"] == 1
    assert report["fixtures"]["II"]["used_after_where"] == 5


def test_fixture_does_not_deserialize_state_or_read_sample(source, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("filesystem opened")

    monkeypatch.setattr(Path, "open", forbidden)
    for kind in ("I", "II"):
        extra = {} if kind == "I" else {"group_id": 0}
        sim = fixture_bundle(source, kind)
        assert sim.fetch_data(x=[0.5], seed=3, **extra)["n_returned"] == 1


def test_missing_definitions_fail():
    with pytest.raises(ValueError, match="definitions"):
        fixture_bundle("", "I")
