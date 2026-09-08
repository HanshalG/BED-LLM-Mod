from pathlib import Path

import pytest

from scripts.scilaws_source_contract_audit import audit, fixture_bundle, source_bytes
from environments.scilaws.point_measurements import PointMeasurements

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


def test_point_boundary_with_pinned_runtime_fixture(source, tmp_path):
    results = []
    for arm in ("h1", "h3"):
        sim = fixture_bundle(source, "I", budget=4)
        boundary = PointMeasurements(
            sim,
            database=tmp_path / f"{arm}.sqlite",
            bounds={"x": [0, 1]},
            target="y",
            budget=4,
            pairing_key=b"fixture-only-private-key" * 2,
            world_id="synthetic",
            episode_id="pair0",
            arm_id=arm,
            runtime_binding="pinned-scilaws-constant-fixture",
        )
        response = boundary.query(
            dict(request_id="first", point={"x": 0.5}, replicates=4)
        )
        assert response["remaining"] == 0
        assert sim.budget_status()["used"] == 4
        assert set(response["observations"]) <= {9.0, 11.0}
        assert (
            boundary.query(dict(request_id="first", point={"x": 0.5}, replicates=4))
            == response
        )
        assert sim.budget_status()["used"] == 4
        results.append(response)
    assert results[0] == results[1]
