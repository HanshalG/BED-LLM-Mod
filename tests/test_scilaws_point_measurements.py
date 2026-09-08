import sqlite3
from types import SimpleNamespace

import pytest

from environments.scilaws.point_measurements import MeasurementError, PointMeasurements


def make(tmp_path, backend=None, *, arm="a", **changes):
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return dict(
            n_returned=1,
            n_clipped=0,
            rows=[dict(x=kwargs["x"][0], y=2.0)],
            secret="NOT_FOR_POLICY",
        )

    args = dict(
        database=tmp_path / f"{arm}.sqlite",
        bounds={"x": [0, 1]},
        target="y",
        budget=4,
        pairing_key=b"z" * 32,
        world_id="world",
        episode_id="paired-0",
        arm_id=arm,
        runtime_binding="synthetic-fixture",
    )
    args.update(changes)
    return PointMeasurements(
        backend or SimpleNamespace(fetch_data=fetch), **args
    ), calls


def request(rid="r0", x=0.5, n=1):
    return dict(request_id=rid, point={"x": x}, replicates=n)


def test_reservation_precedes_call_and_only_measurements_returned(tmp_path):
    def fetch(**kwargs):
        with sqlite3.connect(tmp_path / "a.sqlite") as db:
            assert db.execute("SELECT exposure,status FROM attempts").fetchone() == (
                2,
                "pending",
            )
        return dict(
            n_returned=1,
            n_clipped=0,
            rows=[dict(x=0.5, y=3.0, hidden="SECRET")],
            secret="SECRET",
        )

    m, _ = make(tmp_path, SimpleNamespace(fetch_data=fetch))
    out = m.query(request(n=2))
    assert out == dict(
        request_id="r0",
        round=0,
        point={"x": 0.5},
        observations=[3.0, 3.0],
        used=2,
        remaining=2,
    )


def test_idempotent_restart_without_simulator_replay(tmp_path):
    a, calls = make(tmp_path)
    out = a.query(request(n=2))
    assert len(calls) == 2
    restarted, after = make(tmp_path)
    assert restarted.query(request(n=2)) == out
    assert after == []
    with pytest.raises(MeasurementError, match="different"):
        restarted.query(request(x=0.6))
    assert restarted.query(request("r1", n=2))["remaining"] == 0
    with pytest.raises(MeasurementError, match="budget"):
        restarted.query(request("r2"))
    assert len(after) == 2


def test_crn_excludes_arm_and_point_but_separates_rounds_and_replicates(tmp_path):
    a, ca = make(tmp_path, arm="a")
    b, cb = make(tmp_path, arm="b")
    a.query(request(x=0.2, n=2))
    b.query(request(x=0.8, n=2))
    assert [x["seed"] for x in ca] == [x["seed"] for x in cb]
    assert ca[0]["seed"] != ca[1]["seed"]
    a.query(request("next", n=1))
    b.query(request("different-id", n=1))
    assert ca[-1]["seed"] == cb[-1]["seed"] != ca[0]["seed"]


@pytest.mark.parametrize(
    "change",
    [
        dict(seed=1),
        dict(where="y>0"),
        dict(replicates=True),
        dict(replicates=0),
        dict(point={"x": float("nan")}),
        dict(point={"x": float("inf")}),
        dict(point={"x": 1.1}),
        dict(point={"x": True}),
        dict(point={"other": 0.5}),
    ],
)
def test_invalid_requests_never_call_or_charge(tmp_path, change):
    m, calls = make(tmp_path)
    r = request()
    r.update(change)
    with pytest.raises(ValueError):
        m.query(r)
    assert calls == []
    assert m.query(request())["round"] == 0


@pytest.mark.parametrize(
    "bad",
    [
        dict(error="SECRET"),
        dict(n_returned=1, n_clipped=1, rows=[dict(x=0.5, y=2.0)]),
        dict(n_returned=True, n_clipped=0, rows=[dict(x=0.5, y=2.0)]),
        dict(n_returned=1, n_clipped=0, rows=[dict(x=0.6, y=2.0)]),
        dict(n_returned=1, n_clipped=0, rows=[dict(x=0.5, y=None)]),
    ],
)
def test_bad_response_charges_full_exposure_and_halts(tmp_path, bad):
    m, _ = make(tmp_path, SimpleNamespace(fetch_data=lambda **kw: bad))
    with pytest.raises(MeasurementError, match="session halted") as e:
        m.query(request(n=3))
    assert "SECRET" not in str(e.value)
    with sqlite3.connect(tmp_path / "a.sqlite") as db:
        assert db.execute("SELECT exposure,status FROM attempts").fetchone() == (
            3,
            "failed",
        )
    restarted, calls = make(tmp_path)
    with pytest.raises(MeasurementError):
        restarted.query(request("next"))
    with pytest.raises(MeasurementError):
        restarted.query(request(n=3))
    assert calls == []


def test_crash_leaves_pending_and_prevents_retry(tmp_path):
    def crash(**kwargs):
        raise KeyboardInterrupt()

    m, _ = make(tmp_path, SimpleNamespace(fetch_data=crash))
    with pytest.raises(KeyboardInterrupt):
        m.query(request(n=3))
    restarted, calls = make(tmp_path)
    with pytest.raises(MeasurementError):
        restarted.query(request("next"))
    assert calls == []
    with sqlite3.connect(tmp_path / "a.sqlite") as db:
        assert db.execute("SELECT exposure,status FROM attempts").fetchone() == (
            3,
            "pending",
        )


def test_reentrant_concurrent_query_cannot_overtake_pending(tmp_path):
    other, calls = make(tmp_path)

    def fetch(**kwargs):
        with pytest.raises(MeasurementError, match="halted"):
            other.query(request("concurrent"))
        return dict(n_returned=1, n_clipped=0, rows=[dict(x=0.5, y=2.0)])

    m, _ = make(tmp_path, SimpleNamespace(fetch_data=fetch))
    m.query(request())
    assert calls == []


@pytest.mark.parametrize(
    "change",
    [
        dict(budget=5),
        dict(pairing_key=b"y" * 32),
        dict(runtime_binding="other"),
        dict(episode_id="other"),
    ],
)
def test_ledger_cannot_rebind(tmp_path, change):
    make(tmp_path)
    with pytest.raises(MeasurementError, match="binding"):
        make(tmp_path, **change)
