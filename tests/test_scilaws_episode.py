import json
from pathlib import Path
import platform
import sqlite3
import sys

import pytest

from environments.scilaws.episode import BoundedPointBackend, run_episode, strict_json
from environments.scilaws.point_measurements import MeasurementError, PointMeasurements

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="real policy isolation"
)
PYTHON = str(Path(sys._base_executable).resolve())


def setup(tmp_path, backend_code=None):
    public = tmp_path / "public"
    private = tmp_path / "private"
    public.mkdir()
    private.mkdir()
    (private / "coefficient.txt").write_text("3")
    code = (
        backend_code
        or "import json,sys; q=json.load(sys.stdin); a=float(open('coefficient.txt').read()); print(json.dumps(dict(n_returned=1,n_clipped=0,rows=[dict(x=q['x'][0],y=a*q['x'][0])],private_marker='HIDDEN')))"
    )
    backend = BoundedPointBackend([PYTHON, "-I", "-c", code], cwd=private, timeout=0.5)
    measurements = PointMeasurements(
        backend,
        database=private / "ledger.sqlite",
        bounds={"x": [0, 1]},
        target="y",
        budget=2,
        pairing_key=b"s" * 32,
        world_id="fixture",
        episode_id="paired",
        arm_id="h1",
        runtime_binding="fixture-only",
    )
    return dict(
        public_root=public,
        public_task={"context": "synthetic linear fixture"},
        target_points=[{"x": 0.75}],
        measurements=measurements,
        rounds=2,
        replicates=1,
        journal_path=private / "episode.jsonl",
    )


POLICY = """import json,sys
p=json.load(sys.stdin)
assert set(p)=={'task','target_points','rounds','replicates','phase','round','history'}
assert 'HIDDEN' not in json.dumps(p)
if p['phase']=='measure':
 print(json.dumps({'point':{'x':(.25,.5)[p['round']]}}))
else:
 a=p['history'][-1]['observations'][0]/p['history'][-1]['point']['x']
 print(json.dumps({'predictions':[a*t['x'] for t in p['target_points']]}))
"""


def test_complete_isolated_episode_and_no_alternate_journal_replay(tmp_path):
    kwargs = setup(tmp_path)
    result = run_episode([PYTHON, "-I", "-c", POLICY], **kwargs)
    assert result["status"] == "episode_complete"
    assert result["predictions"] == [2.25]
    assert [h["observations"] for h in result["history"]] == [[0.75], [1.5]]
    rows = [json.loads(x) for x in kwargs["journal_path"].read_text().splitlines()]
    assert [x["status"] for x in rows] == [
        "started",
        "observed",
        "observed",
        "episode_complete",
    ]
    with pytest.raises(FileExistsError):
        run_episode([PYTHON, "-I", "-c", POLICY], **kwargs)
    kwargs["journal_path"] = kwargs["journal_path"].with_name("alternate.jsonl")
    with pytest.raises(MeasurementError, match="claimed"):
        run_episode([PYTHON, "-I", "-c", POLICY], **kwargs)


def test_evaluator_timeout_charges_and_halts(tmp_path):
    kwargs = setup(tmp_path, "import time; time.sleep(10)")
    with pytest.raises(MeasurementError, match="halted"):
        run_episode([PYTHON, "-I", "-c", POLICY], **kwargs)
    with sqlite3.connect(tmp_path / "private/ledger.sqlite") as db:
        assert db.execute("SELECT exposure,status FROM attempts").fetchone() == (
            1,
            "failed",
        )
    assert (
        json.loads(kwargs["journal_path"].read_text().splitlines()[-1])["status"]
        == "failed_closed"
    )


@pytest.mark.parametrize(
    "reply",
    [
        '{"point":{"x":0.5},"seed":3}',
        '{"point":{"x":0.2},"point":{"x":0.5}}',
        '{"point":{"x":NaN}}',
    ],
)
def test_invalid_policy_never_measures(tmp_path, reply):
    kwargs = setup(tmp_path)
    with pytest.raises(ValueError):
        run_episode([PYTHON, "-I", "-c", f"print({reply!r})"], **kwargs)
    with sqlite3.connect(tmp_path / "private/ledger.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone() == (0,)


def test_wrong_final_vector_is_failed_episode(tmp_path):
    kwargs = setup(tmp_path)
    wrong = POLICY.replace("[a*t['x'] for t in p['target_points']]", "[]")
    with pytest.raises(ValueError, match="fixed-length"):
        run_episode([PYTHON, "-I", "-c", wrong], **kwargs)
    assert (
        json.loads(kwargs["journal_path"].read_text().splitlines()[-1])["status"]
        == "failed_closed"
    )


def test_strict_parser():
    for raw in ['{"x":1,"x":2}', '{"x":Infinity}', '{"x":NaN}']:
        with pytest.raises(ValueError):
            strict_json(raw)
