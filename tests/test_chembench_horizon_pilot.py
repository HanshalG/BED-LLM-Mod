from types import SimpleNamespace
import numpy as np

from environments.chembench_mopen.envelope_belief import EnvelopeGaussianModel
from scripts import chembench_horizon_pilot as pilot


def wire(monkeypatch):
    model = EnvelopeGaussianModel(
        [[-0.2] * 4, [0.3] * 4],
        0.15,
        [[0, 0.2, 0.4], [0.1, 0.3, 0.5]],
        [0.5, 0.5],
        branch_count=64,
    )
    monkeypatch.setattr(
        pilot, "preflight", lambda *args: {"status": "public_preflight_passed"}
    )
    monkeypatch.setattr(pilot, "load_source", lambda *args: None)
    monkeypatch.setattr(
        pilot,
        "build_public_pilot",
        lambda *args, **kwargs: SimpleNamespace(model=model),
    )
    calls = []

    def hidden(*args):
        calls.append("hidden")
        return (
            [("family", {})] * 8,
            np.full((8, 4), 0.1),
            np.full((8, 3), 0.2),
            np.zeros((8, 3, 4)),
        )

    monkeypatch.setattr(pilot, "build_hidden_worlds", hidden)

    def decide(model, state, available, remaining, arm, config, seconds):
        return {
            "action": available[0],
            "root_values": [(a, 0.1) for a in available],
            "effective_horizon": min(3, remaining),
        }

    monkeypatch.setattr(pilot, "decide", decide)
    return calls


def test_root_failure_keeps_hidden_boundary_closed(monkeypatch, tmp_path):
    calls = wire(monkeypatch)
    original = pilot.decide

    def fail(*args):
        if args[4] == "h3":
            raise TimeoutError("injected root cap")
        return original(*args)

    monkeypatch.setattr(pilot, "decide", fail)
    result = pilot.run(tmp_path, tmp_path)
    assert result["status"] == "execution_failed"
    assert result["phase"] == "public_root_planning"
    assert not result["hidden_worlds_opened"]
    assert calls == []
    assert not (tmp_path / "hidden_binding.json").exists()


def test_all_eight_worlds_and_controls_complete_with_paired_data(monkeypatch, tmp_path):
    calls = wire(monkeypatch)
    result = pilot.run(tmp_path, tmp_path)
    assert result["status"] == "engineering_null"
    assert calls == ["hidden"]
    assert len(result["records"]) == 8
    for row in result["records"]:
        assert len(row["core"]) == 6 and len(row["population_oracle"]) == 3
        for arm in row["core"] + row["population_oracle"]:
            assert len(arm["history"]) == 3
            assert len({r["choice"]["action"] for r in arm["history"]}) == 3
            assert all(r["observation"] == 0.1 for r in arm["history"])
    assert result["paid_calls_authorized"] is False


def test_stop_at_first_incomplete_world(monkeypatch, tmp_path):
    wire(monkeypatch)
    original = pilot.episode

    def fail(*args):
        if args[5] == "h3":
            raise TimeoutError("injected world cap")
        return original(*args)

    monkeypatch.setattr(pilot, "episode", fail)
    result = pilot.run(tmp_path, tmp_path)
    assert result["completed_worlds"] == 0
    assert result["phase"] == "world_0"
    assert (tmp_path / "world0_h2.json").exists()
    assert not (tmp_path / "world1_h1.json").exists()
