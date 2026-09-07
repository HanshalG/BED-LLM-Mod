from types import SimpleNamespace
import numpy as np
import pytest

from environments.chembench_mopen import pilot_data


def fake_source():
    config, _ = pilot_data.read_protocol()
    params = {
        family: {
            "easy": {
                version: {"kcat": float(i + 1), "Km": float(i + 2)}
                for i, version in enumerate(config["public_parameter_versions"])
            }
        }
        for family in config["families"]
    }
    fns = {
        family: lambda p, *x: p["kcat"] * x[0] / (p["Km"] + x[0])
        for family in config["families"]
    }
    return SimpleNamespace(_PARAMS=params, _RATE_FNS=fns)


def test_public_builder_never_requests_hidden_seed(monkeypatch):
    original = pilot_data.draw_parameters
    config, _ = pilot_data.read_protocol()
    calls = []

    def guarded(source, settings, count, seed):
        assert seed == config["prior_seed"]
        calls.append((count, seed))
        return original(source, settings, count, seed)

    monkeypatch.setattr(pilot_data, "draw_parameters", guarded)
    monkeypatch.setattr(
        pilot_data,
        "build_hidden_worlds",
        lambda *args: pytest.fail("hidden boundary opened"),
    )
    public = pilot_data.build_public_pilot(fake_source())
    assert len(calls) == 1
    assert public.model.num_particles == 16
    assert public.model.num_actions == 4
    assert public.model.targets.shape == (16, 64)
    assert not public.target_inputs.flags.writeable
    assert (
        len(
            {tuple(sorted(params.items())) for _, params in public.candidate_parameters}
        )
        == 16
    )


def test_targets_and_prior_are_reproducible_and_independent_of_truth_seed():
    source = fake_source()
    config, _ = pilot_data.read_protocol()
    a = pilot_data.draw_parameters(source, config, 4, config["prior_seed"])
    changed = dict(config, world_seed=0)
    assert a == pilot_data.draw_parameters(source, changed, 4, config["prior_seed"])
    np.testing.assert_array_equal(
        pilot_data.target_inputs(config), pilot_data.target_inputs(changed)
    )


def test_frozen_protocol_rejects_mutation(tmp_path, monkeypatch):
    path = tmp_path / "protocol.json"
    path.write_text("{}")
    monkeypatch.setattr(pilot_data, "PROTOCOL_PATH", path)
    with pytest.raises(ValueError, match="frozen"):
        pilot_data.read_protocol()


def test_invalid_rate_rejected():
    source = fake_source()
    config, _ = pilot_data.read_protocol()
    family = config["families"][0]
    source._RATE_FNS[family] = lambda *args: -1
    with pytest.raises(ValueError, match="rates"):
        pilot_data.build_public_pilot(source)
