from copy import deepcopy

import pytest

from scripts.scilaws_measurement_design import build, geometry, radical_inverse


def fixture(dim=1):
    names = ["x", "y", "z"][:dim]
    return (
        dict(
            task_id="test",
            inputs=[dict(name=n, public_train_range=[1, 1000]) for n in names],
        ),
        dict(
            task_id="test",
            status="source_contract_valid",
            used_inputs=names,
            bounds={n: [2, 800] for n in names},
            fetch_budget_rows=2000,
        ),
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_fixed_full_geometry(dim):
    m, c = fixture(dim)
    result = geometry(m, c)
    assert len(result["action_points"]) == 8
    assert len(result["target_points"]) == 64
    assert sum(result["target_weights"]) == 1
    assert len(result["initial_points"]) == 1 + 2 * dim
    assert result["total_rows_per_arm"] == 2 * (1 + 2 * dim) + 4
    for key in ("action_points", "target_points", "initial_points"):
        for p in result[key]:
            assert all(2 <= x <= 800 for x in p.values())
    assert all(a["transform"] == "log" for a in result["axes"])
    changed = deepcopy(c)
    changed.update(noise_scale=999, hidden_formula="ignored", endpoint_loss=-999)
    assert geometry(m, changed) == result


def test_linear_zero_bound_and_known_halton():
    m, c = fixture()
    m["inputs"][0]["public_train_range"] = [0, 1000]
    c["bounds"]["x"] = [0, 800]
    r = geometry(m, c)
    assert r["axes"][0]["transform"] == "linear"
    assert r["action_points"][0]["x"] == 400
    assert [radical_inverse(i, 2) for i in (1, 2, 3)] == [0.5, 0.25, 0.75]


def test_empty_intersection_and_budget_fail():
    m, c = fixture()
    c["bounds"]["x"] = [1001, 2000]
    with pytest.raises(ValueError, match="intersection"):
        geometry(m, c)
    m, c = fixture()
    c["fetch_budget_rows"] = 1
    with pytest.raises(ValueError, match="budget"):
        geometry(m, c)


def test_bound_panel_replay():
    prefix = "results/nonmyopic/"
    r = build(
        prefix + "SCILAWS_PUBLIC_METADATA_20260908.json",
        prefix + "SCILAWS_DEVELOPMENT_STATE_CONTRACT_20260908.json",
    )
    assert len(r["tasks"]) == 8
    assert not r["paid_authorization"]
    assert not r["policy_endpoint_authorization"]
    assert r["measurements_generated"] == 0
