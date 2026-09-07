from types import SimpleNamespace
import sys

import numpy as np
import pytest

from environments.chembench_mopen.ir import INPUT_NAMES, RateLaw, RateLawError
from environments.chembench_mopen.symbolic_proposer import (
    export_program,
    propose_from_history,
)


BOUNDS = [[0.01, 100], [0, 50], [0.01, 100], [0, 20], [0.01, 10], [278, 368], [4, 10]]


@pytest.fixture(autouse=True)
def scipy_compatible_test_torch_stub(monkeypatch):
    # The repository stubs torch; SciPy's array dispatch expects its Tensor type.
    stub = sys.modules.get("torch")
    if stub is not None and not hasattr(stub, "Tensor"):
        monkeypatch.setattr(stub, "Tensor", type("Tensor", (), {}), raising=False)


def op(name):
    return SimpleNamespace(name=name, arity=2)


def test_export_keeps_rational_structure_and_constant_precision():
    constant = 1.2345678912345
    program = SimpleNamespace(program=[op("div"), 0, op("add"), constant, 0])
    payload = export_program(program, BOUNDS)
    assert payload["params"] == [
        {"name": "k0", "low": 0.01, "high": 10.0, "transform": "log"}
    ]
    evaluate = RateLaw.from_payload(payload).compile()
    inputs = dict(zip(INPUT_NAMES, [2, 0, 1, 0, 1, 310, 7]))
    assert evaluate(inputs, {"k0": constant}) == pytest.approx(
        2 / (constant + 2), abs=1e-15
    )


def test_division_rejected_when_protection_can_activate():
    with pytest.raises(RateLawError, match="protected division"):
        export_program(SimpleNamespace(program=[op("div"), 0, 1]), BOUNDS)
    with pytest.raises(RateLawError, match="protected division"):
        export_program(
            SimpleNamespace(program=[op("div"), 0, op("sub"), 2.0, 3.0]), BOUNDS
        )


def test_no_parameter_tree_gets_uncertain_amplitude():
    result = export_program(SimpleNamespace(program=[0]), BOUNDS)
    assert result["expr"] == "k0 * (C_A)"
    assert result["params"][0]["low"] < 1 < result["params"][0]["high"]


def test_fitted_positive_difference_cannot_hide_negative_parameter_draws():
    with pytest.raises(RateLawError, match="positivity"):
        export_program(SimpleNamespace(program=[op("sub"), 8.0, 1.0]), BOUNDS)


@pytest.mark.parametrize(
    "nodes", [[op("add"), 0], [0, 1], [op("max"), 0, 1], [-1], [float("nan")], [0] * 32]
)
def test_bad_trees_fail_export(nodes):
    with pytest.raises(RateLawError):
        export_program(SimpleNamespace(program=nodes), BOUNDS)


def history():
    a = np.geomspace(0.05, 10, 12)
    x = np.array([[v, 0, 1, 0, 1, 310, 7] for v in a])
    return x, np.log1p(2 * a)


def test_actual_pinned_search_is_deterministic_and_connects_to_inference():
    pytest.importorskip("gplearn")
    from environments.chembench_mopen.executable_belief import ExecutableBeliefPool

    x, y = history()
    a = propose_from_history(
        x, y, input_bounds=BOUNDS, seed=17, population_size=32, generations=2
    )
    b = propose_from_history(
        x, y, input_bounds=BOUNDS, seed=17, population_size=32, generations=2
    )
    assert a.payloads == b.payloads
    assert a.rejected_exports == b.rejected_exports
    assert 0 < len(a.payloads) <= 4
    assert a.attempted_programs == 32 * a.search_generations
    assert a.approximate_training_scalar_nodes > 0
    assert len(a.final_population_audit) == 32
    assert a.final_population_audit == b.final_population_audit
    assert sum(
        r["status"] == "export_rejected" for r in a.final_population_audit
    ) == len(a.rejected_exports)
    pool = ExecutableBeliefPool(particles_per_law=8)
    for payload in a.payloads:
        pool.add(payload)
    result = pool.snapshot(
        history_inputs=x, observations=y, designs=x[:2], targets=x[-2:], sigma=0.15
    )
    assert np.isfinite(result.model.forecast(result.state)).all()


def test_search_responds_to_observations_not_fixed_domain_names():
    pytest.importorskip("gplearn")
    x, y = history()
    settings = dict(input_bounds=BOUNDS, seed=17, population_size=64, generations=3)
    increasing = propose_from_history(x, y, **settings)
    decreasing = propose_from_history(x, y[::-1], **settings)
    assert increasing.payloads != decreasing.payloads


def test_invalid_history_and_caps_fail_before_fit(monkeypatch):
    pytest.importorskip("gplearn")
    from gplearn.genetic import SymbolicRegressor

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid request must not fit")

    monkeypatch.setattr(SymbolicRegressor, "fit", forbidden)
    x, y = history()
    for options in [dict(generations=11), dict(population_size=513), dict(seed=-1)]:
        with pytest.raises(ValueError):
            propose_from_history(x, y, input_bounds=BOUNDS, **options)
    with pytest.raises(ValueError):
        propose_from_history(x[:1], y[:1], input_bounds=BOUNDS)
