import copy
import math

import numpy as np
import pytest

from environments.chembench_mopen.executable_belief import ExecutableBeliefPool
from environments.chembench_mopen.ir import RateLawError


def law(expr="k * C_A", name="linear"):
    return {
        "name": name,
        "expr": expr,
        "params": [{"name": "k", "low": 0.1, "high": 4.0}],
    }


def inputs(a):
    return [a, 0, 1, 0, 1, 310, 7]


def snapshot(pool, observations=()):
    return pool.snapshot(
        history_inputs=[inputs(1)] * len(observations),
        observations=observations,
        designs=[inputs(1), inputs(2)],
        targets=[inputs(3)],
        sigma=0.15,
    )


def test_parameter_uncertainty_and_full_gaussian_posterior_match_manual():
    pool = ExecutableBeliefPool(particles_per_law=16, seed=13)
    pool.add(law())
    result = snapshot(pool, [0.7, 0.8])
    k = np.array(result.parameter_values)[:, 0]
    assert len(set(k)) == 16
    assert np.all((k >= 0.1) & (k <= 4))
    means = np.log1p(k)
    log_lik = sum(
        -0.5 * ((y - means) / 0.15) ** 2 - math.log(0.15 * math.sqrt(2 * math.pi))
        for y in [0.7, 0.8]
    )
    weights = np.exp(log_lik - log_lik.max())
    weights /= weights.sum()
    np.testing.assert_allclose(np.exp(result.state), weights, atol=1e-14)
    assert result.conditional_log_evidence == pytest.approx(
        log_lik.max() + np.log(np.exp(log_lik - log_lik.max()).mean())
    )
    assert result.model.forecast(result.state)[0] == pytest.approx(
        weights @ np.log1p(3 * k)
    )
    assert result.model.risk(result.state) > 0


def test_refresh_replays_history_once_and_new_law_has_nonzero_mass():
    pool = ExecutableBeliefPool(seed=4)
    first = pool.add(law())
    old = snapshot(pool, [0.7])
    second = pool.add(law("k * C_A / (1 + C_A)", "saturating"))
    refreshed = snapshot(pool, [0.7])
    fresh = ExecutableBeliefPool(seed=4)
    fresh.add(law("k * C_A / (1 + C_A)", "saturating"))
    fresh.add(law())
    independently_built = snapshot(fresh, [0.7])
    assert refreshed.state == independently_built.state
    assert refreshed.history_sha256 == old.history_sha256
    for key in [first, second]:
        assert sum(np.exp(refreshed.state)[np.array(refreshed.particle_law) == key]) > 0
    assert snapshot(pool, [0.7]).state == refreshed.state


def test_duplicate_names_rationales_and_alpha_renaming_do_not_add_mass():
    pool = ExecutableBeliefPool()
    key = pool.add(law())
    before = snapshot(pool)
    renamed = law("q * C_A", "different name")
    renamed["params"][0]["name"] = "q"
    renamed["rationale"] = "No independent evidence."
    assert pool.add(renamed) == key
    after = snapshot(pool)
    assert after.state == before.state
    assert after.parameter_values == before.parameter_values
    assert len(after.law_keys) == 1


def test_proposal_order_independent_and_incremental_condition_equals_replay():
    a, b = ExecutableBeliefPool(seed=8), ExecutableBeliefPool(seed=8)
    payloads = [law(), law("k * C_A / (1 + C_A)")]
    for p in payloads:
        a.add(p)
    for p in reversed(payloads):
        b.add(p)
    before = snapshot(a, [0.7])
    after = snapshot(b, [0.7, 0.9])
    np.testing.assert_allclose(
        before.model.condition(before.state, 0, 0.9), after.state, atol=1e-13
    )
    assert before.history_sha256 != after.history_sha256


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('id')",
        "C_A.real",
        "k[0]",
        "(lambda x: x)(k)",
        "missing * C_A",
    ],
)
def test_unsafe_expressions_rejected_atomically(expr):
    pool = ExecutableBeliefPool()
    with pytest.raises(RateLawError):
        pool.add(law(expr))
    with pytest.raises(ValueError, match="empty"):
        snapshot(pool)


@pytest.mark.parametrize(
    "expr",
    [
        "k * (2 ** (2 ** 100))",
        "k / (C_A - C_A)",
        "k * sqrt(-C_A)",
        "-k * C_A",
        "exp(k * 1000000)",
    ],
)
def test_domain_overflow_and_integer_explosion_fail_closed(expr):
    pool = ExecutableBeliefPool()
    pool.add(law(expr))
    with pytest.raises(RateLawError, match="failed inference"):
        snapshot(pool)


def test_resource_caps_apply_before_expression_execution():
    for kwargs, message in [
        ({"max_scalar_nodes": 1}, "scalar-node"),
        ({"max_workspace_bytes": 1}, "workspace"),
    ]:
        pool = ExecutableBeliefPool(**kwargs)
        pool.add(law("k / (C_A - C_A)"))
        with pytest.raises(RuntimeError, match=message):
            snapshot(pool)
    pool = ExecutableBeliefPool(max_laws=1)
    pool.add(law())
    with pytest.raises(RateLawError, match="capacity"):
        pool.add(law("k * C_A / (1 + C_A)"))
    assert len(snapshot(pool).law_keys) == 1


def test_invalid_payloads_and_history_rejected():
    for payload in [[], {"expr": "x"}, law("C_A"), law("k" + "+ k" * 100)]:
        with pytest.raises(RateLawError):
            ExecutableBeliefPool().add(payload)
    payload = copy.deepcopy(law())
    payload["params"][0]["low"] = True
    with pytest.raises(RateLawError):
        ExecutableBeliefPool().add(payload)
    pool = ExecutableBeliefPool()
    pool.add(law())
    with pytest.raises(ValueError):
        snapshot(pool, [float("nan")])
    with pytest.raises(ValueError):
        pool.snapshot(
            history_inputs=[inputs(1)],
            observations=[],
            designs=[inputs(1)],
            targets=[inputs(2)],
            sigma=0.15,
        )


def test_predictive_score_is_frozen_before_next_observation():
    pool = ExecutableBeliefPool()
    pool.add(law())
    before = snapshot(pool, [0.7])
    old_forecast = before.model.forecast(before.state).copy()
    old_parameters = before.parameter_values
    pool.add(law("k * C_A / (1 + C_A)"))
    after = snapshot(pool, [0.7, 0.9])
    np.testing.assert_array_equal(before.model.forecast(before.state), old_forecast)
    assert before.parameter_values == old_parameters
    assert before.history_sha256 != after.history_sha256
    assert after.interpretation == "finite_pool_conditional_fit_not_selection_corrected"


def test_snapshot_connects_to_all_three_genuine_horizons_without_proposals(monkeypatch):
    from environments.chembench_mopen.batch_horizon import plan_batched
    from environments.chembench_mopen.envelope_belief import EnvelopeGaussianModel

    pool = ExecutableBeliefPool(particles_per_law=2)
    pool.add(law())
    pool.add(law("k * C_A / (1 + C_A)"))
    result = pool.snapshot(
        history_inputs=[inputs(1)],
        observations=[0.7],
        designs=[inputs(0.2), inputs(2), inputs(4)],
        targets=[inputs(0.5), inputs(3)],
        sigma=0.15,
    )
    model = EnvelopeGaussianModel(
        result.model.means,
        result.model.sigmas,
        result.model.targets,
        np.full(result.model.num_particles, 1 / result.model.num_particles),
        branch_count=8,
    )

    def forbid(*args, **kwargs):
        raise AssertionError("planning must not refresh executable laws")

    monkeypatch.setattr(pool, "add", forbid)
    monkeypatch.setattr(pool, "snapshot", forbid)
    before = result.state
    for h in [1, 2, 3]:
        plan = plan_batched(model, result.state, h, max_seconds=10)
        assert plan.effective_horizon == h
        assert plan.action in [0, 1, 2]
        assert np.isfinite(plan.value)
    assert result.state == before
    np.testing.assert_allclose(
        model.forecast(result.state), result.model.forecast(result.state)
    )


@pytest.mark.parametrize(
    "expr",
    [
        "exp(k * C_A / (1 + C_A))",
        "sqrt(k) + log(1 + C_A)",
        "k ** 0.5 * C_A ** 2",
        "k + (-C_A) + C_A",
        "+k * C_A",
    ],
)
def test_vectorized_evaluator_matches_existing_scalar_interpreter(expr):
    from environments.chembench_mopen.ir import INPUT_NAMES, RateLaw

    payload = law(expr)
    pool = ExecutableBeliefPool(particles_per_law=5)
    pool.add(payload)
    result = snapshot(pool)
    scalar = RateLaw.from_payload(payload).compile()
    for row, parameters in enumerate(result.parameter_values):
        for col, concentration in enumerate([1, 2]):
            expected = scalar(
                dict(zip(INPUT_NAMES, inputs(concentration))), {"k": parameters[0]}
            )
            assert result.model.means[row, col] == pytest.approx(np.log1p(expected))
