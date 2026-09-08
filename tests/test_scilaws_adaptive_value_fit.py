import numpy as np

from environments.scilaws.adaptive_value_fit import fit_adaptive


def test_smooth_actions_fresh_checks_and_cache():
    called = []
    def values(u):
        called.append(u)
        return [2+u*u, 3+u]
    model, nodes, info = fit_adaptive(values, -1, 1)
    assert model is not None
    assert info['final_disjoint']
    assert info['normalized_check_error'] < 1e-12
    assert len(called) == len(set(called)) == info['total_evaluations']
    assert len(nodes) == 17


def test_oscillation_hidden_from_adaptation_fails_fresh_check():
    model, _, info = fit_adaptive(lambda u: [2+np.sin(32*np.pi*u)**2], 0, 1)
    assert model is None
    assert info['fit_status'] == 'fresh_check_failed'
    assert info['final_disjoint']


def test_refinement_cap_does_not_emit_fit():
    model, nodes, info = fit_adaptive(lambda u: [2+np.sin(201*u)], -1, 1)
    assert model is None
    assert info['fit_status'] in ('node_cap', 'refinement_cap', 'fresh_check_failed')
    assert len(nodes) <= 65


def test_signed_residual_mode_does_not_clip():
    model, _, info = fit_adaptive(lambda u: [-1+u/2], -1, 1, nonnegative=False)
    assert model is not None
    assert info['final_disjoint']
    assert model(0)[0] == -1
