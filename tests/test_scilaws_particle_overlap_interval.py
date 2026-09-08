import numpy as np
import pytest

from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.particle_overlap_interval import ParticleOverlapIntervals
from environments.scilaws.particle_reference import ParticleReference


@pytest.mark.parametrize('conditioned', [False, True])
def test_overlap_contains_reference(conditioned):
    m = QuantileGaussianModel([[-1., .2], [2., -.5], [.5, 1.]], [[.3, 1.], [1., .8], [.7, .5]],
        [[0., 1.], [1., 0.], [2., 2.]], [.3, .5, .2], target_conditional_variances=[[.1], [.2], [.3]])
    state = m.condition(m.initial_state, 0, .5) if conditioned else m.initial_state
    rows = ParticleOverlapIntervals(m).actions(state)['intervals']
    ref = ParticleReference(m, state)
    for a, row in enumerate(rows):
        assert row['lower'] <= ref.action(a)['value'] <= row['upper']
        assert row['lower'] > row['noise_floor']


def test_identical_likelihood_recovers_prior_risk():
    m = QuantileGaussianModel([[0.], [0.]], .5, [[0.], [2.]], [.3, .7], target_conditional_variances=.1)
    row = ParticleOverlapIntervals(m).actions(m.initial_state)['intervals'][0]
    assert row['lower'] == pytest.approx(.94, abs=2e-12)
    assert row['upper'] == pytest.approx(.94, abs=2e-12)


def test_memory_and_streaming_equivalence():
    m = QuantileGaussianModel([[0.], [1.], [2.]], .4, [[0.], [2.], [1.]], [.2, .3, .5])
    evaluator = ParticleOverlapIntervals(m)
    first = evaluator.actions(m.initial_state)
    second = evaluator.actions(m.initial_state, max_workspace_bytes=8*3*(4+24)+48)
    assert second['block_rows'] == 1
    assert second['pair_terms'] == 9
    np.testing.assert_allclose([r['lower'] for r in first['intervals']],
                               [r['lower'] for r in second['intervals']], atol=1e-14)
    with pytest.raises(SearchLimitExceeded):
        evaluator.actions(m.initial_state, max_workspace_bytes=1)
