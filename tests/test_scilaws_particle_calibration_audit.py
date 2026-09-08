import pytest

from scripts.scilaws_particle_calibration_audit import assess


def test_all_gates_required():
    m = dict(mean_standardized_max=0., variance_relative_max=0., family_tv=0.,
             log_density_error_max=0., ess_fraction=1.)
    assert assess(m)
    for key in m:
        bad = m.copy()
        bad[key] = 0. if key == 'ess_fraction' else 1.
        assert not assess(bad)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.1])
def test_nonfinite_or_negative_error_fails(bad):
    assert not assess(dict(mean_standardized_max=bad, variance_relative_max=0.,
                           family_tv=0., log_density_error_max=0., ess_fraction=1.))
