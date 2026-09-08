import pytest

from environments.program_induction.rollout_risk import expected_brier


def test_manual_mismatched_support_and_decomposition():
    p, q = {'a': .25, 'b': .75}, {'a': .5, 'c': .5}
    result = expected_brier(p, q)
    manual = sum(prob * .5 * (1+sum(v*v for v in q.values())-2*q.get(y, 0))
                 for y, prob in p.items())
    assert result['expected_loss'] == pytest.approx(manual)
    assert manual == pytest.approx(.625)
    assert result['reference_bayes_risk'] == pytest.approx(.1875)
    assert result['excess_risk'] == pytest.approx(.4375)


def test_calibrated_forecast_reduces_to_bayes_risk():
    p = {'a': .2, 'b': .8}
    r = expected_brier(p, p)
    assert r['excess_risk'] == 0
    assert r['expected_loss'] == pytest.approx(.16)
    assert r['expected_loss'] == r['forecast_self_risk']


def test_confident_collapse_reverses_self_risk_ranking():
    p = {'a': .5, 'b': .5}
    calibrated = expected_brier(p, p)
    collapsed = expected_brier(p, {'a': 1.})
    assert collapsed['forecast_self_risk'] < calibrated['forecast_self_risk']
    assert collapsed['expected_loss'] > calibrated['expected_loss']
    assert expected_brier({'b': 1.}, {'a': 1.})['expected_loss'] == 1


def test_branch_conditioning_matters():
    # An observation reveals the target perfectly in either equally likely branch.
    branches = [{'a': 1.}, {'b': 1.}]
    assert sum(expected_brier(p, p)['expected_loss'] / 2 for p in branches) == 0
    assert sum(expected_brier({'a': .5, 'b': .5}, p)['expected_loss'] / 2
               for p in branches) == .5


def test_abstention_has_no_fake_confidence():
    r = expected_brier({'a': 1.}, None)
    assert r['expected_loss'] == 1
    assert r['forecast_self_risk'] is None
    assert r['excess_risk'] is None


@pytest.mark.parametrize('bad', [{}, {'a': True}, {'a': float('nan')},
                                {'a': -.1, 'b': 1.1}, {'a': .9}, {1: 1.}])
def test_invalid_mass_rejected(bad):
    with pytest.raises(ValueError):
        expected_brier(bad, {'a': 1.})
    with pytest.raises(ValueError):
        expected_brier({'a': 1.}, bad)
