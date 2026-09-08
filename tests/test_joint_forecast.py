from copy import deepcopy

import pytest

from environments.program_induction.joint_forecast import condition, validate
from environments.program_induction.rollout_risk import expected_brier


def fixture():
    return dict(interpretation='restricted_syntax_prior_not_full_posterior', worlds=[
        dict(key='p0', weight=.2, answers=['a', 'x'], targets=['0', '1']),
        dict(key='p1', weight=.3, answers=['a', 'y'], targets=['1', '1']),
        dict(key='p2', weight=.5, answers=['b', 'y'], targets=['2', '0'])])


def test_exact_conditional_law_and_unsupported_branch():
    law = fixture()
    assert condition(law, {})['distributions'][0] == {'0': .2, '1': .3, '2': .5}
    branch = condition(law, {0: 'a'})
    assert branch['evidence_probability'] == .5
    assert branch['distributions'] == [{'0': .4, '1': .6}, {'1': 1.}]
    assert condition(law, {0: 'b', 1: 'x'}) == dict(
        evidence_probability=0., distributions=None, support_size=0)
    assert condition(law, {0: 'a', 1: 'y'})['evidence_probability'] == .3


def test_joint_not_product_of_marginals():
    law = fixture()
    joint = condition(law, {0: 'a', 1: 'x'})['evidence_probability']
    product = (condition(law, {0: 'a'})['evidence_probability'] *
               condition(law, {1: 'x'})['evidence_probability'])
    assert joint == .2
    assert product == .1


def test_branch_probability_weighted_loss_matches_world_enumeration():
    law = fixture()
    forecasts = {'a': {'0': .1, '1': .9}, 'b': {'2': .4, '3': .6}}
    by_branch = sum(condition(law, {0: y})['evidence_probability'] *
        expected_brier(condition(law, {0: y})['distributions'][0], q)['expected_loss']
        for y, q in forecasts.items())
    direct = sum(w['weight'] * expected_brier({w['targets'][0]: 1.},
        forecasts[w['answers'][0]])['expected_loss'] for w in law['worlds'])
    assert by_branch == pytest.approx(direct)


@pytest.mark.parametrize('obs', [{True:'a'}, {-1:'a'}, {2:'a'}, {0:None}, []])
def test_bad_observations_rejected(obs):
    with pytest.raises(ValueError):
        condition(fixture(), obs)


@pytest.mark.parametrize('weight', [True, -.1, 0, float('inf'), float('nan'), .7])
def test_bad_weights_rejected(weight):
    law = fixture()
    law['worlds'][0]['weight'] = weight
    with pytest.raises(ValueError):
        validate(law)


def test_dimensions_and_duplicate_worlds_rejected():
    law = fixture()
    bad = deepcopy(law)
    bad['worlds'][0]['answers'].pop()
    with pytest.raises(ValueError):
        validate(bad)
    law['worlds'][0]['key'] = 'p1'
    with pytest.raises(ValueError):
        validate(law)


def test_executable_forecast_deduplication_and_history_filter():
    from environments.program_induction import constrained
    from environments.program_induction.joint_forecast import forecast
    from scripts.deepcoder_opportunity import load_dsl
    dsl = load_dsl()
    programs = constrained.decode(dsl, '{"programs":[{"statement":"x2 = Reverse x0",'
        '"next":{"statement":"x3 = Head x2","next":null}},'
        '{"statement":"x2 = Reverse x0","next":{"statement":"x3 = Last x2","next":null}}]}')
    x = [[1, 2], [3]]
    law = forecast(dsl, programs+programs, [], [x], [x])
    assert len(law['worlds']) == 2
    assert condition(law, {0:'2'})['distributions'] == [{'2':1.}]
    filtered = forecast(dsl, programs, [dict(inputs=x, output=2)], [x], [x])
    assert len(filtered['worlds']) == 1
    assert forecast(dsl, programs, [dict(inputs=x, output=9)], [x], [x]) is None
