from fractions import Fraction

import pytest

from core.research_public_context import condition_on_public_context


def test_shared_context_preserves_prior_and_input():
    weights = {'a': 1, 'b': 3}
    result = condition_on_public_context(weights, {'a': 'public', 'b': 'public'}, 'public')
    assert result['weights'] == {'a': Fraction(1, 4), 'b': Fraction(3, 4)}
    assert result['context_probability'] == 1
    assert weights == {'a': 1, 'b': 3}
    assert not result['scientific_pass'] and not result['paid_calls_authorized']


def test_visible_code_can_identify_world_before_any_probe():
    result = condition_on_public_context({'a': 1, 'b': 1}, {'a': 'code-a', 'b': 'code-b'}, 'code-a')
    assert result['weights'] == {'a': 1, 'b': 0}
    assert result['positive_support_size'] == 1
    assert result['context_probability'] == Fraction(1, 2)


def test_partial_context_conditions_without_uniformizing():
    result = condition_on_public_context(
        {'a': '1/2', 'b': '1/3', 'c': '1/6'},
        {'a': 'same', 'b': 'same', 'c': 'other'}, 'same')
    assert result['weights'] == {'a': Fraction(3, 5), 'b': Fraction(2, 5), 'c': 0}
    assert result['context_probability'] == Fraction(5, 6)


@pytest.mark.parametrize('weights,contexts,observed', [
    ({}, {}, 'x'),
    ({'a': 1}, {}, 'x'),
    ({'a': 1}, {'a': 'x', 'b': 'x'}, 'x'),
    ({'a': 0}, {'a': 'x'}, 'x'),
    ({'a': 1}, {'a': 'x'}, 'y'),
    ({'a': -1}, {'a': 'x'}, 'x'),
    ({'a': True}, {'a': 'x'}, 'x'),
    ({'a': 0.5}, {'a': 'x'}, 'x'),
    ({'a': 'nan'}, {'a': 'x'}, 'x'),
    ({'a': 1}, {'a': ''}, ''),
    ({'': 1}, {'': 'x'}, 'x'),
])
def test_invalid_contract_rejected(weights, contexts, observed):
    with pytest.raises(ValueError):
        condition_on_public_context(weights, contexts, observed)
