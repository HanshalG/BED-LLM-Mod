import pytest

from scripts.deepcoder_support_audit import moments, summarize


def test_correct_and_confident_wrong_brier():
    assert moments([['a', 'x']], ['a', 'x'], 1) == (0, 0, 0)
    assert moments([['a', 'x']], ['a', 'y'], 1) == (1, 0, 1)


def test_hand_calculated_multiclass():
    loss, risk, zeros = moments([['q', 'a'], ['q', 'b'], ['q', 'b']], ['q', 'c'], 1)
    assert loss == pytest.approx(7/9)
    assert risk == pytest.approx(2/9)
    assert zeros == 1


def test_unsupported_cases_not_silently_scored_as_correct():
    result = summarize([['a', 'x']], [['b', 'y'], ['a', 'z']], 1)
    assert result['unsupported_query_conditions'] == 1
    assert result['supported_conditions'] == 1
    assert result['unsupported_by_query'] == [1]
    assert result['supported_only_brier'] == 1
    assert result['supported_only_internal_risk'] == 0
    assert result['prior_brier'] == 1


def test_all_unsupported_is_explicitly_undefined():
    result = summarize([['a', 'x']], [['b', 'y']], 1)
    assert result['supported_only_brier'] is None
    assert result['supported_only_internal_risk'] is None
    assert result['supported_only_target_count'] == 0


def test_self_population_matches_bayes_risk():
    rows = [['a', 'x'], ['a', 'y'], ['b', 'z']]
    result = summarize(rows, rows, 1)
    assert result['prior_brier'] == pytest.approx(result['prior_internal_risk'])
    assert result['supported_only_brier'] == pytest.approx(result['supported_only_internal_risk'])
    assert result['unsupported_query_conditions'] == 0


def test_shape_failure():
    with pytest.raises(ValueError):
        summarize([['a', 'x']], [['b']], 1)
