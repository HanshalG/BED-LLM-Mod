from fractions import Fraction
import pytest

from scripts.number_game_fixed_target_risk import (
    EmptyPredictiveSupport, best_predictive_query, continuation, terminal_brier,
)


def test_target_risk_breaks_entropy_tie_for_target_relevant_query():
    support = [(False, True, False, False), (False, False, False, False),
               (False, False, True, True)]
    assert best_predictive_query(support, queries=(1, 2), targets=(3,)) == (2, 0)
    assert best_predictive_query(support, queries=(1,), targets=(3,)) == (1, Fraction(1, 6))


def test_expected_brier_matches_explicit_world_enumeration():
    support = [(False, True, False, False), (False, False, False, False),
               (False, False, True, True)]
    for query in (1, 2):
        _, score = best_predictive_query(support, queries=(query,), targets=(0, 1, 2, 3))
        realized = sum(terminal_brier([row for row in support if row[query] == truth[query]],
                       targets=(0, 1, 2, 3), truth=truth) for truth in support)/len(support)
        assert realized == score


def test_history_only_selection_filters_and_excludes_without_target_deletion():
    support = [(False, True, False, False), (False, False, True, True)]
    args = dict(history=((0, False),), queries=(0, 1, 2), targets=(0, 1, 2, 3))
    assert continuation(support, **args) == continuation(list(reversed(support)), **args)
    assert continuation(support, **args)[0] == 1
    assert terminal_brier(support, targets=(0, 1, 2, 3), truth=support[0]) == Fraction(3, 16)


def test_empty_and_duplicate_support_fail_without_repair():
    support = [(False, False, False)]
    with pytest.raises(EmptyPredictiveSupport):
        continuation(support, history=((0, True),), queries=(1,), targets=(2,))
    with pytest.raises(ValueError, match='canonical'):
        best_predictive_query(support*2, queries=(1,), targets=(2,))
