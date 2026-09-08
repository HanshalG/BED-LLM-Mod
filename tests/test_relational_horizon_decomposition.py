from fractions import Fraction

import numpy as np
import pytest

from scripts.relational_concept_opportunity import compare
from scripts.relational_horizon_decomposition import decompose


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_exact_decomposition(seed):
    rows = (np.random.default_rng(seed).random((20, 10)) > .5).tolist()
    banked = compare(rows, query_count=4, budget=4)
    result = decompose(rows, banked, query_count=4)
    optimal = Fraction(result['exact_optimal_budget_risk'])
    for part in result['decomposition'].values():
        root = Fraction(part['exact_root_regret'])
        continuation = Fraction(part['exact_continuation_excess'])
        assert root >= 0 and continuation >= 0
        assert root+continuation == Fraction(part['exact_achieved'])-optimal
    assert result['decomposition']['h3']['exact_continuation_excess'] == '0'


def test_banked_choice_tampering_rejected():
    rows = [(False, False, False, False, False), (True, True, True, True, True)]
    banked = compare(rows, query_count=4, budget=4)
    banked['first_queries']['h1'] = 1
    with pytest.raises(ValueError, match='replay'):
        decompose(rows, banked, query_count=4)
