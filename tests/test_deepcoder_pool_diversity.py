import math
import pytest
from environments.program_induction.prediction import forecast
from scripts.deepcoder_pool_diversity import summarize


def test_syntax_duplicates_are_not_behavioral_diversity():
    h = [{'inputs': [[1], [2]], 'output': 1}]
    f = forecast([('a', lambda x: 1), ('b', lambda x: 1)], h, [[[3], [4]]])
    r = summarize(f, [['1'], ['1']])
    assert r['distinct_behaviors'] == 1 and r['compatible_programs'] == 2
    assert r['separating_queries'] == 0 and r['max_query_entropy_nats'] == 0


def test_distinguishable_query_and_empty_support():
    h = [{'inputs': [[1], [2]], 'output': 1}]
    f = forecast([('a', lambda x: 1), ('b', lambda x: x[0][0])], h, [[[3], [4]]])
    r = summarize(f, [['1'], ['3']])
    assert r['distinct_behaviors'] == 2 and r['separating_queries'] == 1
    assert r['max_query_entropy_nats'] == pytest.approx(math.log(2))
    assert summarize(None, [])['status'] == 'no_compatible_support'
    with pytest.raises(ValueError, match='disagreement'):
        summarize(f, [['1'], ['1']])
