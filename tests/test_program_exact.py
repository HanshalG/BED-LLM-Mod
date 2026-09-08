from fractions import Fraction
from itertools import product
from types import SimpleNamespace

import pytest

from environments.program_induction.exact import predict, EnumerationLimit
from environments.program_induction.prediction import category


class Op:
    inputs_type = (list,)
    output_type = list

    def __init__(self, fn):
        self.fn = fn

    def run(self, args):
        return self.fn(args[0])


def grammar():
    return SimpleNamespace(OPERATIONS=[Op(lambda x: x[::-1]), Op(lambda x: x[:1])], LAMBDAS=[])


def brute(dsl, inputs, history):
    result = {}
    for length in (2, 3, 4):
        for first in (0, 1):
            for ops in product(dsl.OPERATIONS, repeat=length):
                outputs = []
                for inp in inputs:
                    value = inp[first]
                    for op in ops:
                        value = op.run([value])
                    outputs.append(category(value))
                if outputs[:len(history)] == history:
                    key = tuple(outputs[len(history):])
                    result[key] = result.get(key, Fraction())+Fraction(1, 3*2*2**length)
    evidence = sum(result.values(), Fraction())
    return evidence, {k: v/evidence for k, v in result.items()}


@pytest.mark.parametrize('observed', [[], [{'inputs': [[1, 2], [3, 4]], 'output': [1]}]])
def test_exact_matches_independent_syntax_enumeration(observed):
    dsl = grammar()
    targets = [[[2, 3], [4, 5]], [[0, 1], [2, 3]]]
    expected = brute(dsl, [r['inputs'] for r in observed]+targets,
                     [category(r['output']) for r in observed])
    result = predict(dsl, observed, targets)
    assert (result['evidence'], result['probabilities']) == expected
    assert result['stats']['completed_depth'] == 4


def test_duplicate_syntax_mass_preserved():
    dsl = grammar()
    dsl.OPERATIONS.append(dsl.OPERATIONS[0])
    # All operations coincide on singleton inputs: their mass must sum to one.
    result = predict(dsl, [], [[[1], [1]]])
    assert result['evidence'] == 1
    assert result['probabilities'] == {('[1]',): Fraction(1)}


@pytest.mark.parametrize('limits', [dict(max_states=1), dict(max_transitions=1), dict(seconds=1e-12)])
def test_caps_do_not_return_partial_forecasts(limits):
    with pytest.raises(EnumerationLimit) as exc:
        predict(grammar(), [], [[[1, 2], [2, 3]]], **limits)
    assert exc.value.stats['completed_depth'] < 4


def test_error_mass_is_not_discarded():
    dsl = SimpleNamespace(OPERATIONS=[Op(lambda _: None)], LAMBDAS=[])
    result = predict(dsl, [{'inputs': [[1], [2]], 'output': None}], [[[3], [4]]])
    assert result['evidence'] == 1
    assert result['probabilities'] == {('ERROR',): Fraction(1)}
