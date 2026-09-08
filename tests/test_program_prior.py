from fractions import Fraction
import random

import pytest

from environments.program_induction.prior import statement_options, program_probability, restricted_weights
from scripts.deepcoder_opportunity import load_dsl, sample_program


def make(dsl, *steps):
    return dsl.Program(['x0', 'x1'], [dsl.Statement.from_str(s) for s in steps])


def test_manual_counts_and_probabilities():
    dsl = load_dsl()
    variables = [('x0', list), ('x1', list)]
    assert len(statement_options(dsl, variables, False)) == 80
    assert len(statement_options(dsl, variables+[('x2', list)], True)) == 55
    assert len(statement_options(dsl, variables+[('x2', int)], True)) == 6
    reverse = make(dsl, 'x2 = Reverse x0', 'x3 = Head x2')
    access = make(dsl, 'x2 = Head x0', 'x3 = Access x2 x0')
    assert program_probability(dsl, reverse) == Fraction(1, 13200)
    assert program_probability(dsl, access) == Fraction(1, 1440)
    weights = restricted_weights(dsl, [reverse, access, access])
    assert weights == {str(reverse): Fraction(6, 61), str(access): Fraction(55, 61)}
    for p in (reverse, access):
        assert p.run([[0], [1]]).get_output() == 0
    assert reverse.run([[0, 1], [1]]).get_output() == 1
    assert access.run([[0, 1], [1]]).get_output() == 0


def test_source_sampler_choice_sequence_and_probability_match():
    dsl = load_dsl()
    for seed in range(100):
        rng = random.Random(seed)
        count = rng.choice((2, 3, 4))
        variables, steps, probability = [('x0', list), ('x1', list)], [], Fraction(1, 3)
        for i in range(count):
            choices = statement_options(dsl, variables, i > 0)
            op, args = rng.choice(choices)
            probability /= len(choices)
            steps.append(dsl.Statement(f'x{i+2}', op, args))
            variables.append((f'x{i+2}', op.output_type))
        reconstructed = dsl.Program(['x0', 'x1'], steps)
        original = sample_program(dsl, seed)
        assert str(reconstructed) == str(original)
        assert program_probability(dsl, original) == probability


@pytest.mark.parametrize('steps', [
    ('x2 = Head x0',),
    ('x2 = Reverse x0', 'x3 = Head x1'),
    ('x2 = Head x0', 'x3 = Head x2'),
])
def test_outside_prior_rejected(steps):
    dsl = load_dsl()
    with pytest.raises(ValueError):
        program_probability(dsl, make(dsl, *steps))


def test_empty_set_rejected():
    with pytest.raises(ValueError):
        restricted_weights(load_dsl(), [])
