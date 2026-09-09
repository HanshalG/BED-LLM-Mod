import pytest

from environments.program_induction.execution_steps import (
    completed_program, next_menu, public_state,
)
from environments.program_induction.prior import program_probability
from scripts.deepcoder_opportunity import load_dsl, sample_program


def path(dsl, program):
    choices = []
    for statement in program.statements:
        choices.append(next(x['choice'] for x in next_menu(dsl, choices)
                            if x['statement'] == str(statement)))
    return choices


def test_source_roundtrip_and_actual_state():
    d = load_dsl()
    history = [dict(inputs=[[1, 2], [3]], output=[99])]  # invalid label must fail
    with pytest.raises(ValueError):
        public_state(d, [], history)
    history[0]['output'] = None
    for seed in range(20):
        p = sample_program(d, seed)
        choices = path(d, p)
        q = completed_program(d, choices)
        assert str(q) == str(p)
        assert program_probability(d, q) > 0
        s = public_state(d, choices, history)
        actual = p.run(history[0]['inputs'])
        output = None if actual is None else actual.get_output()
        assert s['rows'][0]['values'][f'x{len(choices)+1}'] == output
        assert s['fits_observations'] == (output is None)
        if len(choices) == 4:
            assert next_menu(d, choices) == []


def test_type_change_and_no_premature_output_pruning():
    d = load_dsl()
    choices = [next(x['choice'] for x in next_menu(d, []) if x['statement'] == 'x2 = Last x0')]
    history = [dict(inputs=[[4, 1], [2]], output=[1])]
    s = public_state(d, choices, history)
    assert s['types']['x2'] == 'int'
    assert not s['can_stop'] and not s['fits_observations']
    choices.append(next(x['choice'] for x in next_menu(d, choices)
                        if x['statement'] == 'x3 = Drop x2 x0'))
    assert public_state(d, choices, history)['fits_observations']


def test_absorbing_error_retained_as_valid_observation():
    d = load_dsl()
    p = d.Program.from_str('x0 = INPUT | x1 = INPUT | x2 = Last x0 | x3 = Access x2 x1')
    choices = path(d, p)
    s = public_state(d, choices, [dict(inputs=[[9], [1]], output=None)])
    assert s['rows'][0]['execution_error'] and s['fits_observations']


def test_fail_closed_and_empty_history():
    d = load_dsl()
    for choices in ([True], [-1], [100000], [0] * 5, (0,)):
        with pytest.raises(ValueError):
            next_menu(d, choices)
    with pytest.raises(ValueError):
        completed_program(d, [0])
    with pytest.raises(ValueError):
        public_state(d, [], [dict(inputs=[[1], [2]], output=1, hidden_truth='bad')])
    assert not public_state(d, [], [])['fits_observations']
