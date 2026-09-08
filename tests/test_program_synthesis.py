import pytest

from environments.program_induction.synthesis import synthesize, evaluate_expression
from scripts.deepcoder_opportunity import load_dsl


def test_source_replay_and_new_input():
    dsl = load_dsl()
    history = [([[1, 2, 3], [9]], [3, 2, 1]), ([[4, 5], [0]], [5, 4])]
    result = synthesize(dsl, history)
    assert result['status'] == 'compatible_expression_found'
    assert evaluate_expression(result['expression'], [[7, 8], [3]]) == [8, 7]
    assert not result['posterior_samples'] and not result['paid_calls_authorized']


def test_error_is_not_dropped_and_typing_survives_it():
    dsl = load_dsl()
    result = synthesize(dsl, [([[], [9]], None), ([[2, 3], [7]], 2)])
    assert result['status'] == 'compatible_expression_found'
    assert evaluate_expression(result['expression'], [[], [0]]) is None
    assert evaluate_expression(result['expression'], [[4], [0]]) == 4


def test_per_operation_cap_is_hard():
    result = synthesize(load_dsl(), [([[1], [2]], [49, 48, 47])], max_attempts=1)
    assert result['expression'] is None
    assert result['operation_attempts'] <= 1


def test_input_identity_solution_and_no_generated_eval():
    result = synthesize(load_dsl(), [([[1, 2], [9]], [1, 2])])
    assert result['status'] == 'compatible_expression_found'
    assert result['operation_attempts'] == 0
    assert evaluate_expression(result['expression'], [[5], [8]]) == [5]


@pytest.mark.parametrize('history', [[], [([[1]], 1)], [([[True], [1]], 1)],
                                   [([[1], [2]], float('nan'))]])
def test_invalid_history(history):
    with pytest.raises(ValueError):
        synthesize(load_dsl(), history)
