import copy
import json

import pytest

from environments.program_induction.prediction import ARMS, forecast, seal, score


HISTORY = [{'inputs': [[1], [2]], 'output': 1}]
TARGETS = [[[3], [4]]]


def f(value):
    return forecast([('a', lambda x: 1 if x == HISTORY[0]['inputs'] else value)], HISTORY, TARGETS)


def test_seal_then_score_confident_wrong(tmp_path):
    path = tmp_path/'forecasts.json'
    cases = {'case': {arm: f(9 if arm == 'history_aware' else 3) for arm in ARMS}}
    sha = seal(path, cases)
    cases['case']['history_aware']['counts'][0] = {'3': 1}
    calls = []
    def outcomes():
        assert path.exists()
        calls.append(1)
        return {'case': {'target_inputs': TARGETS, 'outputs': [3]}}
    result = score(path, sha, outcomes)
    assert calls == [1]
    assert result['cases']['case']['history_aware'] == dict(
        brier=1, nll=None, nll_is_infinite=True, zero_mass_targets=1)
    assert result['cases']['case']['history_blind']['brier'] == 0
    with pytest.raises(FileExistsError):
        seal(path, cases)


def test_mass_and_duplicates_and_full_history():
    def eval_a(x):
        return 1
    result = forecast([('a', eval_a), ('a', eval_a), ('b', lambda x: 1),
                       ('bad', lambda x: 0)], HISTORY, TARGETS)
    assert result['counts'] == [{'1': 2}]
    assert result['proposed_unique_count'] == 3
    with pytest.raises(ValueError, match='compatible'):
        forecast([('bad', lambda x: 0)], HISTORY, TARGETS)


def test_missing_control_or_mismatched_targets_fail_before_seal(tmp_path):
    cases = {'case': {arm: f(3) for arm in ARMS}}
    cases['case']['history_blind']['target_inputs'] = [[[5], [6]]]
    with pytest.raises(ValueError):
        seal(tmp_path/'a', cases)
    del cases['case']['history_blind']
    with pytest.raises(ValueError):
        seal(tmp_path/'b', cases)


def test_tamper_or_malformed_distribution_never_opens_outcomes(tmp_path):
    path = tmp_path/'f'
    cases = {'case': {arm: f(3) for arm in ARMS}}
    sha = seal(path, cases)
    def bomb():
        raise AssertionError('outcomes opened')
    with pytest.raises(ValueError):
        score(path, 'wrong', bomb)
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError):
        score(path, sha, bomb)
    invalid = copy.deepcopy(cases)
    invalid['case']['history_aware']['counts'] = [{'3': 2}]
    with pytest.raises(ValueError):
        seal(tmp_path/'invalid', invalid)


def test_brier_mixture_manual(tmp_path):
    pool = forecast([('a', lambda x: 1), ('b', lambda x: 1 if x == HISTORY[0]['inputs'] else 2)],
                    HISTORY, TARGETS)
    path = tmp_path/'f'
    sha = seal(path, {'case': {arm: pool for arm in ARMS}})
    result = score(path, sha, lambda: {'case': {'target_inputs': TARGETS, 'outputs': [1]}})
    assert result['cases']['case']['history_aware']['brier'] == .25
    assert result['cases']['case']['history_aware']['nll'] == pytest.approx(.6931471805599453)
    assert json.loads(path.read_text())['interpretation'] == 'uniform_compatible_pool_not_full_posterior'


def test_executable_three_arm_fixture(tmp_path):
    from environments.program_induction.proposals import parse_proposals
    from environments.program_induction.synthesis import synthesize, evaluate_expression
    from scripts.deepcoder_opportunity import load_dsl

    dsl = load_dsl()
    history = [{'inputs': [[1, 2], [9]], 'output': 2}]
    targets = [[[4, 3], [0]]]
    text = json.dumps({'programs': [
        {'steps': [{'op': 'Reverse', 'args': ['x0']}, {'op': 'Head', 'args': ['x2']}]},
        {'steps': [{'op': 'Sort', 'args': ['x0']}, {'op': 'Last', 'args': ['x2']}]},
    ]})
    programs = parse_proposals(dsl, text)
    def evaluator(p):
        def evaluate(inputs):
            state = p.run(inputs)
            return None if state is None else state.get_output()
        return evaluate
    aware = forecast([(str(p), evaluator(p)) for p in programs], history, targets)
    blind = forecast([(str(programs[1]), evaluator(programs[1]))], history, targets)
    result = synthesize(dsl, [(history[0]['inputs'], 2)])
    expression = result['expression']
    assert expression is not None
    symbolic = forecast([(expression.expression(), lambda x: evaluate_expression(expression, x))],
                        history, targets)
    path = tmp_path/'three-arms.json'
    sha = seal(path, {'case': dict(zip(ARMS, (aware, blind, symbolic)))})
    result = score(path, sha, lambda: {'case': {'target_inputs': targets, 'outputs': [3]}})
    assert result['cases']['case']['history_aware']['brier'] == .25
    assert result['cases']['case']['history_blind']['brier'] == 1
