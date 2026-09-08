import json

import pytest

from environments.program_induction.proposals import messages, parse_proposals
from scripts.deepcoder_opportunity import load_dsl


def candidate():
    return {'steps': [{'op': 'Reverse', 'args': ['x0']}, {'op': 'Head', 'args': ['x2']}]}


def test_strict_compile_execution_and_deduplication():
    dsl = load_dsl()
    result = parse_proposals(dsl, json.dumps({'programs': [candidate(), candidate()]}))
    assert len(result) == 1
    assert result[0].run([[1, 2], [4]]).get_output() == 2


def test_lambda_types_and_error_execution():
    dsl = load_dsl()
    c = {'steps': [{'op': 'Filter', 'lambda': '(<0)', 'args': ['x0']},
                   {'op': 'Head', 'args': ['x2']}]}
    result = parse_proposals(dsl, json.dumps({'programs': [c]}))[0]
    assert result.run([[1, 2], [0]]) is None


def test_blind_prompt_invariance_and_history_copy():
    dsl = load_dsl()
    a = [{'inputs': [[1], [2]], 'output': 1}]
    b = [{'inputs': [[3], [4]], 'output': None}]
    assert messages(dsl, a, history_blind=True) == messages(dsl, b, history_blind=True)
    assert messages(dsl, a) != messages(dsl, b)
    saved = messages(dsl, a)
    a[0]['inputs'][0][0] = 9
    assert messages(dsl, a) != saved
    assert json.loads(saved[1]['content'])['history'][0]['inputs'][0] == [1]


@pytest.mark.parametrize('text', ['{}', '{"programs":[],"programs":[]}',
                                 '{"programs":NaN}', '```json\n{}\n```',
                                 '{"programs":[]}', 'x'*32769])
def test_bad_envelope(text):
    with pytest.raises(ValueError):
        parse_proposals(load_dsl(), text)


@pytest.mark.parametrize('step', [
    {'op': '__import__', 'args': ['x0']},
    {'op': 'Head', 'args': ['x99']},
    {'op': 'Head', 'args': ['x1']},
    {'op': 'Take', 'args': ['x0', 'x2']},
    {'op': 'Map', 'lambda': '(<0)', 'args': ['x2']},
    {'op': 'Head', 'args': ['x2'], 'weight': 1},
])
def test_any_invalid_candidate_rejects_entire_batch(step):
    bad = candidate()
    bad['steps'][1] = step
    with pytest.raises(ValueError):
        parse_proposals(load_dsl(), json.dumps({'programs': [candidate(), bad]}))


@pytest.mark.parametrize('history', [[{'inputs': [[True], [1]], 'output': 1}],
                                    [{'inputs': [[1], [2]], 'output': float('nan')}],
                                    [{'inputs': [[1], [2]], 'output': 1, 'truth': 'hidden'}]])
def test_public_context_rejects_uncontrolled_fields(history):
    with pytest.raises(ValueError):
        messages(load_dsl(), history)
