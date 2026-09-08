import copy

import pytest

from environments.program_induction.constrained_request import request, advertised_route, MODEL, PROVIDER
from environments.program_induction.prediction import canonical
from scripts.deepcoder_opportunity import load_dsl


def endpoint():
    return dict(model_id=MODEL, tag=PROVIDER, status=0, context_length=1048576,
                max_completion_tokens=393216, supported_parameters=[
                    'structured_outputs', 'response_format', 'reasoning', 'seed', 'temperature', 'max_tokens'],
                pricing={'prompt': '.00000005', 'completion': '.00000016'})


def test_full_request_contract_and_blind_isolation():
    dsl = load_dsl()
    history = [{'inputs': [[1], [2]], 'output': 1}]
    other = [{'inputs': [[3], [4]], 'output': None}]
    a = request(dsl, history, 11)
    b = request(dsl, history, 11, history_blind=True)
    c = request(dsl, other, 11, history_blind=True)
    assert b == c
    assert a['messages'][0] == b['messages'][0]
    assert a['response_format'] == b['response_format']
    assert a['response_format']['type'] == 'json_schema'
    assert a['response_format']['json_schema']['strict'] is True
    assert a['provider']['only'] == [PROVIDER]
    assert not a['provider']['allow_fallbacks'] and not a['reasoning']['enabled']
    assert 'steps' not in a['messages'][0]['content']
    assert len(canonical(a).encode()) <= 32768
    assert set(a) == {'model', 'messages', 'seed', 'temperature', 'max_tokens', 'stream',
                      'reasoning', 'provider', 'response_format'}


def test_metadata_cannot_authorize_paid_or_claim_actual_schema_support():
    result = advertised_route(endpoint())
    assert result['metadata_eligible']
    assert not result['schema_serving_verified']
    assert not result['paid_authorized']


@pytest.mark.parametrize('kind', ['provider', 'schema', 'price', 'nan', 'context', 'output', 'status'])
def test_bad_route_rejected(kind):
    e = copy.deepcopy(endpoint())
    if kind == 'provider':
        e['tag'] = 'different'
    if kind == 'schema':
        e['supported_parameters'].remove('structured_outputs')
    if kind == 'price':
        e['pricing']['prompt'] = '.01'
    if kind == 'nan':
        e['pricing']['prompt'] = 'NaN'
    if kind == 'context':
        e['context_length'] = 100
    if kind == 'output':
        e['max_completion_tokens'] = 100
    if kind == 'status':
        e['status'] = 1
    with pytest.raises(ValueError):
        advertised_route(e)
