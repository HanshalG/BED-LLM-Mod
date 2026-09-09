import json
import pytest
from scripts.rearc_proposal_interface import build_messages, parse_response, response_format

DSL = 'Grid = tuple\ndef identity(grid: Grid) -> Grid:\n """Return the input grid."""\n return grid'
GRAPH = {'steps':[{'id':'x0','op':'identity','args':['I']}],'output':'x0'}


def test_public_payload_has_only_observed_outputs_and_generic_docs():
    inputs = [[[0]],[[1]]]
    messages = build_messages(inputs=inputs, observations=[{'index':0,'output':[[0]]}],dsl_source=DSL)
    payload = json.loads(messages[1]['content'])
    assert set(payload) == {'dsl','public_inputs','observations'}
    assert payload['observations'] == [{'index':0,'output':[[0]]}]
    assert 'return grid' not in payload['dsl']
    assert payload['public_inputs'] == inputs
    with pytest.raises(TypeError):
        build_messages(inputs=inputs,observations=[],dsl_source=DSL,task_id='private')


def test_response_count_and_no_survivor_filtering():
    assert len(parse_response(json.dumps({'hypotheses':[GRAPH]*4}),DSL)) == 4
    with pytest.raises(ValueError):
        parse_response(json.dumps({'hypotheses':[GRAPH]*3}),DSL)
    bad = {'steps':[{'id':'x0','op':'eval','args':['I']}],'output':'x0'}
    with pytest.raises(ValueError):
        parse_response(json.dumps({'hypotheses':[GRAPH]*3+[bad]}),DSL)


def test_strict_duplicate_keys_and_schema_count():
    with pytest.raises(ValueError,match='duplicate'):
        parse_response('{"hypotheses":[],"hypotheses":[]}',DSL)
    schema = response_format()['json_schema']['schema']
    assert schema['properties']['hypotheses']['minItems'] == 4
    assert schema['additionalProperties'] is False
