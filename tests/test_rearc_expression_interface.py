import json
import pytest
from scripts.rearc_expression_interface import build_messages, parse_response, response_format
from scripts.rearc_expression_cohort import select

DSL = 'def identity(x: Any) -> Any:\n return x\ndef vmirror(x: Any) -> Any:\n return x\n'


def test_public_inputs_do_not_imply_output_access():
    messages = build_messages(inputs=[[[0]], [[1]]], observations=[{'index': 0, 'output': [[0]]}], dsl_source=DSL)
    content = json.loads(messages[1]['content'])
    assert content['observations'] == [{'index': 0, 'output': [[0]]}]
    assert len(content['public_inputs']) == 2
    assert 'ONE input grid' in messages[0]['content']
    assert response_format()['json_schema']['strict']


def test_whole_batch_rejection():
    good = json.dumps({'hypotheses': ['identity(I)']*4})
    assert len(parse_response(good, DSL)['graphs']) == 4
    with pytest.raises(ValueError):
        parse_response(json.dumps({'hypotheses': ['identity(I)']*3+['eval(I)']}), DSL)
    with pytest.raises(ValueError, match='duplicate'):
        parse_response('{"hypotheses":[],"hypotheses":[]}', DSL)


def test_cohort_metadata_order_and_exclusion():
    inventory = [f'{i:08x}' for i in range(30)]
    selected = select(inventory, inventory[:12])
    assert selected == select(list(reversed(inventory)), inventory[:12])
    assert len(selected) == 4 and not set(selected) & set(inventory[:12])
