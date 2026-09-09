import json

import pytest

from environments.program_induction.decomposed_request import request, decode
from environments.program_induction.execution_steps import next_menu
from scripts.deepcoder_luna_medium_probe import bumped
from scripts.deepcoder_opportunity import load_dsl


H = [dict(inputs=[[4, 1], [2]], output=[1])]


def test_paired_public_state_and_actual_replacement():
    d = load_dsl()
    paths = [[] for _ in range(8)]
    a = request(d, H, paths, 123, subgoals=True)
    b = request(d, H, paths, 123, subgoals=False)
    assert a['messages'][1] == b['messages'][1]
    assert a['seed'] == b['seed']
    c = next(x['choice'] for x in next_menu(d, []) if x['statement'] == 'x2 = Last x0')
    response = {str(i): dict(choice=c, subgoals=[[-9]]) for i in range(8)}
    new, audit = decode(d, json.dumps(response), H, paths, subgoals=True)
    assert audit['subgoal_disagreements'] == [1]*8
    follow = request(d, H, new, 124, subgoals=True)
    public = json.loads(follow['messages'][1]['content'])
    assert public['branches'][0]['state']['rows'][0]['values']['x2'] == 1
    assert paths == [[] for _ in range(8)]
    assert bumped(a)['reasoning']['effort'] == 'medium'


def test_whole_batch_rejection_and_no_hidden_keys():
    d = load_dsl()
    paths = [[] for _ in range(8)]
    response = {str(i): dict(choice=0) for i in range(8)}
    response['7']['choice'] = -1
    with pytest.raises(Exception):
        decode(d, json.dumps(response), H, paths, subgoals=False)
    assert paths == [[] for _ in range(8)]
    with pytest.raises(ValueError):
        request(d, [dict(H[0], truth='forbidden')], paths, 0, subgoals=False)
    with pytest.raises(ValueError):
        decode(d, '{"0":{},"0":{}}', H, paths, subgoals=False)


def test_mixed_types_four_steps_and_body_cap():
    d = load_dsl()
    paths = [[] for _ in range(8)]
    for step in range(4):
        body = request(d, H*3, paths, step, subgoals=True)
        assert len(json.dumps(bumped(body), separators=(',', ':')).encode()) <= 32768
        response = {str(i): dict(choice=(i*17+step)%len(next_menu(d,p)), subgoals=[None]*3)
                    for i,p in enumerate(paths)}
        paths, _ = decode(d, json.dumps(response), H*3, paths, subgoals=True)
    with pytest.raises(ValueError):
        request(d, H, paths, 4, subgoals=True)
