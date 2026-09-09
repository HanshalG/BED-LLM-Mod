import json
import pytest
from scripts import physgym_semantic_probe as probe


def public():
    return {t:dict(names=['a'],descriptions={'a':'scale'},context='A scalar experiment.',
                   history=[[{'a':1.},0.]]*4,targets=[{'a':2.}]*32) for t in probe.IDS}


def response():
    return dict(model='openai/gpt-5.6-luna',provider='OpenAI',usage=dict(cost=.001,prompt_tokens=100,
                completion_tokens=30,completion_tokens_details={'reasoning_tokens':20}),
                choices=[dict(finish_reason='stop',message={'content':'{"expressions":["1"]}'})])


def test_full_sixteen_call_seal_and_equal_update(tmp_path):
    class Block:
        root=tmp_path
        report={'status':'incomplete'}
        tags=[]
        def request(self,tag,body):
            self.tags.append(tag)
            return response()
    block=Block()
    def endpoint():
        assert len(block.tags)==16
        assert (tmp_path/'forecasts.json').exists()
        return {t:[0.]*32 for t in probe.IDS}
    probe.collect(block,public(),endpoint)
    assert not block.report['gate_passed']
    assert len(set(block.tags))==16
    p=json.loads((tmp_path/'forecasts.json').read_text())
    assert all(p[t]['forecasts']['refresh']==p[t]['forecasts']['redraw'] for t in probe.IDS)


def test_requests_hide_new_observation_only_from_redraw_proposer():
    case=public()[probe.IDS[0]]
    a=probe.body(case,'refresh',0)
    b=probe.body(case,'redraw',0)
    assert a['seed']==b['seed']
    assert len(json.loads(a['messages'][1]['content'])['history'])==4
    assert len(json.loads(b['messages'][1]['content'])['history'])==3


def test_schema_failure_never_opens_endpoints(tmp_path):
    class Block:
        root=tmp_path
        report={}
        def request(self,*args):
            r=response()
            r['choices'][0]['message']['content']='{"bad":1}'
            return r
    with pytest.raises(ValueError):
        probe.collect(Block(),public(),lambda:pytest.fail('endpoint opened'))
    assert not (tmp_path/'forecasts.json').exists()


def test_score_pass_null_and_empty():
    truth={t:[0.]*32 for t in probe.IDS}
    forecasts={t:{'forecasts':{a:dict(status='complete',mean=[v]*32)
                  for a,v in dict(semantic=.1,blind=1,refresh=.1,redraw=1,symbolic=.2).items()}}
                for t in probe.IDS}
    assert probe.score(forecasts,truth)['gate_passed']
    forecasts[probe.IDS[0]]['forecasts']['refresh']=dict(status='empty_support',mean=None)
    r=probe.score(forecasts,truth)
    assert not r['gate_passed'] and r['reason']=='empty_support'


def test_full_replay_rejects_request_tamper(tmp_path,monkeypatch):
    worlds={t:dict(equation='1',input_variables={'a':'scale'},dummy_variables={},content='public') for t in probe.IDS}
    monkeypatch.setattr(probe,'source',lambda:worlds)
    class Block:
        root=tmp_path
        report=dict(status='incomplete',protocol_sha256=probe.PROTOCOL_SHA,
                    implementation_sha256={},calls=0,accepted_cost_usd=.016)
        def request(self,tag,body):
            self.report['calls']+=1
            raw=response()
            (self.root/(tag+'.request.json')).write_text(json.dumps(body))
            (self.root/(tag+'.response.json')).write_text(json.dumps(raw))
            return raw
    block=Block()
    cases=probe.cases(worlds)
    probe.collect(block,cases,lambda:probe.outcomes(worlds,cases))
    (tmp_path/'result.json').write_text(json.dumps(block.report))
    assert probe.replay(tmp_path)['status']=='replay_valid'
    p=tmp_path/(probe.IDS[0]+'_semantic.request.json')
    data=json.loads(p.read_text())
    data['seed']+=1
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='request mismatch'):
        probe.replay(tmp_path)
