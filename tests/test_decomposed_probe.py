from datetime import datetime
import hashlib
import json
from zoneinfo import ZoneInfo

import pytest

from scripts import deepcoder_decomposed_probe as g, paid_program_probe as paid


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root, ledger = tmp_path/'run', tmp_path/'ledger.json'
    protocol = tmp_path/'protocol.md'
    protocol.write_text('synthetic protocol')
    monkeypatch.setattr(g,'PROTOCOL',protocol)
    monkeypatch.setattr(g,'PROTOCOL_SHA',hashlib.sha256(protocol.read_bytes()).hexdigest())
    monkeypatch.setattr(g,'ROOT',root)
    g.save(ledger,dict(date=datetime.now(ZoneInfo('Europe/London')).date().isoformat(),
        timezone='Europe/London',daily_cap_usd=5,opening_total_usage_usd=10,
        recorded_actual_spend_usd=.04,pending_reservations={'old:uncertain':.04}))
    monkeypatch.setattr(paid,'route',lambda: {'fixture':True})
    reads,posts = [],[]
    def live():
        reads.append(1)
        return dict(total_usage_usd=10,balance_usd=20)
    monkeypatch.setattr(paid,'read_live_credits',live)
    public = {str(i):dict(history=[dict(inputs=[[1],[2]],output=1)]*3,
                          targets=[[[3],[4]]]*32) for i in range(8)}
    monkeypatch.setattr(g,'cases',lambda _:public)
    def execute(body):
        tag=json.loads((root/'result.json').read_text())['current']
        key,step,arm=tag.split('_')
        assert body['seed']==41100000+100*int(key)+int(step)
        assert not (root/'outcomes.json').exists()
        assert root.name+':'+tag in json.loads(ledger.read_text())['pending_reservations']
        shown=json.loads(body['messages'][1]['content'])
        assert len(shown['worked_examples'])==4
        if arm=='whole':
            parsed={'programs':[{'statement':'x2 = Reverse x0',
                                 'next':{'statement':'x3 = Head x2','next':None}}]}
        else:
            assert len(shown['branches'][0]['state']['choices'])==int(step)
            parsed={str(i):dict(choice=0,**({'subgoals':[None]*3} if arm=='subgoal' else {}))
                    for i in range(8)}
        posts.append(body)
        return dict(model='openai/gpt-5.6-luna',provider='OpenAI',usage=dict(cost=.001,
            prompt_tokens=1000,completion_tokens=100,completion_tokens_details={'reasoning_tokens':50}),
            choices=[dict(finish_reason='stop',message={'content':json.dumps(parsed)})])
    monkeypatch.setattr(paid,'execute',execute)
    def outcomes(dsl,cases):
        assert len(posts)==96 and (root/'forecasts.json').exists()
        return {k:dict(target_inputs=c['targets'],outputs=[3]*32) for k,c in cases.items()}
    monkeypatch.setattr(g,'outcomes',outcomes)
    return root,ledger,reads,posts


def test_full_run(setup):
    root,ledger,reads,posts=setup
    r=g.run(ledger)
    assert r['status']=='decomposed_screen_complete' and r['calls']==96
    assert len(reads)==193 and r['accepted_cost_usd']==.096
    assert json.loads(ledger.read_text())['recorded_actual_spend_usd']==.136
    assert not r['depth_authorized']
    panel=json.loads((root/'forecasts.json').read_text())
    for row in panel.values():
        assert len(row['construction'])==12
        for arm in ('subgoal','execution','whole'):
            assert set(row['supports']['short']) <= set(row['supports'][arm])
    with pytest.raises(RuntimeError,match='already opened'):
        g.run(ledger)


def test_independent_replay_and_tampering(setup,monkeypatch):
    from scripts.deepcoder_decomposed_replay import replay
    root,ledger,_,_=setup
    g.run(ledger)
    assert replay(root)['replay_valid']
    path=root/'0_0_subgoal.request.json'
    body=json.loads(path.read_text())
    body['seed']+=1
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='request mismatch'):
        replay(root)


def test_nonterminal_replay_cannot_load_source(tmp_path,monkeypatch):
    from scripts import deepcoder_decomposed_replay as r
    (tmp_path/'result.json').write_text('{"status":"failed_closed"}')
    monkeypatch.setattr(r,'load_dsl',lambda:pytest.fail('source opened before status check'))
    with pytest.raises(ValueError,match='completed screen'):
        r.replay(tmp_path)


@pytest.mark.parametrize('failure',['race','timeout','schema'])
def test_failure_closes_before_endpoints(setup,monkeypatch,failure):
    root,ledger,reads,posts=setup
    original=paid.execute
    if failure=='race':
        def live():
            reads.append(1)
            return dict(total_usage_usd=15 if len(reads)==3 else 10,balance_usd=20)
        monkeypatch.setattr(paid,'read_live_credits',live)
    else:
        def bad(body):
            if failure=='timeout':
                raise TimeoutError()
            raw=original(body)
            raw['choices'][0]['message']['content']='{}'
            return raw
        monkeypatch.setattr(paid,'execute',bad)
    r=g.run(ledger)
    assert r['status']=='failed_closed' and not r['endpoints_opened']
    assert not (root/'forecasts.json').exists()
    assert r['calls']==(0 if failure=='race' else 1)
    assert r['uncertain_exposure_usd']==(.04 if failure=='timeout' else 0)


def test_seal_and_candidate_gates(tmp_path):
    good=dict(distributions=[{'2':1.}]*32)
    bad=dict(distributions=[{'3':1.}]*32)
    panel={str(i):dict(case=dict(history=[dict(inputs=[[1],[2]],output=1)]*3,
        targets=[[[1],[2]]]*32),forecasts=dict(subgoal=good,execution=bad,whole=bad,short=bad))
           for i in range(8)}
    p=tmp_path/'panel.json'
    p.write_text(json.dumps(panel))
    labels={k:dict(target_inputs=r['case']['targets'],outputs=[2]*32) for k,r in panel.items()}
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    r=g.screen.score_sealed(p,sha,lambda:labels)
    assert r['candidate_gates']==dict(subgoal=True,execution=False)
    assert not r['scientific_pass']
    with pytest.raises(ValueError,match='seal'):
        g.screen.score_sealed(p,'bad',lambda:pytest.fail('opened labels'))
    panel.pop('7')
    p.write_text(json.dumps(panel))
    with pytest.raises(ValueError,match='eight-case'):
        g.screen.score_sealed(p,hashlib.sha256(p.read_bytes()).hexdigest(),lambda:pytest.fail('opened labels'))
