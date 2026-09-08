from datetime import datetime, timedelta
import hashlib
import json
from zoneinfo import ZoneInfo

import pytest

from scripts import deepcoder_independent_joint as g


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root, ledger = tmp_path/'run', tmp_path/'ledger.json'
    g.save(ledger,dict(date=datetime.now(ZoneInfo('Europe/London')).date().isoformat(),
        timezone='Europe/London',daily_cap_usd=5,opening_total_usage_usd=10,
        recorded_actual_spend_usd=.04,pending_reservations={'old:uncertain':.04}))
    monkeypatch.setattr(g,'ROOT',root)
    monkeypatch.setattr(g,'route',lambda:{'fixture':True})
    public = {str(i):dict(history=[dict(inputs=[[1],[2]],output=1)]*2,
        query=[[1],[2]],targets=[[[3],[4]]]*32) for i in range(8)}
    monkeypatch.setattr(g,'cases',lambda _:public)
    reads, posts = [], []
    def credits():
        reads.append(1)
        return dict(total_usage_usd=10,balance_usd=20)
    monkeypatch.setattr(g,'read_live_credits',credits)
    def observe(dsl,key,x):
        path = root/(key+'.preanswer.json')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == json.loads(
            (root/(key+'.preanswer.seal.json')).read_text())['sha256']
        assert not (root/'outcomes.json').exists()
        return dict(inputs=x,output=1)
    monkeypatch.setattr(g,'observe',observe)
    def execute(body):
        tag = json.loads((root/'result.json').read_text())['current']
        i, arm = tag.split('_',1)
        history = json.loads(body['messages'][1]['content'])['history']
        assert len(history) == (3 if arm=='regenerated' else 2)
        assert body['seed'] == 34100000+10*int(i)+({'a':0,'b':1}.get(arm,2))
        assert json.loads(ledger.read_text())['pending_reservations']['independent_joint:'+tag] == .04
        assert not (root/'outcomes.json').exists()
        posts.append(body)
        parsed = {'programs':[{'statement':'x2 = Reverse x0','next':{'statement':'x3 = Head x2','next':None}}]}
        return dict(model='openai/gpt-5.6-luna',provider='OpenAI',usage=dict(cost=.001,
            prompt_tokens=1000,completion_tokens=100,completion_tokens_details={'reasoning_tokens':50}),
            choices=[dict(finish_reason='stop',message={'content':json.dumps(parsed)})])
    monkeypatch.setattr(g,'execute',execute)
    def outcomes(dsl,cases):
        assert len(posts)==32 and (root/'forecasts.json').exists()
        return {k:dict(target_inputs=c['targets'],outputs=[3]*32) for k,c in cases.items()}
    monkeypatch.setattr(g,'outcomes',outcomes)
    return root,ledger,reads,posts


def test_full_run_seals_and_null_gate(setup):
    root,ledger,reads,posts=setup
    r=g.run(ledger)
    assert r['status']=='independent_joint_complete'
    assert r['calls']==len(posts)==32 and len(reads)==65
    assert r['accepted_cost_usd']==.032 and r['uncertain_exposure_usd']==0
    assert not r['updater_ranking_gate'] and not r['multi_query_screen_allowed']
    stored=json.loads(ledger.read_text())
    assert stored['recorded_actual_spend_usd']==.072
    assert stored['pending_reservations']=={'old:uncertain':.04}
    with pytest.raises(RuntimeError,match='already opened'):
        g.run(ledger)


def test_transport_failure_retains_exposure(setup,monkeypatch):
    root,ledger,_,posts=setup
    def fail(body):
        raise TimeoutError()
    monkeypatch.setattr(g,'execute',fail)
    r=g.run(ledger)
    assert r['calls']==1 and not posts and r['status']=='failed_closed'
    assert r['uncertain_exposure_usd']==.04 and not r['endpoints_opened']
    assert json.loads(ledger.read_text())['recorded_actual_spend_usd']==.08


def test_dispatch_race(setup,monkeypatch):
    root,ledger,_,posts=setup
    reads=[]
    def live():
        reads.append(1)
        return dict(total_usage_usd=15 if len(reads)==3 else 10,balance_usd=20)
    monkeypatch.setattr(g,'read_live_credits',live)
    r=g.run(ledger)
    assert r['status']=='failed_closed' and r['calls']==0 and not posts
    assert r['uncertain_exposure_usd']==0
    assert json.loads(ledger.read_text())['recorded_actual_spend_usd']==5


def test_day_rollover_stops_before_dispatch(setup,monkeypatch):
    root,ledger,reads,posts=setup
    status=g.budget_status
    def shifted(ledger,**kwargs):
        if len(reads)>=3:
            kwargs['now']=datetime.now(ZoneInfo('Europe/London'))+timedelta(days=1)
        return status(ledger,**kwargs)
    monkeypatch.setattr(g,'budget_status',shifted)
    r=g.run(ledger)
    assert r['status']=='failed_closed' and r['calls']==0 and not posts


def test_schema_failure_stops_without_endpoints(setup,monkeypatch):
    root,ledger,_,_=setup
    execute=g.execute
    def bad(body):
        raw=execute(body)
        raw['choices'][0]['message']['content']='{"programs":[]}'
        return raw
    monkeypatch.setattr(g,'execute',bad)
    r=g.run(ledger)
    assert r['status']=='failed_closed' and r['calls']==1
    assert r['accepted_cost_usd']==.001 and not (root/'outcomes.json').exists()


def test_empty_support_is_retained_as_null(setup,monkeypatch):
    root,ledger,_,posts=setup
    monkeypatch.setattr(g,'expand',lambda *args:([],{'fixture_empty':True}))
    r=g.run(ledger)
    assert len(posts)==32 and r['status']=='independent_joint_complete'
    assert not r['joint_gate'] and not r['updater_ranking_gate']
    assert r['mean_conditional_brier']=={'a':1.,'insertion':1.,'ab':1.}
