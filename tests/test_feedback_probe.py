from datetime import datetime, timedelta
import hashlib
import json
from zoneinfo import ZoneInfo
import pytest

from scripts import deepcoder_feedback_probe as g, paid_program_probe as paid


@pytest.fixture
def setup(tmp_path,monkeypatch):
    root,ledger=tmp_path/'run',tmp_path/'ledger.json'
    g.save(ledger,dict(date=datetime.now(ZoneInfo('Europe/London')).date().isoformat(),
        timezone='Europe/London',daily_cap_usd=5,opening_total_usage_usd=10,
        recorded_actual_spend_usd=.04,pending_reservations={'old:uncertain':.04}))
    monkeypatch.setattr(g,'ROOT',root)
    monkeypatch.setattr(paid,'route',lambda:{'fixture':True})
    public={str(i):dict(history=[dict(inputs=[[1],[2]],output=1)]*3,targets=[[[3],[4]]]*32) for i in range(8)}
    monkeypatch.setattr(g,'cases',lambda _:public)
    reads,posts=[],[]
    def live():
        reads.append(1)
        return dict(total_usage_usd=10,balance_usd=20)
    monkeypatch.setattr(paid,'read_live_credits',live)
    def execute(body):
        tag=json.loads((root/'result.json').read_text())['current']
        key,arm=tag.split('_')
        shown=json.loads(body['messages'][1]['content'])
        assert len(shown['history'])==3
        assert body['seed']==37100000+10*int(key)+(arm!='initial')
        assert not (root/'outcomes.json').exists()
        assert root.name+':'+tag in json.loads(ledger.read_text())['pending_reservations']
        if arm!='initial':
            assert shown['previous_programs'][0]['statement']=='x2 = Reverse x0'
            assert ('execution_feedback' in shown)==(arm=='feedback')
        parsed={'programs':[{'statement':'x2 = Reverse x0' if arm=='initial' else 'x2 = Sort x0',
            'next':{'statement':'x3 = Head x2','next':None}}]}
        posts.append(body)
        return dict(model='openai/gpt-5.6-luna',provider='OpenAI',usage=dict(cost=.001,
            prompt_tokens=1000,completion_tokens=100,completion_tokens_details={'reasoning_tokens':50}),
            choices=[dict(finish_reason='stop',message={'content':json.dumps(parsed)})])
    monkeypatch.setattr(paid,'execute',execute)
    def outcomes(dsl,cases):
        assert len(posts)==24 and (root/'forecasts.json').exists()
        return {k:dict(target_inputs=c['targets'],outputs=[3]*32) for k,c in cases.items()}
    monkeypatch.setattr(g,'outcomes',outcomes)
    return root,ledger,reads,posts


def test_full_matched_run(setup):
    root,ledger,reads,posts=setup
    r=g.run(ledger)
    assert r['status']=='feedback_screen_complete' and r['calls']==24 and len(reads)==49
    assert not r['feedback_gate'] and not r['depth_authorized']
    assert r['accepted_cost_usd']==.024 and r['uncertain_exposure_usd']==0
    assert json.loads(ledger.read_text())['recorded_actual_spend_usd']==.064
    for key in range(8):
        a=json.loads((root/f'{key}_feedback.request.json').read_text())
        b=json.loads((root/f'{key}_control.request.json').read_text())
        pa=json.loads(a['messages'][1]['content'])
        pb=json.loads(b['messages'][1]['content'])
        pa.pop('execution_feedback')
        assert pa==pb
        a['messages'][1]=b['messages'][1]
        assert a==b
    with pytest.raises(RuntimeError,match='already opened'):
        g.run(ledger)


def test_timeout_banks_uncertainty(setup,monkeypatch):
    root,ledger,_,posts=setup
    def fail(body):
        raise TimeoutError()
    monkeypatch.setattr(paid,'execute',fail)
    r=g.run(ledger)
    assert r['status']=='failed_closed' and r['calls']==1 and not posts
    assert r['uncertain_exposure_usd']==.04 and not r['endpoints_opened']
    assert json.loads(ledger.read_text())['recorded_actual_spend_usd']==.08


def test_budget_race(setup,monkeypatch):
    root,ledger,_,posts=setup
    reads=[]
    def live():
        reads.append(1)
        return dict(total_usage_usd=15 if len(reads)==3 else 10,balance_usd=20)
    monkeypatch.setattr(paid,'read_live_credits',live)
    r=g.run(ledger)
    assert r['status']=='failed_closed' and r['calls']==0 and not posts
    assert r['uncertain_exposure_usd']==0
    assert json.loads(ledger.read_text())['recorded_actual_spend_usd']==5


def test_rollover(setup,monkeypatch):
    root,ledger,reads,posts=setup
    status=paid.budget_status
    def shifted(ledger,**kwargs):
        if len(reads)>=3:
            kwargs['now']=datetime.now(ZoneInfo('Europe/London'))+timedelta(days=1)
        return status(ledger,**kwargs)
    monkeypatch.setattr(paid,'budget_status',shifted)
    r=g.run(ledger)
    assert r['status']=='failed_closed' and r['calls']==0 and not posts


def test_empty_support_not_dropped(setup,monkeypatch):
    root,ledger,_,posts=setup
    monkeypatch.setattr(g,'expand',lambda *args:([],{}))
    r=g.run(ledger)
    assert len(posts)==24 and r['mean_brier']=={'initial':1.,'feedback':1.,'control':1.}
    assert r['feedback_coverage']==0 and not r['feedback_gate']


def test_positive_gate_and_seal_guard(tmp_path):
    good=dict(distributions=[{'2':1.}]*32)
    bad=dict(distributions=[{'3':1.}]*32)
    panel={str(i):dict(case=dict(targets=[[[1],[2]]]*32),
        forecasts=dict(initial=bad,feedback=good,control=bad)) for i in range(8)}
    labels={k:dict(target_inputs=r['case']['targets'],outputs=[2]*32) for k,r in panel.items()}
    p=tmp_path/'panel.json'
    p.write_text(json.dumps(panel))
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    r=g.screen.score_sealed(p,sha,lambda:labels)
    assert r['feedback_gate'] and not r['scientific_pass'] and not r['depth_authorized']
    def bomb():
        raise AssertionError('outcomes opened')
    with pytest.raises(ValueError,match='seal'):
        g.screen.score_sealed(p,'wrong',bomb)


def test_schema_failure_is_terminal(setup,monkeypatch):
    root,ledger,_,_=setup
    execute=paid.execute
    def bad(body):
        r=execute(body)
        r['choices'][0]['message']['content']='{"programs":[]}'
        return r
    monkeypatch.setattr(paid,'execute',bad)
    r=g.run(ledger)
    assert r['calls']==1 and r['status']=='failed_closed' and r['accepted_cost_usd']==.001
    assert not (root/'outcomes.json').exists()
