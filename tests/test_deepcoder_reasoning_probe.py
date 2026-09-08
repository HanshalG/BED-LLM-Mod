import copy
from decimal import Decimal

import pytest

from scripts import deepcoder_reasoning_probe as g


def test_only_reasoning_and_completion_ceiling_change():
    body = dict(reasoning={'enabled':False,'exclude':True},max_tokens=4096,
                messages=[{'role':'user','content':'fixed'}],seed=123,response_format={'schema':'fixed'})
    old = copy.deepcopy(body)
    new = g.bumped(body)
    assert body == old
    assert {k for k in new if new[k]!=old[k]} == {'reasoning','max_tokens'}
    assert new['reasoning'] == dict(enabled=True,effort='high',exclude=True)
    assert new['max_tokens'] == 16384
    assert Decimal(65536)*Decimal('.0000001')+Decimal(16384)*Decimal('.0000002') < g.RESERVE
    assert 16*g.RESERVE < g.CAP


def raw():
    return dict(model=g.cr.MODEL,provider='OpenInference',usage=dict(cost=.003,prompt_tokens=10000,
        completion_tokens=2000,completion_tokens_details={'reasoning_tokens':1500}),
        choices=[dict(finish_reason='stop',message={'content':'{}'})])


def test_reasoning_usage_required():
    assert g.validate_response(raw()) == '{}'
    for reasoning in [0,-1,True,2001,None]:
        r=raw()
        r['usage']['completion_tokens_details']['reasoning_tokens']=reasoning
        with pytest.raises(ValueError):
            g.validate_response(r)


def test_truncation_and_cost_fail():
    r=raw()
    r['choices'][0]['finish_reason']='length'
    with pytest.raises(ValueError):
        g.validate_response(r)


def test_reasoning_exhausts_entire_completion_budget():
    r = raw()
    r['usage']['completion_tokens'] = 16384
    r['usage']['completion_tokens_details']['reasoning_tokens'] = 16384
    r['choices'][0]['finish_reason'] = 'length'
    r['choices'][0]['message']['content'] = None
    with pytest.raises(ValueError, match='incomplete response'):
        g.validate_response(r)


def test_full_paired_probe_without_target_access(tmp_path,monkeypatch):
    import json
    import hashlib
    from datetime import datetime
    from zoneinfo import ZoneInfo
    parent,root=tmp_path/'parent',tmp_path/'run'
    parent.mkdir()
    ledger,protocol=tmp_path/'ledger.json',tmp_path/'protocol.md'
    protocol.write_text('fixture')
    case=dict(history=[dict(inputs=[[1],[2]],output=1)]*2,targets=[[[3],[4]]]*32)
    panel=dict(version=1,cases={str(i):case for i in range(8)},
               forecasts={str(i):{a:None for a in g.pred.ARMS} for i in range(8)})
    g.save(parent/'forecasts.json',panel,exclusive=True)
    dsl=g.load_dsl()
    for i in range(8):
        for arm in g.pred.ARMS[:2]:
            g.save(parent/f'{i}_{arm}.request.json',g.cr.request(dsl,case['history'],14100000+i,history_blind=arm=='history_blind'))
    for name,value in [('ROOT',root),('PARENT',parent),('LEDGER',ledger),('PROTOCOL',protocol),
                       ('PARENT_SHA',hashlib.sha256((parent/'forecasts.json').read_bytes()).hexdigest())]:
        monkeypatch.setattr(g,name,value)
    g.save(ledger,dict(date=datetime.now(ZoneInfo('Europe/London')).date().isoformat(),
                      timezone='Europe/London',daily_cap_usd=5,opening_total_usage_usd=10,
                      recorded_actual_spend_usd=0,high_reasoning_authorization=dict(approved=True,consumed=False,cap_usd=.25)))
    monkeypatch.setattr(g,'read_live_credits',lambda:dict(total_usage_usd=10,balance_usd=20))
    monkeypatch.setattr(g,'route',lambda:{})
    calls=[]
    def post(body):
        assert json.loads(ledger.read_text())['pending_reservations']
        assert body['reasoning']['effort']=='high' and body['max_tokens']==16384
        calls.append(body)
        r=raw()
        r['choices'][0]['message']['content']=json.dumps({'programs':[{'statement':'x2 = Reverse x0','next':{'statement':'x3 = Head x2','next':None}}]})
        return r
    monkeypatch.setattr(g,'post',post)
    result=g.run()
    assert result['status']=='paired_observed_history_diagnostic_complete'
    assert len(calls)==result['calls']==16
    assert result['coverage']=={'history_aware':8,'history_blind':8}
    assert result['accepted_cost_usd']==.048
    assert not result['endpoints_opened'] and not (root/'outcomes.json').exists()
    assert json.loads(ledger.read_text())['recorded_actual_spend_usd']==.048
    with pytest.raises(RuntimeError,match='already opened'):
        g.run()
    r=raw()
    r['usage']['cost']=.016
    with pytest.raises(ValueError):
        g.validate_response(r)
