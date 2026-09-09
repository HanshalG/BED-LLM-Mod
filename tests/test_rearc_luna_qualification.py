import json
from decimal import Decimal
import pytest
from scripts import rearc_luna_qualification as q
from scripts.rearc_qualification_panel import request_body


def endpoint():
    return {'pricing':{'prompt':'.0000002','completion':'.0000012',
        'input_cache_write':'.00000025','input_cache_read':'.00000002',
        'web_search':'.01','discount':0,'overrides':[{'min_prompt_tokens':272000}]}}


def test_applicable_prices_and_body():
    q.validate_route(endpoint())
    for patch in ({'request':'.01'}, {'prompt':'.0000003'}, {'overrides':[{'min_prompt_tokens':100}]}):
        e=endpoint(); e['pricing'].update(patch)
        with pytest.raises(ValueError): q.validate_route(e)
    case={'inputs':[[[0]]]*3,'outputs':[[[0]]]*3}
    body=request_body(case,'initial',0,'def identity(x: Grid) -> Grid: pass')
    q.validate_body(body)
    with pytest.raises(ValueError): q.validate_body({**body,'plugins':[{'id':'web'}]})


def test_source_channel_and_no_target_labels():
    from scripts.rearc_examples import Examples
    x=Examples(); x.tasks=['a','b','c','d']; calls=[]
    def one(task,seed,mode):
        calls.append((task,seed,mode))
        row={'input':[[0]],'output_sha256':'sealed'}
        if mode=='demonstration': row['output']=[[1]]
        return row
    x.one=one
    cases=x.public()
    assert len(calls)==44
    assert all(set(c)=={'inputs','outputs','target_inputs','target_hashes'} for c in cases)
    assert all(mode!='output' for _,_,mode in calls)


def test_seal_tampering_stops_target_loader(tmp_path):
    class Source:
        def targets(self,cases): raise AssertionError('label release')
    bank=q.Bank(tmp_path,Source())
    with pytest.raises(ValueError,match='not sealed'): bank.targets([])
    bank.seal([{'a':1}])
    (tmp_path/'forecasts.json').write_text('[]')
    with pytest.raises(ValueError,match='changed'): bank.targets([])


def test_cache_and_replay_do_not_execute_again(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(q,'execute_graph',lambda g,x:calls.append(x) or {'status':'ok','output':x})
    b=q.Bank(tmp_path)
    assert b.evaluate({},[[[0]],[[0]]])==[[[0]],[[0]]]
    assert len(calls)==1
    b=q.Bank(tmp_path,replay=True)
    assert b.evaluate({},[[[0]]])==[[[0]]]
    assert len(calls)==1


def test_dispatch_reserves_and_reauthorizes(tmp_path,monkeypatch):
    p=q.Probe.__new__(q.Probe)
    p.root=tmp_path; p.path=tmp_path/'ledger.json'; p.carry={}; p.ledger={}
    p.report={'calls':0}; p.accepted=Decimal(); p.base=Decimal(); p.reserve=q.RESERVE; p.cap=Decimal('.72')
    events=[]
    def account():
        events.append('reserved' if p.ledger.get('pending_reservations') else 'initial')
        p.ledger['recorded_actual_spend_usd']=0
    p.account=account
    monkeypatch.setattr(q,'route',endpoint)
    monkeypatch.setattr(q,'validate_body',lambda b:None)
    def execute(body):
        assert events==['initial','reserved']
        assert list(json.loads(p.path.read_text())['pending_reservations'].values())==[.06]
        raise TimeoutError('no response')
    monkeypatch.setattr(q,'execute',execute)
    with pytest.raises(TimeoutError): p.request('0_initial',{})
    p.__exit__(TimeoutError,TimeoutError(),None)
    assert p.report['uncertain_exposure_usd']==.06
    assert p.report['status']=='failed_closed'


def test_receipt_limit():
    raw={'model':'openai/gpt-5.6-luna','provider':'OpenAI','usage':{'prompt_tokens':6000,
        'completion_tokens':10000,'completion_tokens_details':{'reasoning_tokens':5000},'cost':.05},
        'choices':[{'finish_reason':'stop','message':{'content':'{}'}}]}
    assert q.response_text(raw)=='{}'
    raw['usage']['cost']=.061
    with pytest.raises(ValueError): q.response_text(raw)


@pytest.mark.parametrize('initial_failure',[False,True])
def test_exact_terminal_replay(tmp_path,monkeypatch,initial_failure):
    import hashlib
    import subprocess
    from scripts import rearc_graph_runtime as runtime
    dsl='def identity(x: Grid) -> Grid: pass'
    monkeypatch.setattr(runtime,'DSL_SHA',hashlib.sha256(dsl.encode()).hexdigest())
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:dsl.encode())
    graph={'steps':[{'id':'x0','op':'identity','args':['I']}],'output':'x0'}
    target_hash=hashlib.sha256(b'[[0]]').hexdigest()
    case={'inputs':[[[0]]]*3,'outputs':[[[0]]]*3,'target_inputs':[[[0]]]*8,'target_hashes':[target_hash]*8}
    q.save(tmp_path/'public.json',[case]*4)
    monkeypatch.setattr(q,'execute_graph',lambda g,x:{'status':'failed'} if initial_failure else {'status':'ok','output':x})
    monkeypatch.setattr(q,'execute_symbolic',lambda x,y:{'status':'ok','graphs':[graph]})
    class Source:
        def targets(self,cases): return [[[[0]]]*8 for _ in range(4)]
    bank=q.Bank(tmp_path,Source())
    calls=[]
    def request(tag,body):
        calls.append(tag)
        text=json.dumps({'hypotheses':[graph]*4})
        raw={'model':'openai/gpt-5.6-luna','provider':'OpenAI','usage':{'prompt_tokens':100,
            'completion_tokens':100,'completion_tokens_details':{'reasoning_tokens':10},'cost':.001},
            'choices':[{'finish_reason':'stop','message':{'content':text}}]}
        for kind,value in [('request',body),('response',raw),('route',endpoint())]:
            q.save(tmp_path/(tag+'.'+kind+'.json'),value)
        return text
    try:
        report=q.collect([case]*4,dsl,request,bank.evaluate,bank.symbolic,bank.seal,lambda:bank.targets([case]*4))
    except ValueError as exc:
        assert initial_failure
        report={'status':'failed_closed','panel_failure':{'type':'ValueError','message':str(exc)}}
    report.update(calls=len(calls),accepted_cost_usd=len(calls)*.001,implementation_sha256={},
        artifact_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob('*.json')})
    q.save(tmp_path/'result.json',report)
    assert q.replay(tmp_path)['status']=='exact_replay'
    (tmp_path/'public.json').write_text('[]')
    with pytest.raises(ValueError,match='artifact identity'): q.replay(tmp_path)
