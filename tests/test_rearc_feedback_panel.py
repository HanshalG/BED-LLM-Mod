import hashlib
import json
import pytest
from scripts import rearc_feedback_run as run
from scripts.rearc_feedback_panel import collect
from scripts import rearc_luna_qualification as old

DSL='def identity(x: Grid) -> Grid: pass'
G={'steps':[{'id':'x0','op':'identity','args':['I']}],'output':'x0'}
TEXT=json.dumps({'hypotheses':[G]*4})


@pytest.mark.parametrize('invalid',[False,True])
def test_full_disk_bank_replay_or_initial_null(tmp_path,monkeypatch,invalid):
    monkeypatch.setattr(run,'DSL_SHA',hashlib.sha256(DSL.encode()).hexdigest())
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**kw:DSL.encode())
    monkeypatch.setattr(old,'execute_graph',lambda g,x:{'status':'ok','output':x})
    monkeypatch.setattr(old,'execute_symbolic',lambda x,y:{'status':'ok','graphs':[G]})
    monkeypatch.setattr(run,'diagnose',lambda g,x:{'status':'ok','output':x})
    case={'inputs':[[[0]]]*3,'outputs':[[[0]]]*3,'target_inputs':[[[0]]]*8,
          'target_hashes':[hashlib.sha256(b'[[0]]').hexdigest()]*8}
    class Source:
        def targets(self,cases):
            assert not invalid
            assert (tmp_path/'forecast_seal.json').exists()
            return [[[[0]]]*8 for _ in range(8)]
    bank=run.FeedbackBank(tmp_path,Source())
    cases=bank.saved('public',lambda:[case]*8)
    calls=[]
    endpoint={'pricing':{'prompt':'.0000002','completion':'.0000012',
                        'input_cache_write':'.00000025','input_cache_read':'.00000002'}}
    def request(tag,body):
        calls.append(tag)
        raw={'model':'openai/gpt-5.6-luna','provider':'OpenAI','usage':{'prompt_tokens':100,
             'completion_tokens':100,'completion_tokens_details':{'reasoning_tokens':20},'cost':.001},
             'choices':[{'finish_reason':'stop','message':{'content':'{}' if invalid else TEXT}}]}
        for key,value in [('request',body),('response',raw),('route',endpoint)]:
            run.save(tmp_path/(tag+'.'+key+'.json'),value)
        return run.response_text(raw)
    result=collect(cases,DSL,request,bank.evaluate,bank.diagnose,bank.symbolic,
                   bank.update,bank.seal,lambda:bank.targets(cases))
    assert len(calls)==(16 if invalid else 48)
    assert result['status']==('initial_coverage_null' if invalid else 'complete')
    assert result['qualification_passed'] is False
    result.update(calls=len(calls),accepted_cost_usd=len(calls)*.001,implementation_sha256={},
                  artifact_sha256=run.artifacts(tmp_path))
    run.save(tmp_path/'result.json',result)
    assert run.replay(tmp_path)['calls']==len(calls)
    if not invalid:
        aware=json.loads((tmp_path/'0_aware_repair.request.json').read_text())
        blind=json.loads((tmp_path/'0_blind_repair.request.json').read_text())
        assert aware['seed']==blind['seed']==33401
    (tmp_path/'public.json').write_text('[]')
    with pytest.raises(ValueError,match='artifact'): run.replay(tmp_path)


def test_collector_uses_only_new_seeds_and_input_channel():
    from scripts.rearc_feedback_examples import FeedbackExamples
    x=FeedbackExamples(); x.tasks=list(range(8)); calls=[]
    def one(task,seed,mode):
        calls.append((task,seed,mode))
        return {'input':[[0]],'output':[[0]],'output_sha256':'hash'}
    x.one=one
    cases=x.public()
    assert len(cases)==8 and len(calls)==88
    assert {s for _,s,m in calls if m=='demonstration'}=={33100,33101,33102}
    assert {s for _,s,m in calls if m=='input'}==set(range(33200,33208))
    assert all(m!='output' for _,_,m in calls)


def test_48_request_cap_prevents_dispatch(tmp_path,monkeypatch):
    from decimal import Decimal
    p=run.FeedbackProbe.__new__(run.FeedbackProbe)
    p.root=tmp_path; p.accepted=Decimal(0); p.reserve=Decimal('.06'); p.cap=Decimal('2.88')
    p.report={'calls':48}
    monkeypatch.setattr(run,'validate_body',lambda b:None)
    monkeypatch.setattr(run,'validate_route',lambda e:None)
    monkeypatch.setattr(run,'route',lambda:{})
    monkeypatch.setattr(run,'execute',lambda b:pytest.fail('paid dispatch'))
    with pytest.raises(RuntimeError,match='block cap'): p.request('extra',{})
