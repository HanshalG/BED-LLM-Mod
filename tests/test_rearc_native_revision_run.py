import json
from pathlib import Path
from decimal import Decimal
import pytest
from scripts import rearc_native_revision_run as run
from scripts.rearc_luna_qualification import digest
from scripts.rearc_named_plan_contract import body
from scripts.rearc_named_plan_interface import schema


def test_dependency_mismatch_stops_before_preflight(monkeypatch):
    monkeypatch.setattr(run, 'version', lambda package: 'wrong')
    with pytest.raises(ValueError, match='dependency version mismatch'):
        run.run()


@pytest.mark.parametrize('failure',[None,'first_authorization','second_authorization','http'])
def test_reserve_reauthorize_dispatch_reconcile(tmp_path,monkeypatch,failure):
    probe = object.__new__(run.RevisionProbe)
    probe.root,probe.path = tmp_path,tmp_path/'ledger.json'
    probe.cap,probe.reserve = Decimal('2.88'),Decimal('.08')
    probe.base=probe.accepted=probe.uncertain=Decimal()
    probe.carry={}
    probe.pending=probe.dispatched=False
    probe.ledger={'recorded_actual_spend_usd':0.,'pending_reservations':{}}
    probe.report={'status':'incomplete','calls':0,'endpoints_opened':False}
    calls=[]
    endpoint=json.loads(Path('results/nonmyopic/rearc_slot_qualification_20260909/route.json').read_text())
    monkeypatch.setattr(run,'route',lambda:endpoint)
    def account():
        calls.append('account')
        n=calls.count('account')
        if n==2:
            assert list(probe.ledger['pending_reservations'].values())==[.08]
        if failure==('first_authorization' if n==1 else 'second_authorization'):
            raise RuntimeError('budget closed')
    probe.account=account
    def execute(value):
        calls.append('http')
        assert calls==['account','account','http']
        assert list(json.loads(probe.path.read_text())['pending_reservations'].values())==[.08]
        if failure=='http':
            raise RuntimeError('uncertain HTTP')
        return {'model':'openai/gpt-5.6-luna','provider':'OpenAI',
            'usage':{'cost':.001,'prompt_tokens':1,'completion_tokens':2,
                     'completion_tokens_details':{'reasoning_tokens':1}},
            'choices':[{'finish_reason':'stop','message':{'content':'ok'}}]}
    monkeypatch.setattr(run,'execute',execute)
    with probe:
        probe.request('test',body([],43000,schema('plan')))
        probe.report['status']='complete'
    assert calls.count('http')==(failure in (None,'http'))
    assert probe.report['uncertain_exposure_usd']==(.08 if failure=='http' else 0.)
    assert probe.report['accepted_cost_usd']==(.001 if failure is None else 0.)
    assert probe.ledger['recorded_actual_spend_usd']==(.08 if failure=='http' else .001 if failure is None else 0.)


def test_call_cap_and_invalid_schema_before_route(tmp_path,monkeypatch):
    probe=object.__new__(run.RevisionProbe)
    probe.report={'calls':36}
    probe.accepted=Decimal()
    probe.cap,probe.reserve=Decimal('2.88'),Decimal('.08')
    monkeypatch.setattr(run,'route',lambda:pytest.fail('catalog after cap'))
    with pytest.raises(RuntimeError,match='cap'):
        probe.request('x',body([],43000,schema('plan')))
    probe.report['calls']=0
    with pytest.raises(ValueError):
        probe.request('x',{'response_format':{}})

@pytest.mark.parametrize('mode', ['complete','initial_null','interrupted','interrupted_after_reveal'])
def test_terminal_response_replay_without_calls(tmp_path,monkeypatch,mode):
    import hashlib
    dsl = 'def identity(x: Any) -> Any:\n return x\n'
    cases = [{'inputs': [[[0]]], 'outputs': [[[0]]], 'query_inputs': [[[3]]]*2,
              'reveal_input':[[2]],'reveal_hash':hashlib.sha256(b'[[2]]').hexdigest(),
              'target_inputs': [[[3]]]*7,
              'target_hashes': [hashlib.sha256(b'[[3]]').hexdigest()]*9} for _ in range(4)]
    g = {'steps':[{'id':'x0','op':'identity','args':['I']}],'output':'x0'}
    class Source:
        def reveal(self,index):
            return [[2]]
        def targets(self,cases):
            return [[[[3]]]*9 for _ in cases]
    class SyntheticBank(run.RevisionBank):
        def evaluate(self,graph,inputs):
            return self.saved('e_'+digest([graph,inputs]),lambda:
                [None]*len(inputs) if mode=='initial_null' else inputs)
        def diagnose(self,graph,x):
            return self.saved('d_'+digest([graph,x]),lambda: {'status':'ok','output':x})
        evaluate_python = evaluate
        diagnose_python = diagnose
    # Repeated deterministic calls share saved records as the real bank does.
    original_saved = SyntheticBank.saved
    def cached(self,name,build):
        if name not in self.cache:
            self.cache[name] = original_saved(self,name,build)
        return self.cache[name]
    monkeypatch.setattr(SyntheticBank,'saved',cached)
    bank = SyntheticBank(tmp_path,Source())
    bank.saved('public',lambda:cases)
    route = json.loads(Path('results/nonmyopic/rearc_expression_qualification_20260909/route.json').read_text())
    report = {'calls':0,'accepted_cost_usd':0.,'uncertain_exposure_usd':0.,
              'endpoints_opened':False,'implementation_sha256':{}}
    def request(tag,body):
        report.update(calls=report['calls']+1,current=tag,phase='http_attempt')
        run.save(tmp_path/(tag+'.request.json'),body,exclusive=True)
        run.save(tmp_path/(tag+'.route.json'),route,exclusive=True)
        if (mode=='interrupted' and report['calls']==3) or (mode=='interrupted_after_reveal' and report['calls']==4):
            report['uncertain_exposure_usd']=.08
            raise RuntimeError('synthetic interrupted HTTP')
        raw = {'model':'openai/gpt-5.6-luna','provider':'OpenAI',
               'usage':{'cost':0.,'prompt_tokens':1,'completion_tokens':2,
                        'completion_tokens_details':{'reasoning_tokens':1}},
               'choices':[{'finish_reason':'stop','message':{'content':json.dumps({f'p{i}':'Rule' for i in range(4)} if tag.endswith('_plan') else {'hypotheses':['def transform(g): return g']*8})}}]}
        run.save(tmp_path/(tag+'.response.json'),raw,exclusive=True)
        report['phase']='feedback'
        return run.response_text(raw)
    def targets():
        report['endpoints_opened']=True
        return bank.targets(cases)
    try:
        report.update(run.collect(cases,dsl,request,bank.evaluate_python,bank.diagnose_python,lambda i:bank.reveal(i,cases),
            bank.update,bank.seal,targets))
    except RuntimeError as error:
        report.update(status='failed_closed',panel_failure={'type':'RuntimeError','message':str(error)})
    report['artifact_sha256']=run.artifacts(tmp_path)
    run.save(tmp_path/'result.json',report)
    assert len(list(tmp_path.glob('reveal_*.json')))==(0 if mode=='interrupted' else 1 if mode=='interrupted_after_reveal' else 4)
    monkeypatch.setattr(run,'RevisionBank',SyntheticBank)
    monkeypatch.setattr(run,'DSL_SHA',hashlib.sha256(dsl.encode()).hexdigest())
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**k:dsl.encode())
    monkeypatch.setattr(run,'route',lambda:pytest.fail('catalog request during replay'))
    assert run.replay(tmp_path) == {'status':'exact_replay',
        'calls':{'complete':36,'initial_null':36,'interrupted':3,'interrupted_after_reveal':4}[mode],
        'new_calls':0,'depth_authorized':False}


def test_native_execution_cache_and_offline_replay(tmp_path,monkeypatch):
    calls=[]
    def execute(code,x):
        calls.append((code,x))
        return {'status':'ok','output':x}
    monkeypatch.setattr(run,'execute_python',execute)
    bank=run.RevisionBank(tmp_path)
    code='def transform(g): return g'
    assert bank.diagnose_python(code,[[1]])['output']==[[1]]
    assert bank.evaluate_python(code,[[[1]],[[1]]])==[[[1]],[[1]]]
    assert len(calls)==1
    monkeypatch.setattr(run,'execute_python',lambda *a:pytest.fail('execution during replay'))
    replay=run.RevisionBank(tmp_path,replay=True)
    assert replay.evaluate_python(code,[[[1]]])==[[[1]]]


def test_qualified_source_binds_only_retained_four():
    cohort=run.qualified_cohort()
    assert len(cohort['selected_ids'])==4
    assert cohort['selected_ids']==['4612dd53','6cf79266','d4f3cd78','88a10436']


def test_reveal_cannot_run_before_initial_bank(tmp_path):
    class Source:
        def reveal(self,i):pytest.fail('unearned reveal')
    bank=run.RevisionBank(tmp_path,Source())
    with pytest.raises(ValueError,match='not banked'):
        bank.reveal(0,[{'reveal_hash':'unused'}])
