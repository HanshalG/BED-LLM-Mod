import json
import shutil
from pathlib import Path
import pytest
from scripts import rearc_slot_run as run
from scripts.rearc_expression_interface import response_format


def test_caps_before_transport(monkeypatch):
    monkeypatch.setattr(run.FeedbackProbe,'request',lambda *a: pytest.fail('HTTP opened'))
    probe = object.__new__(run.SlotProbe)
    probe.report = {'calls':24}
    with pytest.raises(ValueError,match='cap'):
        probe.request('x',{'response_format':response_format()})
    probe.report = {'calls':0}
    with pytest.raises(ValueError,match='cap'):
        probe.request('x',{'response_format':{}})


def test_source_only_prefix_reused_and_replayed_without_worker(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'search',lambda *a: pytest.fail('repeated base worker'))
    bank = run.SlotBank(tmp_path)
    result = bank.search([],128,100000)
    assert len(result['slots']) == 128
    assert result['origin'] == 'banked_source_only_prefix'
    assert run.SlotBank(tmp_path,replay=True).search([],128,100000) == result


def test_nested_actual_worker_bank_replay_and_request_binding(tmp_path,monkeypatch):
    source = Path('results/nonmyopic/herb_slot_runtime_smoke_20260909')
    request = json.loads((source/'request.json').read_text())
    def worker(proposals,count,cap,destination):
        assert (proposals,count,cap) == (request['proposals'],56,50000)
        shutil.copytree(source,destination)
        return json.loads((destination/'result.json').read_text())
    monkeypatch.setattr(run,'search',worker)
    result = run.SlotBank(tmp_path).search(request['proposals'],56,50000)
    hashes = run.artifacts(tmp_path)
    assert any(name.endswith('/stdout.toml') for name in hashes)
    monkeypatch.setattr(run,'search',lambda *a: pytest.fail('worker replayed'))
    assert run.SlotBank(tmp_path,replay=True).search(request['proposals'],56,50000) == result
    recorded = next(tmp_path.glob('search_*/request.json'))
    request['proposals'] = []
    recorded.write_text(json.dumps(request))
    assert hashes != run.artifacts(tmp_path)
    original = json.loads((source/'request.json').read_text())['proposals']
    with pytest.raises(ValueError,match='request binding'):
        run.SlotBank(tmp_path,replay=True).search(original,56,50000)


def test_failure_prefix_does_not_rerun_worker(tmp_path,monkeypatch):
    def fail(*args):
        raise RuntimeError('fixed worker failure')
    monkeypatch.setattr(run,'search',fail)
    with pytest.raises(RuntimeError,match='fixed worker failure'):
        run.SlotBank(tmp_path).search([None]*8,56,50000)
    monkeypatch.setattr(run,'search',lambda *a: pytest.fail('failure retried'))
    with pytest.raises(RuntimeError,match='fixed worker failure'):
        run.SlotBank(tmp_path,replay=True).search([None]*8,56,50000)


@pytest.mark.parametrize('mode', ['complete','initial_null','interrupted'])
def test_terminal_response_replay_without_calls(tmp_path,monkeypatch,mode):
    import hashlib
    dsl = 'def identity(x: Any) -> Any:\n return x\n'
    cases = [{'inputs': [[[0]],[[1]],[[2]]], 'outputs': [[[0]],[[1]],[[2]]],
              'target_inputs': [[[3]]]*8,
              'target_hashes': [hashlib.sha256(b'[[3]]').hexdigest()]*8} for _ in range(4)]
    g = {'steps':[{'id':'x0','op':'identity','args':['I']}],'output':'x0'}
    class Source:
        def targets(self,cases):
            return [[[[3]]]*8 for _ in cases]
    class SyntheticBank(run.SlotBank):
        def evaluate(self,graph,inputs):
            return self.saved('e_'+run.digest([graph,inputs]),lambda:
                [None]*len(inputs) if mode=='initial_null' else inputs)
        def diagnose(self,graph,x):
            return self.saved('d_'+run.digest([graph,x]),lambda: {'status':'ok','output':x})
        def search(self,proposals,count,cap):
            return self.saved('s_'+run.digest([proposals,count,cap]),lambda:
                {'status':'complete','replacement_candidates':0,
                 'slots':[{'slot':i,'graph':g} for i in range(count)]})
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
        if mode=='interrupted' and report['calls']==3:
            report['uncertain_exposure_usd']=.06
            raise RuntimeError('synthetic interrupted HTTP')
        raw = {'model':'openai/gpt-5.6-luna','provider':'OpenAI',
               'usage':{'cost':0.,'prompt_tokens':1,'completion_tokens':2,
                        'completion_tokens_details':{'reasoning_tokens':1}},
               'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'hypotheses':['identity(I)']*4})}}]}
        run.save(tmp_path/(tag+'.response.json'),raw,exclusive=True)
        report['phase']='feedback'
        return run.response_text(raw)
    def targets():
        report['endpoints_opened']=True
        return bank.targets(cases)
    try:
        report.update(run.collect(cases,dsl,request,bank.evaluate,bank.diagnose,
            bank.search,bank.update,bank.seal,targets))
    except RuntimeError as error:
        report.update(status='failed_closed',panel_failure={'type':'RuntimeError','message':str(error)})
    report['artifact_sha256']=run.artifacts(tmp_path)
    run.save(tmp_path/'result.json',report)
    monkeypatch.setattr(run,'SlotBank',SyntheticBank)
    monkeypatch.setattr(run,'DSL_SHA',hashlib.sha256(dsl.encode()).hexdigest())
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**k:dsl.encode())
    monkeypatch.setattr(run,'route',lambda:pytest.fail('catalog request during replay'))
    assert run.replay(tmp_path) == {'status':'exact_replay',
        'calls':{'complete':24,'initial_null':8,'interrupted':3}[mode],
        'new_calls':0,'depth_authorized':False}
