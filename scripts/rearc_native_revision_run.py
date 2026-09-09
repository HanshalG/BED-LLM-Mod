"""One-shot nested-expression qualification with budget reservations and replay."""
import argparse
from decimal import Decimal
import fcntl
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import subprocess
from scripts.paid_program_probe import PaidProbe, execute
from scripts.rearc_expression_run import ExpressionBank, replay_failure
from scripts.rearc_slot_run import artifacts
from scripts.rearc_luna_qualification import LEDGER
from scripts.rearc_named_plan_contract import validate_route, validate_body, response_text, RESERVE
from scripts.paid_program_probe import save, money, route
from scripts.rearc_native_revision_source import RevisionExamples, ROOT as SOURCE_ROOT, COHORT, schedule
from scripts.rearc_source_qualified_pool import replay_pool
from scripts.rearc_native_revision_source import revision_cases
from scripts.rearc_python_runtime import execute as execute_python
from scripts.rearc_luna_qualification import digest
from scripts.rearc_native_revision_panel import collect
from scripts.rearc_source_scope import SOURCE, COMMIT
from scripts.rearc_graph_runtime import DSL_SHA

ROOT = Path('results/nonmyopic/rearc_native_revision_qualification_20260909')
PREFLIGHT = Path('results/nonmyopic/rearc_native_revision_preflight_20260909')
PROTOCOL = Path('results/nonmyopic/REARC_NATIVE_REVISION_PROTOCOL_20260909.md')
SHA = '025ede58c7cfba665914ff724a718d8b52637d7262b9dbf3104b6d9594056235'


def validate_dependencies():
    for line in Path('requirements-plan-contract.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            package, expected = line.split('==')
            if version(package) != expected:
                raise ValueError('dependency version mismatch: ' + package)


class RevisionProbe(PaidProbe):
    def request(self, tag, body):
        validate_body(body)
        if self.report['calls'] >= 36 or self.accepted+self.reserve > self.cap:
            raise RuntimeError('block cap')
        endpoint = route()
        validate_route(endpoint)
        save(self.root/(tag+'.route.json'),endpoint,exclusive=True)
        save(self.root/(tag+'.request.json'),body,exclusive=True)
        self.account()
        self.ledger['pending_reservations'] = dict(self.carry,**{self.root.name+':'+tag:float(self.reserve)})
        save(self.path,self.ledger)
        self.pending,self.dispatched = True,False
        self.account()
        self.report.update(current=tag,phase='http_attempt',calls=self.report['calls']+1)
        save(self.root/'result.json',self.report)
        self.dispatched = True
        raw = execute(body)
        save(self.root/(tag+'.response.json'),raw,exclusive=True)
        self.accepted += money(raw['usage']['cost'])
        self.ledger['recorded_actual_spend_usd'] = float(max(money(self.ledger['recorded_actual_spend_usd']),self.base+self.accepted))
        self.ledger['pending_reservations'] = dict(self.carry)
        save(self.path,self.ledger)
        self.pending = False
        self.report.update(phase='feedback',accepted_cost_usd=float(self.accepted))
        save(self.root/'result.json',self.report)
        return response_text(raw)


class RevisionBank(ExpressionBank):
    def reveal(self,index,cases):
        if type(index) is not int or not 0<=index<len(cases):
            raise ValueError('reveal index')
        initial=self.root/f'update_{index}_initial.json'
        if not initial.exists():
            raise ValueError('initial proposals not banked before reveal')
        value=self.saved(f'reveal_{index}',lambda:self.source.reveal(index))
        if digest(value)!=cases[index]['reveal_hash']:
            raise ValueError('revealed observation identity')
        return value

    def diagnose_python(self, code, x):
        key = 'python_execution_'+digest([code,x])
        if key not in self.cache:
            self.cache[key] = self.saved(key,lambda:execute_python(code,x))
        return self.cache[key]

    def evaluate_python(self, code, inputs):
        rows = [self.diagnose_python(code,x) for x in inputs]
        return [row['output'] if row['status']=='ok' else None for row in rows]

    def search(self,*args):
        raise RuntimeError('symbolic search not authorized in mechanism protocol')


def qualified_cohort():
    candidate = json.loads(COHORT.read_text())
    rows = schedule(candidate)
    manifest = json.loads((SOURCE_ROOT/'pool/manifest.json').read_text())
    if manifest != {'needed':4,'schedules':[rows[i*11:(i+1)*11] for i in range(24)]}:
        raise ValueError('pool schedule binding')
    result = replay_pool(SOURCE_ROOT/'pool')
    if result['status'] != 'source_pool_qualified':
        raise ValueError('source pool closed')
    expected = {**candidate,'selected_ids':[r['task'] for r in result['accepted']],
        'candidate_manifest_sha256':hashlib.sha256(COHORT.read_bytes()).hexdigest(),
        'selection':'first four exact-schedule reference-valid tasks in frozen candidate order'}
    if json.loads((SOURCE_ROOT/'selected_cohort.json').read_text()) != expected:
        raise ValueError('retained cohort binding')
    return expected


def run():
    validate_dependencies()
    preflight = json.loads((PREFLIGHT/'result.json').read_text())
    public = (PREFLIGHT/'public.json').read_bytes()
    if preflight['status'] != 'initial_public_prompt_preflight_pass' or hashlib.sha256(public).hexdigest() != preflight['public_sha256']:
        raise ValueError('public preflight changed')
    retained = qualified_cohort()
    if revision_cases(SOURCE_ROOT/'pool') != json.loads(public):
        raise ValueError('cached schedule differs from prompt public data')
    validate_route(route())
    bindings = sorted(str(p) for pattern in ('rearc_*.py','herb_*.py','herb_*.jl') for p in Path('scripts').glob(pattern))
    bindings += ['requirements-plan-contract.txt', str(PROTOCOL), str(PREFLIGHT/'result.json'), str(PREFLIGHT/'public.json'),
        'results/nonmyopic/REARC_NATIVE_REVISION_CANDIDATES_20260909.json',
        'results/nonmyopic/herb_full_grammar_20260909/grammar.jl',
        'results/nonmyopic/herb_full_grammar_20260909/source.json',
        'results/nonmyopic/herb_bridge_runtime_20260909/Manifest.toml',
        'scripts/paid_program_probe.py','scripts/deepcoder_proposal_gate.py',
        'scripts/deepcoder_luna_medium_probe.py','scripts/deepcoder_luna_transition.py','scripts/openrouter_daily_budget.py']
    bindings += sorted(str(p) for p in SOURCE_ROOT.rglob('*') if p.is_file())
    bindings.append('results/nonmyopic/REARC_REPRESENTATION_PROTOCOL_20260909.md')
    with RevisionProbe(ROOT,LEDGER,'2.88',PROTOCOL,SHA,bindings) as probe:
        probe.reserve = RESERVE
        try:
            with RevisionExamples() as source:
                if source.bindings != json.loads((SOURCE_ROOT/'bindings.json').read_text()):
                    raise ValueError('source binding changed')
                source.cohort = retained
                source.tasks = retained['selected_ids']
                bank = RevisionBank(ROOT,source)
                cases = bank.saved('public',lambda: json.loads(public))
                def targets():
                    probe.report['endpoints_opened'] = True
                    save(ROOT/'result.json',probe.report)
                    return bank.targets(cases)
                probe.report.update(collect(cases,source.dsl,probe.request,
                    bank.evaluate_python,bank.diagnose_python,lambda i:bank.reveal(i,cases),
                    bank.update,bank.seal,targets))
        except Exception as error:
            probe.report['panel_failure'] = {'type':type(error).__name__,'message':str(error)[:200]}
            raise
        finally:
            probe.report['artifact_sha256'] = artifacts(ROOT)
    return probe.report


def replay(root):
    validate_dependencies()
    report = json.loads((root/'result.json').read_text())
    if artifacts(root) != report['artifact_sha256']:
        raise ValueError('artifact identity')
    for path, sha in report['implementation_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('implementation identity')
    dsl = subprocess.check_output(['git','-C',str(SOURCE),'show',COMMIT+':dsl.py']).decode()
    if hashlib.sha256(dsl.encode()).hexdigest() != DSL_SHA:
        raise ValueError('DSL binding')
    cases = json.loads((root/'public.json').read_text())
    bank = RevisionBank(root,replay=True)
    calls, costs = [], []
    def request(tag, body):
        validate_body(body)
        if json.loads((root/(tag+'.request.json')).read_text()) != body:
            raise ValueError('request replay')
        validate_route(json.loads((root/(tag+'.route.json')).read_text()))
        calls.append(tag)
        response = root/(tag+'.response.json')
        if not response.exists():
            if (report['status']=='failed_closed' and report.get('phase')=='http_attempt'
                    and report.get('current')==tag and report['calls']==len(calls)
                    and money(report['uncertain_exposure_usd'])==RESERVE):
                replay_failure(report['panel_failure'])
            raise ValueError('unaccounted missing response')
        raw = json.loads(response.read_text())
        costs.append(money(raw['usage']['cost']))
        return response_text(raw)
    try:
        result = collect(cases,dsl,request,bank.evaluate_python,bank.diagnose_python,
            lambda i:bank.reveal(i,cases),bank.update,bank.seal,lambda: bank.targets(cases))
    except Exception as error:
        if report['status'] != 'failed_closed' or report.get('panel_failure') != {'type':type(error).__name__,'message':str(error)[:200]}:
            raise
    else:
        if any(report[k] != v for k,v in result.items()):
            raise ValueError('result replay')
    forbidden = ('forecasts','forecast_seal','targets') if report['status']=='initial_coverage_null' else ('targets',)
    if not report['endpoints_opened'] and any((root/(p+'.json')).exists() for p in forbidden):
        raise ValueError('closed endpoint artifacts')
    if len(calls) != report['calls'] or sum(costs,Decimal()) != money(report['accepted_cost_usd']):
        raise ValueError('receipt coverage')
    return {'status':'exact_replay','calls':len(calls),'new_calls':0,'depth_authorized':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--replay',action='store_true')
    args = parser.parse_args()
    if args.execute == args.replay:
        parser.error('choose execute or replay')
    if args.replay:
        print(json.dumps(replay(ROOT)))
    else:
        with LEDGER.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            print(json.dumps(run()))
