"""One-shot nested-expression qualification with budget reservations and replay."""
import argparse
import builtins
from decimal import Decimal
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
from scripts.rearc_feedback_run import FeedbackProbe, FeedbackBank, artifacts
from scripts.rearc_luna_qualification import validate_route, validate_body, response_text, digest, RESERVE, LEDGER
from scripts.paid_program_probe import save, money, route
from scripts.rearc_expression_examples import ExpressionExamples
from scripts.rearc_expression_panel import collect
from scripts.rearc_expression_interface import response_format
from scripts.herb_search_runtime import search
from scripts.rearc_source_scope import SOURCE, COMMIT
from scripts.rearc_graph_runtime import DSL_SHA

ROOT = Path('results/nonmyopic/rearc_expression_qualification_20260909')
PREFLIGHT = Path('results/nonmyopic/rearc_expression_preflight_20260909')
PROTOCOL = Path('results/nonmyopic/REARC_EXPRESSION_QUALIFICATION_PROTOCOL_20260909.md')
SHA = '65d045f418706c8ff115e29af60b5f8c85f5fb0ff58fe6d48559baed3bb26368'
SYMBOLIC = Path('results/nonmyopic/REARC_EXPRESSION_SYMBOLIC_PREFLIGHT_20260909.json')
SYMBOLIC_SHA = '195a0299bfa706f5fad61fd66fc4060b5bba0d231890efb328755137f1100f04'


def replay_failure(error):
    cls = getattr(builtins,error['type'],None)
    if not isinstance(cls,type) or not issubclass(cls,Exception):
        cls = type(error['type'],(Exception,),{})
    raise cls(error['message'])


class ExpressionProbe(FeedbackProbe):
    def request(self, tag, body):
        if self.report['calls'] >= 24 or body['response_format'] != response_format():
            raise ValueError('call or schema cap')
        return super().request(tag, body)


class ExpressionBank(FeedbackBank):
    def saved(self, name, build):
        failure = self.root/(name+'.error.json')
        if self.replay and failure.exists():
            if (self.root/(name+'.json')).exists():
                raise ValueError('conflicting bank result')
            replay_failure(json.loads(failure.read_text()))
        try:
            return super().saved(name,build)
        except Exception as error:
            if not self.replay:
                save(failure,{'type':type(error).__name__,'message':str(error)[:200]},exclusive=True)
            raise

    def search(self, proposals, count, cap):
        key = 'search_'+digest([proposals,count,cap])
        def build():
            if not proposals and count == 128:
                if hashlib.sha256(SYMBOLIC.read_bytes()).hexdigest() != SYMBOLIC_SHA:
                    raise ValueError('symbolic preflight changed')
                return json.loads(SYMBOLIC.read_text())
            return search(proposals,count,cap)
        if key not in self.cache:
            self.cache[key] = self.saved(key,build)
        row = self.cache[key]
        if row['status'] != 'complete' or len(row['expressions']) != count or row['expansions'] > cap:
            raise ValueError('search runtime failure')
        return row['expressions']


def run():
    preflight = json.loads((PREFLIGHT/'result.json').read_text())
    public = (PREFLIGHT/'public.json').read_bytes()
    if preflight['status'] != 'public_prompt_preflight_pass' or hashlib.sha256(public).hexdigest() != preflight['public_sha256']:
        raise ValueError('public preflight changed')
    validate_route(route())
    bindings = sorted(str(p) for pattern in ('rearc_*.py','herb_*.py','herb_*.jl') for p in Path('scripts').glob(pattern))
    bindings += [str(PROTOCOL), str(SYMBOLIC), str(PREFLIGHT/'result.json'), str(PREFLIGHT/'public.json'),
        'results/nonmyopic/REARC_EXPRESSION_COHORT_20260909.json',
        'results/nonmyopic/REARC_EXPRESSION_SOURCE_SMOKE_20260909.json',
        'results/nonmyopic/herb_full_grammar_20260909/grammar.jl',
        'results/nonmyopic/herb_full_grammar_20260909/source.json',
        'results/nonmyopic/herb_bridge_runtime_20260909/Manifest.toml',
        'scripts/paid_program_probe.py','scripts/deepcoder_proposal_gate.py',
        'scripts/deepcoder_luna_medium_probe.py','scripts/deepcoder_luna_transition.py','scripts/openrouter_daily_budget.py']
    with ExpressionProbe(ROOT,LEDGER,'1.44',PROTOCOL,SHA,bindings) as probe:
        probe.reserve = RESERVE
        try:
            with ExpressionExamples() as source:
                bank = ExpressionBank(ROOT,source)
                cases = bank.saved('public',lambda: json.loads(public))
                def targets():
                    probe.report['endpoints_opened'] = True
                    save(ROOT/'result.json',probe.report)
                    return bank.targets(cases)
                probe.report.update(collect(cases,source.dsl,probe.request,bank.evaluate,bank.diagnose,
                    bank.search,bank.update,bank.seal,targets))
        except Exception as error:
            probe.report['panel_failure'] = {'type':type(error).__name__,'message':str(error)[:200]}
            raise
        finally:
            probe.report['artifact_sha256'] = artifacts(ROOT)
    return probe.report


def replay(root):
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
    bank = ExpressionBank(root,replay=True)
    calls, costs = [], []
    def request(tag, body):
        validate_body(body)
        if body['response_format'] != response_format() or json.loads((root/(tag+'.request.json')).read_text()) != body:
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
        result = collect(cases,dsl,request,bank.evaluate,bank.diagnose,bank.search,bank.update,bank.seal,lambda: bank.targets(cases))
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
