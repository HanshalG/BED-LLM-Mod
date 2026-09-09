"""One-shot eight-task feedback study with saved-data replay and hard budgets."""
import argparse
from decimal import Decimal
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
from scripts.paid_program_probe import PaidProbe,save,money,execute,route
from scripts.rearc_luna_qualification import Bank,validate_route,validate_body,response_text,digest,RESERVE,LEDGER
from scripts.rearc_feedback_examples import FeedbackExamples
from scripts.rearc_feedback_panel import collect
from scripts.rearc_public_diagnostic import diagnose
from scripts.rearc_source_scope import COMMIT,SOURCE
from scripts.rearc_graph_runtime import DSL_SHA

ROOT=Path('results/nonmyopic/rearc_feedback_qualification_20260909')
PROTOCOL=Path('results/nonmyopic/REARC_FEEDBACK_QUALIFICATION_PROTOCOL_20260909.md')
SHA='f860c49c02a13454a5d9dc7256bc5a4c2936b68d9c130f25ec055ee2539e03d2'


class FeedbackProbe(PaidProbe):
    def request(self,tag,body):
        validate_body(body)
        endpoint=route(); validate_route(endpoint)
        save(self.root/(tag+'.route.json'),endpoint,exclusive=True)
        save(self.root/(tag+'.request.json'),body,exclusive=True)
        if self.accepted+self.reserve>self.cap or self.report['calls']>=48:
            raise RuntimeError('block cap')
        self.account()
        self.ledger['pending_reservations']=dict(self.carry,**{self.root.name+':'+tag:float(self.reserve)})
        save(self.path,self.ledger)
        self.pending,self.dispatched=True,False
        self.account()
        self.report.update(current=tag,phase='http_attempt',calls=self.report['calls']+1)
        save(self.root/'result.json',self.report)
        self.dispatched=True
        raw=execute(body)
        save(self.root/(tag+'.response.json'),raw,exclusive=True)
        self.accepted+=money(raw['usage']['cost'])
        self.ledger['recorded_actual_spend_usd']=float(max(money(self.ledger['recorded_actual_spend_usd']),self.base+self.accepted))
        self.ledger['pending_reservations']=dict(self.carry)
        save(self.path,self.ledger)
        self.pending=False
        self.report.update(phase='feedback',accepted_cost_usd=float(self.accepted))
        save(self.root/'result.json',self.report)
        return response_text(raw)


class FeedbackBank(Bank):
    def diagnose(self,g,x):
        key='diagnostic_'+digest([g,x])
        if key not in self.cache: self.cache[key]=self.saved(key,lambda:diagnose(g,x))
        return self.cache[key]

    def update(self,tag,result):
        if self.saved('update_'+tag,lambda:result)!=result:
            raise ValueError('update replay mismatch')


def artifacts(root):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.json') if p.name!='result.json'}


def run():
    validate_route(route())
    bindings=sorted(str(p) for p in Path('scripts').glob('rearc_*.py'))+[
        str(PROTOCOL),'results/nonmyopic/REARC_FEEDBACK_COHORT_20260909.json',
        'results/nonmyopic/REARC_FEEDBACK_SOURCE_SMOKE_20260909.json',
        'scripts/paid_program_probe.py','scripts/deepcoder_proposal_gate.py',
        'scripts/deepcoder_luna_medium_probe.py','scripts/deepcoder_luna_transition.py',
        'scripts/openrouter_daily_budget.py']
    with FeedbackProbe(ROOT,LEDGER,'2.88',PROTOCOL,SHA,bindings) as probe:
        probe.reserve=RESERVE
        try:
            with FeedbackExamples() as source:
                bank=FeedbackBank(ROOT,source)
                cases=bank.saved('public',source.public)
                def targets():
                    probe.report['endpoints_opened']=True
                    save(ROOT/'result.json',probe.report)
                    return bank.targets(cases)
                probe.report.update(collect(cases,source.dsl,probe.request,bank.evaluate,bank.diagnose,
                    bank.symbolic,bank.update,bank.seal,targets))
        except Exception as exc:
            probe.report['panel_failure']={'type':type(exc).__name__,'message':str(exc)[:200]}
            raise
        finally:
            probe.report['artifact_sha256']=artifacts(ROOT)
    return probe.report


def replay(root):
    report=json.loads((root/'result.json').read_text())
    if artifacts(root)!=report['artifact_sha256']: raise ValueError('artifact identity')
    for path,sha in report['implementation_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha: raise ValueError('implementation identity')
    dsl=subprocess.check_output(['git','-C',str(SOURCE),'show',COMMIT+':dsl.py']).decode()
    if hashlib.sha256(dsl.encode()).hexdigest()!=DSL_SHA: raise ValueError('DSL identity')
    cases=json.loads((root/'public.json').read_text())
    bank=FeedbackBank(root,replay=True); calls=[]; costs=[]
    def request(tag,body):
        validate_body(body)
        if json.loads((root/(tag+'.request.json')).read_text())!=body: raise ValueError('request replay')
        validate_route(json.loads((root/(tag+'.route.json')).read_text()))
        raw=json.loads((root/(tag+'.response.json')).read_text())
        calls.append(tag); costs.append(money(raw['usage']['cost']))
        return response_text(raw)
    try:
        result=collect(cases,dsl,request,bank.evaluate,bank.diagnose,bank.symbolic,
                       bank.update,bank.seal,lambda:bank.targets(cases))
    except Exception as exc:
        if report['status']!='failed_closed' or report.get('panel_failure')!={'type':type(exc).__name__,'message':str(exc)[:200]}:
            raise
    else:
        if any(report[k]!=v for k,v in result.items()): raise ValueError('result replay')
    if not report['endpoints_opened'] and any((root/(p+'.json')).exists() for p in ('forecasts','forecast_seal','targets')):
        raise ValueError('closed endpoint artifacts')
    if len(calls)!=report['calls'] or sum(costs,Decimal())!=money(report['accepted_cost_usd']):
        raise ValueError('receipt coverage')
    return {'status':'exact_replay','calls':len(calls),'new_calls':0,'depth_authorized':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    if args.execute==args.replay: parser.error('choose one mode')
    if args.replay: print(json.dumps(replay(ROOT)))
    else:
        with LEDGER.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            print(json.dumps(run()))
