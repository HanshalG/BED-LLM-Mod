"""One-shot frozen source-function semantic/history diagnostic."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import urllib.request

from environments.program_induction import physics_proposals as model
from environments.program_induction.scalar_expression import ScalarExpression
from scripts import deepcoder_luna_medium_probe as luna
from scripts.deepcoder_proposal_gate import save
from scripts.paid_program_probe import PaidProbe

ROOT=Path('results/nonmyopic/physgym_semantic_history_20260909')
PROTOCOL=Path('results/nonmyopic/PHYSGYM_SEMANTIC_HISTORY_PROTOCOL_20260909.md')
PROTOCOL_SHA='28fe71dca2edef2b37a8e8bf8f00fc15b406730608ea29f512de8885844af54e'
SOURCE_SHA='7903440c59e4379f414b88704a01e9ba7f56adca7a5d6d363a614bd2f92afa12'
IDS=('103','458','457','653')
BINDINGS=[__file__,'environments/program_induction/physics_proposals.py',
          'environments/program_induction/scalar_expression.py','scripts/paid_program_probe.py',
          'scripts/deepcoder_luna_medium_probe.py','scripts/deepcoder_luna_transition.py',
          'scripts/openrouter_daily_budget.py']


def source():
    url='https://raw.githubusercontent.com/principia-ai/PhysGym/fe68079c0921029dde679ed44bc3192dd3b270ab/physgym/samples/full_samples.json'
    with urllib.request.urlopen(url,timeout=30) as response:
        raw=response.read(5000001)
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise ValueError('source mismatch')
    rows=json.loads(raw)
    return {t:next(r for r in rows if str(r['id'])==t) for t in IDS}


def output(row, point):
    names=sorted(row['input_variables'])
    value=ScalarExpression(row['equation'],names)({n:point[n] for n in names})
    if value<=0:
        raise ValueError('nonpositive source response')
    return math.log(value)


def cases(worlds):
    public={}
    for i,t in enumerate(IDS):
        row=worlds[t]
        descriptions=dict(row['input_variables'],**row['dummy_variables'])
        names=sorted(descriptions)
        rng=random.Random(53100000+i)
        noise=random.Random(54100000+i)
        points=[{n:rng.randint(3,12) if n=='N' else math.exp(rng.uniform(math.log(.5),math.log(2)))
                 for n in names} for _ in range(36)]
        history=[[p,output(row,p)+noise.gauss(0,model.SIGMA)] for p in points[:4]]
        public[t]=dict(names=names,descriptions=descriptions,context=row['content'],
                       history=history,targets=points[4:])
    return public


def outcomes(worlds,public):
    result={}
    for i,t in enumerate(IDS):
        noise=random.Random(54100000+i)
        for _ in range(4):
            noise.gauss(0,model.SIGMA)
        result[t]=[output(worlds[t],p)+noise.gauss(0,model.SIGMA) for p in public[t]['targets']]
    return result


def body(case,arm,index):
    schema={'type':'object','properties':{'expressions':{'type':'array','minItems':1,'maxItems':8,
             'items':{'type':'string','maxLength':8192}}},'required':['expressions'],'additionalProperties':False}
    return luna.bumped(dict(model='unused',temperature=0,max_tokens=4096,
        reasoning={'enabled':False,'exclude':True},seed=55100000+10*index+int(arm in ('refresh','redraw')),
        messages=model.messages(case['names'],case['context'],case['descriptions'],
                                case['history'][:4 if arm=='refresh' else 3],arm!='blind'),
        response_format={'type':'json_schema','json_schema':{'name':'scalar_proposals','strict':True,'schema':schema}}))


def panel(public,request):
    results={}
    for i,t in enumerate(IDS):
        case=public[t]
        pools={}
        order=(['semantic','blind'] if i%2==0 else ['blind','semantic'])
        order+=(['refresh','redraw'] if i%2==0 else ['redraw','refresh'])
        for arm in order:
            raw=request(t+'_'+arm,body(case,arm,i))
            pools[arm]=model.decode(luna.validate_response(raw),len(case['names']))
        forecasts={arm:model.predict(pools[arm] if arm in ('semantic','blind') else pools['semantic']+pools[arm],
                     case['names'],case['history'][:3 if arm in ('semantic','blind') else 4],case['targets'])
                   for arm in ('semantic','blind','refresh','redraw')}
        forecasts['symbolic']=dict(status='complete',mean=model.symbolic(case['names'],case['history'],case['targets']))
        results[t]=dict(pools=pools,forecasts=forecasts)
    return results


def score(forecasts,truth):
    if set(forecasts)!=set(IDS) or set(truth)!=set(IDS):
        raise ValueError('incomplete task coverage')
    losses={}
    for t in IDS:
        losses[t]={}
        if len(truth[t])!=32 or not all(math.isfinite(y) for y in truth[t]):
            raise ValueError('invalid endpoints')
        for arm in ('semantic','blind','refresh','redraw','symbolic'):
            f=forecasts[t]['forecasts'][arm]
            if f['status']!='complete':
                losses[t][arm]=None
                continue
            if len(f['mean'])!=32 or not all(math.isfinite(y) for y in f['mean']):
                raise ValueError('invalid forecast')
            losses[t][arm]=sum((p-y)**2 for p,y in zip(f['mean'],truth[t]))/32
    if any(v is None for row in losses.values() for v in row.values()):
        return dict(status='complete',gate_passed=False,reason='empty_support',losses=losses)
    means={a:sum(row[a] for row in losses.values())/4 for a in next(iter(losses.values()))}
    context_wins=sum(r['blind']-r['semantic']>.01 for r in losses.values())
    history_wins=sum(r['redraw']-r['refresh']>.01 for r in losses.values())
    context=means['semantic']<=.9*means['blind'] and context_wins>=2
    history=means['refresh']<=.9*means['redraw'] and history_wins>=2 and means['refresh']<=means['symbolic']
    return dict(status='complete',gate_passed=context and history,context_gate=context,history_gate=history,
                context_wins=context_wins,history_wins=history_wins,means=means,losses=losses)


def collect(block,public,load_outcomes):
    save(block.root/'public.json',public,exclusive=True)
    forecasts=panel(public,block.request)
    path=block.root/'forecasts.json'
    save(path,forecasts,exclusive=True)
    block.report['forecast_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    save(block.root/'result.json',block.report)
    truth=load_outcomes()
    save(block.root/'outcomes.json',truth,exclusive=True)
    block.report.update(endpoints_opened=True,**score(forecasts,truth))


def run(ledger):
    worlds=source()
    public=cases(worlds)
    with PaidProbe(ROOT,ledger,.64,PROTOCOL,PROTOCOL_SHA,BINDINGS) as block:
        collect(block,public,lambda:outcomes(worlds,public))
    return block.report


def replay(root):
    report=json.loads((root/'result.json').read_text())
    if report['status']!='complete' or report['protocol_sha256']!=PROTOCOL_SHA:
        raise ValueError('not a complete frozen run')
    for p,h in report['implementation_sha256'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:
            raise ValueError('implementation changed')
    worlds=source()
    public=cases(worlds)
    if public!=json.loads((root/'public.json').read_text()):
        raise ValueError('public reconstruction failed')
    receipts=[]
    def request(tag,expected):
        if expected!=json.loads((root/(tag+'.request.json')).read_text()):
            raise ValueError('request mismatch')
        raw=json.loads((root/(tag+'.response.json')).read_text())
        receipts.append(raw['usage']['cost'])
        return raw
    forecasts=panel(public,request)
    path=root/'forecasts.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['forecast_sha256'] or forecasts!=json.loads(path.read_text()):
        raise ValueError('forecast mismatch')
    truth=outcomes(worlds,public)
    if truth!=json.loads((root/'outcomes.json').read_text()):
        raise ValueError('endpoint mismatch')
    measured=score(forecasts,truth)
    if any(report.get(k)!=v for k,v in measured.items()) or len(receipts)!=16 or report['calls']!=16:
        raise ValueError('score/call mismatch')
    if abs(sum(receipts)-report['accepted_cost_usd'])>1e-10:
        raise ValueError('cost mismatch')
    return dict(status='replay_valid',calls=16,accepted_cost_usd=report['accepted_cost_usd'],gate_passed=report['gate_passed'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ledger',type=Path)
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    if args.replay:
        print(json.dumps(replay(ROOT),sort_keys=True))
    else:
        if args.ledger is None:
            parser.error('--ledger required')
        with args.ledger.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            print(json.dumps(run(args.ledger),sort_keys=True))
