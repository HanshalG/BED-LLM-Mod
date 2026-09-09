"""One-shot frozen finite-string development opportunity and real-path diagnostic."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import urllib.request

from scripts.author_strings_audit import COMMIT, HASHES, inspect, parse
from environments.string_induction.finite import support
from environments.program_induction.reference import ProgramReference
from environments.program_induction.rollout_risk import expected_brier

ROOT=Path('results/nonmyopic/author_strings_opportunity_20260909')
PROTOCOL=Path('results/nonmyopic/AUTHOR_STRINGS_OPPORTUNITY_PROTOCOL_20260909.md')
SHA='e6db22df793b2199b7a74869e6fff21673e181db616693e1a29ffd0cded317f0'


def save(path, value):
    with path.open('x') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')


def source(task):
    data={}
    for name,sha in HASHES[task].items():
        with urllib.request.urlopen(f'https://huggingface.co/datasets/andrewcropper/ilp-datasets/resolve/{COMMIT}/strings/{task}/train/{name}',timeout=30) as r:
            raw=r.read(1000001)
        if len(raw)>1000000 or hashlib.sha256(raw).hexdigest()!=sha:
            raise ValueError('source binding mismatch')
        data[name]=raw.decode()
    inspect(data['bk.pl'],data['exs.pl'])
    xs,ys=defaultdict(dict),defaultdict(dict)
    for name,args in parse(data['bk.pl']):
        if name=='in':
            e,pos,char=args
            xs[e][pos]=char
    for name,args in parse(data['exs.pl']):
        if name=='pos':
            e,pos,char=args[0][1]
            ys[e][pos]=char
    if set(xs)!={f'e{i}' for i in range(1,11)}:
        raise ValueError('frozen example IDs absent')
    def string(chars):
        return ''.join(chars[i] for i in range(1,len(chars)+1))
    return [(string(xs[f'e{i}']),string(ys[f'e{i}'])) for i in range(1,11)]


def actual(ref, answers, targets, horizon):
    state,menu=ref.initial_state,tuple(range(6))
    trace=[]
    for round_index in range(3):
        action,_=ref.root(state,menu,min(horizon,3-round_index),'adaptive')
        trace.append(dict(query_id=f'e{action+2}',answer=answers[action]))
        branch=next((b for b in ref.branches(state,action) if b.observation==answers[action]),None)
        if branch is None:
            return dict(status='unsupported_actual_answer',trace=trace,brier=1.,abstained=True)
        state=branch.state
        menu=tuple(a for a in menu if a!=action)
    qs=[{y:n/len(state) for y,n in Counter(ref.rows[i][j] for i in state).items()}
        for j in range(6,9)]
    return dict(status='complete',trace=trace,brier=sum(expected_brier({y:1.},q)['expected_loss']
        for y,q in zip(targets,qs))/3,abstained=False,distributions=qs)


def measure(rows, answers, targets):
    if not rows:
        return dict(status='empty_support',opportunity_pass=False)
    ref=ProgramReference(rows,6,seconds=120)
    try:
        state,menu=ref.initial_state,tuple(range(6))
        values={f'h{h}':ref.deployed(state,menu,3,h) for h in (1,2,3)}
        values['random']=ref.random(state,menu,3)
        values['fixed_openloop']=ref.planner.plan(state,3,available=menu,mode='open_loop').root.expected_risk
        values['optimal_b3']=ref.planner.plan(state,3,available=menu).root.expected_risk
        return dict(status='complete',initial_risk=ref.risk(state),expected_risk=values,
                    root_actions={f'h{h}':ref.root(state,menu,h,'adaptive')[0] for h in (1,2,3)},
                    actual_paths={f'h{h}':actual(ref,answers,targets,h) for h in (1,2,3)})
    except (TimeoutError,RuntimeError) as exc:
        return dict(status='incomplete',error_type=type(exc).__name__,opportunity_pass=False)
    finally:
        ref.clear()


def gate(results):
    if set(results)!={'1','10'} or any(r['status']!='complete' for r in results.values()):
        return dict(passed=False,reason='incomplete_reference')
    means={k:sum(r['expected_risk'][k] for r in results.values())/2
           for k in next(iter(results.values()))['expected_risk']}
    ordered=all(r['expected_risk']['h3']<=r['expected_risk']['h2']+1e-12
                and r['expected_risk']['h2']<=r['expected_risk']['h1']+1e-12 for r in results.values())
    passed=(ordered and means['h1']>0 and means['h2']>0 and means['h2']<=.95*means['h1']
            and means['h3']<=.95*means['h2'] and means['h3']<means['random']-1e-12
            and means['h3']<means['fixed_openloop']-1e-12)
    return dict(passed=passed,mean_expected_risk=means,ordered_both=ordered)


def run():
    if ROOT.exists() or hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()!=SHA:
        raise ValueError('already opened or protocol changed')
    ROOT.mkdir()
    report=dict(status='incomplete',protocol_sha256=SHA,model_calls=0,cost_usd=0,paid_authority=False,
        implementation_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
            __file__,'scripts/author_strings_audit.py','environments/string_induction/finite.py',
            'environments/program_induction/reference.py','environments/chembench_mopen/horizon.py']})
    try:
        cases={t:source(t) for t in HASHES}
        matrices={}
        for task,pairs in cases.items():
            rows,work=support(pairs[0],[x for x,y in pairs[1:]])
            matrices[task]=dict(rows=rows,work=work,initial=pairs[0],inputs=[x for x,y in pairs[1:]])
        save(ROOT/'forecasts.json',matrices)
        report['forecast_sha256']=hashlib.sha256((ROOT/'forecasts.json').read_bytes()).hexdigest()
        report['results']={t:measure(m['rows'],[y for x,y in cases[t][1:7]],
                                    [y for x,y in cases[t][7:]]) for t,m in matrices.items()}
        report.update(status='complete',gate=gate(report['results']))
    except Exception as exc:
        report.update(status='failed_closed',error_type=type(exc).__name__)
    save(ROOT/'result.json',report)
    return report


if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True))
