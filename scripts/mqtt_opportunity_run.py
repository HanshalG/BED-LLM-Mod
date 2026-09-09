"""One-shot complete MQTT finite-prior screen, with banked input-only targets."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import subprocess

from scripts.mqtt_source_metadata import SOURCE,COMMIT,metadata,git
from scripts.mqtt_reference import evaluate

BASE=Path('results/nonmyopic')
META=BASE/'MQTT_SOURCE_METADATA_20260909.json'
META_SHA='b2ddd4eb02218334f6d00331e87b0b8ca9652c483f4452873e4b4b2479598e62'
MENUS=BASE/'MQTT_ORACLE_TARGET_MENUS_20260909.json'
ROOT=BASE/'mqtt_oracle_opportunity_20260909'


def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)


def read_metadata():
    raw=META.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=META_SHA:raise ValueError('metadata binding')
    v=json.loads(raw)
    if len(v['groups'])!=8 or any(r['status']!='valid' for r in v['rows']):
        raise ValueError('complete source groups')
    return v


def menus(groups):
    result=[]
    for i,g in enumerate(groups):
        rng=random.Random(61400+i)
        result.append({'scenario':g['scenario'],'alphabet':g['input_alphabet'],
            'words':[[rng.choice(range(len(g['input_alphabet']))) for _ in range(6)] for _ in range(512)]})
    return result


def load_machine(path,expected_sha,alphabet):
    raw=git('show',COMMIT+':'+path)
    if hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('machine source binding')
    m=metadata(raw)
    if m['input_alphabet']!=alphabet:raise ValueError('exact action alphabet')
    value=json.loads(subprocess.check_output(['dot','-Tdot_json'],input=raw,timeout=10))
    names={r['_gvid']:r['name'] for r in value['objects']}
    table={};start=None
    for e in value['edges']:
        s,t=names[e['tail']],names[e['head']]
        if s=='__start0':start=t;continue
        a,_,y=e['label'].partition('/')
        table[s,alphabet.index(a.strip())]=t,y.strip()
    return start,table


def aggregate(rows):
    if len(rows)!=8 or any(r['status']!='complete' for r in rows):
        return {'status':'incomplete_reference','opportunity_passed':False,'paid_authorized':False}
    keys=('h1','h2','h3','adaptive_full','committed','random')
    means={k:sum((Fraction(r['risk'][k]) for r in rows),Fraction())/8 for k in keys}
    def gain(a,b):return (means[a]-means[b])/means[a] if means[a]>0 else None
    gains={'h2_over_h1':gain('h1','h2'),'h3_over_h2':gain('h2','h3'),
           'adaptive_over_committed':gain('committed','adaptive_full')}
    wins=sum(Fraction(r['initial_risk'])>0 and Fraction(r['risk']['adaptive_full'])<Fraction(r['risk']['committed']) for r in rows)
    passed=all(v is not None and v>=Fraction(1,20) for v in gains.values()) and wins>=3
    return {'status':'complete','means':{k:str(v) for k,v in means.items()},
            'relative_gains':{k:None if v is None else str(v) for k,v in gains.items()},
            'adaptive_winning_groups':wins,'opportunity_passed':passed,
            'paid_authorized':False,'llm_efficacy_established':False}


def main():
    m=read_metadata();expected=menus(m['groups'])
    if json.loads(MENUS.read_text())!=expected:raise ValueError('target input menu binding')
    ROOT.mkdir(exist_ok=False)
    bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (META,MENUS,Path(__file__),Path('scripts/mqtt_reference.py'),
         Path('scripts/mqtt_source_metadata.py'),BASE/'MQTT_ORACLE_OPPORTUNITY_PROTOCOL_20260909.md')}
    save(ROOT/'bindings.json',bindings)
    hashes={r['path']:r['sha256'] for r in m['rows']}
    rows=[]
    for i,(g,t) in enumerate(zip(m['groups'],expected)):
        try:
            machines=[load_machine(p,hashes[p],g['input_alphabet']) for p in g['paths']]
            row={'scenario':g['scenario'],'models':len(machines),
                 **evaluate(machines,g['input_alphabet'],t['words'],budget=6,seconds=180)}
        except (TimeoutError,ValueError,AssertionError,subprocess.SubprocessError) as e:
            row={'scenario':g['scenario'],'status':'incomplete','error_type':type(e).__name__}
        save(ROOT/f'group_{i}.json',row);rows.append(row)
        print(json.dumps(row),flush=True)
    report={**aggregate(rows),'groups':rows,'model_calls':0,'cost_usd':0,
            'source_prior_only':True,'budget_commands':6}
    save(ROOT/'result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='groups'}),flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--menus']:save(MENUS,menus(read_metadata()['groups']))
    elif not sys.argv[1:]:main()
    else:raise ValueError('arguments')
