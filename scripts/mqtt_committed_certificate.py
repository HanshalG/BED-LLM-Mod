"""Independent execution certificate for the saved zero-risk fixed sequences."""
from collections import Counter
from fractions import Fraction
import hashlib
import json

from scripts.mqtt_opportunity_run import META,MENUS,ROOT,read_metadata,load_machine,menus


def trace(machine,actions,reset):
    start,table=machine
    state=start;outputs=[]
    for action in actions:
        if action==reset:state=start;out='RESET_ACK'
        else:state,out=table[state,action]
        outputs.append(out)
    return tuple(outputs)


def certificate(machines,words,sequence,reset):
    groups={}
    for i,m in enumerate(machines):
        groups.setdefault(trace(m,sequence,reset),[]).append(i)
    loss=Fraction()
    for ids in groups.values():
        for word in words:
            counts=Counter(trace(machines[i],word,reset) for i in ids)
            loss+=Fraction(len(ids),len(machines)*len(words))*(
                Fraction(1,2)-sum((Fraction(n,len(ids))**2/2
                                   for n in counts.values()),Fraction()))
    return loss,sorted(len(ids) for ids in groups.values())


def main():
    for path,digest in json.loads((ROOT/'bindings.json').read_text()).items():
        from pathlib import Path
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError('changed reference binding')
    m=read_metadata();target=json.loads(MENUS.read_text())
    assert target==menus(m['groups'])
    result=json.loads((ROOT/'result.json').read_text())
    assert result['status']=='complete' and len(result['groups'])==8
    hashes={r['path']:r['sha256'] for r in m['rows']}
    rows=[]
    for i,(g,t,r) in enumerate(zip(m['groups'],target,result['groups'])):
        assert json.loads((ROOT/f'group_{i}.json').read_text())==r
        assert r['scenario']==g['scenario'] and r['models']==len(g['paths'])
        sequence=r['committed_sequence'];reset=len(g['input_alphabet'])
        assert len(sequence)==6 and all(type(a)is int and 0<=a<=reset for a in sequence)
        machines=[load_machine(p,hashes[p],g['input_alphabet']) for p in g['paths']]
        value,sizes=certificate(machines,t['words'],sequence,reset)
        assert value==Fraction(r['risk']['committed'])==0
        assert Fraction(r['risk']['adaptive_full'])==0
        rows.append({'scenario':g['scenario'],'risk':str(value),
                     'posterior_partition_sizes':sizes,'sequence':sequence})
    report={'status':'verified','groups':rows,'all_committed_risks_zero':True,
            'interpretation':'Nonnegative Brier risk and feasible zero-risk fixed sequences certify no adaptive advantage on this exact panel.',
            'model_calls':0,'cost_usd':0,
            'result_sha256':hashlib.sha256((ROOT/'result.json').read_bytes()).hexdigest()}
    with (ROOT/'committed_certificate.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__':main()
