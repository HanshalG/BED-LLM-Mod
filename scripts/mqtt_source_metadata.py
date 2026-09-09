"""Pinned complete-family DOT metadata audit; no protocol interaction or scoring."""
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE=Path('/private/tmp/bed-mqtt-source-audit')
COMMIT='24b5535f37ca92745ddb4c3a5a5381b6ccfa87ce'
PREFIX='eval/src/main/resources/mqtt/'
BASE=Path('results/nonmyopic')
MANIFEST=BASE/'MQTT_SOURCE_MANIFEST_20260909.json'
OUT=BASE/'MQTT_SOURCE_METADATA_20260909.json'


def save(path,value):
    with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False)


def git(*args):
    return subprocess.check_output(['git','-C',str(SOURCE),*args],timeout=30)


def select():
    paths=git('ls-tree','-r','--name-only',COMMIT,PREFIX).decode().splitlines()
    if len(paths)!=32 or any(not p.startswith(PREFIX) or not p.endswith('.dot') for p in paths):
        raise ValueError('complete 32-file family required')
    value={'commit':COMMIT,'paths':paths,'selection':'all32MQTTfiles; no transition-dependent selection',
           'grouping':'exact input alphabet plus literal scenario basename; no normalization'}
    save(MANIFEST,value)
    return value


def metadata(raw):
    if len(raw)>1024*1024: raise ValueError('model byte cap')
    value=json.loads(subprocess.check_output(['dot','-Tdot_json'],input=raw,timeout=10))
    nodes={r['_gvid']:r['name'] for r in value['objects']}
    if len(set(nodes.values()))!=len(nodes) or '__start0' not in nodes.values():
        raise ValueError('unique nodes and explicit start required')
    states=set(nodes.values())-{'__start0'}
    if not states or any(not s.startswith('s') or not s[1:].isdigit() for s in states):
        raise ValueError('state names')
    start=[]; transitions={}; outputs=set(); alphabet=set()
    for edge in value['edges']:
        src,dst=nodes[edge['tail']],nodes[edge['head']]
        if src=='__start0':
            if edge.get('label','') or dst not in states: raise ValueError('start edge')
            start.append(dst);continue
        if src not in states or dst not in states: raise ValueError('state reference')
        inp,sep,out=edge.get('label','').partition('/')
        inp,out=inp.strip(),out.strip()
        if not sep or not inp or not out or (src,inp) in transitions:
            raise ValueError('deterministic labelled transition')
        transitions[src,inp]=dst;outputs.add(out);alphabet.add(inp)
    if len(start)!=1: raise ValueError('exactly one start')
    if len(transitions)!=len(states)*len(alphabet): raise ValueError('incomplete machine')
    reached={start[0]};todo=[start[0]]
    while todo:
        s=todo.pop()
        for a in alphabet:
            t=transitions[s,a]
            if t not in reached: reached.add(t);todo.append(t)
    return {'states':len(states),'reachable_states':len(reached),
            'transitions':len(transitions),'input_alphabet':sorted(alphabet),
            'output_symbols':len(outputs),'total_deterministic':True,
            'explicit_reset_state':True}


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    manifest=json.loads(MANIFEST.read_text())
    if manifest['commit']!=COMMIT or manifest['paths']!=git('ls-tree','-r','--name-only',COMMIT,PREFIX).decode().splitlines():
        raise ValueError('manifest identity')
    rows=[];groups={}
    for path in manifest['paths']:
        raw=git('show',COMMIT+':'+path)
        row={'path':path,'sha256':hashlib.sha256(raw).hexdigest()}
        try:
            row.update(status='valid',**metadata(raw))
            key=(Path(path).stem,tuple(row['input_alphabet']))
            groups.setdefault(key,[]).append(path)
        except (ValueError,KeyError,subprocess.SubprocessError) as e:
            row.update(status='invalid',error_type=type(e).__name__)
        rows.append(row)
    report={'status':'complete_metadata_audit','source_commit':COMMIT,
            'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            'graphviz_version':subprocess.check_output(['dot','-V'],stderr=subprocess.STDOUT).decode().strip(),
            'rows':rows,'groups':[{'scenario':k[0],'input_alphabet':list(k[1]),'paths':v}
                                  for k,v in sorted(groups.items())],
            'model_calls':0,'protocol_steps':0,'cost_usd':0,'paid_authorized':False,
            'scope':'source metadata; transition bytes parsed locally but no response trace generated'}
    save(OUT,report)
    print(json.dumps({'status':report['status'],'files':len(rows),'valid':sum(r['status']=='valid' for r in rows),
        'groups':[(g['scenario'],len(g['paths'])) for g in report['groups']]}))


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--select']:print(json.dumps(select()))
    elif not sys.argv[1:]:main()
    else:raise ValueError('arguments')
