"""Static diagnostic of banked proposals; never executes or rescues a program."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from scripts.rearc_program_graph import exports


def errors(graph,functions,constants):
    known={'I'}|functions|constants
    rows=[]
    for j,step in enumerate(graph['steps']):
        if step['id']!=f'x{j}': rows.append({'kind':'step_identifier','step':j,'value':step['id']})
        if step['op'] not in known: rows.append({'kind':'unknown_operator','step':j,'value':step['op']})
        for arg in step['args']:
            if arg not in known:
                try:
                    node=ast.parse(arg,mode='eval').body
                    kind='nested_expression' if isinstance(node,ast.Call) else 'unknown_reference'
                except SyntaxError:
                    kind='unknown_reference'
                rows.append({'kind':kind,'step':j,'value':arg})
        known.add(step['id'])
    if graph['output'] not in {s['id'] for s in graph['steps']}:
        rows.append({'kind':'output_reference','value':graph['output']})
    return rows


def main():
    parent=Path('results/nonmyopic/rearc_feedback_qualification_20260909')
    report=json.loads((parent/'result.json').read_text())
    source=subprocess.check_output(['git','-C','/private/tmp/bed-rearc-source-audit','show',
        'e5b7f1d06362a76f9d3b8c25154ff1fafca897ce:dsl.py']).decode()
    functions,constants=exports(source)
    records=[]
    for path in sorted(parent.glob('*.response.json')):
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=report['artifact_sha256'][path.name]:
            raise ValueError('response identity')
        graphs=json.loads(json.loads(raw)['choices'][0]['message']['content'])['hypotheses']
        records.append({'response':path.name,'programs':[errors(g,functions,constants) for g in graphs]})
    if len(records)!=16: raise ValueError('receipt coverage')
    result={'records':records,'response_count':16,
        'batches_with_nested_expressions':sum(any(e['kind']=='nested_expression' for g in r['programs'] for e in g) for r in records),
        'batches_with_reference_errors':sum(any(g for g in r['programs']) for r in records),
        'model_calls':0,'programs_executed':0,'cost_usd':0,'endpoint_reopened':False,
        'scope':'retrospective lexical/reference categories, not type checking or repaired accuracy'}
    with Path('results/nonmyopic/REARC_REFERENCE_ERROR_AUDIT_20260909.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__': main()
