"""Saved-data-only diagnostics; never execute candidates or request new outcomes."""
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from scripts.rearc_predictive_score import outcome, mixture_scores
from scripts.rearc_public_source_journal import save

ROOT=Path('results/nonmyopic/rearc_qualified_representation_qualification_20260909')
RESULT_SHA='6b17fa058fe6fd35decef71a8bf0979476c6f108a2f3f770afc7554574528ccd'
OUTPUT=Path('results/nonmyopic/REARC_REPRESENTATION_ERROR_AUDIT_20260909.json')


def contains_literal_grid(code, observed):
    if code is None:
        return False
    expected=outcome(observed)
    for node in ast.walk(ast.parse(code)):
        if isinstance(node,(ast.List,ast.Tuple)):
            try:
                value=ast.literal_eval(node)
            except (ValueError,TypeError,SyntaxError):
                continue
            if isinstance(value,(list,tuple)) and value and isinstance(value[0],(list,tuple)):
                if outcome(value)==expected:
                    return True
    return False


def diagnostics(outputs,weights,truth):
    truth=outcome(truth)
    if truth is None:
        raise ValueError('valid truth required')
    score=mixture_scores(outputs,truth,weights)
    mass=defaultdict(float)
    same_shape=0.
    errors=[]
    for value,w in zip(outputs,weights):
        value=outcome(value)
        mass[value]+=w
        if w>0 and value is not None and (len(value),len(value[0]))==(len(truth),len(truth[0])):
            same_shape+=w
            errors.append(sum(a!=b for row,trow in zip(value,truth) for a,b in zip(row,trow)))
    return {'truth_probability':score['exact_grid_probability'],
        'concentration':sum(p*p for p in mass.values()),
        'whole_grid_brier':score['whole_grid_brier'],
        'failure_probability':score['failure_probability'],
        'same_shape_probability':same_shape,
        'best_supported_same_shape_wrong_cells':min(errors) if errors else None,
        'truth_cells':len(truth)*len(truth[0])}


def audit(root):
    if hashlib.sha256((root/'result.json').read_bytes()).hexdigest()!=RESULT_SHA:
        raise ValueError('terminal result binding')
    report=json.loads((root/'result.json').read_text())
    if report['status']!='complete' or not report['endpoints_opened']:
        raise ValueError('endpoints not previously opened')
    names=['public.json','forecasts.json','targets.json']+[f'update_{i}.json' for i in range(6)]
    for name in names:
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=report['artifact_sha256'][name]:
            raise ValueError('saved artifact binding')
    public=json.loads((root/'public.json').read_text())
    forecasts=json.loads((root/'forecasts.json').read_text())
    targets=json.loads((root/'targets.json').read_text())
    tasks=[]
    all_gain=[]
    for i,(f,ys) in enumerate(zip(forecasts,targets)):
        u=json.loads((root/f'update_{i}.json').read_text())
        literal=[contains_literal_grid(code,public[i]['outputs'][0]) for code in u['arms']['python']['slots']]
        rows=[]
        for j,y in enumerate(ys):
            row={'index':j,'partition':'query' if j<2 else 'target'}
            for arm in ('python','dsl'):
                row[arm]=diagnostics(f[arm]['outputs'][j],f[arm]['weights'],y)
            truth_gain=row['python']['truth_probability']-row['dsl']['truth_probability']
            spreading_gain=(row['dsl']['concentration']-row['python']['concentration'])/2
            row['brier_improvement_decomposition']={'truth_mass':truth_gain,'concentration':spreading_gain}
            all_gain.append((truth_gain,spreading_gain))
            rows.append(row)
        tasks.append({'task_index':i,'literal_demo_output_slots':[j for j,v in enumerate(literal) if v],
            'literal_demo_output_posterior_mass':sum(w for w,v in zip(f['python']['weights'],literal) if v),
            'rows':rows})
    return {'status':'saved_diagnostic','terminal_sha256':RESULT_SHA,'tasks':tasks,
        'mean_brier_improvement_decomposition':{k:sum(r[j] for r in all_gain)/len(all_gain)
            for j,k in enumerate(('truth_mass','concentration'))},
        'interpretation':'literal presence is a flag, not proof that a memorized branch executes',
        'model_calls':0,'candidate_executions':0,'new_endpoint_labels':0,'cost_usd':0,
        'changes_gate':False,'depth_authorized':False}


if __name__=='__main__':
    result=audit(ROOT)
    save(OUTPUT,result)
    print(json.dumps({'status':result['status'],'decomposition':result['mean_brier_improvement_decomposition']}))
