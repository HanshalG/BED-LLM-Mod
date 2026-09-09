"""Retrospective per-formula diagnosis on banked outputs, never new forecasts."""
import ast
import hashlib
import json
import math
from pathlib import Path

from environments.program_induction.scalar_expression import ScalarExpression
from scripts import physgym_semantic_probe as probe


def evaluate(expr,names,points):
    f=ScalarExpression(expr,[f'x{i}' for i in range(len(names))])
    results=[]
    for p in points:
        try:
            v=f({f'x{i}':p[n] for i,n in enumerate(names)})
            results.append(math.log(v) if v>0 else None)
        except ValueError:
            results.append(None)
    return results


def mse(predictions,labels):
    if len(predictions)!=len(labels) or not labels:
        raise ValueError('shape mismatch')
    return None if any(p is None for p in predictions) else sum((p-y)**2 for p,y in zip(predictions,labels))/len(labels)


def main():
    path=Path('results/nonmyopic/PHYSGYM_PROPOSAL_ERROR_AUDIT_20260909.json')
    if path.exists():
        raise ValueError('already banked')
    verified=probe.replay(probe.ROOT)
    public=json.loads((probe.ROOT/'public.json').read_text())
    forecasts=json.loads((probe.ROOT/'forecasts.json').read_text())
    truth=json.loads((probe.ROOT/'outcomes.json').read_text())
    terminal=json.loads((probe.ROOT/'result.json').read_text())
    cases={}
    for t,c in public.items():
        rows={}
        for arm,exprs in forecasts[t]['pools'].items():
            rows[arm]=[]
            for expr in exprs:
                history=evaluate(expr,c['names'],[p for p,y in c['history']])
                target=evaluate(expr,c['names'],c['targets'])
                valid=[j for j,v in enumerate(target) if v is not None]
                rows[arm].append(dict(expr=expr,initial_mse=mse(history[:3],[y for p,y in c['history'][:3]]),
                    four_observation_mse=mse(history,[y for p,y in c['history']]),
                    invalid_target_indices=[j for j,v in enumerate(target) if v is None],
                    full_target_mse=mse(target,truth[t]),
                    valid_subset_mse=mse([target[j] for j in valid],[truth[t][j] for j in valid]) if valid else None,
                    valid_subset_not_comparable_to_full_score=True))
        arms={}
        for arm in ('semantic','blind','refresh','redraw'):
            pool=rows[arm] if arm in ('semantic','blind') else rows['semantic']+rows[arm]
            unique={ast.dump(ast.parse(r['expr'],mode='eval')):r for r in pool}
            hkey='initial_mse' if arm in ('semantic','blind') else 'four_observation_mse'
            valid=[r for r in unique.values() if r[hkey] is not None and r['full_target_mse'] is not None]
            best=min((r['full_target_mse'] for r in valid),default=None)
            arms[arm]=dict(valid_formulas=len(valid),saved_mse=terminal['losses'][t][arm],
                           hindsight_best_single_mse=best,
                           not_convex_mixture_optimum=True,
                           best_training_formula_target_mse=(min(valid,key=lambda r:r[hkey])['full_target_mse'] if valid else None))
        cases[t]=dict(formulas=rows,arms=arms)
    result=dict(cases=cases,source_replay=verified,model_calls=0,cost_usd=0,
                retrospective=True,new_outcomes_opened=False,paid_authority=False,
                source_terminal_sha256=hashlib.sha256((probe.ROOT/'result.json').read_bytes()).hexdigest())
    with path.open('x') as output:
        json.dump(result,output,indent=2,sort_keys=True,allow_nan=False)
        output.write('\n')
    print(json.dumps({t:r['arms'] for t,r in cases.items()},sort_keys=True))


if __name__=='__main__':
    main()
