"""Training-only scale calibration diagnosis; not a rerun of the frozen gate."""
import json
from pathlib import Path

from scripts.physgym_proposal_error_audit import evaluate, mse


def scale_fit(predictions,labels):
    if not labels or len(predictions)!=len(labels) or any(v is None for v in predictions):
        raise ValueError('invalid training arrays')
    return sum(y-p for p,y in zip(predictions,labels))/len(labels)


def main():
    root=Path('results/nonmyopic/physgym_semantic_history_20260909')
    path=Path('results/nonmyopic/PHYSGYM_SCALE_ERROR_AUDIT_20260909.json')
    if path.exists():
        raise ValueError('already banked')
    public=json.loads((root/'public.json').read_text())
    forecasts=json.loads((root/'forecasts.json').read_text())
    outcomes=json.loads((root/'outcomes.json').read_text())
    result={}
    for t,case in public.items():
        result[t]={}
        for arm,pool in forecasts[t]['pools'].items():
            result[t][arm]=[]
            for e in pool:
                h=evaluate(e,case['names'],[p for p,y in case['history']])
                target=evaluate(e,case['names'],case['targets'])
                if any(v is None for v in h):
                    result[t][arm].append(dict(status='invalid_training_formula'))
                    continue
                labels=[y for p,y in case['history']]
                offset=scale_fit(h,labels)
                result[t][arm].append(dict(status='complete',expression=e,log_scale=offset,
                    training_mse=mse([v+offset for v in h],labels),
                    target_mse=mse([None if v is None else v+offset for v in target],outcomes[t]),
                    invalid_targets=sum(v is None for v in target)))
    with path.open('x') as output:
        json.dump(dict(cases=result,model_calls=0,cost_usd=0,retrospective=True,
                       gate_recomputed=False,paid_authority=False),output,indent=2,sort_keys=True,allow_nan=False)
        output.write('\n')
    print(json.dumps(result['457'],sort_keys=True))


if __name__=='__main__':
    main()
