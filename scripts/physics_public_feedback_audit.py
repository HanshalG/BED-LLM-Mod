"""Exercise prospective feedback on banked histories; no endpoint file access."""
import hashlib
import json
from pathlib import Path
from environments.program_induction.physics_feedback import feedback


def main():
    root=Path('results/nonmyopic/physgym_semantic_history_20260909')
    destination=Path('results/nonmyopic/PHYSICS_PUBLIC_FEEDBACK_AUDIT_20260909.json')
    if destination.exists():
        raise ValueError('already banked')
    raw=(root/'forecasts.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!='630a1eda7068e0087cbec526da4b929f3bb125efd1e4f13e9123e98882b54a27':
        raise ValueError('banked proposal binding mismatch')
    pools=json.loads(raw)
    public=json.loads((root/'public.json').read_text())
    cases={t:{arm:feedback(expressions,public[t]['names'],public[t]['history'])
              for arm,expressions in panel['pools'].items()} for t,panel in pools.items()}
    result=dict(cases=cases,model_calls=0,cost_usd=0,endpoint_labels_read=False,
                paid_authority=False,implementation_sha256=hashlib.sha256(
                    Path('environments/program_induction/physics_feedback.py').read_bytes()).hexdigest())
    with destination.open('x') as out:
        json.dump(result,out,indent=2,sort_keys=True,allow_nan=False)
        out.write('\n')
    print(json.dumps({t:{arm:[dict(invalid_guard_count=r['invalid_guard_count'],
                    maximum_standardized_residual=max(map(abs,r.get('standardized_residuals',[0])))) for r in rs]
                  for arm,rs in arms.items()} for t,arms in cases.items()},sort_keys=True))


if __name__=='__main__':
    main()
