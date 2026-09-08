"""Audit only displayed histories from the completed independent-joint run."""
import hashlib
import json
from pathlib import Path

from environments.program_induction.execution_feedback import checks
from scripts import deepcoder_independent_joint as g


def main():
    terminal = (g.ROOT/'result.json').read_bytes()
    if hashlib.sha256(terminal).hexdigest() != '0c361b625830d76644739fd73064a52639b85d8c9cbed93a7e34199fb7687a27':
        raise ValueError('terminal identity mismatch')
    dsl=g.load_dsl()
    public=json.loads((g.ROOT/'public.json').read_text())
    cases={}
    for key,case in public.items():
        cases[key]={}
        obs=json.loads((g.ROOT/(key+'.observation.json')).read_text())
        for arm in ('a','b','regenerated','repeat'):
            raw=json.loads((g.ROOT/(key+'_'+arm+'.response.json')).read_text())
            ps=g.constrained.decode(dsl,g.luna.validate_response(raw))
            history=case['history']+[obs] if arm=='regenerated' else case['history']
            evidence=checks(dsl,ps,history)
            cases[key][arm]=dict(raw_programs=len(ps),compatible=sum(r['fits_observed_history'] for r in evidence),
                execution_feedback=evidence)
    report=dict(cases=cases,model_calls=0,cost_usd=0,target_outcomes_read=False,
                raw_programs=sum(r['raw_programs'] for c in cases.values() for r in c.values()),
                compatible=sum(r['compatible'] for c in cases.values() for r in c.values()))
    g.save(Path('results/nonmyopic/DEEPCODER_PUBLIC_FIT_AUDIT_20260909.json'),report,exclusive=True)
    print(g.pred.canonical({k:v for k,v in report.items() if k!='cases'}))


if __name__=='__main__':
    main()
