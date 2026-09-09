"""Retrospective exact-short symbolic control on all opened feedback cases."""
import json
from pathlib import Path
from time import monotonic

from environments.program_induction.short_support import enumerate_two_statement
from environments.program_induction.independent_joint_screen import _loss
from scripts import deepcoder_feedback_probe as g
from scripts.deepcoder_feedback_replay import verify


def main():
    parent=verify(g.ROOT)
    d=g.load_dsl()
    public=json.loads((g.ROOT/'public.json').read_text())
    truth=json.loads((g.ROOT/'outcomes.json').read_text())
    old=json.loads((g.ROOT/'result.json').read_text())
    cases={}
    for key,case in public.items():
        start=monotonic()
        ps,work=enumerate_two_statement(d,case['history'])
        forecast=g.program_forecast(d,ps,case)
        seconds=monotonic()-start
        labels=[g.pred.category(y) for y in truth[key]['outputs']]
        score=_loss(forecast['distributions'] if forecast else None,labels)
        # Target behavior equality is descriptive on the opened panel only.
        behaviors={tuple(g.pred.category(g.evaluate(p,x)) for x in case['targets']) for p in ps}
        cases[key]=dict(work=work,seconds=seconds,forecast=forecast,score=score,
            target_behaviors=len(behaviors),initial_luna_score=old['scores'][key]['initial'])
    report=dict(cases=cases,source_replay=parent,model_calls=0,cost_usd=0,
        mean_brier=sum(c['score']['brier'] for c in cases.values())/8,
        zero_mass_targets=sum(c['score']['zero_mass_targets'] for c in cases.values()),
        complete_support_cases=sum(c['forecast'] is not None for c in cases.values()),
        depth_authorized=False,new_hidden_answers_opened=False,
        interpretation='retrospective_two_statement_component_control_not_full_posterior_or_llm_result')
    g.save(Path('results/nonmyopic/DEEPCODER_SHORT_CONTROL_AUDIT_20260909.json'),report,exclusive=True)
    print(g.pred.canonical({k:v for k,v in report.items() if k not in ['cases','source_replay']}))


if __name__=='__main__':
    main()
