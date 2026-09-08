"""Retrospective best-possible reweighting of already-opened feedback supports."""
import json
from pathlib import Path

from environments.program_induction.support_ceiling import oracle_mixture
from scripts import deepcoder_feedback_probe as g
from scripts.deepcoder_feedback_replay import verify


def main():
    verified=verify(g.ROOT)
    d=g.load_dsl()
    public=json.loads((g.ROOT/'public.json').read_text())
    truth=json.loads((g.ROOT/'outcomes.json').read_text())
    terminal=json.loads((g.ROOT/'result.json').read_text())
    cases={}
    for key,case in public.items():
        cases[key]={}
        for arm in ('initial','feedback','control'):
            raw=json.loads((g.ROOT/(key+'_'+arm+'.response.json')).read_text())
            ps=g.constrained.decode(d,g.luna.validate_response(raw))
            pool,_=g.expand(d,ps,case['history'])
            if arm=='initial':
                initial=pool
            support={str(p):p for p in (pool if arm=='initial' else initial+pool)}
            rows=[[g.pred.category(g.evaluate(p,x)) for x in case['targets']] for p in support.values()]
            labels=[g.pred.category(y) for y in truth[key]['outputs']]
            ceiling=oracle_mixture(rows,labels)
            current=terminal['scores'][key][arm]['brier']
            if ceiling['lower']>current+1e-9:
                raise ValueError('lower bound exceeds feasible saved forecast')
            cases[key][arm]=dict(saved_brier=current,oracle=ceiling)
    means={a:{name:sum(r[a]['oracle'][name] for r in cases.values())/8 for name in ['lower','upper','pointwise_floor']}
           for a in g.screen.ARMS}
    report=dict(cases=cases,mean_oracle=means,source_replay=verified,model_calls=0,cost_usd=0,
        new_hidden_answers_opened=False,depth_authorized=False,
        interpretation='retrospective_optimistic_ceiling_not_calibrated_weights_or_new_efficacy')
    g.save(Path('results/nonmyopic/DEEPCODER_FEEDBACK_SUPPORT_CEILING_20260909.json'),report,exclusive=True)
    print(g.pred.canonical(means))


if __name__=='__main__':
    main()
