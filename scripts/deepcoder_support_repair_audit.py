"""Retrospective recovery mechanics on all already-opened predictive outcomes."""
import hashlib
import json
from pathlib import Path
from environments.program_induction import constrained
from environments.program_induction.local_support import expand
from environments.program_induction.support_repair import prepare, condition
from scripts import deepcoder_luna_predictive as g


def main():
    result = g.ROOT/'result.json'
    if hashlib.sha256(result.read_bytes()).hexdigest() != '48e5cb3d5dc2c2af3ff6958ef0c392660d9f34ad2ef56c2dd88076a224f19e45':
        raise ValueError('banked result mismatch')
    cases = json.loads((g.ROOT/'public.json').read_text())
    answers = json.loads((g.ROOT/'outcomes.json').read_text())
    d, rows = g.load_dsl(), {}
    for key, case in cases.items():
        raw = json.loads((g.ROOT/(key+'_history_aware.response.json')).read_text())
        roots = constrained.decode(d, g.luna.validate_response(raw))
        previous, _ = expand(d, roots, case['history'])
        # The candidate-generation interface has no new observation argument.
        neighbors, work = prepare(d, previous, case['history'])
        trials = []
        for x,y in zip(answers[key]['target_inputs'],answers[key]['outputs']):
            obs = dict(inputs=x, output=y)
            before = condition(previous, case['history'], obs)
            after = condition(neighbors, case['history'], obs)
            trials.append(dict(original_survivors=len(before), repair_survivors=len(after)))
        rows[key] = dict(previous_support=len(previous), pre_answer_neighbor_support=len(neighbors),
            work=work, trials=trials, unsupported_before=sum(r['original_survivors']==0 for r in trials),
            unsupported_after=sum(r['repair_survivors']==0 for r in trials))
    report = dict(cases=rows, model_calls=0, cost_usd=0, new_hidden_answers_opened=False,
                  interpretation='retrospective_one_observation_repair_not_generalization', depth_authorized=False)
    g.save(Path('results/nonmyopic/DEEPCODER_SUPPORT_REPAIR_AUDIT_20260909.json'),report,exclusive=True)
    print(g.pred.canonical({k:{a:v for a,v in r.items() if a!='trials'} for k,r in rows.items()}))


if __name__ == '__main__':
    main()
