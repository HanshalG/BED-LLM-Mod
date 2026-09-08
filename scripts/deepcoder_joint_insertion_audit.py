"""Reconstruct pre-answer insertion worlds and audit their joint predictions."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from environments.program_induction import constrained
from environments.program_induction.local_support import evaluate, expand
from environments.program_induction.prior import restricted_weights
from environments.program_induction.rollout_risk import expected_brier
from environments.program_induction.structural_support import prepare
from scripts import deepcoder_luna_transition as g


def main():
    terminal_bytes = (g.ROOT/'result.json').read_bytes()
    terminal_sha = hashlib.sha256(terminal_bytes).hexdigest()
    if terminal_sha != 'a3aa0e8cb9cc92012b4419817c0108befdd9983ed1494e2b8e3701221cc53942':
        raise ValueError('banked terminal changed')
    terminal = json.loads(terminal_bytes)
    replay = g.screen.score_sealed(g.ROOT/'forecasts.json', terminal['forecast_sha256'],
        lambda: json.loads((g.ROOT/'outcomes.json').read_text()))
    if replay['scores'] != terminal['scores']:
        raise ValueError('score replay mismatch')
    public = json.loads((g.ROOT/'public.json').read_text())
    forecasts = json.loads((g.ROOT/'forecasts.json').read_text())['forecasts']
    dsl, rows = g.load_dsl(), {}
    for key, case in public.items():
        raw = json.loads((g.ROOT/(key+'_initial.response.json')).read_text())
        roots = constrained.decode(dsl, g.luna.validate_response(raw))
        previous, _ = expand(dsl, roots, case['history'])
        worlds, work = prepare(dsl, previous, case['history'])
        saved = json.loads((g.ROOT/(key+'.repair.json')).read_text())
        if [str(p) for p in worlds] != saved['programs'] or work != saved['work']:
            raise ValueError('pre-answer support reconstruction mismatch')
        weights = restricted_weights(dsl, worlds)
        answer_law = {}
        for p in worlds:
            label = g.pred.category(evaluate(p, case['query']))
            answer_law[label] = answer_law.get(label, Fraction()) + weights[str(p)]
        # This retrospective answer lookup occurs only after rebuilding its law.
        obs = json.loads((g.ROOT/(key+'.observation.json')).read_text())
        if obs['inputs'] != case['query']:
            raise ValueError('query mismatch')
        probability = answer_law.get(g.pred.category(obs['output']), Fraction())
        branch = dict(case, history=case['history']+[obs])
        reference = g.screen.program_forecast(dsl, worlds, branch)
        if reference != forecasts[key]['insertion_repair']:
            raise ValueError('conditional target reconstruction mismatch')
        risks = {}
        if reference is not None:
            for arm, q in forecasts[key].items():
                distributions = q['distributions'] if q else [None]*32
                risks[arm] = sum(expected_brier(p, q)['expected_loss'] for p, q in
                    zip(reference['distributions'], distributions))/32
        rows[key] = dict(pre_answer_worlds=len(worlds), answer_probability=float(probability),
            answer_probability_exact=str(probability),
            answer_law={k:float(v) for k,v in answer_law.items()},
            conditional_target_support=reference['support_size'] if reference else 0,
            simulator_expected_brier=risks,
            realized_brier={a:r['brier'] for a,r in terminal['scores'][key].items()})
    report = dict(cases=rows, parent_terminal_sha256=terminal_sha,
        model_calls=0, cost_usd=0, new_hidden_answers_opened=False,
        interpretation='retrospective_joint_law_diagnostic_not_fresh_calibration',
        depth_authorized=False)
    g.save(Path('results/nonmyopic/DEEPCODER_JOINT_INSERTION_AUDIT_20260909.json'), report, exclusive=True)
    print(g.pred.canonical({k:{a:v for a,v in row.items() if a!='answer_law'}
                           for k,row in rows.items()}))


if __name__ == '__main__':
    main()
