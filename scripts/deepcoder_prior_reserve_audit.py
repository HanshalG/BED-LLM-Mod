"""Retrospective independent-prior reserve diagnostic on opened transition cases."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import monotonic
from environments.program_induction.local_support import evaluate
from scripts import deepcoder_luna_transition as g


def reserve(draw, execute, history, draws):
    if type(draws) is not int or draws <= 0:
        raise ValueError('positive fixed draw count required')
    retained = []
    evaluations = 0
    for i in range(draws):
        p = draw(i)
        fits = True
        for h in history:
            evaluations += 1
            if execute(p,h['inputs']) != h['output']:
                fits = False
                break
        if fits:
            retained.append(p)
    return retained, evaluations


def main():
    terminal = g.ROOT/'result.json'
    if hashlib.sha256(terminal.read_bytes()).hexdigest() != 'a3aa0e8cb9cc92012b4419817c0108befdd9983ed1494e2b8e3701221cc53942':
        raise ValueError('banked transition result changed')
    public = json.loads((g.ROOT/'public.json').read_text())
    d, rows = g.load_dsl(), {}
    start = monotonic()
    for key, case in public.items():
        seed = 27100000+10000*int(key)
        pool, work = reserve(lambda i:g.sample_program(d,seed+i), evaluate, case['history'],4096)
        # Only after sampling/conditioning, compare the already-opened third answer.
        predictions = Counter(g.pred.category(evaluate(p,case['query'])) for p in pool)
        obs = json.loads((g.ROOT/(key+'.observation.json')).read_text())
        if obs['inputs'] != case['query']:
            raise ValueError('observation identity mismatch')
        count = predictions.get(g.pred.category(obs['output']),0)
        rows[key] = dict(draws=4096, history_evaluations=work, retained=len(pool),
                         unique_syntax=len({str(p) for p in pool}),
                         third_prediction_counts=dict(predictions), realized_answer_count=count,
                         realized_answer_probability=count/len(pool) if pool else None,
                         programs_sha256=g.pred.digest([str(p) for p in pool]), start_seed=seed)
    report = dict(cases=rows, elapsed_seconds=monotonic()-start, model_calls=0,cost_usd=0,
                  new_hidden_answers_opened=False, predictive_mixture_authorized=False,
                  interpretation='fixed_draw_conditional_prior_reserve_retrospective_diagnostic')
    g.save(Path('results/nonmyopic/DEEPCODER_PRIOR_RESERVE_AUDIT_20260909.json'),report,exclusive=True)
    print(g.pred.canonical(rows))


if __name__ == '__main__':
    main()
