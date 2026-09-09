"""Frozen four-task scene-interface qualification with sealed future outcomes."""
from scripts.rearc_scene_update import scene_update, ARMS
from scripts.rearc_named_plan_contract import body
from scripts.rearc_python_forecast import forecast_python_slots
from scripts.rearc_predictive_score import mixture_scores, outcome

PARTS = {'all': range(9), 'query': range(2), 'target': range(2, 9)}
METRICS = ('whole_grid_brier', 'fixed_canvas_brier',
           'exact_grid_probability', 'failure_probability')


def score(forecasts, labels):
    if len(forecasts) != 4 or len(labels) != 4 or any(len(y) != 9 for y in labels):
        raise ValueError('four-task endpoint coverage')
    losses = []
    covered = diverse = 0
    for row, ys in zip(forecasts, labels):
        if set(row) != set(ARMS) or any(len(f['outputs']) != 9 for f in row.values()):
            raise ValueError('two-arm forecast coverage')
        metrics = {a: [mixture_scores(o, y, row[a]['weights'])
                      for o, y in zip(row[a]['outputs'], ys)] for a in ARMS}
        covered += sum(r['exact_grid_probability'] > 0 for r in metrics['inventory'][:2])
        inv = row['inventory']
        diverse += any(len({outcome(o) for o, w in zip(outputs, inv['weights'])
                            if w > 0 and outcome(o) is not None}) >= 2
                       for outputs in inv['outputs'][:2])
        losses.append({a: {part: {k: sum(metrics[a][i][k] for i in indices) / len(indices)
                                  for k in METRICS}
                           for part, indices in PARTS.items()} for a in ARMS})
    means = {a: {part: {k: sum(t[a][part][k] for t in losses) / 4
                       for k in METRICS} for part in PARTS} for a in ARMS}
    gains = [t['raw']['all']['whole_grid_brier'] - t['inventory']['all']['whole_grid_brier']
             for t in losses]
    wins = sum(g > 0 for g in gains)
    gain = means['raw']['all']['whole_grid_brier'] - means['inventory']['all']['whole_grid_brier']
    truth_gain = (means['inventory']['all']['exact_grid_probability']
                  - means['raw']['all']['exact_grid_probability'])
    gates = {'answer_support': covered >= 6, 'predictive_disagreement': diverse >= 2,
             'paired_score': gain >= .01 and wins >= 2,
             'nonworse_canvas': means['inventory']['all']['fixed_canvas_brier']
                                <= means['raw']['all']['fixed_canvas_brier']}
    return {'status': 'complete', 'means': means, 'task_losses': losses,
            'paired_task_brier_gains': gains, 'task_wins': wins,
            'brier_gain_decomposition': {'truth_mass': truth_gain,
                                         'concentration': gain - truth_gain},
            'covered_query_answers': covered, 'diverse_tasks': diverse, 'gates': gates,
            'qualification_passed': all(gates.values()), 'depth_authorized': False,
            'endpoints_opened': True}


def collect(cases, dsl, request, evaluate, diagnose, bank_update, seal, targets):
    expected = {'inputs', 'outputs', 'query_inputs', 'target_inputs', 'target_hashes'}
    if len(cases) != 4 or any(set(c) != expected or any(len(c[k]) != n for k, n in
            [('inputs', 2), ('outputs', 2), ('query_inputs', 2),
             ('target_inputs', 7), ('target_hashes', 9)]) for c in cases):
        raise ValueError('public four-task coverage')
    forecasts = []
    for i, c in enumerate(cases):
        future = c['query_inputs'] + c['target_inputs']
        inputs = c['inputs'] + future
        observations = [{'index': j, 'output': y} for j, y in enumerate(c['outputs'])]
        def dispatch(stage, prompt, fmt):
            step = stage.split('_', 1)[1]
            seed = 51400 + 3*i + {'plan': 0, 'compile': 1, 'repair': 2}[step]
            return request(f'{i}_{stage}', body(prompt, seed, fmt))
        result = scene_update(inputs=inputs, observations=observations, dsl_source=dsl,
            request=dispatch, diagnose=diagnose,
            bank_update=lambda arm, value: bank_update(f'{i}_{arm}', value),
            order=ARMS if i % 2 == 0 else ARMS[::-1])
        case = {'inputs': c['inputs'], 'outputs': c['outputs'], 'target_inputs': future}
        forecasts.append({a: forecast_python_slots(result['arms'][a]['slots'], case, evaluate)
                          for a in ARMS})
    coverage = [not f['inventory']['conditioning']['failed'] for f in forecasts]
    if sum(coverage) < 3:
        return {'status': 'initial_coverage_null', 'two_example_covered': coverage,
                'qualification_passed': False, 'endpoints_opened': False,
                'depth_authorized': False}
    seal(forecasts)
    return {**score(forecasts, targets()), 'two_example_covered': coverage}
