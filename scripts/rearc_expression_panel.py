"""Four-task prospective qualification; no target access before sealed forecasts."""
from scripts.rearc_expression_update import update
from scripts.rearc_expression_interface import response_format
from scripts.rearc_luna_qualification import validate_body
from scripts.herb_candidate_expression import to_graph
from scripts.rearc_program_graph import exports
from scripts.rearc_program_weights import condition_programs
from scripts.rearc_expression_forecast import forecast
from scripts.rearc_predictive_score import mixture_scores


def body(messages, seed):
    value = {'model': 'openai/gpt-5.6-luna', 'max_tokens': 16384,
             'reasoning': {'enabled': True, 'effort': 'medium', 'exclude': True}, 'seed': seed,
             'provider': {'only': ['openai'], 'allow_fallbacks': False, 'require_parameters': True,
                          'max_price': {'prompt': .2, 'completion': 1.2}},
             'messages': messages, 'response_format': response_format()}
    validate_body(value)
    return value


def collect(cases, dsl, request, evaluate, diagnose, search, bank_update, seal, targets):
    if len(cases) != 4 or any(len(c['inputs']) != 3 or len(c['outputs']) != 3 or len(c['target_inputs']) != 8 for c in cases):
        raise ValueError('four complete cases required')
    functions, constants = exports(dsl)
    def build(i, arm):
        case = cases[i]
        visible = 3 if arm == 'aware' else 1
        seed = (34300 if arm == 'initial' else 34400)+2*i
        proposed = update(inputs=case['inputs']+case['target_inputs'],
            observations=[{'index': j, 'output': case['outputs'][j]} for j in range(visible)],
            dsl_source=dsl, request=lambda phase,m: request(f'{i}_{arm}_{phase}', body(m, seed+(phase=='repair'))),
            diagnose=diagnose)
        generated = search(proposed['graphs'], 56, 50000)
        if len(generated) != 56 or len(proposed['slots']) != 8:
            raise ValueError('candidate coverage')
        graphs = proposed['graphs']+[to_graph(expr, functions, constants) for expr in generated]
        bank_update(f'{i}_{arm}', {'proposal': proposed, 'search_expressions': generated, 'graphs': graphs})
        return graphs, proposed['graphs']
    initial = [build(i, 'initial') for i in range(4)]
    covered = [condition_programs(graphs, [evaluate(g,c['inputs'][:1]) for g in graphs], c['outputs'][:1])['consistent_programs'] > 0
               for (graphs,_), c in zip(initial, cases)]
    if sum(covered) < 3:
        return {'status': 'initial_coverage_null', 'initial_covered': covered,
                'qualification_passed': False, 'endpoints_opened': False, 'depth_authorized': False}
    forecasts = []
    for i, case in enumerate(cases):
        pools = {'initial': initial[i][0]}
        for arm in (('aware', 'blind') if i%2 == 0 else ('blind', 'aware')):
            graphs, direct = build(i, arm)
            pools[arm] = initial[i][0]+graphs
            if arm == 'aware':
                pools['direct'] = initial[i][1]+direct
        baseline = search([], 128, 100000)
        if len(baseline) != 128:
            raise ValueError('symbolic candidate coverage')
        pools['symbolic'] = [to_graph(expr, functions, constants) for expr in baseline]
        forecasts.append({arm: forecast(graphs, case, evaluate) for arm,graphs in pools.items()})
    seal(forecasts)
    labels = targets()
    if len(labels) != 4 or any(len(row) != 8 for row in labels):
        raise ValueError('target coverage')
    keys = ('whole_grid_brier', 'fixed_canvas_brier')
    losses = []
    for fs, ys in zip(forecasts, labels):
        row = {}
        for arm,f in fs.items():
            scored = [mixture_scores(pred, y, f['weights']) for pred,y in zip(f['outputs'], ys)]
            row[arm] = {k: sum(s[k] for s in scored)/8 for k in keys}
        losses.append(row)
    means = {a: {k: sum(r[a][k] for r in losses)/4 for k in keys} for a in forecasts[0]}
    a,b = means['aware']['whole_grid_brier'], means['blind']['whole_grid_brier']
    wins = sum(r['blind']['whole_grid_brier']-r['aware']['whole_grid_brier'] >= .01 for r in losses)
    passed = b-a >= .01 and a <= .9*b and wins >= 2 and all(
        means['aware'][k] <= means[c][k] for k in keys for c in ('initial','blind','symbolic'))
    return {'status': 'complete', 'initial_covered': covered, 'means': means, 'task_losses': losses,
            'task_wins': wins, 'qualification_passed': passed, 'endpoints_opened': True, 'depth_authorized': False}
