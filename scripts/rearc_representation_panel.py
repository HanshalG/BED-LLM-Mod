"""Prospective six-task representation panel using unchanged predictive metrics."""
from scripts.rearc_representation_update import representation_update, ARMS
from scripts.rearc_named_plan_panel import score as predictive_score
from scripts.rearc_named_plan_contract import body
from scripts.rearc_program_graph import exports
from scripts.herb_candidate_expression import to_graph
from scripts.rearc_slot_forecast import forecast_slots
from scripts.rearc_python_forecast import forecast_python_slots


def score(forecasts, labels):
    mapping = {'contrasting':'python', 'ordinary':'dsl'}
    result = predictive_score([{k:row[v] for k,v in mapping.items()} for row in forecasts], labels)
    result['means'] = {mapping[k]:v for k,v in result['means'].items()}
    result['task_losses'] = [{mapping[k]:v for k,v in row.items()} for row in result['task_losses']]
    result['scope'] = 'representation qualification only; no sequential efficacy claim'
    return result


def collect(cases, dsl, request, evaluate_python, evaluate_dsl,
            diagnose_python, diagnose_dsl, bank_update, seal, targets):
    expected = {'inputs','outputs','query_inputs','target_inputs','target_hashes'}
    if len(cases)!=6 or any(set(c)!=expected or any(len(c[k])!=n for k,n in
            [('inputs',1),('outputs',1),('query_inputs',2),('target_inputs',8),('target_hashes',10)]) for c in cases):
        raise ValueError('public six-task coverage')
    functions, constants = exports(dsl)
    forecasts = []
    for i, case in enumerate(cases):
        def dispatch(stage, prompt, response_schema):
            offset = 0 if stage=='plan' else 1 if stage.endswith('_compile') else 2
            return request(f'{i}_{stage}', body(prompt,46400+3*i+offset,response_schema))
        result = representation_update(inputs=case['inputs']+case['query_inputs']+case['target_inputs'],
            observations=[{'index':0,'output':case['outputs'][0]}], dsl_source=dsl,
            request=dispatch, diagnose_python=diagnose_python, diagnose_dsl=diagnose_dsl,
            order=ARMS if i%2==0 else ARMS[::-1])
        bank_update(str(i), result)
        if any(len(result['arms'][a]['slots'])!=16 for a in ARMS):
            raise ValueError('program slot coverage')
        scoring_case = {'inputs':case['inputs'], 'outputs':case['outputs'],
                        'target_inputs':case['query_inputs']+case['target_inputs']}
        slots = [{'slot':j,'graph':to_graph(expr,functions,constants) if expr is not None else None}
                 for j,expr in enumerate(result['arms']['dsl']['slots'])]
        forecasts.append({
            'python':forecast_python_slots(result['arms']['python']['slots'],scoring_case,evaluate_python),
            'dsl':forecast_slots(slots,scoring_case,evaluate_dsl)})
    coverage = [not row['python']['conditioning']['failed'] for row in forecasts]
    if sum(coverage)<4:
        return {'status':'initial_coverage_null', 'initial_covered':coverage,
            'qualification_passed':False, 'depth_authorized':False, 'endpoints_opened':False}
    seal(forecasts)
    return {**score(forecasts, targets()), 'initial_covered':coverage}
