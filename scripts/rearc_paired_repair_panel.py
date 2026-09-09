"""Shared-proposal feedback comparison with frozen root-predictive gates."""
from scripts.rearc_paired_repair import paired_update, ARMS
from scripts.rearc_named_plan_panel import score as predictive_score
from scripts.rearc_named_plan_contract import body
from scripts.rearc_program_graph import exports
from scripts.herb_candidate_expression import to_graph
from scripts.rearc_slot_forecast import forecast_slots


def score(forecasts, labels, active):
    if len(active)!=6 or any(type(x) is not bool for x in active):
        raise ValueError('intervention coverage')
    # Reuse the identical mathematical gates, with explicit arm-name translation.
    mapping = {'contrasting':'actionable', 'ordinary':'generic'}
    old = [{key:row[value] for key,value in mapping.items()} for row in forecasts]
    result = predictive_score(old, labels)
    result['means'] = {mapping[k]:v for k,v in result['means'].items()}
    result['task_losses'] = [{mapping[k]:v for k,v in row.items()} for row in result['task_losses']]
    result['intervention_active'] = active
    result['gates']['intervention_identifiable'] = sum(active)>=2
    result['qualification_passed'] = all(result['gates'].values())
    return result


def collect(cases, dsl, request, evaluate, diagnose, bank_update, seal, targets):
    expected = {'inputs','outputs','query_inputs','target_inputs','target_hashes'}
    if len(cases)!=6 or any(set(c)!=expected or any(len(c[k])!=n for k,n in
            [('inputs',1),('outputs',1),('query_inputs',2),('target_inputs',8),('target_hashes',10)]) for c in cases):
        raise ValueError('public six-task coverage')
    functions, constants = exports(dsl)
    forecasts, active = [], []
    for i,case in enumerate(cases):
        def dispatch(stage, prompt, schema):
            offset = 0 if stage=='plan' else 1 if stage=='compile' else 2
            return request(f'{i}_{stage}', body(prompt,42400+3*i+offset,schema))
        result = paired_update(inputs=case['inputs']+case['query_inputs']+case['target_inputs'],
            observations=[{'index':0,'output':case['outputs'][0]}],dsl_source=dsl,
            request=dispatch,diagnose=diagnose,order=ARMS if i%2==0 else ARMS[::-1])
        bank_update(str(i),result)
        active.append(result['compiler_intervention_active'])
        row = {}
        for arm in ARMS:
            slots = [{'slot':j,'graph':to_graph(expr,functions,constants) if expr is not None else None}
                for j,expr in enumerate(result['arms'][arm]['slots'])]
            row[arm] = forecast_slots(slots,{'inputs':case['inputs'],'outputs':case['outputs'],
                'target_inputs':case['query_inputs']+case['target_inputs']},evaluate)
        forecasts.append(row)
    covered = [not row['actionable']['conditioning']['failed'] for row in forecasts]
    if sum(covered)<4:
        return {'status':'initial_coverage_null','initial_covered':covered,'intervention_active':active,
            'qualification_passed':False,'depth_authorized':False,'endpoints_opened':False}
    seal(forecasts)
    return {**score(forecasts,targets(),active),'initial_covered':covered}
