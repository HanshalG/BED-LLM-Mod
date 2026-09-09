"""Fixed six-task root-predictive qualification; no sequential efficacy claim."""
from scripts.rearc_mechanism_update import update
from scripts.rearc_mechanism_contract import body
from scripts.rearc_program_graph import exports
from scripts.herb_candidate_expression import to_graph
from scripts.rearc_slot_forecast import forecast_slots
from scripts.rearc_predictive_score import mixture_scores,outcome

ARMS = ('contrasting','ordinary')


def diversity(forecasts):
    return sum(any(len({outcome(y) for y,w in zip(values,row['contrasting']['weights'])
                        if w>0 and outcome(y) is not None})>=2
                   for values in row['contrasting']['outputs'][:2]) for row in forecasts)


def score(forecasts,labels):
    if len(forecasts)!=6 or len(labels)!=6 or any(len(row)!=10 for row in labels):
        raise ValueError('endpoint coverage')
    losses = []
    covered = 0
    keys = ('whole_grid_brier','fixed_canvas_brier')
    for f,truth in zip(forecasts,labels):
        if set(f)!=set(ARMS) or any(len(f[a]['outputs'])!=10 for a in ARMS):
            raise ValueError('forecast coverage')
        record = {}
        for arm in ARMS:
            values = [mixture_scores(outputs,y,f[arm]['weights']) for outputs,y in zip(f[arm]['outputs'],truth)]
            record[arm] = {part:{k:sum(values[i][k] for i in indices)/len(indices) for k in keys}
                for part,indices in (('all',list(range(10))),('query',[0,1]),('target',list(range(2,10))))}
            if arm=='contrasting':
                covered += sum(row['exact_grid_probability']>0 for row in values[:2])
        losses.append(record)
    means = {arm:{part:{k:sum(row[arm][part][k] for row in losses)/6 for k in keys}
                   for part in ('all','query','target')} for arm in ARMS}
    wins = sum(row['ordinary']['all']['whole_grid_brier']-row['contrasting']['all']['whole_grid_brier']>=.01 for row in losses)
    diverse = diversity(forecasts)
    gates = {'answer_support':covered>=8,'predictive_disagreement':diverse>=3,
        'paired_score':means['ordinary']['all']['whole_grid_brier']-means['contrasting']['all']['whole_grid_brier']>=.01 and wins>=2,
        'nonworse_canvas':means['contrasting']['all']['fixed_canvas_brier']<=means['ordinary']['all']['fixed_canvas_brier']}
    return {'status':'complete','means':means,'task_losses':losses,'covered_query_answers':covered,
        'diverse_tasks':diverse,'task_wins':wins,'gates':gates,'qualification_passed':all(gates.values()),
        'endpoints_opened':True,'depth_authorized':False}


def collect(cases,dsl,request,evaluate,diagnose,bank_update,seal,targets):
    expected = {'inputs','outputs','query_inputs','target_inputs','target_hashes'}
    if (len(cases)!=6 or any(set(c)!=expected or len(c['inputs'])!=1 or len(c['outputs'])!=1
            or len(c['query_inputs'])!=2 or len(c['target_inputs'])!=8 or len(c['target_hashes'])!=10 for c in cases)):
        raise ValueError('public six-task coverage')
    functions,constants = exports(dsl)
    forecasts = []
    for i,case in enumerate(cases):
        row = {}
        for arm in (ARMS if i%2==0 else ARMS[::-1]):
            def dispatch(stage,messages,format):
                seed = 38400+3*i+{'plan':0,'compile':1,'repair':2}[stage]
                return request(f'{i}_{arm}_{stage}',body(messages,seed,format))
            result = update(mode=arm,inputs=case['inputs']+case['query_inputs']+case['target_inputs'],
                observations=[{'index':0,'output':case['outputs'][0]}],dsl_source=dsl,
                request=dispatch,diagnose=diagnose)
            if len(result['slots'])!=16:
                raise ValueError('program slot coverage')
            slots = [{'slot':j,'graph':to_graph(expr,functions,constants) if expr is not None else None}
                     for j,expr in enumerate(result['slots'])]
            bank_update(f'{i}_{arm}',{'proposal':result,'slots':slots})
            row[arm] = forecast_slots(slots,{'inputs':case['inputs'],'outputs':case['outputs'],
                'target_inputs':case['query_inputs']+case['target_inputs']},evaluate)
        forecasts.append(row)
    coverage = [not row['contrasting']['conditioning']['failed'] for row in forecasts]
    if sum(coverage)<4:
        return {'status':'initial_coverage_null','initial_covered':coverage,'qualification_passed':False,
                'endpoints_opened':False,'depth_authorized':False}
    seal(forecasts)
    return {**score(forecasts,targets()),'initial_covered':coverage}
