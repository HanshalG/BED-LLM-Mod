"""Dependency-injected qualification logic; no direct network or source access."""
import json
from scripts.rearc_proposal_interface import build_messages, response_format, parse_response
from scripts.rearc_program_weights import condition_programs
from scripts.rearc_predictive_score import mixture_scores


def request_body(case, arm, index, dsl_source):
    visible = 3 if arm == 'aware' else 1
    if arm not in {'initial','aware','blind'}:
        raise ValueError('unknown arm')
    body = {'model':'openai/gpt-5.6-luna','max_tokens':16384,
            'reasoning':{'enabled':True,'effort':'medium','exclude':True},
            'seed':31300+index+(100 if arm!='initial' else 0),
            'provider':{'only':['openai'],'allow_fallbacks':False,'require_parameters':True,
                        'max_price':{'prompt':.2,'completion':1.2}},
            'messages':build_messages(inputs=case['inputs'],
                observations=[{'index':i,'output':case['outputs'][i]} for i in range(visible)],
                dsl_source=dsl_source),'response_format':response_format()}
    if len(json.dumps(body).encode())>32768:
        raise ValueError('complete request byte cap')
    return body


def forecast(graphs, case, evaluate):
    predictions=[evaluate(graph,case['inputs']+case['target_inputs']) for graph in graphs]
    if any(len(row)!=11 for row in predictions):
        raise ValueError('complete demonstration and target predictions required')
    conditioned=condition_programs(graphs,[row[:3] for row in predictions],case['outputs'])
    if conditioned['failed']:
        return {'conditioning':conditioned,'outputs':[[None] for _ in range(8)],'weights':[1.]}
    return {'conditioning':conditioned,'outputs':[[row[3+j] for row in predictions] for j in range(8)],
            'weights':conditioned['weights']}


def collect(cases, dsl_source, request_text, evaluate, symbolic, seal_forecasts, open_targets):
    if len(cases)!=4 or any(len(c['inputs'])!=3 or len(c['outputs'])!=3 or len(c['target_inputs'])!=8 for c in cases):
        raise ValueError('exact four-task panel required')
    initial=[]
    for i,case in enumerate(cases):
        graphs=parse_response(request_text(f'{i}_initial',request_body(case,'initial',i,dsl_source)),dsl_source)
        first=[evaluate(graph,case['inputs'][:1]) for graph in graphs]
        if not condition_programs(graphs,first,case['outputs'][:1])['consistent_programs']:
            raise ValueError('initial semantic gate failed before refresh or endpoints')
        initial.append(graphs)
    predictions=[]
    for i,case in enumerate(cases):
        pools={'initial':initial[i]}
        for arm in (('aware','blind') if i%2==0 else ('blind','aware')):
            pools[arm]=initial[i]+parse_response(request_text(f'{i}_{arm}',request_body(case,arm,i,dsl_source)),dsl_source)
        pools['symbolic']=symbolic(case['inputs'],case['outputs'])
        if not pools['symbolic']:
            raise ValueError('symbolic runtime returned no programs')
        predictions.append({arm:forecast(pool,case,evaluate) for arm,pool in pools.items()})
    seal_forecasts(predictions)
    targets=open_targets()
    if len(targets)!=4 or any(len(row)!=8 for row in targets):
        raise ValueError('endpoint coverage')
    losses=[]
    for task,truths in zip(predictions,targets):
        row={}
        for arm,forecasted in task.items():
            scores=[mixture_scores(values,truth,forecasted['weights']) for values,truth in zip(forecasted['outputs'],truths)]
            row[arm]={key:sum(s[key] for s in scores)/8 for key in ('whole_grid_brier','fixed_canvas_brier')}
        losses.append(row)
    means={arm:{key:sum(row[arm][key] for row in losses)/4 for key in ('whole_grid_brier','fixed_canvas_brier')}
           for arm in ('initial','aware','blind','symbolic')}
    a,b=means['aware'],means['blind']
    wins=sum(row['blind']['whole_grid_brier']-row['aware']['whole_grid_brier']>=.01 for row in losses)
    passed=(b['whole_grid_brier']-a['whole_grid_brier']>=.01 and
            a['whole_grid_brier']<=.9*b['whole_grid_brier'] and wins>=2 and
            all(a[key]<=means[control][key] for key in a for control in ('initial','blind','symbolic')))
    return {'status':'complete','qualification_passed':passed,'means':means,'task_losses':losses,
            'task_wins':wins,'depth_authorized':False}
