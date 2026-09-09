"""Frozen eight-task execution-feedback qualification controller."""
from scripts.rearc_feedback_update import propose_and_repair
from scripts.rearc_proposal_interface import response_format
from scripts.rearc_luna_qualification import validate_body
from scripts.rearc_program_weights import condition_programs
from scripts.rearc_qualification_panel import forecast
from scripts.rearc_predictive_score import mixture_scores


def body(messages, seed):
    value={'model':'openai/gpt-5.6-luna','max_tokens':16384,
           'reasoning':{'enabled':True,'effort':'medium','exclude':True},'seed':seed,
           'provider':{'only':['openai'],'allow_fallbacks':False,'require_parameters':True,
                       'max_price':{'prompt':.2,'completion':1.2}},
           'messages':messages,'response_format':response_format()}
    validate_body(value)
    return value


def collect(cases,dsl,request,evaluate,diagnose,symbolic,bank_update,seal,targets):
    if len(cases)!=8 or any(len(c['inputs'])!=3 or len(c['outputs'])!=3 or len(c['target_inputs'])!=8 for c in cases):
        raise ValueError('eight complete cases required')
    def update(i,arm):
        visible=3 if arm=='aware' else 1
        seed=(33300 if arm=='initial' else 33400)+2*i
        result=propose_and_repair(inputs=cases[i]['inputs'],
            observations=[{'index':j,'output':cases[i]['outputs'][j]} for j in range(visible)],
            dsl_source=dsl,request=lambda phase,m:request(f'{i}_{arm}_{phase}',body(m,seed+(phase=='repair'))),
            evaluate=diagnose)
        bank_update(f'{i}_{arm}',result)
        return result['graphs']
    initial=[update(i,'initial') for i in range(8)]
    covered=[]
    for graphs,case in zip(initial,cases):
        matched=condition_programs(graphs,[evaluate(g,case['inputs'][:1]) for g in graphs],case['outputs'][:1])
        covered.append(matched['consistent_programs']>0)
    if sum(covered)<6:
        return {'status':'initial_coverage_null','initial_coverage':sum(covered),'initial_covered':covered,
                'qualification_passed':False,'endpoints_opened':False,'depth_authorized':False}
    forecasts=[]
    for i,case in enumerate(cases):
        pools={'initial':initial[i]}
        for arm in (('aware','blind') if i%2==0 else ('blind','aware')):
            pools[arm]=initial[i]+update(i,arm)
        pools['symbolic']=symbolic(case['inputs'],case['outputs'])
        if not pools['symbolic']: raise ValueError('symbolic runtime empty')
        forecasts.append({arm:forecast(graphs,case,evaluate) for arm,graphs in pools.items()})
    seal(forecasts)
    labels=targets()
    if len(labels)!=8 or any(len(row)!=8 for row in labels): raise ValueError('target coverage')
    losses=[]
    keys=('whole_grid_brier','fixed_canvas_brier')
    for fs,ys in zip(forecasts,labels):
        row={}
        for arm,f in fs.items():
            scored=[mixture_scores(pred,y,f['weights']) for pred,y in zip(f['outputs'],ys)]
            row[arm]={k:sum(s[k] for s in scored)/8 for k in keys}
        losses.append(row)
    means={a:{k:sum(r[a][k] for r in losses)/8 for k in keys} for a in ('initial','aware','blind','symbolic')}
    a,b=means['aware']['whole_grid_brier'],means['blind']['whole_grid_brier']
    wins=sum(r['blind']['whole_grid_brier']-r['aware']['whole_grid_brier']>=.01 for r in losses)
    passed=b-a>=.01 and a<=.9*b and wins>=4 and all(means['aware'][k]<=means[c][k] for k in keys for c in ('initial','blind','symbolic'))
    return {'status':'complete','initial_coverage':sum(covered),'initial_covered':covered,
            'qualification_passed':passed,'means':means,'task_losses':losses,'task_wins':wins,
            'endpoints_opened':True,'depth_authorized':False}
