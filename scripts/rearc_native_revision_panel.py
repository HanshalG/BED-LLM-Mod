"""Four-task transfer gate for evidence-conditioned native hypothesis revision."""
import hashlib
import json
from scripts.rearc_native_revision import paired_revision,forecast_revision,ARMS
from scripts.rearc_named_plan_contract import body
from scripts.rearc_python_forecast import forecast_python_slots
from scripts.rearc_predictive_score import mixture_scores,outcome


def score(forecasts,labels):
    if len(forecasts)!=4 or len(labels)!=4 or any(len(ys)!=9 for ys in labels):
        raise ValueError('four-task endpoint coverage')
    losses=[]; covered=0; diverse=0; recovery=0
    for f,ys in zip(forecasts,labels):
        if set(f)!={'aware','blind','initial'} or any(len(f[a]['outputs'])!=9 for a in f):
            raise ValueError('three-arm forecast coverage')
        metrics={a:[mixture_scores(o,y,f[a]['weights']) for o,y in zip(f[a]['outputs'],ys)] for a in f}
        covered+=sum(r['exact_grid_probability']>0 for r in metrics['aware'][:2])
        recovery+=any(a['exact_grid_probability']>0 and b['exact_grid_probability']==0
                      for a,b in zip(metrics['aware'][:2],metrics['initial'][:2]))
        diverse+=any(len({outcome(o) for o,w in zip(row,f['aware']['weights']) if w>0 and outcome(o) is not None})>=2
                     for row in f['aware']['outputs'][:2])
        losses.append({a:{part:{k:sum(metrics[a][i][k] for i in indices)/len(indices)
            for k in ('whole_grid_brier','fixed_canvas_brier')}
            for part,indices in [('all',range(9)),('query',range(2)),('target',range(2,9))]} for a in f})
    means={a:{part:{k:sum(t[a][part][k] for t in losses)/4
        for k in ('whole_grid_brier','fixed_canvas_brier')} for part in ('all','query','target')}
        for a in ('aware','blind','initial')}
    wins=sum(t['blind']['all']['whole_grid_brier']-t['aware']['all']['whole_grid_brier']>=.01 for t in losses)
    gates={'answer_support':covered>=6,'predictive_disagreement':diverse>=2,
        'paired_score':means['blind']['all']['whole_grid_brier']-means['aware']['all']['whole_grid_brier']>=.01 and wins>=2,
        'nonworse_canvas':means['aware']['all']['fixed_canvas_brier']<=means['blind']['all']['fixed_canvas_brier'],
        'new_support_recovery':recovery>=2}
    return {'status':'complete','means':means,'task_losses':losses,'gates':gates,
        'covered_query_answers':covered,'diverse_tasks':diverse,'recovery_tasks':recovery,'task_wins':wins,
        'qualification_passed':all(gates.values()),'depth_authorized':False,'endpoints_opened':True}


def collect(cases,dsl,request,evaluate,diagnose,reveal,bank_update,seal,targets):
    expected={'inputs','outputs','reveal_input','reveal_hash','query_inputs','target_inputs','target_hashes'}
    if len(cases)!=4 or any(set(c)!=expected or any(len(c[k])!=n for k,n in
        [('inputs',1),('outputs',1),('query_inputs',2),('target_inputs',7),('target_hashes',9)]) for c in cases):
        raise ValueError('public four-task coverage')
    forecasts=[]
    for i,c in enumerate(cases):
        inputs=c['inputs']+[c['reveal_input']]+c['query_inputs']+c['target_inputs']
        def dispatch(stage,prompt,fmt):
            prefix,step=stage.split('_',1)
            offset={'plan':0,'compile':1,'repair':2}[step]+(0 if prefix=='initial' else 3)
            return request(f'{i}_{stage}',body(prompt,50400+6*i+offset,fmt))
        def reveal_observation():
            y=reveal(i)
            if hashlib.sha256(json.dumps(outcome(y),separators=(',',':')).encode()).hexdigest()!=c['reveal_hash']:
                raise ValueError('second observation binding')
            return {'index':1,'output':y}
        result=paired_revision(inputs=inputs,observation={'index':0,'output':c['outputs'][0]},
            dsl_source=dsl,request=dispatch,diagnose=diagnose,reveal=reveal_observation,
            bank_update=lambda tag,value:bank_update(f'{i}_{tag}',value),order=ARMS if i%2==0 else ARMS[::-1])
        future=c['query_inputs']+c['target_inputs']
        row=forecast_revision(result,inputs,future,evaluate)
        row['initial']=forecast_python_slots(result['initial']['slots'],
            {'inputs':inputs[:2],'outputs':[o['output'] for o in result['observations']],
             'target_inputs':future},evaluate)
        forecasts.append(row)
    coverage=[not row['aware']['conditioning']['failed'] for row in forecasts]
    if sum(coverage)<3:
        return {'status':'initial_coverage_null','two_example_covered':coverage,
            'qualification_passed':False,'depth_authorized':False,'endpoints_opened':False}
    seal(forecasts)
    return {**score(forecasts,targets()),'two_example_covered':coverage}
