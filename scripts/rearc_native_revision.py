"""Native program revision: new-answer proposer versus history-blind proposer."""
import copy
import json
from scripts.rearc_named_plan_interface import messages,schema,parse_plan
from scripts.rearc_representation_update import compile_messages,interpret_python
from scripts.rearc_python_forecast import forecast_python_slots

ARMS=('aware','blind')


def native_update(*,inputs,observations,dsl_source,request,diagnose,previous=None):
    common=dict(inputs=inputs,observations=observations,dsl_source=dsl_source)
    prompt=messages('plan','contrasting',**common)
    if previous is not None:
        if set(previous)!={'plan','slots'} or len(previous['slots'])!=16:
            raise ValueError('fixed initial proposal state')
        payload=json.loads(prompt[1]['content'])
        payload['previous_proposals']=copy.deepcopy(previous)
        prompt[1]['content']=json.dumps(payload,sort_keys=True)
        prompt[0]['content']+=' Previous proposals are fallible hypotheses, not additional observations. Revise their mechanisms using only listed observed outputs.'
    if len(json.dumps(prompt).encode())>65536:
        raise ValueError('revision plan message budget')
    plan=parse_plan(request('plan',prompt,schema('plan')))
    base=compile_messages('python',**common,plan=plan)
    original=request('compile',copy.deepcopy(base),schema('compile'))
    slots,feedback=interpret_python(original,inputs,observations,diagnose)
    repair=copy.deepcopy(base)+[{'role':'assistant','content':original},
        {'role':'user','content':json.dumps({'public_execution_feedback':feedback,
         'instruction':'Return eight repaired or alternative Python hypotheses with the same plan-slot mapping. Only listed observations are facts.'},sort_keys=True)}]
    if len(json.dumps(repair).encode())>65536:
        raise ValueError('revision repair message budget')
    revised=request('repair',repair,schema('compile'))
    new,final_feedback=interpret_python(revised,inputs,observations,diagnose)
    return {'plan':plan,'slots':slots+new,'compile_feedback':feedback,
            'repair_feedback':final_feedback,'calls':3}


def paired_revision(*,inputs,observation,dsl_source,request,diagnose,reveal,
                    bank_update,order=ARMS):
    if tuple(order) not in (ARMS,ARMS[::-1]):
        raise ValueError('exact revision arms')
    common=dict(inputs=inputs,dsl_source=dsl_source,diagnose=diagnose)
    initial=native_update(**common,observations=[observation],
        request=lambda stage,prompt,fmt:request('initial_'+stage,prompt,fmt))
    bank_update('initial',initial)
    added=reveal()
    # Validate the newly revealed observation before either revision call.
    observations=[observation,added]
    messages('plan','contrasting',inputs=inputs,observations=observations,dsl_source=dsl_source)
    previous={'plan':initial['plan'],'slots':initial['slots']}
    results={}
    for arm in order:
        revised=native_update(**common,observations=observations if arm=='aware' else [observation],
            previous=copy.deepcopy(previous),
            request=lambda stage,prompt,fmt:request(arm+'_'+stage,prompt,fmt))
        bank_update(arm,revised)
        results[arm]={'slots':initial['slots']+revised['slots'],'revision':revised}
    return {'initial':initial,'arms':results,'observations':observations,'calls':9}


def forecast_revision(result,inputs,target_inputs,evaluate):
    observations=result['observations']
    case={'inputs':[inputs[o['index']] for o in observations],
          'outputs':[o['output'] for o in observations],'target_inputs':target_inputs}
    # Both arms condition on identical evidence. Only the proposal generator differs.
    return {arm:forecast_python_slots(result['arms'][arm]['slots'],case,evaluate) for arm in ARMS}
