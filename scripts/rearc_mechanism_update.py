"""One plan, one compilation, one fixed public-feedback repair; no paid transport."""
import json
from scripts.rearc_mechanism_interface import messages,schema,parse_plan,parse_programs,MESSAGE_BYTES
from scripts.rearc_feedback_update import example_feedback


def update(*,mode,inputs,observations,dsl_source,request,diagnose):
    common = dict(inputs=inputs,observations=observations,dsl_source=dsl_source)
    plan = parse_plan(request('plan',messages('plan',mode,**common),schema('plan')))
    base = messages('compile',mode,**common,plan=plan)
    def interpret(text):
        try:
            result = parse_programs(text,dsl_source)
        except (ValueError,TypeError,KeyError):
            return [None]*8, [], {'status':'invalid_batch'}
        feedback = [[{'example_index':observation['index'],**example_feedback(
            g,inputs[observation['index']],observation['output'],diagnose)}
            for observation in observations] for g in result['graphs']]
        return result['expressions'],result['graphs'],{'status':'evaluated','programs':feedback}
    compiled = request('compile',base,schema('compile'))
    first,graphs,feedback = interpret(compiled)
    repair = base+[{'role':'assistant','content':compiled},{'role':'user','content':json.dumps({
        'public_execution_feedback':feedback,
        'instruction':'Return eight repaired or alternative expressions. Retain the original plan-slot mapping when one was requested. Only listed observations are facts.'},sort_keys=True)}]
    if len(json.dumps(repair).encode())>MESSAGE_BYTES:
        raise ValueError('repair message budget')
    second,repaired,repair_feedback = interpret(request('repair',repair,schema('compile')))
    return {'mode':mode,'plan':plan,'slots':first+second,'graphs':graphs+repaired,
            'compile_feedback':feedback,'repair_feedback':repair_feedback,'calls':3}
