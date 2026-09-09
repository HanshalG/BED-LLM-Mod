"""Prospective paired repair: one shared plan/proposal, two independent repairs."""
import copy
import json
from scripts.rearc_named_plan_interface import messages, schema, parse_plan, parse_programs
from scripts.rearc_feedback_update import example_feedback
from scripts.rearc_compile_feedback import compiler_feedback

ARMS = ('actionable', 'generic')


def interpret(text, dsl, inputs, observations, diagnose):
    try:
        parsed = parse_programs(text, dsl)
    except (ValueError, TypeError, KeyError):
        return [None]*8, {'status': 'invalid_batch'}
    feedback = [[{'example_index': o['index'], **example_feedback(
        graph, inputs[o['index']], o['output'], diagnose)} for o in observations]
        for graph in parsed['graphs']]
    return parsed['expressions'], {'status': 'evaluated', 'programs': feedback}


def paired_update(*, inputs, observations, dsl_source, request, diagnose,
                  order=ARMS):
    if tuple(order) not in (ARMS, ARMS[::-1]):
        raise ValueError('exact paired arm coverage')
    common = dict(inputs=inputs, observations=observations, dsl_source=dsl_source)
    plan = parse_plan(request('plan', messages('plan', 'contrasting', **common), schema('plan')))
    base = messages('compile', 'contrasting', **common, plan=plan)
    original = request('compile', base, schema('compile'))
    slots, feedback = interpret(original, dsl_source, inputs, observations, diagnose)
    structured = compiler_feedback(original, dsl_source) if feedback['status']=='invalid_batch' else None
    results = {}
    for arm in order:
        payload = {'public_execution_feedback': copy.deepcopy(feedback),
            'instruction': 'Return eight repaired or alternative expressions. Retain the original plan-slot mapping when one was requested. Only listed observations are facts.'}
        if arm=='actionable' and structured is not None:
            payload['compiler_feedback'] = structured
        # Each arm sees only the shared original response, never the other repair.
        prompt = copy.deepcopy(base)+[{'role':'assistant','content':original},
            {'role':'user','content':json.dumps(payload, sort_keys=True)}]
        if len(json.dumps(prompt).encode()) > 65536:
            raise ValueError('repair message budget')
        repaired = request(arm+'_repair', prompt, schema('compile'))
        revised, runtime_feedback = interpret(repaired, dsl_source, inputs, observations, diagnose)
        results[arm] = {'slots': slots+revised, 'repair_feedback': runtime_feedback}
    return {'plan':plan, 'compile_feedback':feedback, 'arms':results,
            'compiler_intervention_active':structured is not None, 'calls':4}
