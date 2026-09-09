"""Shared-plan, equal-call native Python versus DSL proposal experiment."""
import copy
import json

from scripts.rearc_named_plan_interface import messages, schema, parse, parse_plan
from scripts.rearc_paired_repair import interpret as interpret_dsl
from scripts.rearc_compile_feedback import compiler_feedback
from scripts.rearc_feedback_update import example_feedback
from scripts.rearc_python_contract import validate, MODULES

ARMS = ('python', 'dsl')


def compile_messages(arm, *, inputs, observations, dsl_source, plan):
    if arm not in ARMS:
        raise ValueError('representation')
    base = messages('compile', 'contrasting', inputs=inputs,
        observations=observations, dsl_source=dsl_source, plan=plan)
    if arm == 'dsl':
        return base
    payload = json.loads(base[1]['content'])
    payload.pop('dsl')
    instruction = (
        'Infer the grid transformation using only explicitly observed outputs. '
        'Other public inputs have unknown outputs. Return eight self-contained '
        'Python 3.8 programs, each defining transform(grid) for ONE list-of-lists '
        'integer grid and returning one rectangular grid of colors 0..9, dimensions '
        '1..30. Slots 0-1 implement p0, 2-3 p1, 4-5 p2, 6-7 p3, two plausible '
        'implementations per mechanism. Ordinary helper functions, loops and '
        'comprehensions are allowed. Only these modules may be imported: '
        + ', '.join(sorted(MODULES)) + '. '
        'No file, network, process, clock or random access; no private attributes, '
        'dunder names, classes, decorators, async functions, dynamic eval or exec. '
        'Each program runs independently in a fresh bounded container on each input. '
        'Use ordinary safe builtins; no reflection. Type annotations are postponed. '
        'Keep each program within 8192 characters and 16384 UTF-8 bytes. '
        'Return code strings in hypotheses, without Markdown fences. '
        'Do not include task identifiers, reference solutions or other examples.'
    )
    base = [{'role':'system', 'content':instruction},
            {'role':'user', 'content':json.dumps(payload, sort_keys=True)}]
    if len(json.dumps(base).encode()) > 65536:
        raise ValueError('compile message budget')
    return base


def interpret_python(text, inputs, observations, diagnose):
    try:
        codes = parse(text, 'compile')['hypotheses']
    except (ValueError, TypeError, KeyError):
        return [None]*8, {'status':'invalid_batch', 'error':'response_schema'}
    errors = []
    for i, code in enumerate(codes):
        try:
            validate(code)
        except (ValueError, SyntaxError, TypeError, RecursionError) as exc:
            # No candidate-controlled exception text or hidden values in feedback.
            errors.append({'slot':i, 'error_type':type(exc).__name__})
    if errors:
        return [None]*8, {'status':'invalid_batch', 'accepted_slots':0,
                         'slot_errors':errors}
    def diagnostic(code, x):
        result = diagnose(code, x)
        if result.get('status') in ('ok', 'runtime_failed'):
            return result
        if result.get('status') == 'failed':
            # Python worker failures become generic execution errors for both scores
            # and bounded public feedback, never arbitrary exception messages.
            return {'status':'execution_error', 'trace':[]}
        raise ValueError('unknown Python diagnostic outcome')
    feedback = [[{'example_index':o['index'], **example_feedback(
        code, inputs[o['index']], o['output'], diagnostic)} for o in observations]
        for code in codes]
    return codes, {'status':'evaluated', 'programs':feedback}


def representation_update(*, inputs, observations, dsl_source, request,
                          diagnose_python, diagnose_dsl, order=ARMS):
    if tuple(order) not in (ARMS, ARMS[::-1]):
        raise ValueError('exact representation coverage')
    if not observations:
        raise ValueError('observed example required')
    common = dict(inputs=inputs, observations=observations, dsl_source=dsl_source)
    plan = parse_plan(request('plan', messages('plan', 'contrasting', **common), schema('plan')))
    results = {}
    for arm in order:
        base = compile_messages(arm, **common, plan=plan)
        original = request(arm+'_compile', copy.deepcopy(base), schema('compile'))
        def interpret(text):
            if arm == 'python':
                return interpret_python(text, inputs, observations, diagnose_python)
            return interpret_dsl(text, dsl_source, inputs, observations, diagnose_dsl)
        initial, feedback = interpret(original)
        payload = {'public_execution_feedback':feedback,
            'instruction':'Return eight repaired or alternative hypotheses in the same representation. '
                'Retain the original plan-slot mapping. Only listed observations are facts.'}
        if arm == 'dsl' and feedback['status'] == 'invalid_batch':
            payload['compiler_feedback'] = compiler_feedback(original, dsl_source)
        prompt = copy.deepcopy(base)+[{'role':'assistant','content':original},
            {'role':'user','content':json.dumps(payload, sort_keys=True)}]
        if len(json.dumps(prompt).encode()) > 65536:
            raise ValueError('repair message budget')
        repaired = request(arm+'_repair', prompt, schema('compile'))
        revised, revised_feedback = interpret(repaired)
        results[arm] = {'slots':initial+revised, 'compile_feedback':feedback,
                        'repair_feedback':revised_feedback}
    return {'plan':plan, 'arms':results, 'calls':5,
            'scope':'shared-plan representation qualification; not a depth experiment'}
