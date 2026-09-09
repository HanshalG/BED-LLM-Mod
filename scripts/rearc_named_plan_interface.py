"""Prospective shared-schema interface; old paid plan parser stays immutable."""
import json
from copy import deepcopy
from jsonschema import Draft202012Validator
from scripts.rearc_mechanism_interface import load,parse_programs as executable_programs
from scripts.rearc_proposal_interface import build_messages as public_messages

MESSAGE_BYTES = 65536
FIELDS = ('p0','p1','p2','p3')
PLAN_SCHEMA = {'type':'object','additionalProperties':False,'required':list(FIELDS),
    'properties':{key:{'type':'string','minLength':1,'maxLength':512} for key in FIELDS}}
PROGRAM_SCHEMA = {'type':'object','additionalProperties':False,'required':['hypotheses'],
    'properties':{'hypotheses':{'type':'array','minItems':8,'maxItems':8,
        'items':{'type':'string','minLength':1,'maxLength':8192}}}}


def schema(stage):
    if stage not in ('plan','compile'):
        raise ValueError('stage')
    value = PLAN_SCHEMA if stage=='plan' else PROGRAM_SCHEMA
    return {'type':'json_schema','json_schema':{'name':'named_'+stage,'strict':True,'schema':deepcopy(value)}}


def parse(text,stage):
    value = load(text)
    validator = Draft202012Validator(schema(stage)['json_schema']['schema'])
    if not validator.is_valid(value):
        raise ValueError('schema invalid')
    return value


def parse_plan(text):
    value = parse(text,'plan')
    return {key:value[key] for key in FIELDS}


def plan_diagnostics(plan):
    value = parse_plan(json.dumps(plan))
    return {'blank_fields':[key for key,text in value.items() if not text.strip()],
            'semantic_correctness':'not_established_by_schema'}


def parse_programs(text,dsl_source):
    parse(text,'compile')
    # DSL compilation is an additional executable-validity layer, not JSON schema.
    return executable_programs(text,dsl_source)


def messages(stage,mode,*,inputs,observations,dsl_source,plan=None):
    if stage not in ('plan','compile') or mode not in ('contrasting','ordinary'):
        raise ValueError('stage or mode')
    base = public_messages(inputs=inputs,observations=observations,dsl_source=dsl_source)
    payload = json.loads(base[1]['content'])
    common = ('Infer the grid transformation using only the explicitly observed outputs. '
        'Other input grids are public but their outputs are unknown. Do not assume '
        'unobserved outputs or reference solutions. ')
    if stage=='plan':
        if plan is not None:
            raise ValueError('plan not yet available')
        payload.pop('dsl')
        instruction = ('Give four plausible contrasting mechanisms in named fields p0 through p3. '
            'Each should fit the observed examples and differ in an assumption that could '
            'change a prediction on another input. State the mechanism and distinguishing '
            'assumption, without asserting it is true. Do not merely rename one explanation. '
            if mode=='contrasting' else
            'Give your ordinary best plan for solving the transformation in four notes in '
            'named fields p0 through p3. Describe your interpretation and implementation '
            'using whichever reasoning organization you find most useful. ')
        instruction += 'Use complete concise statements well within 512 characters per field.'
    else:
        payload['plan'] = parse_plan(json.dumps(plan))
        instruction = ('Return eight nested expressions. Slots 0-1 implement p0, 2-3 p1, '
            '4-5 p2, 6-7 p3, two plausible implementations per mechanism. '
            if mode=='contrasting' else
            'Use all four notes to return eight plausible nested-expression hypotheses. ')
        instruction += ('Use the full supplied DSL and I, which is one grid, not a list '
            'of examples. Nested calls only; no assignments, literals, attributes or task '
            'IDs. Computed callables use __bed_call1 through __bed_call4. Respect function '
            'arities. Each expression must return a grid on an input independently.')
    result = [{'role':'system','content':common+instruction},
              {'role':'user','content':json.dumps(payload,sort_keys=True)}]
    if len(json.dumps(result).encode())>MESSAGE_BYTES:
        raise ValueError('message budget')
    return result
