"""Prospective contrasting-mechanism versus ordinary-plan proposal interface."""
import json
from scripts.rearc_proposal_interface import build_messages as public_messages
from scripts.rearc_program_graph import exports
from scripts.herb_candidate_expression import to_graph

MODES = ('contrasting', 'ordinary')
MESSAGE_BYTES = 65536


def load(text):
    if not isinstance(text,str) or len(text.encode()) > 131072:
        raise ValueError('response size')
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(text,object_pairs_hook=unique)


def schema(stage):
    if stage == 'plan':
        item = {'type':'object','additionalProperties':False,'required':['id','description'],
                'properties':{'id':{'type':'string','enum':['p0','p1','p2','p3']},
                              'description':{'type':'string','minLength':1,'maxLength':512}}}
        field,count = 'plans',4
    elif stage == 'compile':
        item = {'type':'string','minLength':1,'maxLength':8192}
        field,count = 'hypotheses',8
    else:
        raise ValueError('stage')
    return {'type':'json_schema','json_schema':{'name':'mechanism_'+stage,'strict':True,
        'schema':{'type':'object','additionalProperties':False,'required':[field],
                  'properties':{field:{'type':'array','minItems':count,'maxItems':count,'items':item}}}}}


def parse_plan(text):
    value = load(text)
    if not isinstance(value,dict) or set(value)!={'plans'} or not isinstance(value['plans'],list) or len(value['plans'])!=4:
        raise ValueError('four plan slots')
    for i,row in enumerate(value['plans']):
        if (not isinstance(row,dict) or set(row)!={'id','description'} or row['id']!=f'p{i}'
                or not isinstance(row['description'],str) or not row['description'].strip()
                or len(row['description'].encode())>512):
            raise ValueError('ordered bounded plan')
    return value


def parse_programs(text,dsl_source):
    value = load(text)
    if not isinstance(value,dict) or set(value)!={'hypotheses'} or not isinstance(value['hypotheses'],list) or len(value['hypotheses'])!=8:
        raise ValueError('eight program slots')
    functions,constants = exports(dsl_source)
    graphs = [to_graph(expr,functions,constants) for expr in value['hypotheses']]
    return {'expressions':value['hypotheses'],'graphs':graphs}


def messages(stage,mode,*,inputs,observations,dsl_source,plan=None):
    if mode not in MODES or stage not in ('plan','compile'):
        raise ValueError('stage or mode')
    base = public_messages(inputs=inputs,observations=observations,dsl_source=dsl_source)
    payload = json.loads(base[1]['content'])
    common = ('Infer a grid transformation using only the explicitly observed outputs. '
              'Other input grids are public but their outputs are unknown. Do not assume '
              'unobserved outputs or reference solutions. ')
    if stage == 'plan':
        if plan is not None:
            raise ValueError('plan not yet available')
        payload.pop('dsl')
        instruction = ('Give four plausible contrasting mechanisms in slots p0 through p3. '
            'Each should explain how inputs determine outputs, fit the observed examples, '
            'and differ in an assumption that could change a prediction on another input. '
            'Do not merely rename one mechanism or alter syntax. State each mechanism and '
            'its distinguishing assumption, without asserting it is true.' if mode=='contrasting' else
            'Give your ordinary best plan for solving the transformation in four concise '
            'notes p0 through p3. Describe your interpretation and how you would implement it. '
            'Use whichever reasoning organization you find most useful.')
    else:
        payload['plan'] = parse_plan(json.dumps(plan))
        instruction = ('Return eight executable nested expressions. Slots 0-1 implement plan '
            'p0, 2-3 p1, 4-5 p2, 6-7 p3, with two plausible implementations per mechanism. '
            if mode=='contrasting' else
            'Use all four plan notes to return eight plausible executable nested-expression '
            'hypotheses for the transformation. ')
        instruction += ('Use the full supplied DSL and I, which is one grid, not a list of '
            'examples. Nested calls only; no assignments, literals, attributes or task IDs. '
            'Computed callables use __bed_call1 through __bed_call4. Respect function arities. '
            'Each expression must return a grid when applied to an input independently.')
    result = [{'role':'system','content':common+instruction},
              {'role':'user','content':json.dumps(payload,sort_keys=True)}]
    if len(json.dumps(result).encode())>MESSAGE_BYTES:
        raise ValueError('message budget')
    return result
