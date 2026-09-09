"""Source-independent public demonstration prompts and strict graph responses."""
import ast
import json
from scripts.rearc_graph_worker import grid
from scripts.rearc_program_graph import exports, validate_graph

COUNT = 4


def dsl_documentation(source):
    functions, constants = exports(source)
    lines = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            args = ', '.join(a.arg+(': '+ast.unparse(a.annotation) if a.annotation else '') for a in node.args.args)
            returns = ast.unparse(node.returns) if node.returns else 'unspecified'
            lines.append(f'{node.name}({args}) -> {returns}: {ast.get_docstring(node) or ""}')
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    lines.append(target.id+' = '+ast.unparse(node.value))
    return '\n'.join(lines), functions, constants


def response_format():
    name = {'type':'string', 'minLength':1, 'maxLength':64}
    step = {'type':'object','additionalProperties':False,'required':['id','op','args'],
            'properties':{'id':name,'op':name,'args':{'type':'array','maxItems':4,'items':name}}}
    graph = {'type':'object','additionalProperties':False,'required':['steps','output'],
             'properties':{'steps':{'type':'array','minItems':1,'maxItems':128,'items':step},'output':name}}
    return {'type':'json_schema','json_schema':{'name':'grid_hypotheses','strict':True,
        'schema':{'type':'object','additionalProperties':False,'required':['hypotheses'],
                  'properties':{'hypotheses':{'type':'array','minItems':COUNT,'maxItems':COUNT,'items':graph}}}}}


def build_messages(*, inputs, observations, dsl_source):
    if not 1 <= len(inputs) <= 16:
        raise ValueError('1..16 public input grids required')
    public_inputs = [grid(value) for value in inputs]
    observed, seen = [], set()
    for item in observations:
        if not isinstance(item, dict) or set(item) != {'index','output'}:
            raise ValueError('observation fields')
        index = item['index']
        if type(index) is not int or not 0 <= index < len(inputs) or index in seen:
            raise ValueError('unique in-range observation index required')
        observed.append({'index':index,'output':grid(item['output'])})
        seen.add(index)
    documentation, _, _ = dsl_documentation(dsl_source)
    messages = [{'role':'system','content':
        'Infer the unknown grid transformation from the observed examples. Return exactly four '
        'distinct plausible executable hypotheses using only the supplied generic DSL. '
        'Each hypothesis is a straight-line graph: input I, fresh step ids x0,x1,..., named '
        'positional arguments referring to I, DSL functions/constants or earlier steps, and '
        'an output step. Higher-order callable steps are allowed. No Python code, literals, '
        'reference solutions or task identifiers. Do not infer unobserved output labels as facts.'},
        {'role':'user','content':json.dumps({'dsl':documentation,'public_inputs':public_inputs,
                                           'observations':observed},sort_keys=True)}]
    if len(json.dumps(messages).encode()) > 131072:
        raise ValueError('public prompt size cap')
    return messages


def parse_response(text, dsl_source):
    if not isinstance(text,str) or len(text.encode()) > 131072:
        raise ValueError('response size cap')
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    result = json.loads(text,object_pairs_hook=unique)
    if not isinstance(result,dict) or set(result) != {'hypotheses'} or not isinstance(result['hypotheses'],list) or len(result['hypotheses']) != COUNT:
        raise ValueError('exact hypothesis count required')
    functions, constants = exports(dsl_source)
    for graph in result['hypotheses']:
        validate_graph(graph,functions,constants)
    return result['hypotheses']
