"""Prospective Luna-medium request/receipt caps for mechanism qualification."""
import json
from decimal import Decimal
from scripts.paid_program_probe import money
from scripts.rearc_named_plan_interface import schema

RESERVE = Decimal('.08')
PROVIDER = {'only':['openai'],'allow_fallbacks':False,'require_parameters':True,
            'max_price':{'prompt':.2,'completion':1.2}}
REASONING = {'enabled':True,'effort':'medium','exclude':True}


def validate_route(endpoint):
    prices = endpoint['pricing']
    ceilings = {'prompt':'.0000002','completion':'.0000012',
                'input_cache_write':'.00000025','input_cache_read':'.0000002'}
    for key,ceiling in ceilings.items():
        if money(prices[key])>Decimal(ceiling):
            raise ValueError('price ceiling')
    for key,value in prices.items():
        if key not in {*ceilings,'web_search','discount','overrides'} and money(value):
            raise ValueError('unbudgeted fee')
    for override in prices.get('overrides',[]):
        if type(override.get('min_prompt_tokens')) is not int or override['min_prompt_tokens']<=131072:
            raise ValueError('applicable price override')
    if 131072*(money(prices['prompt'])+money(prices['input_cache_write']))+16384*money(prices['completion'])>RESERVE:
        raise ValueError('exposure exceeds reservation')


def validate_body(value):
    if (set(value)!={'model','max_tokens','reasoning','seed','provider','messages','response_format'}
            or value['model']!='openai/gpt-5.6-luna' or value['max_tokens']!=16384
            or value['reasoning']!=REASONING or value['provider']!=PROVIDER
            or type(value['seed']) is not int or not 0<=value['seed']<2**31
            or value['response_format'] not in (schema('plan'),schema('compile'))
            or len(json.dumps(value).encode())>65536):
        raise ValueError('request cap or route')


def body(messages,seed,response_format):
    value = {'model':'openai/gpt-5.6-luna','max_tokens':16384,'reasoning':dict(REASONING),
        'seed':seed,'provider':dict(PROVIDER),'messages':messages,'response_format':response_format}
    validate_body(value)
    return value


def response_text(raw):
    usage = raw['usage']
    reasoning = usage['completion_tokens_details']['reasoning_tokens']
    if (type(reasoning) is not int or type(usage['completion_tokens']) is not int
            or not 0<reasoning<=usage['completion_tokens']<=16384
            or type(usage['prompt_tokens']) is not int or not 0<=usage['prompt_tokens']<=131072
            or money(usage['cost'])>RESERVE or raw['model']!='openai/gpt-5.6-luna'
            or raw['provider']!='OpenAI'):
        raise ValueError('receipt mismatch')
    choice, = raw['choices']
    if choice['finish_reason']!='stop' or not isinstance(choice['message']['content'],str):
        raise ValueError('incomplete response')
    return choice['message']['content']

