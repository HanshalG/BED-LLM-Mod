"""Transport-free request and cost contract for a future constrained gate."""
from decimal import Decimal

from .constrained import schema
from .prediction import canonical
from .proposals import messages as legacy_messages


MODEL = 'deepseek/deepseek-v4-flash-0731'
PROVIDER = 'open-inference/fp8'
INPUT_RESERVATION_TOKENS = 65536
OUTPUT_TOKENS = 4096
RESERVATION_USD = Decimal('.015')


def request(dsl, history, seed, *, history_blind=False):
    if type(seed) is not int:
        raise ValueError('integer request seed required')
    # Reuse only the validated public vocabulary/history, not old shape instructions.
    public = legacy_messages(dsl, history, history_blind=history_blind)[1]
    instructions = (
        'Infer alternative executable programs explaining the observed examples. '
        'Return only an object with programs, an array of 1 to 8 programs. '
        'Each program is a nested chain of objects with statement and next. '
        'statement is one exact allowed assignment string from the response schema. '
        'next is the next statement object, or null after the final statement. '
        'Use 2 to 4 statements. The schema enforces types, variable scope and use '
        'of each previous result. Inputs x0,x1 are integer lists. The last result '
        'is the output. Any undefined operation or intermediate integer outside '
        '[-50,50] makes the entire program ERROR. Observed null means ERROR. '
        'Lambda division is floor division, including negative integers. '
        'The source prior chooses length uniformly from 2,3,4 then each statement '
        'uniformly among all type-valid operation/lambda/argument tuples consuming '
        'the previous result (except the first statement). Input lengths are '
        'uniform 1..5 and integer elements uniform -10..10, independently of programs. '
        'No prose, weights, fitted constants or Python.'
    )
    body = dict(model=MODEL, messages=[{'role': 'system', 'content': instructions}, public],
                seed=seed, temperature=.7, max_tokens=OUTPUT_TOKENS, stream=False,
                reasoning={'enabled': False, 'exclude': True},
                provider={'only': [PROVIDER], 'allow_fallbacks': False, 'require_parameters': True,
                          'max_price': {'prompt': .1, 'completion': .2}},
                response_format={'type': 'json_schema', 'json_schema': {
                    'name': 'source_valid_programs', 'strict': True, 'schema': schema(dsl)}})
    # Include the schema and every parameter, not just message content.
    if len(canonical(body).encode('utf-8')) > 32768:
        raise ValueError('complete request exceeds byte reservation contract')
    return body


def advertised_route(endpoint):
    """Check metadata eligibility, NOT actual support for this particular schema."""
    required = {'structured_outputs', 'response_format', 'reasoning', 'seed',
                'temperature', 'max_tokens'}
    if (endpoint.get('model_id') != MODEL or endpoint.get('tag') != PROVIDER
            or endpoint.get('status') != 0
            or not required <= set(endpoint.get('supported_parameters', []))):
        raise ValueError('route identity or advertised parameters failed')
    context, completion = endpoint.get('context_length'), endpoint.get('max_completion_tokens')
    if (type(context) is not int or context < INPUT_RESERVATION_TOKENS+OUTPUT_TOKENS
            or type(completion) is not int or completion < OUTPUT_TOKENS):
        raise ValueError('advertised context/output reservation failed')
    prices = endpoint.get('pricing', {})
    for key, ceiling in [('prompt', Decimal('.0000001')), ('completion', Decimal('.0000002'))]:
        value = prices.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError('missing/invalid price')
        price = Decimal(str(value))
        if not price.is_finite() or not 0 <= price <= ceiling:
            raise ValueError('route exceeds frozen price ceiling')
    worst = INPUT_RESERVATION_TOKENS*Decimal('.0000001')+OUTPUT_TOKENS*Decimal('.0000002')
    if worst > RESERVATION_USD:
        raise ValueError('reservation does not cover token ceilings')
    return dict(metadata_eligible=True, schema_serving_verified=False, paid_authorized=False,
                token_price_exposure_usd=str(worst), reservation_usd=str(RESERVATION_USD))
