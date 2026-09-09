"""Prospective raw-grid versus factual-inventory native proposal comparison."""
import copy
import json

from scripts.rearc_native_revision import native_update
from scripts.rearc_named_plan_interface import messages
from scripts.rearc_scene_inventory import grid_inventory

ARMS = ('raw', 'inventory')


def scene_update(*, inputs, observations, dsl_source, request, diagnose,
                 bank_update, order=ARMS):
    if tuple(order) not in (ARMS, ARMS[::-1]):
        raise ValueError('exact scene arms')
    # Validate the public history before deriving facts or dispatching any call.
    messages('plan', 'contrasting', inputs=inputs, observations=observations,
             dsl_source=dsl_source)
    if not observations:
        raise ValueError('observed example required')
    facts = {
        'inputs': [{'index': i, 'inventory': grid_inventory(g)}
                   for i, g in enumerate(inputs)],
        'observed_outputs': [
            {'index': o['index'], 'inventory': grid_inventory(o['output'])}
            for o in observations],
    }
    results = {}
    for arm in order:
        def dispatch(stage, prompt, fmt):
            prompt = copy.deepcopy(prompt)
            if arm == 'inventory':
                payload = json.loads(prompt[1]['content'])
                payload['public_scene_facts'] = copy.deepcopy(facts)
                prompt[1]['content'] = json.dumps(payload, sort_keys=True)
                prompt[0]['content'] += (
                    ' Public scene facts are deterministic measurements of the '
                    'listed grids, not new examples or inferred semantic roles. '
                    'Color roles can differ across inputs; frequency does not '
                    'establish background. Component lists may be truncated as '
                    'explicitly recorded. Infer and implement roles from evidence.')
            if len(json.dumps(prompt).encode()) > 65536:
                raise ValueError('scene message budget')
            return request(arm + '_' + stage, prompt, fmt)
        result = native_update(inputs=inputs, observations=observations,
            dsl_source=dsl_source, request=dispatch, diagnose=diagnose)
        bank_update(arm, result)
        results[arm] = result
    return {'arms': results, 'calls': 6,
            'scope': 'observation-interface qualification, not planning efficacy'}
