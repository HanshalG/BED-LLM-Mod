"""Prospective adapter: reject invalid guides atomically without replacing slots."""
from scripts.herb_search_guidance import weights
from scripts.rearc_program_graph import validate_graph


def prepare(metadata, slots):
    rules = metadata['rules']
    if len(rules) != len(set(rules)):
        raise ValueError('duplicate grammar rules')
    functions = {row['name'] for row in metadata['signatures']}
    terminals = {r.removeprefix('Value = ') for r in rules if '(' not in r}
    accepted, audit = [], []
    for index, graph in enumerate(slots):
        reason = None
        try:
            validate_graph(graph, functions, terminals-functions-{'I'})
        except (ValueError, TypeError, KeyError):
            reason = 'invalid_graph'
        if reason is None:
            for step in graph['steps']:
                direct = step['op'] in functions
                operation = step['op'] if direct else f"__bed_call{len(step['args'])}"
                arity = len(step['args']) + int(not direct)
                rule = f"Value = {operation}({', '.join(['Value']*arity)})"
                if rule not in rules:
                    reason = 'unsupported_call_arity'
                    break
        if reason is None:
            accepted.append(graph)
        audit.append({'slot': index, 'eligible': reason is None, 'reason': reason})
    base, guided = weights(metadata, accepted)
    return {'graphs': accepted, 'slots': audit, 'base': base, 'guided': guided,
            'replacement_proposals': 0}


def search(slots, count, max_expansions):
    """Keep the frozen worker/candidate limits; only eligible graphs guide it."""
    import json
    from scripts.herb_search_runtime import GRAMMAR, search as bounded_search
    metadata = json.loads((GRAMMAR/'source.json').read_text())
    metadata['rules'] = [line.strip() for line in (GRAMMAR/'grammar.jl').read_text().splitlines()
                         if line.strip().startswith('Value = ')]
    prepared = prepare(metadata, slots)
    result = bounded_search(prepared['graphs'], count, max_expansions)
    return {'guidance_slots': prepared['slots'], 'replacement_proposals': 0,
            'search': result}
