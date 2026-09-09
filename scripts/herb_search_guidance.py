"""Positive search weights over all DSL rules; never Bayesian hypothesis priors."""
import json
from pathlib import Path

from scripts.rearc_program_graph import validate_graph


def weights(metadata, proposals):
    signatures = sorted(metadata['signatures'], key=lambda row: row['name'])
    functions = {row['name'] for row in signatures}
    # The grammar order is authoritative; do not reconstruct constants by value.
    rules = metadata['rules']
    terminals = {r.removeprefix('Value = ') for r in rules if '(' not in r}
    constants = terminals - functions - {'I'}
    base = []
    for rule in rules:
        rhs = rule.removeprefix('Value = ')
        if rhs == 'I':
            base.append(64.)
        elif '(' not in rhs:
            base.append(1.)
        else:
            name = rhs.split('(', 1)[0]
            signature = next((s for s in signatures if s['name'] == name), None)
            base.append(16. if signature and signature['returns'] == 'Grid' else 4.)
    base = [value / sum(base) for value in base]
    counts = [0.] * len(rules)
    for proposal in proposals:
        validate_graph(proposal, functions, constants)
        for step in proposal['steps']:
            op = step['op']
            call = op if op in functions else f"__bed_call{len(step['args'])}"
            arity = len(step['args']) + int(op not in functions)
            index = rules.index(f"Value = {call}({', '.join(['Value'] * arity)})")
            counts[index] += 1
            for arg in step['args']:
                if arg in terminals:
                    counts[rules.index(f'Value = {arg}')] += 1
    if not sum(counts):
        return base, base.copy()
    guided = [.5 * b + .5 * c / sum(counts) for b, c in zip(base, counts)]
    return base, guided


if __name__ == '__main__':
    import sys
    source_dir, output_dir = map(Path, sys.argv[1:3])
    output_dir.mkdir(exist_ok=False)
    metadata = json.loads((source_dir / 'source.json').read_text())
    metadata['rules'] = [line.strip() for line in (source_dir / 'grammar.jl').read_text().splitlines()
                         if line.strip().startswith('Value = ')]
    # Handcrafted positive-control guide, not an LLM result or task reference.
    proposal = {'steps': [{'id': 'x0', 'op': 'vmirror', 'args': ['I']},
                          {'id': 'x1', 'op': 'hconcat', 'args': ['x0', 'I']}], 'output': 'x1'}
    base, guided = weights(metadata, [proposal])
    for name, values in [('base', base), ('guided', guided)]:
        (output_dir / f'{name}.toml').write_text('weights = ' + json.dumps(values) + '\n')
    (output_dir / 'guide.json').write_text(json.dumps(proposal, indent=2))
