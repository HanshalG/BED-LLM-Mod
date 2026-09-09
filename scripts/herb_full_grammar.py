"""Source-only full-export grammar; annotations are metadata, not pruning rules."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from scripts.rearc_graph_runtime import DSL_SHA
from scripts.rearc_program_graph import exports


def compile_grammar(source):
    functions, constants = exports(source)
    declarations = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    signatures = []
    for node in declarations:
        args = node.args
        if args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.posonlyargs:
            raise ValueError('unsupported signature')
        if not 1 <= len(args.args) <= 4:
            raise ValueError('unsupported arity')
        signatures.append({'name': node.name, 'arity': len(args.args),
                           'arguments': [ast.unparse(a.annotation) for a in args.args],
                           'returns': ast.unparse(node.returns)})
    if any(not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*', name) for name in functions | constants):
        raise ValueError('invalid symbol')
    terminals = ['I'] + sorted(constants) + sorted(functions)
    rules = [f'Value = {name}' for name in terminals]
    for signature in sorted(signatures, key=lambda row: row['name']):
        rules.append(f"Value = {signature['name']}({', '.join(['Value'] * signature['arity'])})")
    # Calling a computed function is distinct from passing it as an argument.
    for arity in range(1, 5):
        rules.append(f"Value = __bed_call{arity}({', '.join(['Value'] * (arity + 1))})")
    text = 'grammar = @csgrammar begin\n' + '\n'.join('    ' + r for r in rules) + '\nend\n'
    arities = Counter(row['arity'] for row in signatures)
    return text, {'function_count': len(functions), 'constant_count': len(constants),
                  'terminal_count': len(terminals), 'rule_count': len(rules),
                  'signatures': signatures,
                  'direct_one_call_combinations': sum(count * len(terminals)**arity for arity, count in arities.items()),
                  'sound_type_filter': False, 'benchmark_examples': 0, 'calls': 0}


if __name__ == '__main__':
    import sys
    directory = Path(sys.argv[1])
    directory.mkdir(exist_ok=False)
    source = subprocess.check_output(['git', '-C', '/private/tmp/bed-rearc-source-audit', 'show',
                                     'e5b7f1d06362a76f9d3b8c25154ff1fafca897ce:dsl.py'])
    if hashlib.sha256(source).hexdigest() != DSL_SHA:
        raise ValueError('source binding')
    grammar, report = compile_grammar(source.decode())
    report['source_sha256'] = DSL_SHA
    (directory / 'grammar.jl').write_text(grammar)
    (directory / 'source.json').write_text(json.dumps(report, indent=2))
