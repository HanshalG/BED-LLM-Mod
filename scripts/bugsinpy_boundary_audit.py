"""Inspect named pre-fix functions as syntax, without importing project code."""
import ast
import hashlib
import json
from pathlib import Path

from scripts.bugsinpy_contract_audit import fetch

OUTPUT = Path('results/nonmyopic/BUGSINPY_BOUNDARY_AUDIT_20260909.json')
SOURCES = (
    ('nvbn/thefuck', '42a8b4f639269886e468762e6d100b6f01aad8ab',
     'thefuck/rules/mkdir_p.py', ('match', 'get_new_command')),
    ('nvbn/thefuck', '42a8b4f639269886e468762e6d100b6f01aad8ab',
     'thefuck/utils.py', ('sudo_support',)),
    ('nvbn/thefuck', '42a8b4f639269886e468762e6d100b6f01aad8ab',
     'thefuck/types.py', ()),
    ('cookiecutter/cookiecutter', '5c282f020a8db7e5e7c4e7b51b010556ca31fb7f',
     'cookiecutter/prompt.py', ('read_user_choice',)),
    ('psf/black', '026c81b83454f176a9f9253cbfb70be2c159d822',
     'black.py', ('format_str', 'format_file_contents')),
)


def inspect_functions(raw, names):
    tree = ast.parse(raw)
    result = {}
    for name in names:
        matches = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(matches) != 1:
            raise ValueError('missing or duplicate function')
        node = matches[0]
        walks = list(ast.walk(node))
        arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        reads = {n.id for n in walks if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        result[name] = {
            'start_line': node.lineno, 'end_line': node.end_lineno,
            'ast_nodes': len(walks),
            'decorators': [ast.unparse(d) for d in node.decorator_list],
            'unused_arguments_syntactically': [a.arg for a in arguments if a.arg not in reads],
            'call_targets': sorted({ast.unparse(n.func) for n in walks if isinstance(n, ast.Call)}),
            'command_attributes': sorted({n.attr for n in walks if isinstance(n, ast.Attribute)
                                          and isinstance(n.value, ast.Name) and n.value.id == 'command'}),
        }
    return result


def run():
    if OUTPUT.exists():
        raise RuntimeError('already banked')
    records = []
    for repo, revision, path, names in SOURCES:
        url = f'https://raw.githubusercontent.com/{repo}/{revision}/{path}'
        raw = fetch(url)
        records.append({'url': url, 'sha256': hashlib.sha256(raw).hexdigest(),
                        'functions': inspect_functions(raw, names)})
    result = {'records': records, 'network_reads': len(records), 'model_calls': 0,
              'cost_usd': 0, 'source_executed': False, 'fixed_source_read': False,
              'test_contents_read': False, 'runtime_equivalence_verified': False,
              'opportunity_measured': False, 'paid_authorized': False}
    with OUTPUT.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
