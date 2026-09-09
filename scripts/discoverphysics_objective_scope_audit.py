"""Read-only source scope audit; no simulator, model, or endpoint imports."""
import ast
import hashlib
import json
from pathlib import Path


SOURCE = Path('scripts/discoverphysics_dark_matter_structured_replication.py')


def audit_source():
    raw = SOURCE.read_bytes()
    tree = ast.parse(raw)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'selection_metrics']
    if len(functions) != 1:
        raise ValueError('selection function changed')
    result = {'source': str(SOURCE), 'sha256': hashlib.sha256(raw).hexdigest(),
              'selectors': {}}
    for node in functions[0].body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in ('myopic_root', 'lookahead_root'):
            continue
        result['selectors'][target.id] = {'line': node.lineno, 'expression': ast.unparse(node.value)}
    if set(result['selectors']) != {'myopic_root', 'lookahead_root'}:
        raise ValueError('missing root selection')
    result['model_calls'] = result['cost_usd'] = 0
    return result


if __name__ == '__main__':
    print(json.dumps(audit_source(), indent=2))
