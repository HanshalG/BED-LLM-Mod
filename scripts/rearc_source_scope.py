"""Metadata-only prospective task selection; never import task implementations."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE = Path('/private/tmp/bed-rearc-source-audit')
COMMIT = 'e5b7f1d06362a76f9d3b8c25154ff1fafca897ce'


def select_ids(source):
    ids = sorted(node.name.removeprefix('verify_') for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name.startswith('verify_'))
    if len(ids) != len(set(ids)) or any(len(key) != 8 or any(c not in '0123456789abcdef' for c in key) for key in ids):
        raise ValueError('invalid verifier identifier inventory')
    return ids, sorted(ids, key=lambda key: hashlib.sha256(('bed-rearc-source-v1:'+key).encode()).digest())[:4]


def main():
    path = Path('results/nonmyopic/REARC_SOURCE_SCOPE_20260909.json')
    if path.exists():
        raise FileExistsError(path)
    source = subprocess.check_output(['git', '-C', str(SOURCE), 'show', COMMIT+':verifiers.py'])
    ids, selected = select_ids(source)
    assert len(ids) == 400
    result = {'source_commit': COMMIT, 'source_url': 'https://github.com/michaelhodel/re-arc',
              'verifiers_sha256': hashlib.sha256(source).hexdigest(), 'all_ids': ids,
              'selected_ids': selected, 'model_calls': 0, 'cost_usd': 0,
              'scope': 'four source contracts only; no paid permission or task execution',
              'selection_rule': 'first four SHA256(bed-rearc-source-v1:+id), no replacement'}
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'all_ids'}, indent=2))


if __name__ == '__main__':
    main()
