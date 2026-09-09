"""Frozen source-only disagreement audit, not a policy evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import uuid

from scripts.black_prefixed_runtime_smoke import source_paths
from scripts.bugsinpy_contract_audit import fetch

PARENT = 'sha256:017df47b29fcfc1eeb456f5967ed271adacfac91ffe5757e0efe959cc494ff3c'
PARENT_TAG = 'bed-black-prefixed-smoke:20260909'
FIXED = 'd6db1c12a8e14833fe22da377cddc2bd1f43dc14'
OUTPUT = Path('results/nonmyopic/BLACK_REFERENCE_DISAGREEMENT_V3_20260909.json')
PROTOCOL = Path('results/nonmyopic/BLACK_REFERENCE_DISAGREEMENT_PROTOCOL_20260909.md')
AMENDMENT = Path('results/nonmyopic/BLACK_REFERENCE_BUILD_AMENDMENT_20260909.md')
ERROR_AMENDMENT = Path('results/nonmyopic/BLACK_REFERENCE_SAFE_ERROR_AMENDMENT_20260909.md')
TEMPLATES = (
    'x = {e}\n', 'f({e})\n', 'print({e})\n', 'print {e}\n',
    'print >>stream, {e}\n', 'exec({e})\n', 'exec {e}\n',
    'def print(x):\n    return x\nprint({e})\n',
)
CONTEXTS = ('', 'from __future__ import print_function\n',
            'from __future__ import unicode_literals\n')


def inputs():
    rows = []
    for t, template in enumerate(TEMPLATES):
        for c, context in enumerate(CONTEXTS):
            for e, expression in enumerate(('x', '1', '(1, 2)', '"hello"')):
                source = context + template.format(e=expression)
                rows.append({'id': f't{t}c{c}e{e}', 'template': t, 'source': source})
    rows.sort(key=lambda r: hashlib.sha256(('black-domain-20260909:' + r['source']).encode()).hexdigest())
    for i, row in enumerate(rows):
        row['role'] = 'query' if i < 32 else 'diagnostic_target'
    return rows


def summarize(rows, old, new):
    if len(old) != len(rows) or len(new) != len(rows):
        raise ValueError('incomplete outcomes')
    changed = [r for r, a, b in zip(rows, old, new) if a != b]
    queries = sum(r['role'] == 'query' for r in changed)
    targets = sum(r['role'] == 'diagnostic_target' for r in changed)
    families = len({r['template'] for r in changed})
    return {'changed_ids': [r['id'] for r in changed], 'changed_queries': queries,
            'changed_targets': targets, 'changed_templates': families,
            'target_disagreement_rate': targets / 64,
            'source_eligibility': queries >= 2 and targets >= 4 and families >= 2}


RUNNER = '''import black, json
rows = json.load(open('/app/inputs.json'))
result = []
for row in rows:
    try:
        text = black.format_file_contents(row['source'], fast=False, mode=black.FileMode())
        result.append({'kind': 'formatted', 'text': text})
    except black.NothingChanged:
        result.append({'kind': 'formatted', 'text': row['source']})
    except black.InvalidInput:
        result.append({'kind': 'InvalidInput'})
    except AssertionError as error:
        if not str(error).startswith('cannot use --safe with this file; failed to parse source file'):
            raise
        result.append({'kind': 'SourceAstUnsupported'})
print(json.dumps(result))
'''


def checked(args, timeout):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout).stdout


def parent_binding():
    value = json.loads(checked(['docker', 'image', 'inspect', PARENT_TAG], 20))[0]
    if value['Id'] != PARENT:
        raise ValueError('parent image changed')
    return value


def run_container(image):
    name = 'bed-black-audit-' + uuid.uuid4().hex
    args = ['docker', 'run', '--name', name, '--rm', '--network=none', '--read-only',
            '--cap-drop=ALL', '--security-opt=no-new-privileges', '--pids-limit=32',
            '--memory=256m', '--cpus=1', '--tmpfs=/tmp:rw,noexec,nosuid,size=16m', image,
            'python', '/app/runner.py']
    try:
        return json.loads(checked(args, 60))
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=20)


def run():
    if OUTPUT.exists():
        raise RuntimeError('already banked')
    rows = inputs()
    result = {'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              'build_amendment_sha256': hashlib.sha256(AMENDMENT.read_bytes()).hexdigest(),
              'error_amendment_sha256': hashlib.sha256(ERROR_AMENDMENT.read_bytes()).hexdigest(),
              'input_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
              'inputs': rows, 'parent_image': PARENT, 'fixed_revision': FIXED,
              'model_calls': 0, 'cost_usd': 0, 'policy_endpoints_opened': False,
              'diagnostic_reference_outcomes_opened': False, 'upstream_tests_read': False}
    try:
        parent_binding()
        with tempfile.TemporaryDirectory(prefix='bed-black-disagreement-') as directory:
            root = Path(directory)
            (root / 'inputs.json').write_text(json.dumps(rows))
            (root / 'runner.py').write_text(RUNNER)
            (root / 'Dockerfile').write_text(f'FROM {PARENT_TAG}\nCOPY inputs.json runner.py /app/\n')
            tag = 'bed-black-diagnostic-old:20260909'
            checked(['docker', 'build', '-t', tag, str(root)], 120)
            old_image = json.loads(checked(['docker', 'image', 'inspect', tag], 20))[0]['Id']
            old = run_container(old_image)
            tree_raw = fetch(f'https://api.github.com/repos/psf/black/git/trees/{FIXED}?recursive=1')
            result['fixed_tree_sha256'] = hashlib.sha256(tree_raw).hexdigest()
            result['fixed_source_hashes'] = {}
            for path in source_paths(json.loads(tree_raw)):
                raw = fetch(f'https://raw.githubusercontent.com/psf/black/{FIXED}/{path}')
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                result['fixed_source_hashes'][path] = hashlib.sha256(raw).hexdigest()
            parent_binding()
            (root / 'Dockerfile').write_text(f'FROM {PARENT_TAG}\nCOPY . /app/\n')
            tag = 'bed-black-diagnostic-reference:20260909'
            checked(['docker', 'build', '-t', tag, str(root)], 120)
            new_image = json.loads(checked(['docker', 'image', 'inspect', tag], 20))[0]['Id']
            new = run_container(new_image)
            result['diagnostic_reference_outcomes_opened'] = True
            result['image_ids'] = [old_image, new_image]
            result['outcome_hashes'] = [hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
                                        for v in (old, new)]
            result.update(summarize(rows, old, new))
            result['status'] = 'audit_complete'
    except Exception as exc:
        result['status'] = 'failed_closed'
        result['error'] = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            result['stderr'] = exc.stderr[-3000:]
    with OUTPUT.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ('inputs', 'fixed_source_hashes')}, indent=2))


if __name__ == '__main__':
    run()
