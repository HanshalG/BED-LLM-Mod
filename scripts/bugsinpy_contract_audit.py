"""Read pinned framework files and metadata only; never execute upstream code."""
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '11c5f1eea954a42132cfd06bf257766a7963e0fd'
OUTPUT = Path('results/nonmyopic/BUGSINPY_CONTRACT_AUDIT_20260909.json')
FILES = ('README.md', 'Dockerfile', 'framework/bin/bugsinpy-checkout',
         'framework/bin/bugsinpy-test')


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read(4000001)
    if len(raw) > 4000000:
        raise ValueError('response cap')
    return raw


def inventory(tree):
    if tree.get('truncated') is not False:
        raise ValueError('incomplete inventory')
    paths = [r['path'] for r in tree['tree'] if r['type'] == 'blob']
    bugs = sorted(p for p in paths if p.startswith('projects/') and p.endswith('/bug.info'))
    return {'bug_metadata_paths': bugs, 'bug_count': len(bugs),
            'projects': sorted({p.split('/')[1] for p in bugs}),
            'root_license_paths': [p for p in paths if '/' not in p and
                                   p.lower().startswith(('license', 'copying'))]}


def legacy_accepts_output(output):
    """Literal Python translation of the pinned relevant-test success predicate."""
    last = output.rsplit('\n', 1)[-1]
    return 'OK' in last or 'pass' in last or 'passed' in output or 'OK ' in output


def run():
    if OUTPUT.exists():
        raise RuntimeError('audit already banked')
    base = f'https://raw.githubusercontent.com/soarsmu/BugsInPy/{REVISION}/'
    files = {path: fetch(base + path) for path in FILES}
    raw_tree = fetch(f'https://api.github.com/repos/soarsmu/BugsInPy/git/trees/{REVISION}?recursive=1')
    test = files['framework/bin/bugsinpy-test'].decode()
    checkout = files['framework/bin/bugsinpy-checkout'].decode()
    predicate = '[[ ${res_first##*$\'\\n\'} == *"OK"* || ${res_first##*$\'\\n\'} == *"pass"* || $res_first == *"passed"* || $res_first == *"OK "* ]]'
    if predicate not in test:
        raise ValueError('success predicate changed; translation needs review')
    required = ['res_first=$(pytest $single_test 2>&1)', 'echo "$res_first"']
    if not all(s in test for s in required):
        raise ValueError('test interface changed')
    if '###Copy test file from fixed to buggy' not in checkout:
        raise ValueError('checkout interface changed')
    result = dict(revision=REVISION,
                  files={p: {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
                         for p, raw in files.items()},
                  tree_sha256=hashlib.sha256(raw_tree).hexdigest(),
                  inventory=inventory(json.loads(raw_tree)),
                  single_test_supported=True, raw_test_stdout_exposed=True,
                  fixed_tests_copied_to_buggy_checkout=True,
                  mixed_failure_example='1 failed, 3 passed in 0.10s',
                  mixed_failure_accepted=legacy_accepts_output('1 failed, 3 passed in 0.10s'),
                  model_calls=0, cost_usd=0, network_reads=5,
                  task_contents_read=False, upstream_code_executed=False,
                  opportunity_measured=False, paid_authorized=False)
    with OUTPUT.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('revision', 'mixed_failure_accepted', 'model_calls')}))


if __name__ == '__main__':
    run()
