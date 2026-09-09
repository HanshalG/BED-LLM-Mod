"""Metadata and driver-only audit; never import source tasks or inspect labels."""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '4257f44b0ff1181dedaedee6a447e133219fcebf'
ROOT = Path('results/nonmyopic/QUIXBUGS_CONTRACT_AUDIT_20260909.json')


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read(1000001)
    if len(raw) > 1000000:
        raise ValueError('source response cap')
    return raw


def analyze_driver(text):
    tree = ast.parse(text)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    imported = [ast.unparse(c.args[0]) for c in calls
                if isinstance(c.func, ast.Name) and c.func.id == '__import__' and c.args]
    processes = [c for c in calls if ast.unparse(c.func) == 'subprocess.Popen']
    prints = [ast.unparse(c) for c in calls if isinstance(c.func, ast.Name) and c.func.id == 'print']
    return dict(dynamic_import_arguments=imported,
                corrected_module_import=any('correct_python_programs.' in v for v in imported),
                prints_complete_test_record='print(py_testcase)' in prints,
                subprocess_launch_count=len(processes),
                popen_calls_without_timeout=sum(not any(k.arg == 'timeout' for k in c.keywords) for c in processes),
                source_code_executed_by_audit=False)


def inventory(tree):
    if tree.get('truncated') is not False:
        raise ValueError('incomplete inventory')
    paths = [r['path'] for r in tree['tree'] if r['type'] == 'blob']
    groups = {}
    for directory, suffix in [('python_programs', '.py'), ('correct_python_programs', '.py'),
                              ('json_testcases', '.json'), ('python_testcases', '.py')]:
        groups[directory] = sorted(p for p in paths if p.startswith(directory+'/') and p.endswith(suffix))
    return groups


def run():
    if ROOT.exists():
        raise RuntimeError('banked audit already exists')
    base = f'https://raw.githubusercontent.com/jkoppel/QuixBugs/{REVISION}/'
    files = {p: fetch(base+p) for p in ('README.md', 'LICENSE', 'tester.py')}
    raw_tree = fetch(f'https://api.github.com/repos/jkoppel/QuixBugs/git/trees/{REVISION}?recursive=1')
    groups = inventory(json.loads(raw_tree))
    report = dict(revision=REVISION, files={p: dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)) for p, raw in files.items()},
                  inventory_sha256=hashlib.sha256(raw_tree).hexdigest(), paths=groups,
                  counts={k: len(v) for k, v in groups.items()}, driver=analyze_driver(files['tester.py'].decode()),
                  model_calls=0, cost_usd=0, network_reads=4, task_contents_read=False,
                  paid_authorized=False, opportunity_measured=False)
    with ROOT.open('x') as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write('\n')
    return report


if __name__ == '__main__':
    report = run()
    print(json.dumps({k: report[k] for k in ('revision', 'counts', 'driver', 'model_calls', 'paid_authorized')}, sort_keys=True))
