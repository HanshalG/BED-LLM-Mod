"""Prespecified metadata-only sample. No patches, tests, or upstream execution."""
import hashlib
import json
from pathlib import Path

from scripts.bugsinpy_contract_audit import REVISION, fetch

ROOT = Path('results/nonmyopic/BUGSINPY_DEVELOPMENT_SOURCE_AUDIT_20260909.json')
CONTRACT = Path('results/nonmyopic/BUGSINPY_CONTRACT_AUDIT_20260909.json')
PROJECTS = ('black', 'cookiecutter', 'thefuck')


def select(paths):
    chosen = []
    for project in PROJECTS:
        available = [p for p in paths if p.startswith(f'projects/{project}/bugs/')]
        if not available:
            raise ValueError(f'missing project {project}')
        chosen.append(min(available, key=lambda p: hashlib.sha256(
            ('bed-source-eligibility-20260909:' + p).encode()).hexdigest()))
    return chosen


def parse_info(raw):
    rows = {}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition('=')
        if not sep or key in rows:
            raise ValueError('malformed metadata')
        rows[key] = value.strip().strip('"')
    return rows


def run():
    if ROOT.exists():
        raise RuntimeError('already banked')
    contract_raw = CONTRACT.read_bytes()
    contract = json.loads(contract_raw)
    if contract['revision'] != REVISION:
        raise ValueError('revision mismatch')
    paths = select(contract['inventory']['bug_metadata_paths'])
    base = f'https://raw.githubusercontent.com/soarsmu/BugsInPy/{REVISION}/'
    records = []
    for path in paths:
        project = path.split('/')[1]
        row = {'path': path, 'project': project, 'files': {}}
        for name in (path, f'projects/{project}/project.info'):
            raw = fetch(base + name)
            row['files'][name] = {'sha256': hashlib.sha256(raw).hexdigest(),
                                  'metadata': parse_info(raw)}
        records.append(row)
    result = {'revision': REVISION, 'selection': paths, 'records': records,
              'contract_sha256': hashlib.sha256(contract_raw).hexdigest(),
              'model_calls': 0, 'cost_usd': 0, 'source_executed': False,
              'patches_or_tests_read': False, 'network_reads': 6}
    with ROOT.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
