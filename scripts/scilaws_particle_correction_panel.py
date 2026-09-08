"""Full correction panel with exact banked-prefix and independent-reference reuse."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.scilaws_particle_correction_audit import run as run_task
from scripts.scilaws_particle_integration_panel import SCENARIOS

REFERENCES = Path('results/nonmyopic/SCILAWS_PARTICLE_INTEGRATION_PANEL_20260908/result.json')
REFERENCE_SHA = '06f7f6c21ceda8a34ee5cc96f7199d3da677179db241d9c6b1f20e5288e0353d'
PREFIX = Path('results/nonmyopic/SCILAWS_PARTICLE_LINEAR_CORRECTION_AUDIT_20260908.json')
PREFIX_SHA = '49ccf51cb49f94bd7fd996013519b89859d28cbcfe04618531b99a0bdd0d772a'
COUNTS = (4, 8, 16, 32)


def read_bound(path, sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('artifact binding mismatch')
    return json.loads(raw)


def assess(cases, references):
    expected = [(t, s, c) for t in range(8) for s in (1304, 1305) for c in SCENARIOS]
    def identity(c):
        return c['task_index'], c['seed'], c['scenario']
    if list(map(identity, cases)) != expected or list(map(identity, references)) != expected:
        raise ValueError('coverage or order mismatch')
    totals = {q: 0 for q in COUNTS}
    for case, reference in zip(cases, references):
        refs = reference['reference']
        if (reference['reference_reason'] is not None or len(refs) != 8 or any(
                not np.isfinite([r['value'], r['error_estimate'], r['tail_bound'], r['mass_error']]).all()
                or not 0 <= r['error_estimate'] + r['tail_bound'] <= 1e-7
                or not 0 <= r['mass_error'] <= 1e-8 for r in refs)):
            raise ValueError('unqualified independent reference')
        if [r['branch_count'] for r in case['candidates']] != list(COUNTS):
            raise ValueError('candidate coverage mismatch')
        for row in case['candidates']:
            passed = False
            if row['status'] == 'complete':
                values = [r['value'] for r in row['roots']]
                if len(values) != 8 or not np.isfinite(values).all():
                    raise ValueError('invalid complete roots')
                action = min(range(8), key=lambda a: (values[a], a))
                if action != row['action']:
                    raise ValueError('selected action mismatch')
                error = max(abs(v-r['value']) for v, r in zip(values, refs))
                regret = refs[action]['value']-min(r['value'] for r in refs)
                passed = (error <= 1e-4 and regret <= 1e-4
                          and 0 <= row['seconds'] <= 5 and 0 <= row['states'] <= 100000)
            totals[row['branch_count']] += int(passed)
    return {q: dict(passed_cases=n, total_cases=48, qualified=n == 48) for q, n in totals.items()}


def run(output):
    references = read_bound(REFERENCES, REFERENCE_SHA)['cases']
    prefix = read_bound(PREFIX, PREFIX_SHA)
    output = Path(output)
    output.mkdir(exist_ok=False)
    cases, shards = [], []
    for t in range(8):
        for seed in (1304, 1305):
            selected = [c for c in references if c['task_index'] == t and c['seed'] == seed]
            if [c['scenario'] for c in selected] != list(SCENARIOS):
                raise ValueError('reference task identity mismatch')
            reused = t == 0 and seed == 1304
            if reused:
                data = prefix
            else:
                path = output / f'task{t}_seed{seed}.json'
                run_task(path, task_index=t, seed=seed, references=selected,
                         reference_sha256=REFERENCE_SHA)
                raw = path.read_bytes()
                shards.append(dict(path=path.name, sha256=hashlib.sha256(raw).hexdigest()))
                data = json.loads(raw)
            cases.extend(dict(c, task_index=t, seed=seed, reused=reused) for c in data['cases'])
            print('completed', t, seed, 'reused', reused, flush=True)
    result = dict(cases=cases, assessment=assess(cases, references), shards=shards,
                  reference_sha256=REFERENCE_SHA, prefix_sha256=PREFIX_SHA,
                  new_cases=45, reused_cases=3, source_measurements=0, model_calls=0,
                  paid_cost_usd=0, deployment_authorized=False)
    with (output / 'result.json').open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(parser.parse_args().output)
