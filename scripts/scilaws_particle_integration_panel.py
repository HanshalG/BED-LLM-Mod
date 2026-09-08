"""Full public-fixture integration panel, reusing the pinned workload prefix."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from environments.chembench_mopen.batch_horizon import plan_batched
from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from scripts.scilaws_particle_integration_audit import run as run_task
from scripts.scilaws_reference_preflight import DESIGN_SHA

BANK = Path('results/nonmyopic/SCILAWS_PARTICLE_INTEGRATION_AUDIT_20260908.json')
BANK_SHA = '4597f7e59916cd633fcd0bddb0c7eac96d1b9239e1c6c55aa2b8dc6722ba1b57'
SCENARIOS = ('zero', 'affine', 'quadratic')
COUNTS = (32, 64)


def read_bank(path=BANK):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BANK_SHA:
        raise ValueError('bank binding mismatch')
    return json.loads(raw)


def assess(cases):
    expected = [(t, s, c) for t in range(8) for s in (1304, 1305) for c in SCENARIOS]
    if [(c['task_index'], c['seed'], c['scenario']) for c in cases] != expected:
        raise ValueError('incomplete or unordered panel')
    totals = {q: 0 for q in COUNTS}
    for case in cases:
        if [r['branch_count'] for r in case['candidates']] != list(COUNTS):
            raise ValueError('candidate coverage mismatch')
        refs = case['reference']
        valid = (len(refs) == 8 and case['reference_reason'] is None and all(
            np.isfinite([r['value'], r['error_estimate'], r['tail_bound'], r['mass_error']]).all()
            and 0 <= r['error_estimate'] + r['tail_bound'] <= 1e-7
            and 0 <= r['mass_error'] <= 1e-8 for r in refs))
        for row in case['candidates']:
            passed = False
            if row['status'] == 'complete' and valid:
                roots = row['roots']
                if [a for a, _ in roots] != list(range(8)):
                    raise ValueError('root coverage mismatch')
                error = max(abs(v - refs[a]['value']) for a, v in roots)
                regret = refs[row['action']]['value'] - min(r['value'] for r in refs)
                passed = bool(np.isfinite([v for _, v in roots]).all()
                              and error <= 1e-4 and regret <= 1e-4
                              and 0 <= row['seconds'] <= 5 and 0 <= row['states'] <= 100000)
            totals[row['branch_count']] += int(passed)
    return {q: dict(passed_cases=n, total_cases=48, qualified=n == 48) for q, n in totals.items()}


def memory_preflight(q):
    # Stop at the first branch call: test the actual planner's memory admission only.
    class Admitted(Exception):
        pass

    p, t = 2048, 64
    model = QuantileGaussianModel(np.zeros((p, 8)), np.ones((p, 8)), np.zeros((p, t)),
                  np.full(p, 1 / p), target_conditional_variances=1., branch_count=q)
    try:
        with patch('environments.chembench_mopen.batch_horizon.posterior_branches_many',
                   side_effect=Admitted):
            plan_batched(model, model.initial_state, 3, risk_backend='centered',
                         allow_repeats=True, max_states=100000, max_seconds=5)
    except Admitted:
        return dict(admitted=True, depth_run=False)
    except SearchLimitExceeded as exc:
        if str(exc) != 'workspace budget too small for one belief':
            raise
        return dict(admitted=False, depth_run=False, reason=str(exc))
    raise AssertionError('memory probe unexpectedly planned')


def run(output):
    bank = read_bank()
    raw = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('geometry binding mismatch')
    output = Path(output)
    output.mkdir(exist_ok=False)
    cases, shards = [], []
    for task in range(8):
        for seed in (1304, 1305):
            reused = task == 0 and seed == 1304
            if reused:
                data = bank
            else:
                path = output / f'task{task}_seed{seed}.json'
                run_task(path, task_index=task, seed=seed, counts=COUNTS)
                payload = path.read_bytes()
                shards.append(dict(path=path.name, sha256=hashlib.sha256(payload).hexdigest()))
                data = json.loads(payload)
            for case in data['cases']:
                cases.append(dict(case, task_index=task, seed=seed, reused=reused,
                                  candidates=[r for r in case['candidates'] if r['branch_count'] in COUNTS]))
            print('completed', task, seed, 'reused', reused, flush=True)
    result = dict(cases=cases, assessment=assess(cases), shards=shards,
                  memory_preflight={q: memory_preflight(q) for q in COUNTS},
                  reused_sha256=BANK_SHA, design_sha256=DESIGN_SHA,
                  new_cases=45, reused_cases=3, source_measurements=0,
                  model_calls=0, paid_cost_usd=0, deployment_authorized=False)
    with (output / 'result.json').open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(parser.parse_args().output)
