"""Opened numerical regression fixtures, no scientific cohort or paid calls."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from environments.chembench_mopen.adaptive_parameter_integral import adaptive_parameter_integral
from scripts.parameter_smc_reference_audit import fixture, OBS, SIGMA


def main():
    path = Path('results/nonmyopic/ADAPTIVE_PARAMETER_REFERENCE_20260909.json')
    if path.exists():
        raise RuntimeError('already banked')
    data = {'model_calls': 0, 'cost_usd': 0, 'rows': [], 'backend_sha256': hashlib.sha256(
        Path('environments/chembench_mopen/adaptive_parameter_integral.py').read_bytes()).hexdigest()}
    for name in ('one_parameter', 'four_modes'):
        prior, likelihood, predict, reference = fixture(name)
        count, sigma = (len(OBS), SIGMA) if name == 'one_parameter' else (2, .1)
        result = adaptive_parameter_integral(prior.lower, prior.upper, likelihood, predict,
                    output_size=4, log_likelihood_bound=-count*math.log(sigma*math.sqrt(2*math.pi)))
        result['fixture'] = name
        if result['status'] == 'agreement':
            ref, check = reference(2048), result['checks'][-1]
            result['reference_errors'] = {
                'log_evidence': abs(check['log_evidence']-ref['log_evidence']),
                'mean': float(np.max(np.abs(np.array(check['mean'])-ref['mean']))),
                'relative_variance': float(np.max(np.abs(np.array(check['variance'])/ref['variance']-1)))}
        data['rows'].append(result)
    with path.open('x') as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write('\n')
    for row in data['rows']:
        print(row['fixture'], row['status'], row['evaluated_rows'], row.get('reference_errors', row.get('reason')))


if __name__ == '__main__':
    main()
