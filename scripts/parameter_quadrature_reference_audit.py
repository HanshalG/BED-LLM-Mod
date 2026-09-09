"""One-shot diagnostic on opened numerical fixtures, not scientific endpoints."""
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.chembench_mopen.parameter_quadrature import (
    IntegrationUnresolved, integrate_parameters,
)
from scripts.parameter_smc_reference_audit import fixture


def main():
    output = Path('results/nonmyopic/PARAMETER_QUADRATURE_REFERENCE_20260909.json')
    if output.exists():
        raise RuntimeError('already banked')
    data = {'model_calls': 0, 'cost_usd': 0, 'scientific_gate_opened': False,
            'backend_sha256': hashlib.sha256(Path(
                'environments/chembench_mopen/parameter_quadrature.py').read_bytes()).hexdigest(),
            'rows': []}
    for name in ('one_parameter', 'four_modes'):
        prior, likelihood, predict, reference = fixture(name)
        ref = reference(2048)
        try:
            result = integrate_parameters(prior.lower, prior.upper, likelihood,
                                          predict, output_size=4)
        except IntegrationUnresolved as error:
            data['rows'].append({'fixture': name, 'status': 'unresolved', 'reason': str(error)})
            continue
        data['rows'].append({
            'fixture': name, 'status': 'resolution_agreement',
            'orders': result.orders, 'evaluated_rows': result.evaluated_rows,
            'log_evidence': result.log_evidence, 'mean': result.mean.tolist(),
            'variance': result.variance.tolist(),
            'log_evidence_error': abs(result.log_evidence - ref['log_evidence']),
            'max_mean_error': float(np.max(np.abs(result.mean-ref['mean']))),
            'max_relative_variance_error': float(np.max(np.abs(result.variance/ref['variance']-1))),
        })
    with output.open('x') as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
