"""Independent numerical references for the existing adaptive SMC backend."""
import hashlib
import json
import math
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.special import roots_legendre

from environments.chembench_mopen.smc import TransformedParameterPrior, adaptive_tempered_smc
from scripts.parameter_integration_reference_audit import X, TARGETS, OBS, SIGMA, reference

OUTPUT = Path('results/nonmyopic/PARAMETER_SMC_REFERENCE_AUDIT_20260909.json')
PROTOCOL = Path('results/nonmyopic/PARAMETER_SMC_REFERENCE_PROTOCOL_20260909.md')


def multimodal_reference(order):
    nodes, mass = roots_legendre(order)
    theta = 3*nodes
    loglik = -.5*((theta**2-1)/.1)**2-math.log(.1*math.sqrt(2*math.pi))
    raw = mass/2*np.exp(loglik)
    z = raw.sum()
    weights = raw/z
    m2, m4 = weights@(theta**2), weights@(theta**4)
    return {'mean': [0., 0., 0., float(2*m2)],
            'variance': [float(m2), float(m2), float(m2*m2), float(2*(m4-m2*m2))],
            'log_evidence': float(2*np.log(z)), 'mode_mass': [.25]*4}


def fixture(name):
    if name == 'one_parameter':
        prior = TransformedParameterPrior(('k',), ('log',), np.array([math.log(.01)]), np.array([math.log(100.)]))
        def likelihood(z):
            return (-.5*((np.log1p(np.exp(z)*X)-OBS)/SIGMA)**2-math.log(SIGMA*math.sqrt(2*math.pi))).sum(axis=1)
        def predict(z):
            return np.log1p(np.exp(z)*TARGETS)
        return prior, likelihood, predict, reference
    if name != 'four_modes':
        raise ValueError('unknown fixture')
    prior = TransformedParameterPrior(('a', 'b'), ('identity', 'identity'), np.array([-3., -3.]), np.array([3., 3.]))
    def likelihood(z):
        return (-.5*((z*z-1)/.1)**2-math.log(.1*math.sqrt(2*math.pi))).sum(axis=1)
    def predict(z):
        a, b = z.T
        return np.column_stack((a, b, a*b, a*a+b*b))
    return prior, likelihood, predict, multimodal_reference


def run():
    if OUTPUT.exists():
        raise RuntimeError('already banked')
    output = {'rows': [], 'references': {}, 'model_calls': 0, 'cost_usd': 0,
              'paid_authorized': False, 'old_gate_changed': False,
              'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              'smc_sha256': hashlib.sha256(Path('environments/chembench_mopen/smc.py').read_bytes()).hexdigest()}
    try:
        for name in ('one_parameter', 'four_modes'):
            prior, likelihood, predict, integrate = fixture(name)
            first, ref = integrate(1024), integrate(2048)
            for key in ('mean', 'variance', 'log_evidence'):
                if not np.allclose(first[key], ref[key], atol=1e-9, rtol=0):
                    raise ValueError('reference not converged')
            output['references'][name] = ref
            for n in (512, 2048):
                for seed in range(8):
                    start = monotonic()
                    calls = 0
                    def counted(z):
                        nonlocal calls
                        calls += len(z)
                        if calls > 600000 or monotonic()-start > 30:
                            raise RuntimeError('frozen evaluation/time cap')
                        return likelihood(z)
                    result = adaptive_tempered_smc(prior, counted, num_particles=n, seed=seed,
                        initialization='sobol', proposal_geometry='full')
                    pred = predict(result.particles)
                    mean = result.weights@pred
                    var = result.weights@((pred-mean)**2)
                    error = np.max(np.abs(mean-ref['mean']))
                    scaled_error = np.max(np.abs(mean-ref['mean'])/np.sqrt(ref['variance']))
                    variance_error = np.max(np.abs(var/ref['variance']-1))
                    evidence_error = abs(result.log_evidence-ref['log_evidence'])
                    row = {'fixture': name, 'particles': n, 'seed': seed,
                        'max_mean_error': float(error), 'max_mean_error_in_posterior_sd': float(scaled_error),
                        'max_relative_variance_error': float(variance_error),
                        'absolute_log_evidence_error': float(evidence_error),
                        'mean': mean.tolist(), 'variance': var.tolist(),
                        'likelihood_rows': calls, 'seconds': monotonic()-start,
                        'rungs': result.diagnostics.num_rungs,
                        'acceptance_rate': result.diagnostics.aggregate_acceptance_rate,
                        'unique_particles': len(np.unique(result.particles, axis=0))}
                    qualifies = evidence_error <= .1 and variance_error <= .2
                    if name == 'one_parameter':
                        qualifies = qualifies and error <= .01
                    else:
                        modes = ((result.particles[:, 0] > 0).astype(int)*2
                                 +(result.particles[:, 1] > 0).astype(int))
                        masses = np.bincount(modes, weights=result.weights, minlength=4)
                        row['mode_masses'] = masses.tolist()
                        qualifies = qualifies and scaled_error <= .15 and np.max(np.abs(masses-.25)) <= .1
                    row['numerical_qualified'] = bool(qualifies)
                    output['rows'].append(row)
        output['status'] = 'complete'
    except Exception as error:
        output['status'] = 'failed_closed'
        output['error'] = str(error)
    with OUTPUT.open('x') as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(output['status'])
    for name in ('one_parameter', 'four_modes'):
        for n in (512, 2048):
            rows = [r for r in output['rows'] if r['fixture'] == name and r['particles'] == n]
            if rows:
                print(name, n, 'qualified', sum(r['numerical_qualified'] for r in rows), '/', len(rows),
                      'worst_logZ', max(r['absolute_log_evidence_error'] for r in rows),
                      'worst_scaled_mean', max(r['max_mean_error_in_posterior_sd'] for r in rows))


if __name__ == '__main__':
    run()
