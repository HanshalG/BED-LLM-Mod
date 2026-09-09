"""Numerical qualification on a known one-parameter fixture, no LLM/world gate."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre, logsumexp

from environments.chembench_mopen.executable_belief import ExecutableBeliefPool

OUTPUT = Path('results/nonmyopic/PARAMETER_INTEGRATION_REFERENCE_AUDIT_20260909.json')
PROTOCOL = Path('results/nonmyopic/PARAMETER_INTEGRATION_REFERENCE_PROTOCOL_20260909.md')
X = np.array([.1, .3, .7, 1.4, 3., 5.])
TARGETS = np.array([.2, .8, 2., 4.])
SIGMA = .05
OBS = np.log1p(1.3*X) + SIGMA*np.array([-.5, .5, 0., 1., -1., .25])


def reference(order, observations=OBS):
    nodes, masses = roots_legendre(order)
    low, high = math.log(.01), math.log(100.)
    theta = np.exp((nodes+1)*(high-low)/2+low)
    loglik = (-.5*((np.log1p(theta[:, None]*X)-observations)/SIGMA)**2
              - math.log(SIGMA*math.sqrt(2*math.pi))).sum(axis=1)
    unnormalized = np.log(masses/2)+loglik
    evidence = logsumexp(unnormalized)
    weights = np.exp(unnormalized-evidence)
    pred = np.log1p(theta[:, None]*TARGETS)
    mean = weights@pred
    variance = weights@((pred-mean)**2)
    return {'mean': mean.tolist(), 'variance': variance.tolist(), 'log_evidence': float(evidence)}


def point(x):
    return [float(x), 0., 1., 0., 1., 310., 7.]


def run():
    if OUTPUT.exists():
        raise RuntimeError('already banked')
    refs = [reference(n) for n in (1024, 2048)]
    for key in refs[0]:
        if not np.allclose(refs[0][key], refs[1][key], rtol=0, atol=1e-9):
            raise ValueError('quadrature reference failed convergence')
    ref = refs[-1]
    rows = []
    for count in (32, 256, 2048):
        for seed in range(8):
            pool = ExecutableBeliefPool(particles_per_law=count, seed=seed)
            pool.add({'name': 'known_linear', 'expr': 'k*C_A',
                      'params': [{'name': 'k', 'low': .01, 'high': 100., 'transform': 'log'}]})
            snap = pool.snapshot(history_inputs=[point(x) for x in X], observations=OBS,
                                 designs=[point(1.)], targets=[point(x) for x in TARGETS], sigma=SIGMA)
            weights = np.exp(snap.state)
            mean = snap.model.forecast(snap.state)
            variance = weights@((snap.model.targets-mean)**2)
            ess = float(1/(weights@weights))
            mean_error = float(np.max(np.abs(mean-ref['mean'])))
            evidence_error = abs(snap.conditional_log_evidence-ref['log_evidence'])
            rows.append({'particles': count, 'seed': seed, 'ess': ess,
                         'mean': mean.tolist(), 'variance': variance.tolist(),
                         'max_mean_error': mean_error, 'absolute_log_evidence_error': evidence_error,
                         'numerical_qualified': mean_error <= .01 and evidence_error <= .1 and ess >= 10})
    result = {'status': 'complete', 'rows': rows, 'reference': ref,
              'reference_orders': [1024, 2048], 'reference_tolerance': 1e-9,
              'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              'fitter_sha256': hashlib.sha256(Path('environments/chembench_mopen/executable_belief.py').read_bytes()).hexdigest(),
              'model_calls': 0, 'cost_usd': 0, 'new_scientific_outcomes': False,
              'paid_authorized': False}
    with OUTPUT.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    for n in (32, 256, 2048):
        selected = [r for r in rows if r['particles'] == n]
        print(n, 'passes', sum(r['numerical_qualified'] for r in selected),
              'ESS', [round(r['ess'], 2) for r in selected],
              'maxmean', max(r['max_mean_error'] for r in selected),
              'maxlogZ', max(r['absolute_log_evidence_error'] for r in selected))


if __name__ == '__main__':
    run()
