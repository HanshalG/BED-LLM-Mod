"""Posterior-particle approximation screen against the exact conjugate model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t

from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.posterior_particles import sample_posterior
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_reference_preflight import DESIGN_SHA


def assess(metrics):
    limits = dict(mean_standardized_max=.05, variance_relative_max=.05,
                  family_tv=.05, log_density_error_max=.1)
    return bool(all(np.isfinite(metrics[k]) and 0 <= metrics[k] <= v for k, v in limits.items())
                and np.isfinite(metrics['ess_fraction']) and metrics['ess_fraction'] >= .1)


def compare(exact, state, particles, logs):
    p = particles.model
    w = np.exp(logs)
    mean, variance = exact.moments(state)
    pm = p.forecast(logs)
    pv = w @ ((p.targets-pm)**2 + p.target_conditional_variances)
    family = np.bincount(particles.family_indices, weights=w, minlength=len(state.components))
    errors = []
    for a in range(exact.num_actions):
        rows = np.array([b.predictive(x[a]) for b, x in zip(state.components, exact.action_features)])
        df, loc, scale2 = rows.T
        weights = np.exp(state.log_weights)
        center = weights @ loc
        sd = np.sqrt(weights @ (scale2*df/(df-2)+(loc-center)**2))
        for z in (-2., 0., 2.):
            y = center + z*sd
            truth = logsumexp(state.log_weights+t.logpdf(y, df, loc=loc, scale=np.sqrt(scale2)))
            estimate = logsumexp(np.asarray(logs)+p.log_likelihood(a, y))
            errors.append(abs(truth-estimate))
    result = dict(mean_standardized_max=float(np.max(abs(pm-mean)/np.sqrt(variance))),
                  variance_relative_max=float(np.max(abs(pv-variance)/variance)),
                  family_tv=float(np.sum(abs(family-np.exp(state.log_weights)))/2),
                  log_density_error_max=float(max(errors)),
                  ess_fraction=float(1/(len(w)*np.sum(w*w))))
    return dict(**result, passed=assess(result))


def run(output, *, sampling='iid'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('geometry binding mismatch')
    results = []
    for index, d in enumerate(json.loads(raw)['tasks']):
        for scenario in ('zero', 'affine', 'quadratic'):
            exact, state, _ = initialize_corrected(d, observations(d, scenario), quadrature_order=4)
            df, loc, scale2 = np.array([b.predictive(x[0]) for b, x in zip(state.components, exact.action_features)]).T
            w = np.exp(state.log_weights)
            center = w @ loc
            sd = np.sqrt(w @ (scale2*df/(df-2)+(loc-center)**2))
            for n in (32, 128, 512):
                for seed in (1304, 1305):
                    rng = np.random.default_rng(np.random.SeedSequence([seed, index, ('zero','affine','quadratic').index(scenario)]))
                    particles = sample_posterior(exact, state, particles_per_family=n, rng=rng, sampling=sampling)
                    checks = [dict(stage='initial', **compare(exact, state, particles, particles.model.initial_state))]
                    for z in (-2., 0., 2.):
                        y = float(center+z*sd)
                        updated = exact.condition(state, 0, y)
                        logs = particles.model.condition(particles.model.initial_state, 0, y)
                        checks.append(dict(stage=f'update_{z:g}', **compare(exact, updated, particles, logs)))
                    count = particles.model.num_particles
                    minimum_workspace = 8*count**2 + 8*3*(20*16*count+8*count*64) + count*65*8
                    results.append(dict(task_id=d['task_id'], scenario=scenario, seed=seed,
                        particles_per_family=n, particles=count, checks=checks,
                        minimum_h3_workspace_bytes=minimum_workspace,
                        h3_workspace_fits=minimum_workspace<=64*1024**2,
                        passed=all(c['passed'] for c in checks)))
            print(d['task_id'], scenario, flush=True)
    with output.open('x') as f:
        json.dump(dict(results=results, design_sha256=DESIGN_SHA, sampling=sampling, source_measurements=0,
                       planning_calls=0, model_calls=0, paid_cost_usd=0,
                       deployment_authorized=False), f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--sampling', choices=['iid', 'sobol'], default='iid')
    args = p.parse_args()
    with threadpool_limits(limits=1, user_api='blas'):
        run(args.output, sampling=args.sampling)
