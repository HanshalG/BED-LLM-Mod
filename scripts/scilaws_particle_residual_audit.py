"""Retrospective residual-tail diagnosis, with no new candidate counts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.particle_residual_reference import ParticleResidualReference
from environments.scilaws.posterior_particles import sample_posterior
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_particle_correction_panel import read_bound, REFERENCES, REFERENCE_SHA
from scripts.scilaws_reference_preflight import DESIGN_SHA


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    references = read_bound(REFERENCES, REFERENCE_SHA)['cases']
    corrected = read_bound('results/nonmyopic/SCILAWS_PARTICLE_CORRECTION_PANEL_20260908/result.json',
        '34f3835cec1b0f7bd3846d056411cd2697b59c9ceb8916000eab926fe6792516')['cases']
    raw = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('geometry binding mismatch')
    designs = json.loads(raw)['tasks']
    cases = []
    for t, seed in ((0, 1304), (2, 1305), (6, 1304)):
        design = designs[t]
        exact, state, _ = initialize_corrected(design, observations(design, 'affine'), quadrature_order=4)
        p = sample_posterior(exact, state, particles_per_family=512,
            rng=np.random.default_rng(np.random.SeedSequence([seed, t, 1])), sampling='sobol').model
        model = QuantileGaussianModel(p.means, p.sigmas, p.targets, np.exp(p.initial_state),
            target_weights=p.target_weights, target_conditional_variances=p.target_conditional_variances,
            branch_count=32)
        def select(data):
            return next(c for c in data if (c['task_index'], c['seed'], c['scenario']) == (t, seed, 'affine'))
        refs = select(references)['reference']
        old = next(r for r in select(corrected)['candidates'] if r['branch_count'] == 32)['roots']
        reference = ParticleResidualReference(model, model.initial_state)
        rows, reason = [], None
        try:
            for a in range(8):
                row = reference.action(a)
                row.update(action=a, independent_risk_difference=row['value']-refs[a]['value'],
                           quantile_advantage_deficit=row['advantage']-old[a]['advantage'],
                           saved_root_error=old[a]['value']-refs[a]['value'])
                rows.append(row)
        except (ValueError, ArithmeticError, SearchLimitExceeded) as exc:
            reason = str(exc)
        cases.append(dict(task_index=t, seed=seed, scenario='affine', rows=rows, reason=reason,
                          evaluations=reference.evaluations))
        print(t, seed, len(rows), reason, flush=True)
    with output.open('x') as f:
        json.dump(dict(cases=cases, source_measurements=0, model_calls=0, paid_cost_usd=0,
                       deployment_authorized=False), f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(parser.parse_args().output)
