"""First-task full-action integration diagnostic at the calibrated particle count."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.chembench_mopen.batch_horizon import plan_batched
from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.particle_reference import ParticleReference
from environments.scilaws.posterior_particles import sample_posterior
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_reference_preflight import DESIGN_SHA


def run(output, *, task_index=0, seed=1304, counts=(4, 8, 16, 32, 64)):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('geometry binding mismatch')
    design = json.loads(raw)['tasks'][task_index]
    cases = []
    for index, scenario in enumerate(('zero', 'affine', 'quadratic')):
        exact, state, _ = initialize_corrected(design, observations(design, scenario), quadrature_order=4)
        rng = np.random.default_rng(np.random.SeedSequence([seed, task_index, index]))
        p = sample_posterior(exact, state, particles_per_family=512, rng=rng, sampling='sobol').model
        reference = ParticleReference(p, p.initial_state)
        refs, reason = [], None
        try:
            for a in range(8):
                refs.append(reference.action(a))
        except (ValueError, SearchLimitExceeded) as exc:
            reason = str(exc)
        candidates = []
        for q in counts:
            model = QuantileGaussianModel(p.means, p.sigmas, p.targets, np.exp(p.initial_state),
                target_weights=p.target_weights, target_conditional_variances=p.target_conditional_variances,
                branch_count=q)
            row = dict(branch_count=q, passed=False)
            try:
                plan = plan_batched(model, model.initial_state, 1, risk_backend='centered',
                                    max_states=100000, max_seconds=5)
                row.update(status='complete', roots=plan.root_values, action=plan.action,
                           seconds=plan.elapsed_seconds, states=plan.processed_states)
                if len(refs) == 8 and reason is None:
                    error = max(abs(v-refs[a]['value']) for a, v in plan.root_values)
                    regret = refs[plan.action]['value']-min(r['value'] for r in refs)
                    row.update(max_error=error, regret=regret,
                        passed=error<=1e-4 and regret<=1e-4 and all(
                            r['error_estimate']+r['tail_bound']<=1e-7 for r in refs))
            except (ValueError, SearchLimitExceeded) as exc:
                row.update(status='incomplete', reason=str(exc))
            candidates.append(row)
        cases.append(dict(scenario=scenario, reference=refs, reference_reason=reason,
                          reference_evaluations=reference.evaluations, candidates=candidates))
        print(scenario, len(refs), reason, [r['passed'] for r in candidates], flush=True)
    with output.open('x') as f:
        json.dump(dict(cases=cases, design_sha256=DESIGN_SHA, particles_per_family=512,
                       seed=seed, source_measurements=0, model_calls=0, paid_cost_usd=0,
                       deployment_authorized=False), f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(p.parse_args().output)
