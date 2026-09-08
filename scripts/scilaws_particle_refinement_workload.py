"""All-root shared-budget refinement workload, not outer quadrature certification."""
import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np

from environments.chembench_mopen.batch_horizon import posterior_branches_many
from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.interval_refinement import refine
from environments.scilaws.particle_risk_interval import ParticleRiskIntervals
from environments.scilaws.posterior_particles import sample_posterior
from environments.scilaws.shared_particle_reference import SharedParticleReference
from environments.scilaws.tail_quantile_belief import TailQuantileGaussianModel
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_particle_correction_panel import read_bound
from scripts.scilaws_particle_integration_panel import SCENARIOS
from scripts.scilaws_reference_preflight import DESIGN_SHA


def run(output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    bank = read_bound('results/nonmyopic/SCILAWS_PARTICLE_CONTINUATION_20260908/result.json',
                     '552743d0b93853c2bb974c179c85b7cc654b4419f8bf556c80d6118360796476')['cases']
    raw = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('geometry binding mismatch')
    design = json.loads(raw)['tasks'][0]
    cases = []
    for index, scenario in enumerate(SCENARIOS):
        exact, state, _ = initialize_corrected(design, observations(design, scenario), quadrature_order=4)
        p = sample_posterior(exact, state, particles_per_family=512,
            rng=np.random.default_rng(np.random.SeedSequence([1304, 0, index])), sampling='sobol').model
        model = TailQuantileGaussianModel(p.means, p.sigmas, p.targets, np.exp(p.initial_state),
            target_weights=p.target_weights, target_conditional_variances=p.target_conditional_variances,
            branch_count=32)
        cached = {c['branch_index']: c['reference'] for c in bank
                  if (c['task_index'], c['seed'], c['scenario']) == (0, 1304, scenario)}
        budget = SharedParticleReference()
        bounds = ParticleRiskIntervals(model)
        roots, trace, reason = [], [], None
        try:
            for root in range(8):
                budget.charge(32)
                _, logs, masses = posterior_branches_many(model, np.asarray(model.initial_state)[None, :],
                                                         root, return_weights=True)
                intervals = []
                for child in logs[0]:
                    budget.check()
                    intervals.append([(r['lower'], r['upper']) for r in bounds.actions(child)])

                def evaluate(branch, action):
                    budget.check()
                    reused = root == 0 and branch in cached
                    row = cached[branch][action] if reused else budget.action(model, logs[0, branch], action)
                    trace.append(dict(root=root, branch=branch, action=action, reused=reused, result=row))
                    budget.check()
                    return row['value'], row['error_estimate']+row['tail_bound']

                result = refine(masses[0].tolist(), intervals, evaluate)
                budget.check()
                roots.append(dict(root=root, result=result))
        except (ValueError, ArithmeticError, SearchLimitExceeded) as exc:
            reason = str(exc)
        result = dict(scenario=scenario, roots=roots, trace=trace, reason=reason,
                      evaluations=budget.evaluations, seconds=monotonic()-budget.start,
                      all_roots_complete=len(roots) == 8 and reason is None,
                      outer_error_bounded=False, cached_reference_work_charged=False,
                      deployment_authorized=False, model_calls=0, source_measurements=0, paid_cost_usd=0)
        cases.append(result)
        with (output / f'{scenario}.json').open('x') as f:
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
        print(scenario, len(roots), len(trace), reason, flush=True)
    with (output / 'result.json').open('x') as f:
        json.dump(dict(cases=cases, source_measurements=0, model_calls=0, paid_cost_usd=0,
                       deployment_authorized=False), f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(parser.parse_args().output)
