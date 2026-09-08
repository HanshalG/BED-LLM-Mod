"""Bounded all-action integral workload against pinned scalar references."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.chembench_mopen.batch_horizon import posterior_branches_many
from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.batched_particle_reference import integrate_actions
from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.posterior_particles import sample_posterior
from environments.scilaws.tail_quantile_belief import TailQuantileGaussianModel
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_particle_correction_panel import read_bound
from scripts.scilaws_particle_integration_panel import SCENARIOS
from scripts.scilaws_reference_preflight import DESIGN_SHA


def run(output):
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
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
        m = TailQuantileGaussianModel(p.means, p.sigmas, p.targets, np.exp(p.initial_state),
            target_weights=p.target_weights, target_conditional_variances=p.target_conditional_variances,
            branch_count=32)
        _, states = posterior_branches_many(m, np.asarray(m.initial_state)[None, :], 0)
        refs = next(c['reference'] for c in bank if (c['task_index'], c['seed'], c['scenario'], c['branch_index'])
                    == (0, 1304, scenario, 15))
        row = dict(scenario=scenario, passed=False)
        try:
            result = integrate_actions(m, states[0, 15])
            discrepancy = max(abs(a['value']-b['value']) for a, b in zip(result['roots'], refs))
            row.update(result, max_discrepancy=discrepancy,
                       passed=discrepancy <= 1e-7 and result['seconds'] <= .04 and result['evaluations'] <= 800)
        except (ValueError, ArithmeticError, SearchLimitExceeded) as exc:
            row['reason'] = str(exc)
        cases.append(row)
        print(scenario, row.get('seconds'), row.get('evaluations'), row.get('max_discrepancy'), flush=True)
    with path.open('x') as f:
        json.dump(dict(cases=cases, workload_passed=all(c['passed'] for c in cases),
                       source_measurements=0, model_calls=0, paid_cost_usd=0, deployment_authorized=False),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(parser.parse_args().output)
