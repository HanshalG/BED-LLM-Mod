"""Analytic terminal interval feasibility; no outer integration certification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.chembench_mopen.batch_horizon import posterior_branches_many
from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.particle_risk_interval import ParticleRiskIntervals
from environments.scilaws.posterior_particles import sample_posterior
from environments.scilaws.tail_quantile_belief import TailQuantileGaussianModel
from environments.scilaws.weighted_intervals import weighted_min_interval
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_particle_correction_panel import read_bound
from scripts.scilaws_particle_integration_panel import SCENARIOS
from scripts.scilaws_reference_preflight import DESIGN_SHA


def run(output):
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    previous = read_bound('results/nonmyopic/SCILAWS_PARTICLE_CONTINUATION_20260908/result.json',
                         '552743d0b93853c2bb974c179c85b7cc654b4419f8bf556c80d6118360796476')['cases']
    raw = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('geometry binding mismatch')
    cases = []
    for t, design in enumerate(json.loads(raw)['tasks']):
        for seed in (1304, 1305):
            for index, scenario in enumerate(SCENARIOS):
                exact, state, _ = initialize_corrected(design, observations(design, scenario), quadrature_order=4)
                p = sample_posterior(exact, state, particles_per_family=512,
                    rng=np.random.default_rng(np.random.SeedSequence([seed, t, index])), sampling='sobol').model
                m = TailQuantileGaussianModel(p.means, p.sigmas, p.targets, np.exp(p.initial_state),
                    target_weights=p.target_weights, target_conditional_variances=p.target_conditional_variances,
                    branch_count=32)
                _, logs, masses = posterior_branches_many(m, np.asarray(m.initial_state)[None, :], 0,
                                                         return_weights=True)
                evaluator = ParticleRiskIntervals(m)
                intervals = [evaluator.actions(row) for row in logs[0]]
                pairs = [[(r['lower'], r['upper']) for r in row] for row in intervals]
                summary = weighted_min_interval(masses[0].tolist(), pairs)
                tail_indices = [i for i, q in enumerate(m._quantiles) if q < 1e-5 or q > 1-1e-5]
                tail_width = sum(summary['weighted_widths'][i] for i in tail_indices)
                refs = [r for r in previous if (r['task_index'], r['seed'], r['scenario']) == (t, seed, scenario)]
                if [r['branch_index'] for r in refs] != [0, 15, 31]:
                    raise ValueError('banked reference coverage mismatch')
                covered = 0
                for saved in refs:
                    if saved['reason'] is not None or len(saved['reference']) != 8:
                        raise ValueError('incomplete banked reference')
                    for bounds, ref in zip(intervals[saved['branch_index']], saved['reference']):
                        radius = ref['error_estimate']+ref['tail_bound']
                        if not bounds['lower']-radius <= ref['value'] <= bounds['upper']+radius:
                            raise ValueError('independent reference contradicts analytic bounds')
                        covered += 1
                cases.append(dict(task_index=t, seed=seed, scenario=scenario, intervals=intervals,
                                  probabilities=masses[0].tolist(), summary=summary,
                                  extreme_tail_indices=tail_indices, extreme_tail_width=tail_width,
                                  covered_saved_references=covered))
            print('completed', t, seed, flush=True)
    with path.open('x') as f:
        json.dump(dict(cases=cases, design_sha256=DESIGN_SHA, source_measurements=0, model_calls=0,
                       paid_cost_usd=0, outer_error_bounded=False, deployment_authorized=False),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(parser.parse_args().output)
