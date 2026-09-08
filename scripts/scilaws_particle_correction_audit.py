"""Fixed first-task correction workload against banked independent references."""
import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np

from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.particle_linear_correction import ParticleLinearCorrection
from environments.scilaws.posterior_particles import sample_posterior
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_particle_integration_panel import read_bank, BANK_SHA, SCENARIOS
from scripts.scilaws_reference_preflight import DESIGN_SHA


def run(output):
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    bank = read_bank()
    raw = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('geometry binding mismatch')
    design = json.loads(raw)['tasks'][0]
    cases = []
    for index, scenario in enumerate(SCENARIOS):
        exact, state, _ = initialize_corrected(design, observations(design, scenario), quadrature_order=4)
        p = sample_posterior(exact, state, particles_per_family=512,
            rng=np.random.default_rng(np.random.SeedSequence([1304, 0, index])), sampling='sobol').model
        saved = bank['cases'][index]
        assert saved['scenario'] == scenario and saved['reference_reason'] is None
        refs = saved['reference']
        candidates = []
        for count in (4, 8, 16, 32):
            model = QuantileGaussianModel(p.means, p.sigmas, p.targets, np.exp(p.initial_state),
                target_weights=p.target_weights, target_conditional_variances=p.target_conditional_variances,
                branch_count=count)
            evaluator = ParticleLinearCorrection(model, model.initial_state)
            rows, row = [], dict(branch_count=count, passed=False)
            started = monotonic()
            try:
                for a in range(8):
                    rows.append(evaluator.action(a))
                action = min(range(8), key=lambda a: (rows[a]['value'], a))
                error = max(abs(rows[a]['value']-refs[a]['value']) for a in range(8))
                regret = refs[action]['value']-min(r['value'] for r in refs)
                row.update(status='complete', action=action, max_error=error, regret=regret,
                           passed=error <= 1e-4 and regret <= 1e-4)
            except (ValueError, ArithmeticError, SearchLimitExceeded) as exc:
                row.update(status='incomplete', reason=str(exc))
            row.update(roots=rows, states=evaluator.states, seconds=monotonic()-started)
            candidates.append(row)
        cases.append(dict(scenario=scenario, candidates=candidates))
        print(scenario, [r['passed'] for r in candidates], flush=True)
    with path.open('x') as f:
        json.dump(dict(cases=cases, reference_sha256=BANK_SHA, design_sha256=DESIGN_SHA,
                       source_measurements=0, model_calls=0, paid_cost_usd=0,
                       deployment_authorized=False), f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(parser.parse_args().output)
