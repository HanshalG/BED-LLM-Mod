"""Fixed continuation-history accuracy screen, including extreme predictive nodes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.chembench_mopen.batch_horizon import posterior_branches_many
from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.particle_linear_correction import ParticleLinearCorrection
from environments.scilaws.particle_reference import ParticleReference
from environments.scilaws.posterior_particles import sample_posterior
from environments.scilaws.tail_quantile_belief import TailQuantileGaussianModel
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_particle_correction_panel import read_bound
from scripts.scilaws_particle_integration_panel import SCENARIOS
from scripts.scilaws_reference_preflight import DESIGN_SHA

BRANCHES = (0, 15, 31)


def assess(cases):
    expected = [(t, seed, c, b) for t in range(8) for seed in (1304, 1305)
                for c in SCENARIOS for b in BRANCHES]
    if [(c['task_index'], c['seed'], c['scenario'], c['branch_index']) for c in cases] != expected:
        raise ValueError('continuation coverage mismatch')
    passed = 0
    for case in cases:
        refs, roots = case['reference'], case['roots']
        if case['reason'] is not None or len(refs) != 8 or len(roots) != 8:
            continue
        if not all(np.isfinite([r['value'], r['error_estimate'], r['tail_bound'], r['mass_error']]).all()
                   and 0 <= r['error_estimate']+r['tail_bound'] <= 1e-7
                   and 0 <= r['mass_error'] <= 1e-8 for r in refs):
            continue
        values = [r['value'] for r in roots]
        if not np.isfinite(values).all():
            continue
        action = min(range(8), key=lambda a: (values[a], a))
        error = max(abs(v-r['value']) for v, r in zip(values, refs))
        regret = refs[action]['value']-min(r['value'] for r in refs)
        passed += int(error <= 1e-4 and regret <= 1e-4)
    return dict(passed_cases=passed, total_cases=144, screen_passed=passed == 144,
                full_tree_qualified=False)


def run(output):
    read_bound('results/nonmyopic/SCILAWS_PARTICLE_TAIL_PANEL_20260908/result.json',
               '95e53384a587fa90cdbbf776eb22613eae1dd72ec3f615a6839fff201dbce588')
    raw = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('geometry binding mismatch')
    output = Path(output)
    output.mkdir(exist_ok=False)
    cases, shards = [], []
    for t, design in enumerate(json.loads(raw)['tasks']):
        for seed in (1304, 1305):
            batch = []
            for index, scenario in enumerate(SCENARIOS):
                exact, state, _ = initialize_corrected(design, observations(design, scenario), quadrature_order=4)
                p = sample_posterior(exact, state, particles_per_family=512,
                    rng=np.random.default_rng(np.random.SeedSequence([seed, t, index])), sampling='sobol').model
                model = TailQuantileGaussianModel(p.means, p.sigmas, p.targets, np.exp(p.initial_state),
                    target_weights=p.target_weights, target_conditional_variances=p.target_conditional_variances,
                    branch_count=32)
                ys, logs = posterior_branches_many(model, np.asarray(model.initial_state)[None, :], 0)
                for branch in BRANCHES:
                    state = logs[0, branch]
                    reference = ParticleReference(model, state)
                    refs, roots, reason = [], [], None
                    try:
                        for a in range(8):
                            refs.append(reference.action(a))
                        evaluator = ParticleLinearCorrection(model, state)
                        for a in range(8):
                            roots.append(evaluator.action(a))
                    except (ValueError, ArithmeticError, SearchLimitExceeded) as exc:
                        reason = str(exc)
                    row = dict(task_index=t, seed=seed, scenario=scenario, branch_index=branch,
                               query_action=0, observation=float(ys[0, branch]),
                               reference=refs, roots=roots, reason=reason,
                               reference_evaluations=reference.evaluations)
                    if reason is None:
                        action = min(range(8), key=lambda a: (roots[a]['value'], a))
                        row.update(action=action, max_error=max(abs(roots[a]['value']-refs[a]['value']) for a in range(8)),
                                   regret=refs[action]['value']-min(r['value'] for r in refs))
                    batch.append(row)
            path = output / f'task{t}_seed{seed}.json'
            with path.open('x') as f:
                json.dump(batch, f, indent=2, sort_keys=True, allow_nan=False)
                f.write('\n')
            shards.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            cases.extend(batch)
            print(t, seed, 'completed', len(batch), 'failures', sum(c['reason'] is not None or c.get('max_error', 1.) > 1e-4 for c in batch), flush=True)
    with (output / 'result.json').open('x') as f:
        json.dump(dict(cases=cases, shards=shards, assessment=assess(cases), design_sha256=DESIGN_SHA,
                       source_measurements=0, model_calls=0, paid_cost_usd=0, deployment_authorized=False),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(parser.parse_args().output)
