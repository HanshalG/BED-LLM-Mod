"""CPU-only microprofile of the fixed particle integrand, not a policy rerun."""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
from time import perf_counter

import numpy as np

from environments.chembench_mopen.centered_risk import CenteredTargetRisk
from environments.chembench_mopen.batch_horizon import posterior_branches_many
from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.posterior_particles import sample_posterior
from environments.scilaws.tail_quantile_belief import TailQuantileGaussianModel
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_particle_integration_panel import SCENARIOS
from scripts.scilaws_reference_preflight import DESIGN_SHA


def density_risk(joint, risk, *, shifted=False):
    if shifted:
        peak = np.max(joint)
        numerator = np.exp(joint-peak)
        normalizer = np.sum(numerator)
        weights = numerator/normalizer
        total = peak+np.log(normalizer)
    else:
        total = np.logaddexp.reduce(joint)
        weights = np.exp(joint-total)
    return np.exp(total)*float(risk(weights[None, :])[0])


def run(output):
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
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
        _, states = posterior_branches_many(model, np.asarray(model.initial_state)[None, :], 0)
        state = states[0, 15]
        risk = CenteredTargetRisk(model)

        def workload(shifted):
            values = []
            for a in range(8):
                mu, sd = model.means[:, a], model.sigmas[:, a]
                constants = state-np.log(sd)-.5*np.log(2*np.pi)
                for y in np.linspace(np.min(mu-8*sd), np.max(mu+8*sd), 257):
                    joint = constants-.5*((y-mu)/sd)**2
                    values.append(density_risk(joint, risk, shifted=shifted))
            return np.asarray(values)

        timings = {False: [], True: []}
        values = {}
        for repetition in range(3):
            for shifted in ((False, True) if repetition % 2 == 0 else (True, False)):
                started = perf_counter()
                values[shifted] = workload(shifted)
                timings[shifted].append(perf_counter()-started)
        difference = float(np.max(np.abs(values[False]-values[True])))
        if not np.isfinite(list(values.values())).all() or difference > 1e-10:
            raise ValueError('normalization microkernel equivalence failed')
        profiler = cProfile.Profile()
        profiler.runcall(workload, False)
        stats = pstats.Stats(profiler)
        rows = [dict(function=f'{key[0]}:{key[1]}:{key[2]}', calls=value[1], own_seconds=value[2],
                     cumulative_seconds=value[3]) for key, value in stats.stats.items()]
        rows.sort(key=lambda r: -r['own_seconds'])
        cases.append(dict(scenario=scenario, evaluations_per_repeat=2056, repeats=3,
                          original_seconds=timings[False], shifted_seconds=timings[True],
                          maximum_integrand_difference=difference, profile=rows[:20]))
        print(scenario, timings, difference, flush=True)
    with path.open('x') as f:
        json.dump(dict(cases=cases, source_measurements=0, model_calls=0, paid_cost_usd=0,
                       policy_run=False, deployment_authorized=False), f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(parser.parse_args().output)
