"""Complete analytic branch-domain allocation audit, no adaptive integrals."""
import argparse
import hashlib
import json
from pathlib import Path

from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.linear_risk_interval import terminal_risk_interval
from environments.scilaws.weighted_intervals import weighted_min_interval
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_reference_preflight import DESIGN_SHA


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('geometry binding mismatch')
    cases = []
    for design in json.loads(raw)['tasks']:
        for scenario in ('zero', 'affine', 'quadratic'):
            model, state, _ = initialize_corrected(design, observations(design, scenario), quadrature_order=4)
            roots = []
            for a in range(8):
                branches = model.branches(state, a)
                intervals = []
                for b in branches:
                    row = [terminal_risk_interval(model, b.state, k) for k in range(8)]
                    intervals.append([(v['lower'], v['upper']) for v in row])
                result = weighted_min_interval([b.probability for b in branches], intervals,
                    correction=model.horizon_chance_risk_correction(state, a, 2))
                roots.append(dict(action=a, branches=len(branches), **result))
            cases.append(dict(task_id=design['task_id'], scenario=scenario, roots=roots))
    with output.open('x') as f:
        json.dump(dict(cases=cases, design_sha256=DESIGN_SHA, outer_order=4,
                       terminal_error_budget=5e-5, blas_threads=1,
                       integrations=0, source_measurements=0, model_calls=0,
                       paid_cost_usd=0, deployment_authorized=False,
                       interpretation='full_outer4_terminal_interval_allocation_not_continuous_root_certificate'),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(p.parse_args().output)
