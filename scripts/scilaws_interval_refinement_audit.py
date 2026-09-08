"""Full order-four outer roots with interval-guided terminal reference work."""
import argparse
import hashlib
import json
from pathlib import Path

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.adaptive_reference import AdaptiveReference
from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.interval_refinement import refine
from environments.scilaws.linear_risk_interval import terminal_risk_interval
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
    for d in json.loads(raw)['tasks']:
        for scenario in ('zero', 'affine', 'quadratic'):
            model, state, _ = initialize_corrected(d, observations(d, scenario), quadrature_order=4)
            ref = AdaptiveReference(model, predictive_coordinates=True)
            roots, reason = [], None
            try:
                for a in range(8):
                    ref.check()
                    branches = model.branches(state, a)
                    intervals = []
                    for b in branches:
                        row = []
                        for k in range(8):
                            ref.check()
                            v = terminal_risk_interval(model, b.state, k)
                            row.append((v['lower'], v['upper']))
                        intervals.append(row)
                    def evaluate(i, k):
                        return ref.terminal(branches[i].state, k)
                    result = refine([b.probability for b in branches], intervals, evaluate,
                                    correction=model.horizon_chance_risk_correction(state, a, 2))
                    ref.check()
                    roots.append(dict(action=a, **result))
            except (ValueError, SearchLimitExceeded) as exc:
                reason = str(exc)
            cases.append(dict(task_id=d['task_id'], scenario=scenario, roots=roots,
                              complete=len(roots)==8 and reason is None, reason=reason,
                              evaluations=ref.evaluations))
            print(d['task_id'], scenario, len(roots), ref.evaluations, reason, flush=True)
    with output.open('x') as f:
        json.dump(dict(cases=cases, design_sha256=DESIGN_SHA, outer_order=4,
                       seconds_cap=5, evaluations_cap=100000, blas_threads=1,
                       terminal_error_budget=5e-5, outer_error_bounded=False,
                       source_measurements=0, model_calls=0, paid_cost_usd=0,
                       deployment_authorized=False), f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(p.parse_args().output)
