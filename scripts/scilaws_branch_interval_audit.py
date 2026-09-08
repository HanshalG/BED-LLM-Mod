"""Zero-integration replay of saved branch scores against analytic risk intervals."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.linear_risk_interval import terminal_risk_interval
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_reference_preflight import DESIGN_SHA


SAVED_SHA = '1e5cb5d540feedf4d27d039703144ac336fabd46ad2b809bf920d5ccc53b7177'


def certificate(score, lower, upper):
    if not all(math.isfinite(x) for x in (score, lower, upper)) or lower > upper:
        raise ValueError('invalid interval or score')
    error = max(abs(score-lower), abs(score-upper))
    return dict(worst_case_error=error, certified=error <= 1e-4)


def run(output):
    from threadpoolctl import threadpool_limits
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path('results/nonmyopic/SCILAWS_BRANCH_ACCURACY_AUDIT_20260908.json').read_bytes()
    geometry = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != SAVED_SHA or hashlib.sha256(geometry).hexdigest() != DESIGN_SHA:
        raise ValueError('binding mismatch')
    designs = {d['task_id']: d for d in json.loads(geometry)['tasks']}
    cases = []
    for case in json.loads(raw)['cases']:
        d = designs[case['task_id']]
        # Preserve the saved run's construction arithmetic before exact identity checks.
        with threadpool_limits(limits=8, user_api='blas'):
            model, state, _ = initialize_corrected(d, observations(d, case['scenario']), quadrature_order=4)
            branches = [(a, b) for a in range(8) for b in model.branches(state, a)]
        records = []
        for r in case['records']:
            a, branch = branches[r['branch_index']]
            if (a != r['action'] or branch.observation != r['observation']
                    or branch.probability != r['probability']):
                raise ValueError('branch identity mismatch')
            interval = terminal_risk_interval(model, branch.state, r['next_action'])
            lo, hi = interval['lower'], interval['upper']
            records.append(dict(branch_index=r['branch_index'], next_action=r['next_action'],
                lower=lo, upper=hi, width=hi-lo,
                reference_contained=lo-r['reference_error'] <= r['reference'] <= hi+r['reference_error'],
                candidates={o: certificate(v, lo, hi) for o, v in r['scores'].items()}))
        cases.append(dict(task_id=case['task_id'], scenario=case['scenario'], records=records))
    with output.open('x') as f:
        json.dump(dict(cases=cases, source_sha256=SAVED_SHA, design_sha256=DESIGN_SHA,
                       integrations=0, model_calls=0, source_measurements=0, paid_cost_usd=0,
                       deployment_authorized=False,
                       interpretation='analytic_working_model_bounds_on_saved_prefix_only'),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(parser.parse_args().output)
