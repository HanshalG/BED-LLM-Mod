"""Full geometry initialization contracts with declared synthetic labels only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.reference_prior import initialize, unit_points
from scripts.scilaws_reference_preflight import DESIGN_SHA


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('geometry binding mismatch')
    rows = []
    for design in json.loads(raw)['tasks']:
        x = unit_points(design['initial_points'], design['axes'])
        for scenario in ('zero', 'affine', 'quadratic'):
            signal = (np.zeros(len(x)) if scenario=='zero' else
                      x.mean(axis=1) if scenario=='affine' else (x*x).mean(axis=1))
            if design['initial_replicates'] != 2:
                raise ValueError('frozen two-replicate contract changed')
            y = np.repeat(signal[:, None], 2, axis=1)
            if scenario != 'zero':
                y += np.array([-.05, .05])
            base, posterior, _ = initialize(design, y, quadrature_order=4)
            model, state, scale = initialize_corrected(design, y, quadrature_order=4)
            error = abs(model.risk(state)-base.risk(posterior))
            passed = (state.components == posterior.components and error <= 1e-12
                      and np.allclose(state.log_weights, posterior.log_weights, atol=1e-14, rtol=0)
                      and np.isfinite(model.forecast(state)).all()
                      and all(c.shape == 3+y.size/2 for c in state.components))
            rows.append(dict(task_id=design['task_id'], scenario=scenario,
                initial_observations=int(y.size), actions=model.num_actions,
                targets=len(model.target_weights), families=len(state.components),
                scale=scale.value, risk_difference=error, passed=bool(passed)))
    with output.open('x') as f:
        json.dump(dict(rows=rows, design_sha256=DESIGN_SHA,
                       all_contracts_pass=all(r['passed'] for r in rows),
                       source_measurements=0, model_calls=0, planning_calls=0,
                       paid_cost_usd=0, deployment_authorized=False,
                       interpretation='synthetic_initialization_contracts_not_runtime_accuracy_or_opportunity'),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    run(p.parse_args().output)
