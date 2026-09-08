"""Initialized eight-action h1 numerical comparison, synthetic labels only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.adaptive_reference import AdaptiveReference
from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.reference_prior import unit_points
from scripts.scilaws_control_variate_deep_audit import plan_row
from scripts.scilaws_reference_preflight import DESIGN_SHA


def observations(design, scenario):
    x = unit_points(design['initial_points'], design['axes'])
    if scenario not in ('zero', 'affine', 'quadratic') or design['initial_replicates'] != 2:
        raise ValueError('invalid synthetic fixture')
    signal = (np.zeros(len(x)) if scenario == 'zero' else
              x.mean(axis=1) if scenario == 'affine' else (x*x).mean(axis=1))
    return np.repeat(signal[:, None], 2, axis=1) + (
        np.zeros(2) if scenario == 'zero' else np.array([-.05, .05]))


def assess(reference, candidate):
    if reference['status'] != 'completed' or candidate['status'] != 'completed':
        return dict(passed=False, reason='incomplete')
    roots = reference['roots']
    scores = dict(candidate['root_action_values'])
    if len(roots) != 8 or len(candidate['root_action_values']) != 8 or set(scores) != set(range(8)):
        raise ValueError('incomplete action coverage')
    if not np.isfinite(roots).all() or not np.isfinite(list(scores.values())).all():
        raise ValueError('nonfinite scores')
    error = max(abs(scores[a]-roots[a][0]) for a in scores)
    regret = roots[candidate['action']][0] - min(v[0] for v in roots)
    valid = max(v[1] for v in roots) <= 1e-7
    return dict(passed=bool(valid and error <= 1e-4 and regret <= 1e-4),
                reference_valid=bool(valid), max_root_error=error, regret=regret)


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('geometry binding mismatch')
    rows = []
    for design in json.loads(raw)['tasks']:
        for scenario in ('zero', 'affine', 'quadratic'):
            y = observations(design, scenario)
            model, state, _ = initialize_corrected(design, y, quadrature_order=8)
            ref = AdaptiveReference(model, predictive_coordinates=True)
            reference = dict(status='incomplete', roots=[])
            try:
                for a in range(8):
                    reference['roots'].append(ref.terminal(state, a))
                reference['status'] = 'completed'
            except (SearchLimitExceeded, ValueError) as exc:
                reference['reason'] = str(exc)
            reference['evaluations'] = ref.evaluations
            candidates = []
            for order in (4, 8, 16, 32, 64):
                model, state, _ = initialize_corrected(design, y, quadrature_order=order)
                candidate = plan_row(model, state, 1)
                candidate.update(order=order, assessment=assess(reference, candidate))
                candidates.append(candidate)
            rows.append(dict(task_id=design['task_id'], scenario=scenario,
                             reference=reference, candidates=candidates))
            print(design['task_id'], scenario, reference['status'],
                  [c['assessment']['passed'] for c in candidates], flush=True)
    with output.open('x') as f:
        json.dump(dict(rows=rows, design_sha256=DESIGN_SHA, depth=1,
                       seconds_cap=5, evaluations_cap=100000, tolerance=1e-8,
                       candidate_tolerance=1e-4, source_measurements=0,
                       model_calls=0, paid_cost_usd=0, deployment_authorized=False,
                       interpretation='initialized_synthetic_h1_not_source_or_h3_qualification'),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
