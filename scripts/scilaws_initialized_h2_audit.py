"""Full initialized h2 refinement without source measurements."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.scilaws.initialized_reference import initialize_corrected
from scripts.scilaws_control_variate_deep_audit import plan_row
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_reference_preflight import DESIGN_SHA


ORDERS = (4, 8, 16, 32, 64)


def assess(rows):
    indexed = {r['order']: r for r in rows}
    if len(rows) != len(ORDERS) or set(indexed) != set(ORDERS):
        raise ValueError('incomplete order coverage')
    for row in rows:
        if row['status'] == 'completed':
            values = row['root_action_values']
            if len(values) != 8 or {a for a, _ in values} != set(range(8)):
                raise ValueError('incomplete action coverage')
            if not np.isfinite([v for _, v in values]).all():
                raise ValueError('nonfinite root value')
    if any(indexed[o]['status'] != 'completed' for o in (32, 64)):
        return dict(reference_valid=False, reason='reference_incomplete', candidates=[])
    ref = dict(indexed[64]['root_action_values'])
    coarse = dict(indexed[32]['root_action_values'])
    delta = max(abs(ref[a]-coarse[a]) for a in ref)
    valid = delta <= 1e-5
    candidates = []
    for order in (4, 8, 16):
        row = indexed[order]
        complete = row['status'] == 'completed'
        scores = dict(row['root_action_values']) if complete else {}
        error = max(abs(scores[a]-ref[a]) for a in ref) if complete else None
        regret = ref[row['action']]-min(ref.values()) if complete else None
        candidates.append(dict(order=order, max_root_error=error, regret=regret,
                               passed=bool(valid and complete and error <= 1e-4
                                           and regret <= 1e-4)))
    return dict(reference_valid=valid, reference_delta=delta, candidates=candidates)


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
            y = observations(design, scenario)
            rows = []
            for order in ORDERS:
                model, state, _ = initialize_corrected(design, y, quadrature_order=order)
                row = plan_row(model, state, 2)
                row['order'] = order
                rows.append(row)
                print(design['task_id'], scenario, order, row['status'], flush=True)
            cases.append(dict(task_id=design['task_id'], scenario=scenario,
                              rows=rows, assessment=assess(rows)))
    with output.open('x') as f:
        json.dump(dict(cases=cases, design_sha256=DESIGN_SHA, depth=2,
                       seconds_cap=5, nodes_cap=100000, reference_tolerance=1e-5,
                       candidate_tolerance=1e-4, source_measurements=0, model_calls=0,
                       paid_cost_usd=0, deployment_authorized=False,
                       interpretation='initialized_h2_refinement_not_independent_proof_or_source_evidence'),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
