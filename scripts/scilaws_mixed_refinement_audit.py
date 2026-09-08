"""All-root mixed-family h2 integration diagnostic, no source measurements."""

import argparse
import json
from pathlib import Path

from environments.scilaws.horizon_control_variate import HorizonControlVariateMixture
from environments.scilaws.regression_belief import RegressionBelief
from scripts.scilaws_control_variate_deep_audit import plan_row


HISTORIES = (
    (), ((0, 0.7),), ((0, -0.7),), ((0, 0.7), (1, -0.7)),
)
ORDERS = (4, 8, 16, 32, 64)


def fixture(order, history):
    m = HorizonControlVariateMixture(
        [[[1.0], [2.0]], [[2.0], [1.0]]],
        [[[1.0], [3.0]], [[1.0], [2.0]]],
        [RegressionBelief([-0.5], [[2.0]], 3.0, 0.2),
         RegressionBelief([0.5], [[1.0]], 3.0, 0.4)],
        [0.4, 0.6], target_weights=[0.25, 0.75],
        quadrature_order=order, include_observation_noise=True,
    )
    state = m.initial_state
    for action, observation in history:
        state = m.condition(state, action, observation)
    return m, state


def assess(rows):
    indexed = {r['order']: r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != set(ORDERS):
        raise ValueError("incomplete or duplicate order coverage")
    for row in rows:
        if row['status'] == 'completed':
            if {a for a, _ in row['root_action_values']} != {0, 1}:
                raise ValueError("missing exact root values")
    if any(indexed[o]['status'] != 'completed' for o in (32, 64)):
        return dict(reference_valid=False, reason='reference_incomplete', candidates=[])
    ref = dict(indexed[64]['root_action_values'])
    previous = dict(indexed[32]['root_action_values'])
    delta = max(abs(ref[a] - previous[a]) for a in ref)
    valid = delta <= 1e-5
    candidates = []
    for order in (4, 8, 16):
        row = indexed[order]
        complete = row['status'] == 'completed'
        values = dict(row['root_action_values']) if complete else {}
        error = max(abs(values[a] - ref[a]) for a in ref) if complete else None
        regret = ref[row['action']] - min(ref.values()) if complete else None
        candidates.append(dict(order=order, max_root_error=error, regret=regret,
                               passed=bool(valid and complete and error <= 1e-4
                                           and regret <= 1e-4)))
    return dict(reference_valid=valid, reference_max_root_delta=delta,
                candidates=candidates)


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    cases = []
    for i, history in enumerate(HISTORIES):
        rows = []
        for order in ORDERS:
            m, state = fixture(order, history)
            row = plan_row(m, state, 2)
            row['order'] = order
            rows.append(row)
            print(i, order, row['status'], flush=True)
        cases.append(dict(case=i, history=history, rows=rows, assessment=assess(rows)))
    result = dict(schema_version=1, cases=cases, depth=2, nodes_cap=100000,
                  seconds_cap=5, reference_tolerance=1e-5, candidate_tolerance=1e-4,
                  source_measurements=0, model_calls=0, paid_cost_usd=0,
                  source_execution_authorized=False, lower_order_deployment_authorized=False,
                  interpretation='synthetic_mixed_h2_refinement_not_h3_or_source_calibration')
    with output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    run(p.parse_args().output)
