"""Both roots per opened history, with a single shared numerical budget."""

import argparse
import json
import math
from pathlib import Path

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.switch_reference import SwitchReference
from scripts.scilaws_mixed_refinement_audit import HISTORIES, fixture


def evaluate(model, state, *, reference_class=SwitchReference):
    ref = reference_class(model, predictive_coordinates=True)
    row = dict(status='incomplete', roots=[], action=None, numerical_check=False)
    try:
        for action in range(model.num_actions):
            value, error = ref.action(state, action, 2)
            if not all(math.isfinite(v) and v >= 0 for v in (value, error)):
                raise ValueError('invalid reference result')
            row['roots'].append(dict(action=action, value=value, error=error))
        row['status'] = 'completed'
        row['action'] = min(row['roots'], key=lambda r: (r['value'], r['action']))['action']
        row['numerical_check'] = all(
            r['error'] + ref.max_inner_error <= 1e-5 for r in row['roots'])
    except (SearchLimitExceeded, ValueError, RuntimeError) as exc:
        row['reason'] = str(exc)
    row.update(evaluations=ref.evaluations, switches=ref.switches,
               max_inner_error_estimate=ref.max_inner_error)
    return row


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for case, history in enumerate(HISTORIES):
        m, state = fixture(8, history)
        row = evaluate(m, state)
        row.update(case=case, history=history)
        rows.append(row)
        print(case, row['status'], row['evaluations'], flush=True)
    result = dict(rows=rows, seconds_cap_per_plan=5, evaluations_cap_per_plan=100000,
                  all_complete=all(r['status']=='completed' for r in rows),
                  all_numerical_checks=all(r['numerical_check'] for r in rows),
                  error_interpretation='reported_estimates_not_rigorous_nested_bounds',
                  source_measurements=0, model_calls=0, paid_cost_usd=0,
                  source_execution_authorized=False, depth_three_authorized=False)
    with output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    run(p.parse_args().output)
