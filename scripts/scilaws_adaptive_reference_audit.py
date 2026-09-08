"""Bounded independent reference on the four already-opened synthetic histories."""

import argparse
import json
from pathlib import Path

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.adaptive_reference import AdaptiveReference
from scripts.scilaws_mixed_refinement_audit import HISTORIES, fixture


def run(output, *, predictive_coordinates=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for case, history in enumerate(HISTORIES):
        for depth in (1, 2):
            m, state = fixture(8, history)
            reference = AdaptiveReference(m, predictive_coordinates=predictive_coordinates)
            row = dict(case=case, depth=depth, history=history)
            try:
                values = [reference.action(state, a, depth) for a in range(2)]
                row.update(status='completed', roots=values,
                           action=min(range(2), key=lambda a: (values[a][0], a)))
                if depth == 1:
                    comparison = []
                    for order in (4, 8, 16, 64):
                        candidate, belief = fixture(order, history)
                        scores = [candidate.expected_terminal_risk(belief, a)[0]
                                  for a in range(2)]
                        comparison.append(dict(order=order, scores=scores,
                            max_error=max(abs(scores[a]-values[a][0]) for a in range(2))))
                    row['comparison'] = comparison
            except (SearchLimitExceeded, ValueError) as exc:
                row.update(status='incomplete', reason=str(exc))
            row.update(evaluations=reference.evaluations,
                       max_inner_error_estimate=reference.max_inner_error)
            rows.append(row)
            print(case, depth, row['status'], flush=True)
    result = dict(rows=rows, tolerance=1e-8, seconds_cap=5, evaluations_cap=100000,
                  predictive_coordinates=predictive_coordinates,
                  source_measurements=0, model_calls=0, paid_cost_usd=0,
                  deployment_authorized=False,
                  interpretation='adaptive_error_estimates_not_rigorous_bounds_or_source_evidence')
    with output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--predictive-coordinates', action='store_true')
    args = p.parse_args()
    run(args.output, predictive_coordinates=args.predictive_coordinates)
