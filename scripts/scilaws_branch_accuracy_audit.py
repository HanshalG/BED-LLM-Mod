"""Bounded independent checks on actual outer-order-four branch posteriors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.adaptive_reference import AdaptiveReference
from environments.scilaws.initialized_reference import initialize_corrected
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_reference_preflight import DESIGN_SHA


def assessment(records, complete, expected):
    if len(records) > expected:
        raise ValueError('excess reference coverage')
    valid = complete and len(records) == expected
    results = []
    for order in (4, 8):
        errors = [abs(r['scores'][str(order)]-r['reference']) for r in records]
        finite = all(np.isfinite(e) for e in errors)
        reference_ok = all(0 <= r['reference_error'] <= 1e-7 for r in records)
        error = max(errors, default=None)
        results.append(dict(order=order, max_observed_error=error,
                            passed=bool(valid and finite and reference_ok
                                        and error is not None and error <= 1e-4)))
    return results


def run(output, *, blas_threads=None):
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
            outer, state, _ = initialize_corrected(design, y, quadrature_order=4)
            fine, _, _ = initialize_corrected(design, y, quadrature_order=8)
            # Enumerate the fixed diagnostic domain before starting the integration budget.
            branches = [(a, b) for a in range(8) for b in outer.branches(state, a)]
            expected = len(branches)*8
            reference = AdaptiveReference(outer, predictive_coordinates=True)
            records, reason = [], None
            try:
                for branch_index, (a, branch) in enumerate(branches):
                    for next_action in range(8):
                        value, error = reference.terminal(branch.state, next_action)
                        scores = {str(o): m.expected_terminal_risk(branch.state, next_action)[0]
                                  for o, m in ((4, outer), (8, fine))}
                        records.append(dict(branch_index=branch_index, action=a,
                            observation=branch.observation, probability=branch.probability,
                            next_action=next_action, reference=value, reference_error=error,
                            scores=scores))
            except (SearchLimitExceeded, ValueError) as exc:
                reason = str(exc)
            result = dict(task_id=design['task_id'], scenario=scenario,
                          expected_integrals=expected, completed_integrals=len(records),
                          evaluations=reference.evaluations, reason=reason, records=records,
                          assessment=assessment(records, reason is None, expected))
            cases.append(result)
            print(design['task_id'], scenario, len(records), expected, reason,
                  [r['max_observed_error'] for r in result['assessment']], flush=True)
    with output.open('x') as f:
        json.dump(dict(cases=cases, design_sha256=DESIGN_SHA, outer_order=4,
                       blas_threads=blas_threads,
                       reference_seconds_cap=5, reference_evaluations_cap=100000,
                       source_measurements=0, model_calls=0, paid_cost_usd=0,
                       deployment_authorized=False,
                       interpretation='actual_order4_branch_diagnostic_not_full_h2_or_h3_accuracy'),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--single-thread-blas', action='store_true')
    args = p.parse_args()
    if args.single_thread_blas:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api='blas'):
            run(args.output, blas_threads=1)
    else:
        run(args.output)
