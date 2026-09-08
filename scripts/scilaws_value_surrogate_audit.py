"""Fixed interpolation gate on all opened histories, shared budget per plan."""
import argparse
import json
from pathlib import Path

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.adaptive_reference import AdaptiveReference
from environments.scilaws.value_surrogate import approximate_root
from scripts.scilaws_mixed_refinement_audit import HISTORIES, fixture


def run(output, *, adaptive=False, linear_baseline=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for case, history in enumerate(HISTORIES):
        m, state = fixture(8, history)
        ref = AdaptiveReference(m, predictive_coordinates=True)
        row = dict(case=case, roots=[], status='incomplete')
        try:
            for action in (0, 1):
                result = approximate_root(ref, state, action, adaptive=adaptive,
                                          linear_baseline=linear_baseline)
                row['roots'].append(dict(action=action, **result))
            row['status'] = 'sample_checks_passed' if all(
                r['status']=='sample_checks_passed' for r in row['roots']) else 'gate_failed'
        except (SearchLimitExceeded, ValueError) as exc:
            row['reason'] = str(exc)
        row['evaluations'] = ref.evaluations
        rows.append(row)
        print(case, row['status'], ref.evaluations, flush=True)
    with output.open('x') as f:
        json.dump(dict(rows=rows, adaptive=adaptive, linear_baseline=linear_baseline,
                       model_calls=0, source_measurements=0, paid_cost_usd=0,
                       deployment_authorized=False, uniform_error_proven=False),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--adaptive', action='store_true')
    p.add_argument('--linear-baseline', action='store_true')
    args = p.parse_args()
    run(args.output, adaptive=args.adaptive, linear_baseline=args.linear_baseline)
