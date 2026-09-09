"""Public-request evidence only: no source laws, target outcomes, or new calls."""
import hashlib
import itertools
import json
import math
from pathlib import Path

from environments.program_induction.scalar_expression import ScalarExpression

ROOT = Path('results/nonmyopic/correction_reasoning_comparison_20260909')
OUTPUT = Path('results/nonmyopic/CORRECTION_PUBLIC_SIGNAL_AUDIT_20260909.json')


def linear_signal(x, residual):
    if len(x) != len(residual) or len(x) != 6:
        raise ValueError('requires six matched public rows')
    if not all(math.isfinite(v) for v in x + residual):
        raise ValueError('nonfinite history')
    centered = [v - sum(x)/len(x) for v in x]
    variation = sum(v*v for v in centered)
    if variation == 0:
        raise ValueError('unvaried input')
    mean = sum(residual)/len(residual)
    centered_y = [v - mean for v in residual]
    covariance = sum(a*b for a, b in zip(centered, centered_y))
    slope = covariance / variation
    before = sum(v*v for v in centered_y)
    after = sum((y-slope*a)**2 for a, y in zip(centered, centered_y))
    total, extreme = 0, 0
    for permutation in itertools.permutations(centered_y):
        total += 1
        statistic = abs(sum(a*b for a, b in zip(centered, permutation)))
        extreme += statistic >= abs(covariance) - 1e-12
    return {'slope': slope, 'constant_sse': before, 'linear_sse': after,
            'sse_reduction_fraction': 1-after/before if before else 0,
            'permutation_extreme': extreme, 'permutation_total': total,
            'exploratory_permutation_p': extreme/total}


def run():
    if OUTPUT.exists():
        raise RuntimeError('already banked')
    result = {'cases': [], 'request_hashes': {}, 'model_calls': 0, 'cost_usd': 0,
              'target_outcomes_read': False, 'true_formulas_read': False,
              'paid_authorized': False, 'old_gate_changed': False}
    for i in range(4):
        requests = {}
        for effort in ('medium', 'high'):
            for arm in ('control', 'refresh'):
                path = ROOT / f'{i}_{effort}_{arm}.request.json'
                raw = path.read_bytes()
                result['request_hashes'][path.name] = hashlib.sha256(raw).hexdigest()
                requests[effort, arm] = json.loads(raw)
        for arm in ('control', 'refresh'):
            if requests['medium', arm]['messages'] != requests['high', arm]['messages']:
                raise ValueError('unmatched reasoning comparison')
        payload = json.loads(requests['medium', 'refresh']['messages'][1]['content'])
        control = json.loads(requests['medium', 'control']['messages'][1]['content'])
        if payload['variables'] != ['x0', 'x1'] or control['history'] != payload['history'][:3]:
            raise ValueError('public payload contract')
        history = payload['history']
        base = ScalarExpression(payload['base_expression'], payload['variables'])
        residual = [r['observed_log_response'] - math.log(base(r['inputs'])) for r in history]
        x = [r['inputs']['x1'] for r in history]
        result['cases'].append({'case': i,
            'control_x1_distinct': len({r['inputs']['x1'] for r in control['history']}),
            'refresh_x1_distinct': len(set(x)), 'residuals_after_base_removal': residual,
            'linear_signal': linear_signal(x, residual),
            'supplied_standardized_residuals': payload['initial_proposal_diagnostics'][0]['standardized_residuals']})
    with OUTPUT.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(result['cases'], indent=2))


if __name__ == '__main__':
    run()
