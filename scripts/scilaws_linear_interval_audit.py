"""Replay analytic intervals against banked inner solves; no new quadrature."""
import argparse
import hashlib
import json
from pathlib import Path

from environments.scilaws.linear_risk_interval import terminal_risk_interval
from scripts.scilaws_mixed_refinement_audit import fixture


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path('results/nonmyopic/SCILAWS_NESTED_WORK_AUDIT_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != 'ba31f479f9a64c9ff6717c792c968b18ba05916d82d546e630c3cb7fe6396011':
        raise ValueError('trace binding mismatch')
    m, state = fixture(8, ())
    rows = []
    for record in json.loads(raw)['records']:
        if record['status'] != 'completed':
            continue
        child = m.condition(state, 0, record['observation'])
        interval = terminal_risk_interval(m, child, record['action'])
        rows.append(dict(observation=record['observation'], action=record['action'],
            lower=interval['lower'], upper=interval['upper'], width=interval['width'],
            contains_recorded_value=interval['lower']-record['error'] <= record['value']
                                    <= interval['upper']+record['error'],
            skip_eligible=interval['width']/2 <= 1e-8,
            recorded_evaluations=record['evaluations']))
    with output.open('x') as f:
        json.dump(dict(rows=rows, midpoint_error_allowance=1e-8,
                       eligible_count=sum(r['skip_eligible'] for r in rows),
                       saved_evaluations=sum(r['recorded_evaluations'] for r in rows if r['skip_eligible']),
                       model_calls=0, source_measurements=0, new_integrals=0,
                       paid_cost_usd=0, deployment_authorized=False),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    run(p.parse_args().output)
