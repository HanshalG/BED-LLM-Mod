"""Fixed analytic tail envelope sweep on opened synthetic histories."""
import argparse
import json
from pathlib import Path

from environments.scilaws.adaptive_reference import AdaptiveReference
from environments.scilaws.tail_risk_bound import tail_risk_bound
from scripts.scilaws_mixed_refinement_audit import HISTORIES, fixture


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for case, history in enumerate(HISTORIES):
        m, state = fixture(8, history)
        ref = AdaptiveReference(m, predictive_coordinates=True)
        for action in (0, 1):
            df, loc, scale2 = ref.density_parameters(state, action)
            c = ref.coordinates(df, loc, scale2, m._state(state))
            bounds = []
            for radius in (4, 8, 16, 32, 64):
                row = tail_risk_bound(m, state, action,
                    c['center']-radius*c['scale'], c['center']+radius*c['scale'])
                row['radius'] = radius
                bounds.append(row)
            selected = next((r['radius'] for r in bounds if r['value'] <= 1e-6), None)
            rows.append(dict(case=case, action=action, bounds=bounds, selected_radius=selected))
    with output.open('x') as f:
        json.dump(dict(rows=rows, tail_allowance=1e-6, model_calls=0,
                       source_measurements=0, paid_cost_usd=0, deployment_authorized=False),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    run(p.parse_args().output)
