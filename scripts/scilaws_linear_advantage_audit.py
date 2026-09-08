"""Fixed eight-action first-branch comparison, not a full planning qualification."""
import argparse
import hashlib
import json
from pathlib import Path

from environments.scilaws.adaptive_reference import AdaptiveReference
from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.linear_advantage_reference import LinearAdvantageReference
from scripts.scilaws_initialized_accuracy_audit import observations
from scripts.scilaws_reference_preflight import DESIGN_SHA


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DESIGN_SHA:
        raise ValueError('geometry binding mismatch')
    design = json.loads(raw)['tasks'][0]
    rows = []
    for scenario in ('zero', 'affine', 'quadratic'):
        model, state, _ = initialize_corrected(design, observations(design, scenario), quadrature_order=4)
        child = model.branches(state, 0)[0].state
        results = []
        for cls in (AdaptiveReference, LinearAdvantageReference):
            ref = cls(model, predictive_coordinates=True)
            roots = [ref.terminal(child, a) for a in range(8)]
            results.append(dict(method=cls.__name__, roots=roots, evaluations=ref.evaluations))
        difference = max(abs(v[0]-w[0]) for v, w in zip(results[0]['roots'], results[1]['roots']))
        rows.append(dict(scenario=scenario, results=results, max_difference=difference,
                         passed=difference<=1e-7 and all(e<=1e-7 for r in results for _, e in r['roots'])))
    with output.open('x') as f:
        json.dump(dict(rows=rows, design_sha256=DESIGN_SHA, source_measurements=0,
                       model_calls=0, paid_cost_usd=0, deployment_authorized=False),
                  f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    with threadpool_limits(limits=1, user_api='blas'):
        run(p.parse_args().output)
