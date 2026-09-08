"""One opened synthetic root; partitioning feasibility, not a policy result."""
import argparse
import json
from pathlib import Path

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.switch_reference import SwitchReference
from scripts.scilaws_mixed_refinement_audit import fixture


def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    m, state = fixture(8, ())
    ref = SwitchReference(m, predictive_coordinates=True)
    result = dict(status='incomplete', case=0, root_action=0, depth=2,
                  model_calls=0, source_measurements=0, paid_cost_usd=0,
                  reference_qualified=False, full_panel_authorized=False)
    try:
        value, error = ref.action(state, 0, 2)
        result.update(status='root_complete', value=value, outer_error_estimate=error)
    except (SearchLimitExceeded, ValueError, RuntimeError) as exc:
        result['reason'] = str(exc)
    result.update(evaluations=ref.evaluations, switches=ref.switches,
                  max_inner_error_estimate=ref.max_inner_error)
    with output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    run(p.parse_args().output)
