import json

import pytest

threadpool_limits = pytest.importorskip('threadpoolctl').threadpool_limits

from environments.scilaws.adaptive_reference import AdaptiveReference
from environments.scilaws.initialized_reference import initialize_corrected
from scripts.scilaws_initialized_accuracy_audit import observations


@pytest.mark.parametrize('scenario', ['zero', 'affine', 'quadratic'])
def test_blas_thread_count_preserves_full_geometry_reference(scenario):
    design = json.load(open('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json'))['tasks'][0]
    results = []
    for threads in (8, 1):
        with threadpool_limits(limits=threads, user_api='blas'):
            model, state, _ = initialize_corrected(design, observations(design, scenario),
                                                    quadrature_order=4)
            child = model.branches(state, 0)[0].state
            reference = AdaptiveReference(model, predictive_coordinates=True)
            roots = [reference.terminal(child, a)[0] for a in range(8)]
            results.append((roots, reference.evaluations))
    assert results[0][0] == pytest.approx(results[1][0], abs=1e-12, rel=0)
    assert results[0][1] == results[1][1]
