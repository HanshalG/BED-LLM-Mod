import numpy as np
import hashlib
import json

from scripts.discoverphysics_candidate_reconstruction import HELPERS, OUT, load_functions


def test_helper_loader_does_not_execute_module_toplevel(tmp_path):
    path = tmp_path/'module.py'
    path.write_text('raise AssertionError("module executed")\ndef selected(x):\n return x+1\n')
    namespace = {}
    load_functions(path, ('selected',), namespace)
    assert namespace['selected'](2) == 3


def test_exact_action_inputs_and_numerical_only_helpers():
    import math
    namespace = {'np': np, 'math': math}
    for path, names in HELPERS.items():
        load_functions(path, names, namespace)
    actions = namespace['action_table']()
    assert len(actions) == 25
    assert actions['center'] == [0., 0.]
    assert len(namespace['heldout_experiments']()) == 2
    assert 'evaluate_actual' not in namespace
    assert 'hidden_halo_family' not in namespace


def test_banked_candidate_provenance_and_same_objective_root():
    result = json.loads((OUT/'RESULT.json').read_text())
    assert result['status'] == 'complete'
    assert result['model_calls'] == result['physical_endpoint_runs'] == result['cost_usd'] == 0
    assert result['candidate_trajectories'] == 216
    assert result['max_eig_replay_error'] < 1e-12
    assert result['cache_sha256'] == hashlib.sha256((OUT/'CANDIDATES.npz').read_bytes()).hexdigest()
    with np.load(OUT/'CANDIDATES.npz', allow_pickle=False) as data:
        assert data['means'].shape == (25, 8, 2)
        assert data['targets'].shape == (8, 120)
        assert data['maps'].shape == (8, 10, 2)
    for scores in result['risk_by_order'].values():
        assert min(scores, key=scores.get) == 'B'
    assert result['risk_max_refinement_delta'] < 1e-4
