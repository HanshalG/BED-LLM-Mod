import numpy as np

from scripts.discoverphysics_candidate_reconstruction import HELPERS, load_functions


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
