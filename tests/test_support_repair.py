from environments.program_induction import constrained
from environments.program_induction.local_support import expand
from environments.program_induction.support_repair import prepare, condition
from scripts.deepcoder_opportunity import load_dsl


def test_contradicted_root_can_seed_a_valid_repair():
    d = load_dsl()
    root = constrained.decode(d, '{"programs":[{"statement":"x2 = Reverse x0","next":{"statement":"x3 = Head x2","next":null}}]}')
    history = [{'inputs': [[1],[2]], 'output':1}]
    observed = {'inputs': [[1,2],[3]], 'output':1}
    assert expand(d, root, history+[observed])[0] == []
    candidates, _ = prepare(d, root, history)
    fixed = condition(candidates, history, observed)
    assert fixed and str(root[0]) not in {str(p) for p in fixed}
    assert condition(candidates, history, {'inputs':[[1,2],[3]],'output':42}) == []
