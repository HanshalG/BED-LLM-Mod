from environments.chembench_mopen.horizon import SearchLimitExceeded
from scripts.scilaws_switch_full_audit import evaluate


def test_budget_is_shared_and_partial_is_not_complete():
    class Model:
        num_actions = 2

    class Reference:
        instances = 0
        def __init__(self, *args, **kwargs):
            Reference.instances += 1
            self.evaluations = 0
            self.switches = []
            self.max_inner_error = 0.0
        def action(self, state, action, depth):
            self.evaluations += 60000
            if self.evaluations > 100000:
                raise SearchLimitExceeded('shared budget exhausted')
            return 0.1, 1e-9

    result = evaluate(Model(), None, reference_class=Reference)
    assert Reference.instances == 1
    assert result['status'] == 'incomplete'
    assert len(result['roots']) == 1
    assert result['action'] is None
    assert not result['numerical_check']
