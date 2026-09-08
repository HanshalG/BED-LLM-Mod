import pytest

from scripts.scilaws_initialized_accuracy_audit import assess


def test_accuracy_gate():
    ref = dict(status='completed', roots=[(float(a), 1e-9) for a in range(8)])
    candidate = dict(status='completed', root_action_values=list(enumerate(range(8))), action=0)
    assert assess(ref, candidate)['passed']
    candidate['action'] = 1
    assert not assess(ref, candidate)['passed']
    candidate['root_action_values'] = [(0, 0)]
    with pytest.raises(ValueError, match='coverage'):
        assess(ref, candidate)


def test_incomplete_or_uncertain_reference_cannot_pass():
    candidate = dict(status='completed', root_action_values=list(enumerate(range(8))), action=0)
    assert not assess(dict(status='incomplete'), candidate)['passed']
    ref = dict(status='completed', roots=[(float(a), 1e-6) for a in range(8)])
    assert not assess(ref, candidate)['passed']
