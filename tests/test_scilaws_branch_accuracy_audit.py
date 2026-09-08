from scripts.scilaws_branch_accuracy_audit import assessment


def test_incomplete_or_error_cannot_pass():
    records = [dict(reference=1., reference_error=1e-9, scores={'4': 1., '8': 1.})]
    assert all(r['passed'] for r in assessment(records, True, 1))
    assert not any(r['passed'] for r in assessment(records, False, 1))
    assert not any(r['passed'] for r in assessment(records, True, 2))
    records[0]['scores']['4'] = 1.001
    assert not assessment(records, True, 1)[0]['passed']
    records[0]['reference_error'] = 1e-6
    assert not any(r['passed'] for r in assessment(records, True, 1))
