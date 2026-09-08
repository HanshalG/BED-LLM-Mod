import pytest

from scripts.scilaws_branch_interval_audit import certificate


def test_certificate_checks_both_endpoints():
    assert certificate(1., .99999, 1.00001)['certified']
    assert not certificate(1., 1., 1.01)['certified']
    assert not certificate(1., .99, 1.)['certified']
    with pytest.raises(ValueError):
        certificate(1., 2., 1.)
    with pytest.raises(ValueError):
        certificate(float('nan'), 0., 1.)
