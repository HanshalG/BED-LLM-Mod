import numpy as np
import sys

from scripts.parameter_integration_reference_audit import reference


def test_independent_quadrature_converges_and_has_uncertainty(monkeypatch):
    # The repo's lightweight torch stub lacks Tensor; SciPy probes that attribute.
    # This numerical test uses NumPy only. Restore the stub via fixture teardown.
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    first, second = reference(1024), reference(2048)
    np.testing.assert_allclose(first['mean'], second['mean'], atol=1e-9, rtol=0)
    np.testing.assert_allclose(first['variance'], second['variance'], atol=1e-9, rtol=0)
    assert abs(first['log_evidence']-second['log_evidence']) < 1e-9
    assert all(v > 0 for v in first['variance'])
