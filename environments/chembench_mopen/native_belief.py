"""Explicit optional compiled engine; never silently changes the default model."""

from pathlib import Path
import numpy as np

from .envelope_belief import EnvelopeGaussianModel
from .crossing_belief import _gauss


class NativeEnvelopeGaussianModel(EnvelopeGaussianModel):
    def __init__(self, *args, **kwargs):
        import pyximport

        pyximport.install(
            language_level=3,
            build_dir=str(Path.home() / ".cache" / "bed-native-quantiles"),
        )
        from ._native_quantiles import envelope_starts, invert, rules

        self._invert_quantiles_many = invert
        self._envelope_starts = envelope_starts
        self._compiled_rules = rules
        super().__init__(*args, **kwargs)
        self._nodes = np.zeros((self.branch_count + 1, self.branch_count))
        self._masses = np.zeros_like(self._nodes)
        self._workspace_fixed_bytes = self._nodes.nbytes + self._masses.nbytes
        for order in range(1, self.branch_count + 1):
            self._nodes[order, :order], self._masses[order, :order] = _gauss(order)

    def _quadrature_rules_many(self, logs, weights, action):
        sigmas = self.sigmas[:, action]
        if not np.all(sigmas == sigmas[0]):
            rules = [self.quadrature_rule(row, action) for row in logs]
            return np.stack([r[0] for r in rules]), np.stack([r[1] for r in rules])
        return self._compiled_rules(
            logs,
            weights,
            self.means[:, action],
            sigmas[0],
            self.branch_count,
            self._nodes,
            self._masses,
        )
