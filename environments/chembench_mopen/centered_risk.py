"""Linear-storage target variance with a two-pass cancellation fallback."""
import numpy as np


class CenteredTargetRisk:
    def __init__(self, model):
        self.centered = (model.targets-model.targets[0])*np.sqrt(model.target_weights)
        self.second = np.sum(self.centered**2, axis=1)
        self.noise = model.target_noise_risk
        if not np.isfinite(self.centered).all() or not np.isfinite(self.second).all():
            raise ValueError('unrepresentable centered target moments')
        self.centered.setflags(write=False)
        self.second.setflags(write=False)

    def __call__(self, weights):
        means = weights @ self.centered
        second = weights @ self.second
        squared_mean = np.sum(means**2, axis=1)
        variance = second-squared_mean
        threshold = 64*np.finfo(float).eps*np.maximum(second+squared_mean, 1.)
        for i in np.flatnonzero(variance <= threshold):
            # Recenter at this posterior mean instead of subtracting near-equal moments.
            residual = self.centered-means[i]
            variance[i] = weights[i] @ np.sum(residual**2, axis=1)
        result = variance + weights @ self.noise
        if not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError('invalid centered target risk')
        return result
