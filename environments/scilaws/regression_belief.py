"""Conjugate parameter/noise inference conditional on fixed feature functions.

Noise variance ~ InvGamma(shape, scale); coefficients | variance are normal
with covariance variance * precision^{-1}. This is a homoscedastic Gaussian
observation model, not a claim about the hidden simulator's empirical noise.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.special import gammaln


@lru_cache(maxsize=1024)
def _feature_solve(precision, features):
    """Precision depends on actions, not observed values; reuse exact solves."""
    value = np.linalg.solve(precision, features)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class RegressionBelief:
    mean: tuple[float, ...]
    precision: tuple[tuple[float, ...], ...]
    shape: float
    scale: float

    def __post_init__(self):
        mean = np.asarray(self.mean, dtype=float)
        precision = np.asarray(self.precision, dtype=float)
        if (
            mean.ndim != 1
            or not 1 <= len(mean) <= 32
            or precision.shape != (len(mean), len(mean))
            or not np.isfinite(mean).all()
            or not np.isfinite(precision).all()
            or not np.allclose(precision, precision.T, atol=1e-12, rtol=0)
        ):
            raise ValueError("finite mean and symmetric precision required")
        for value in (self.shape, self.scale):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("positive finite shape/scale required")
        if self.shape <= 1:
            raise ValueError("shape > 1 required for finite predictive risk")
        try:
            np.linalg.cholesky(precision)
        except np.linalg.LinAlgError as exc:
            raise ValueError("positive definite precision required") from exc
        object.__setattr__(self, "mean", tuple(float(x) for x in mean))
        object.__setattr__(
            self, "precision", tuple(tuple(float(x) for x in row) for row in precision)
        )
        object.__setattr__(self, "shape", float(self.shape))
        object.__setattr__(self, "scale", float(self.scale))

    def _features(self, features):
        x = np.asarray(features, dtype=float)
        if x.shape != (len(self.mean),) or not np.isfinite(x).all():
            raise ValueError("finite feature vector required")
        return x

    def predictive(self, features):
        """Return Student-t degrees of freedom, location and squared scale."""
        x = self._features(features)
        leverage = float(x @ _feature_solve(self.precision, tuple(x)))
        result = (
            2 * self.shape,
            float(x @ self.mean),
            self.scale / self.shape * (1 + leverage),
        )
        if not all(math.isfinite(v) for v in result) or result[2] <= 0:
            raise ValueError("predictive parameters out of numerical range")
        return result

    def log_predictive(self, features, observation):
        if isinstance(observation, bool) or not math.isfinite(observation):
            raise ValueError("finite scalar observation required")
        df, location, scale2 = self.predictive(features)
        with np.errstate(over="raise", invalid="raise"):
            z2 = np.square(np.float64(observation) - location) / scale2
            value = (
                gammaln((df + 1) / 2)
                - gammaln(df / 2)
                - 0.5 * math.log(df * math.pi * scale2)
                - (df + 1) / 2 * np.log1p(z2 / df)
            )
        if not math.isfinite(value):
            raise ValueError("predictive density out of numerical range")
        return float(value)

    def condition(self, features, observation):
        """Return a new state and the pre-update predictive log density."""
        log_density = self.log_predictive(features, observation)
        x = self._features(features)
        solved = _feature_solve(self.precision, tuple(x))
        denominator = 1 + x @ solved
        residual = observation - x @ self.mean
        # Rank-one update avoids subtracting large quadratic sufficient statistics.
        mean = np.asarray(self.mean) + solved * residual / denominator
        precision = np.asarray(self.precision) + np.outer(x, x)
        scale = self.scale + 0.5 * residual**2 / denominator
        return RegressionBelief(mean, precision, self.shape + 0.5, scale), log_density

    def target_moments(self, features):
        """Mean and epistemic variance of the latent regression at each target.

        Add noise_variance to obtain future-observation variance. A log-response
        model must keep this loss in log space: exponentiated Student-t means
        are not finite in general.
        """
        x = np.asarray(features, dtype=float)
        if (
            x.ndim != 2
            or x.shape[1] != len(self.mean)
            or not len(x)
            or not np.isfinite(x).all()
        ):
            raise ValueError("finite target feature matrix required")
        means = x @ self.mean
        variances = self.noise_variance * np.sum(
            x * np.linalg.solve(self.precision, x.T).T, axis=1
        )
        if (
            not np.isfinite(means).all()
            or not np.isfinite(variances).all()
            or np.any(variances < 0)
        ):
            raise ValueError("target moments out of numerical range")
        return means, variances

    @property
    def noise_variance(self):
        return self.scale / (self.shape - 1)
