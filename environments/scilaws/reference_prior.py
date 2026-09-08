"""Frozen classical feature reference; no LLM or simulator access.

Initial-data response scaling is empirical Bayes. Evidence after that scaling
is conditional fitting evidence, not selection-corrected model evidence.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement
import math

import numpy as np

from .regression_belief import RegressionBelief
from .regression_mixture import RegressionMixture

FAMILIES = ("constant", "affine", "quadratic", "additive_rbf")


@dataclass(frozen=True)
class ResponseScale:
    value: float

    def __post_init__(self):
        if (
            isinstance(self.value, bool)
            or not math.isfinite(self.value)
            or self.value <= 0
        ):
            raise ValueError("positive finite response scale required")

    @classmethod
    def from_initial(cls, observations):
        y = np.asarray(observations, dtype=float)
        if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
            raise ValueError("finite nonempty initial data required")
        maximum = float(np.max(np.abs(y)))
        return cls(maximum if maximum > 0 else 1.0)

    def transform(self, observations):
        y = np.asarray(observations, dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("finite observations required")
        # logaddexp form avoids overflow in y/scale and y^2 for extreme values.
        with np.errstate(divide="ignore"):
            log_ratio = np.log(np.abs(y)) - math.log(self.value)
        transformed = np.sign(y) * np.logaddexp(
            log_ratio, 0.5 * np.logaddexp(2 * log_ratio, 0)
        )
        if not np.isfinite(transformed).all():
            raise ValueError("response transform exceeded numeric range")
        return transformed


def unit_points(points, axes):
    names = [axis["name"] for axis in axes]
    if not 1 <= len(names) <= 3 or len(set(names)) != len(names):
        raise ValueError("invalid axes")
    if not points or any(set(p) != set(names) for p in points):
        raise ValueError("point coordinates must match axes")
    columns = []
    for axis in axes:
        lo, hi = axis["bounds"]
        x = np.asarray([p[axis["name"]] for p in points], dtype=float)
        if not np.isfinite(x).all() or np.any(x < lo) or np.any(x > hi):
            raise ValueError("point outside fixed support")
        if axis["transform"] == "log":
            if lo <= 0:
                raise ValueError("log support must be positive")
            x, lo, hi = np.log(x), math.log(lo), math.log(hi)
        elif axis["transform"] != "linear":
            raise ValueError("unknown transform")
        columns.append(2 * (x - lo) / (hi - lo) - 1)
    return np.column_stack(columns)


def features(points, axes, family):
    x = unit_points(points, axes)
    columns = [np.ones(len(x))]
    if family in ("affine", "quadratic"):
        columns.extend(x.T)
    if family == "quadratic":
        columns.extend(
            x[:, i] * x[:, j]
            for i, j in combinations_with_replacement(range(x.shape[1]), 2)
        )
    if family == "additive_rbf":
        columns.extend(
            np.exp(-0.5 * ((x[:, i] - centre) / 0.5) ** 2)
            for i in range(x.shape[1])
            for centre in (-0.5, 0.5)
        )
    if family not in FAMILIES:
        raise ValueError("unknown reference family")
    return np.column_stack(columns)


def make_model(design, *, quadrature_order):
    action = [features(design["action_points"], design["axes"], f) for f in FAMILIES]
    targets = [features(design["target_points"], design["axes"], f) for f in FAMILIES]
    components = []
    for x in action:
        precision = np.full(x.shape[1], 4.0)
        precision[0] = 0.25
        components.append(
            RegressionBelief(np.zeros(x.shape[1]), np.diag(precision), 3.0, 0.2)
        )
    return RegressionMixture(
        action,
        targets,
        components,
        [0.25] * 4,
        target_weights=design["target_weights"],
        quadrature_order=quadrature_order,
        include_observation_noise=True,
    )


def initialize(design, observations, *, quadrature_order):
    """Point-major initial replicates, shared identically by all policy arms."""
    count = len(design["initial_points"])
    replicates = design["initial_replicates"]
    y = np.asarray(observations, dtype=float)
    if y.shape != (count, replicates) or not np.isfinite(y).all():
        raise ValueError("complete point-major initial replicates required")
    scale = ResponseScale.from_initial(y.ravel())
    z = scale.transform(y)
    model = make_model(design, quadrature_order=quadrature_order)
    matrices = [features(design["initial_points"], design["axes"], f) for f in FAMILIES]
    components = list(model.initial_state.components)
    logs = np.asarray(model.initial_state.log_weights)
    from scipy.special import logsumexp
    from .regression_mixture import MixtureState

    for point in range(count):
        for replicate in range(replicates):
            for i, matrix in enumerate(matrices):
                components[i], density = components[i].condition(
                    matrix[point], z[point, replicate]
                )
                logs[i] += density
            logs -= logsumexp(logs)
    return model, MixtureState(tuple(components), tuple(logs)), scale
