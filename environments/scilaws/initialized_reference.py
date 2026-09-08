"""Explicit corrected reference after shared initialization, no source access."""

import numpy as np

from .horizon_control_variate import HorizonControlVariateMixture
from .reference_prior import initialize


def initialize_corrected(design, observations, *, quadrature_order):
    base, posterior, scale = initialize(design, observations, quadrature_order=quadrature_order)
    model = HorizonControlVariateMixture(
        base.action_features, base.target_features, posterior.components,
        np.exp(posterior.log_weights), target_weights=base.target_weights,
        quadrature_order=quadrature_order,
        include_observation_noise=base.include_observation_noise,
    )
    # Even callers using model.initial_state start from the shared-data posterior.
    return model, model.initial_state, scale
