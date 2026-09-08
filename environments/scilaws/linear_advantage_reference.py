"""Independent residual representation: linear risk minus Bayes advantage."""
import math

import numpy as np

from .adaptive_reference import AdaptiveReference, student_log_density
from .linear_risk_interval import terminal_risk_interval


class LinearAdvantageReference(AdaptiveReference):
    def terminal(self, state, action):
        model = self.model
        interval = terminal_risk_interval(model, state, action)
        df, loc, scale2 = self.density_parameters(state, action)
        density = student_log_density(df, loc, scale2)
        logs = model._state(state)
        predictions, slopes = [], []
        for b, features, targets in zip(state.components, model.action_features,
                                        model.target_features, strict=True):
            phi = features[action]
            solve = np.linalg.solve(np.asarray(b.precision), phi)
            predictions.append(targets @ np.asarray(b.mean))
            slopes.append(targets @ (solve/(1+phi @ solve)))
        predictions, slopes = np.asarray(predictions), np.asarray(slopes)
        linear_mean = np.asarray(interval['target_mean'])
        linear_slope = np.asarray(interval['slope'])

        def advantage(y):
            joint = logs + density(y)
            total = np.logaddexp.reduce(joint)
            weights = np.exp(joint-total)
            conditional = weights @ (predictions + slopes*(y-loc)[:, None])
            linear = linear_mean + linear_slope*(y-interval['observation_mean'])
            return np.exp(total) * ((conditional-linear)**2 @ model.target_weights)

        value, error = self.integrate(advantage, **self.coordinates(df, loc, scale2, logs))
        self.max_inner_error = max(self.max_inner_error, error)
        risk = interval['linear_risk']-value
        if not math.isfinite(risk) or risk < 0:
            raise ValueError('invalid linear-advantage risk')
        return risk, error
