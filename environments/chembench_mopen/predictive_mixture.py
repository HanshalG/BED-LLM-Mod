"""Freeze adaptive predictor pools into one ordinary-horizon planning model.

Group weights are predictive-pipeline weights, not physical mechanism priors.
Each pool supplies its own normalized, current-history particle distribution.
The caller guarantees shared action/outcome/target semantics, not just shapes.
No proposals, refitting, or prequential learner mutation occurs inside a plan.
"""

import numpy as np

from .horizon import FiniteBeliefModel


class PredictivePoolMixture(FiniteBeliefModel):
    """Preserve group mass while conditioning particles along imagined branches.

    Construct a fresh snapshot after a REAL update. Do not overwrite a live
    prequential learner with imagined group posteriors or score a new pool against
    the observation that was used to generate it.
    """

    def __init__(self, pools, group_weights):
        if type(pools) is not dict or not pools or len(pools) > 32:
            raise ValueError("1 to 32 named predictor pools required")
        if any(type(key) is not str or not key for key in pools):
            raise ValueError("nonempty string pool IDs required")
        if type(group_weights) is not dict or set(group_weights) != set(pools):
            raise ValueError("exact pool weight keys required")
        self.group_ids = tuple(pools)
        models = tuple(pools.values())
        if any(not isinstance(m, FiniteBeliefModel) for m in models):
            raise ValueError("finite predictive models required")
        weights = np.asarray([group_weights[k] for k in self.group_ids], dtype=float)
        self._validate_weights(weights, len(models))
        first = models[0]
        for model in models[1:]:
            if (
                model.likelihoods.shape[1:] != first.likelihoods.shape[1:]
                or model.targets.shape[1] != first.targets.shape[1]
                or not np.array_equal(model.target_weights, first.target_weights)
            ):
                raise ValueError("pools must share measurement axes and target measure")
        offset = 0
        slices = []
        for model in models:
            slices.append(slice(offset, offset + model.num_particles))
            offset += model.num_particles
        self._group_slices = tuple(slices)
        super().__init__(
            np.concatenate([m.likelihoods for m in models]),
            np.concatenate([m.targets for m in models]),
            np.concatenate(
                [w * np.asarray(m.initial_state) for w, m in zip(weights, models)]
            ),
            target_weights=first.target_weights,
        )

    def group_masses(self, state):
        weights = self._weights(state)
        return {
            key: float(weights[section].sum())
            for key, section in zip(self.group_ids, self._group_slices)
        }

    def group_forecasts(self, state, action):
        """Next-observation probabilities, before collecting its real outcome.

        An eliminated group returns None, not a fabricated conditional forecast.
        The caller must handle that state explicitly in its fixed-expert pipeline.
        """
        action = self._action(action)
        weights = self._weights(state)
        result = {}
        for key, section in zip(self.group_ids, self._group_slices):
            mass = float(weights[section].sum())
            result[key] = (
                None
                if mass == 0
                else tuple(
                    float(p)
                    for p in weights[section]
                    @ self.likelihoods[section, action, :]
                    / mass
                )
            )
        return result
