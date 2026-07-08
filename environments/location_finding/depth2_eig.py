"""Depth-2 location EIG with batched LLM branch belief and candidate generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from core import BeliefState
from .beliefs import _merge_hypotheses, build_location_posteriors_many
from .eig import expected_information_gain
from .generation import _generate_location_hypotheses_many, generate_location_candidates_many
from .physics import signal_intensities_for_hypotheses
from .types import Location, LocationObservation

if TYPE_CHECKING:
    from helpers import Config
    from model import Model


def score_location_candidates_depth2_batched(
    belief_state: BeliefState,
    candidates: list[Location],
    config: "Config",
    questioner: "Model",
    observations: list[LocationObservation],
    *,
    immediate_scores: Sequence[float] | None = None,
) -> list[float]:
    """Depth-2 forward search with batched hypothesis refresh and candidate generation."""
    if not candidates:
        return []

    noise_sd = config.location_noise_sd
    quadrature_order = config.location_eig_quadrature_order
    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    hypotheses = belief_state.hypotheses

    if immediate_scores is not None:
        if len(immediate_scores) != len(candidates):
            raise ValueError("immediate_scores length must match candidates")
        totals = [float(score) for score in immediate_scores]
    else:
        totals = [
            expected_information_gain(belief_state, candidate, noise_sd, quadrature_order, config=config)
            for candidate in candidates
        ]

    branch_specs: list[tuple[int, float, LocationObservation]] = []
    for candidate_idx, candidate in enumerate(candidates):
        means = signal_intensities_for_hypotheses(np.asarray(list(hypotheses), dtype=float), candidate, config=config)
        for mean, hypothesis_probability in zip(means, probabilities):
            if hypothesis_probability == 0.0:
                continue
            branch_specs.append(
                (
                    candidate_idx,
                    float(hypothesis_probability),
                    LocationObservation(query=candidate, value=float(mean)),
                )
            )

    if not branch_specs:
        return totals

    observations_many = [list(observations) + [branch_obs] for _candidate_idx, _prob, branch_obs in branch_specs]
    belief_states_many = [belief_state] * len(branch_specs)
    generated_many = _generate_location_hypotheses_many(
        questioner,
        observations_many,
        belief_states_many,
        config,
        label="depth-2 branch belief generation",
    )
    merged_hypotheses_many = [
        _merge_hypotheses(belief_state, generated) for generated in generated_many
    ]
    future_states = build_location_posteriors_many(
        questioner,
        merged_hypotheses_many,
        observations_many,
        config,
        context_states=[belief_state] * len(branch_specs),
        label="depth-2 branch posterior scoring",
    )
    future_candidates_many = generate_location_candidates_many(
        questioner,
        future_states,
        observations_many,
        config,
    )

    for (candidate_idx, hypothesis_probability, _branch_obs), future_state, future_candidates in zip(
        branch_specs,
        future_states,
        future_candidates_many,
    ):
        if not future_candidates or len(future_state.hypotheses) <= 1:
            continue
        future_scores = [
            expected_information_gain(future_state, future_candidate, noise_sd, quadrature_order, config=config)
            for future_candidate in future_candidates
        ]
        if future_scores:
            totals[candidate_idx] += hypothesis_probability * max(future_scores)

    return totals
