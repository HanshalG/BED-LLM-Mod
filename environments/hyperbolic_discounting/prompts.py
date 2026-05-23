"""Prompt templates for hyperbolic temporal discounting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from helpers import Config

if TYPE_CHECKING:
    from .runner import HyperbolicBeliefState, HyperbolicObservation, HyperbolicParams


def _format_observations(observations: list[HyperbolicObservation]) -> str:
    if not observations:
        return "No experiments run yet."
    lines = []
    for index, observation in enumerate(observations, start=1):
        design = observation.design
        lines.append(
            f"{index}. iR={design.immediate_reward:g}, dR={design.delayed_reward:g}, "
            f"days={design.days}, signal={observation.value:g}"
        )
    return "\n".join(lines)


def _format_hypotheses(hypotheses: list[HyperbolicParams], probabilities: list[float] | None = None) -> str:
    rows = []
    for index, hypothesis in enumerate(hypotheses):
        suffix = ""
        if probabilities is not None and index < len(probabilities):
            suffix = f", p={probabilities[index]:.4f}"
        rows.append(f"H{index + 1}: k={hypothesis.k:g}, alpha={hypothesis.alpha:g}{suffix}")
    return "\n".join(rows) if rows else "No hypotheses."


def belief_generation_messages(
    observations: list[HyperbolicObservation],
    belief_state: HyperbolicBeliefState | None,
    config: Config,
) -> list[dict[str, str]]:
    context = ""
    if belief_state is not None and belief_state.hypotheses:
        context = (
            "Current belief support:\n"
            + _format_hypotheses(belief_state.hypotheses, belief_state.probabilities)
            + "\n\n"
        )
    system = (
        "You help a scientist study hyperbolic temporal discounting. "
        "Propose diverse candidate parameter pairs (k, alpha) consistent with the experiment history. "
        "Return JSON only: {\"hypotheses\": [{\"k\": number, \"alpha\": number}, ...]}."
    )
    user = (
        f"{context}"
        f"Experiment history:\n{_format_observations(observations)}\n\n"
        f"Return up to {config.htd_num_generated_hypotheses} unique hypotheses."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def candidate_generation_messages(
    belief_state: HyperbolicBeliefState,
    observations: list[HyperbolicObservation],
    config: Config,
) -> list[dict[str, str]]:
    system = (
        "You propose informative experiment designs for hyperbolic temporal discounting. "
        "Each design is (immediate_reward, delayed_reward, delay_days). "
        'Return JSON only: {"designs": [{"iR": number, "dR": number, "days": integer}, ...]}.'
    )
    user = (
        f"Belief support:\n{_format_hypotheses(belief_state.hypotheses, belief_state.probabilities)}\n\n"
        f"History:\n{_format_observations(observations)}\n\n"
        f"Bounds: iR in {config.htd_ir_bounds}, dR in {config.htd_dr_bounds}, "
        f"days in {config.htd_days_bounds}.\n"
        f"Return up to {config.htd_target_num_candidates} diverse designs."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def naive_design_messages(observations: list[HyperbolicObservation], config: Config) -> list[dict[str, str]]:
    system = (
        "Pick one next experiment design for hyperbolic temporal discounting. "
        'Return JSON: {"designs": [{"iR": number, "dR": number, "days": integer}]}'
    )
    user = (
        f"History:\n{_format_observations(observations)}\n\n"
        f"Bounds: iR in {config.htd_ir_bounds}, dR in {config.htd_dr_bounds}, "
        f"days in {config.htd_days_bounds}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def naive_estimate_messages(observations: list[HyperbolicObservation], config: Config) -> list[dict[str, str]]:
    del config
    system = (
        "Estimate the hidden hyperbolic discounting parameters from the experiment history. "
        'Return JSON: {"hypotheses": [{"k": number, "alpha": number}]}'
    )
    user = f"History:\n{_format_observations(observations)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def hyperbolic_posterior_distribution_messages(
    observations: list[HyperbolicObservation],
    hypotheses: list[HyperbolicParams],
    context_probabilities: list[float],
    config: Config,
) -> list[dict[str, str]]:
    labels = [f"H{index + 1}" for index in range(len(hypotheses))]
    label_map = {label: hypothesis for label, hypothesis in zip(labels, hypotheses)}
    support = {
        label: {"k": hypothesis.k, "alpha": hypothesis.alpha}
        for label, hypothesis in label_map.items()
    }
    system = (
        "Assign a probability distribution over the labeled hypotheses given the experiment history. "
        f"Return JSON mapping each label to a probability, e.g. {json.dumps({labels[0]: 0.5}) if labels else '{\"H1\": 1.0}'}."
    )
    user = (
        f"Hypotheses:\n{json.dumps(support)}\n\n"
        f"Previous probabilities: {json.dumps(dict(zip(labels, context_probabilities)))}\n\n"
        f"History:\n{_format_observations(observations)}\n\n"
        f"Use labels: {labels}. Temperature context: belief calls={config.belief_distribution_num_calls}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
