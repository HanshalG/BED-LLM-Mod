#!/usr/bin/env python3
"""Provider-constrained LLM support smoke for adaptive force-law BED."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy import special

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, load_config
from scripts.discoverphysics_dark_matter_semantic_smoke import (
    NonReasoningOpenRouterAdapter,
)
from scripts.discoverphysics_extra_dimensions_confirmation import (
    FROZEN_CANDIDATE,
)
from scripts.discoverphysics_extra_dimensions_opportunity import (
    OBSERVATION_NOISE_STD,
    evaluate_scalar_policy,
)


INTERFACE_VERSION = "discoverphysics-extra-dimensions-llm-support-smoke-1"
MODEL_ID = "openai/gpt-5.4"
NUM_HYPOTHESES = 12
NUM_BRANCHES = 2
EXPECTED_REQUESTS = 10
MAX_NEW_TOKENS = 2600
RUN_BUDGET_USD = 0.75
PROJECTED_COST_USD = 0.18
REFRESH_MASS = 0.05
DT = 0.005
DURATION = 1.0
SOFTENING = 0.05
HELDOUT_RADII = np.geomspace(0.3, 9.0, 96)

PARAMETER_BOUNDS = {
    "log_amplitude": (-6.0, -0.5),
    "long_exponent": (0.5, 2.5),
    "short_exponent": (0.5, 3.0),
    "transition_radius": (0.2, 8.0),
    "transition_width": (0.1, 1.5),
    "screening_rate": (0.0, 0.5),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def response_format() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "label": {"type": "string", "minLength": 1, "maxLength": 80},
        "weight": {"type": "integer", "minimum": 1, "maximum": 100},
    }
    for name, (minimum, maximum) in PARAMETER_BOUNDS.items():
        properties[name] = {
            "type": "number",
            "minimum": minimum,
            "maximum": maximum,
        }
    hypothesis = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "radial_force_support",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "items": hypothesis,
                        "minItems": NUM_HYPOTHESES,
                        "maxItems": NUM_HYPOTHESES,
                    }
                },
                "required": ["hypotheses"],
                "additionalProperties": False,
            },
        },
    }


def parse_support(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {"hypotheses"}:
        raise ValueError("support response has unexpected fields")
    hypotheses = payload["hypotheses"]
    if not isinstance(hypotheses, list) or len(hypotheses) != NUM_HYPOTHESES:
        raise ValueError("support must contain exactly 12 hypotheses")
    expected = {"label", "weight", *PARAMETER_BOUNDS}
    parsed: list[dict[str, Any]] = []
    labels: set[str] = set()
    weights: list[float] = []
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, dict) or set(hypothesis) != expected:
            raise ValueError(f"hypothesis {index} has unexpected fields")
        label = hypothesis["label"].strip()
        if not label or label.casefold() in labels:
            raise ValueError("hypothesis labels must be nonempty and unique")
        labels.add(label.casefold())
        weight = hypothesis["weight"]
        if isinstance(weight, bool) or not isinstance(weight, int):
            raise ValueError("weights must be integers")
        item: dict[str, Any] = {"label": label, "weight": float(weight)}
        for name, bounds in PARAMETER_BOUNDS.items():
            value = hypothesis[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            value = float(value)
            if not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"{name} is outside its frozen bounds")
            item[name] = value
        parsed.append(item)
        weights.append(float(weight))
    total = sum(weights)
    for item in parsed:
        item["probability"] = item.pop("weight") / total
    signatures = force_features(parsed)
    if len({tuple(np.round(row, 7)) for row in signatures}) < 10:
        raise ValueError("support compiles to fewer than ten distinct curves")
    transformed_means(parsed)
    return parsed


def log_force(hypothesis: dict[str, Any], radii: np.ndarray) -> np.ndarray:
    radii = np.asarray(radii, dtype=float)
    log_radius = np.log(radii)
    transition_coordinate = (
        np.log(radii / hypothesis["transition_radius"])
        / hypothesis["transition_width"]
    )
    exponent_change = (
        hypothesis["short_exponent"] - hypothesis["long_exponent"]
    )
    return (
        hypothesis["log_amplitude"]
        - hypothesis["long_exponent"] * log_radius
        + exponent_change
        * hypothesis["transition_width"]
        * np.logaddexp(0.0, -transition_coordinate)
        - hypothesis["screening_rate"] * radii
    )


def force_values(
    hypothesis: dict[str, Any],
    radii: np.ndarray,
) -> np.ndarray:
    return np.exp(np.clip(log_force(hypothesis, radii), -40.0, 20.0))


def force_features(hypotheses: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [log_force(hypothesis, HELDOUT_RADII) for hypothesis in hypotheses]
    )


def transformed_means(hypotheses: list[dict[str, Any]]) -> np.ndarray:
    action_radii = np.asarray(FROZEN_CANDIDATE.action_radii)
    num_actions = len(action_radii)
    num_hypotheses = len(hypotheses)
    positions = np.zeros((num_actions, num_hypotheses, 2))
    positions[:, :, 0] = action_radii[:, None]
    velocities = np.zeros_like(positions)
    yoshida_weight = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    drift = (
        0.5 * yoshida_weight,
        0.5 * (1.0 - yoshida_weight),
        0.5 * (1.0 - yoshida_weight),
        0.5 * yoshida_weight,
    )
    kick = (
        yoshida_weight,
        1.0 - 2.0 * yoshida_weight,
        yoshida_weight,
    )

    def acceleration(current_positions: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(current_positions, axis=-1)
        effective = np.sqrt(distances**2 + SOFTENING**2)
        magnitudes = np.empty_like(distances)
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            magnitudes[:, hypothesis_index] = force_values(
                hypothesis,
                effective[:, hypothesis_index],
            )
        return (
            -magnitudes[:, :, None]
            * current_positions
            / np.maximum(distances[:, :, None], 1e-12)
        )

    for _ in range(int(round(DURATION / DT))):
        for coefficient_index, kick_coefficient in enumerate(kick):
            positions += drift[coefficient_index] * DT * velocities
            velocities += (
                kick_coefficient * DT * acceleration(positions)
            )
        positions += drift[-1] * DT * velocities
    displacements = action_radii[:, None] - positions[:, :, 0]
    if np.any(~np.isfinite(displacements)) or np.any(displacements <= 0.0):
        raise ValueError("support produces a nonpositive or invalid displacement")
    return FROZEN_CANDIDATE.measurement_scale * np.log(displacements)


def support_probabilities(hypotheses: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [hypothesis["probability"] for hypothesis in hypotheses],
        dtype=float,
    )


def formula_prompt() -> str:
    return (
        "Each hypothesis defines a positive attractive radial force. Exact code "
        "uses log F(r) = log_amplitude - long_exponent*log(r) + "
        "(short_exponent-long_exponent)*transition_width*"
        "softplus(-log(r/transition_radius)/transition_width) - "
        "screening_rate*r. Equal short and long exponents remove the crossover; "
        "zero screening removes exponential screening. Generate a diverse "
        "Bayesian support spanning simple powers, screened laws, and crossovers. "
        "Weights are positive relative masses and need not sum to a fixed number."
    )


def initial_messages(*, preflight: bool = False) -> list[dict[str, str]]:
    context = (
        "This is a discarded transport preflight for a synthetic empty history."
        if preflight
        else (
            "No measurements have been made. The available experiment radii are "
            "2.4, 3.6, 5.5, and 8.0. A measurement is 0.4 times the log of the "
            "one-second inward radial displacement, with Gaussian noise std 0.075."
        )
    )
    return [
        {
            "role": "system",
            "content": (
                "You are the hypothesis generator inside a sequential Bayesian "
                "experimental-design system. Return only the schema-conforming "
                "support. Do not reason about which action the planner should select."
            ),
        },
        {
            "role": "user",
            "content": f"{context}\n\n{formula_prompt()}",
        },
    ]


def branch_messages(radius: float, observation: float) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You regenerate executable force-law hypotheses after a hypothetical "
                "measurement. Return only the schema-conforming support. Preserve "
                "multiple plausible mechanisms while concentrating on laws compatible "
                "with the complete history."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Complete history: at experiment radius {radius:g}, the observed "
                f"scaled log displacement was {observation:.8f}; observation noise "
                f"standard deviation is {OBSERVATION_NOISE_STD}. Available next radii "
                "are 2.4, 3.6, 5.5, and 8.0.\n\n"
                f"{formula_prompt()}"
            ),
        },
    ]


def mixture_quantile(
    means: np.ndarray,
    probabilities: np.ndarray,
    quantile: float,
) -> float:
    lower = float(np.min(means) - 8.0 * OBSERVATION_NOISE_STD)
    upper = float(np.max(means) + 8.0 * OBSERVATION_NOISE_STD)
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        cdf = float(
            np.sum(
                probabilities
                * special.ndtr(
                    (midpoint - means) / OBSERVATION_NOISE_STD
                )
            )
        )
        if cdf < quantile:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def representative_branches(
    means: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    representatives = np.empty((means.shape[0], NUM_BRANCHES))
    masses = np.empty_like(representatives)
    for action_index in range(means.shape[0]):
        representatives[action_index] = [
            mixture_quantile(means[action_index], probabilities, 0.25),
            mixture_quantile(means[action_index], probabilities, 0.75),
        ]
        boundary = float(np.mean(representatives[action_index]))
        low_mass = float(
            np.sum(
                probabilities
                * special.ndtr(
                    (boundary - means[action_index])
                    / OBSERVATION_NOISE_STD
                )
            )
        )
        masses[action_index] = [low_mass, 1.0 - low_mass]
    return representatives, masses


def posterior_at(
    probabilities: np.ndarray,
    means: np.ndarray,
    observation: float,
) -> np.ndarray:
    logits = np.log(probabilities) - 0.5 * (
        (observation - means) / OBSERVATION_NOISE_STD
    ) ** 2
    logits -= np.max(logits)
    posterior = np.exp(logits)
    return posterior / posterior.sum()


def dynamic_scores(
    initial: list[dict[str, Any]],
    refreshes: list[list[list[dict[str, Any]]]],
    representatives: np.ndarray,
    branch_masses: np.ndarray,
) -> dict[str, Any]:
    initial_means = transformed_means(initial)
    initial_probabilities = support_probabilities(initial)
    initial_values = evaluate_scalar_policy(
        initial_means,
        initial_probabilities,
        quadrature_points=16,
        heldout_features=force_features(initial),
    )
    dynamic = initial_values["immediate_eig_nats"].copy()
    branch_best_eig = np.empty((len(FROZEN_CANDIDATE.action_radii), 2))
    for root_index in range(len(FROZEN_CANDIDATE.action_radii)):
        for branch_index in range(NUM_BRANCHES):
            refreshed = refreshes[root_index][branch_index]
            initial_posterior = posterior_at(
                initial_probabilities,
                initial_means[root_index],
                representatives[root_index, branch_index],
            )
            union = [*initial, *refreshed]
            union_probabilities = np.concatenate(
                [
                    (1.0 - REFRESH_MASS) * initial_posterior,
                    REFRESH_MASS * support_probabilities(refreshed),
                ]
            )
            union_values = evaluate_scalar_policy(
                transformed_means(union),
                union_probabilities,
                quadrature_points=12,
            )
            branch_best_eig[root_index, branch_index] = float(
                np.max(union_values["immediate_eig_nats"])
            )
        dynamic[root_index] += float(
            np.sum(
                branch_masses[root_index]
                * branch_best_eig[root_index]
            )
        )
    return {
        "initial": initial_values,
        "fixed_depth_two_scores": initial_values["total_eig_nats"],
        "dynamic_depth_two_scores": dynamic,
        "branch_best_eig": branch_best_eig,
    }


def build_model(config: Config) -> NonReasoningOpenRouterAdapter:
    if len(config.model_pairs) != 1:
        raise ValueError("smoke requires exactly one model pair")
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID or spec.backend != "openrouter":
        raise ValueError("smoke requires openai/gpt-5.4 through OpenRouter")
    return NonReasoningOpenRouterAdapter(spec, config)


def usage_summary(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot["adapter_requests"]),
        "http_attempts": int(snapshot["http_attempts"]),
        "retry_count": int(snapshot["retry_count"]),
        "reasoning_tokens": int(snapshot["adapter_reasoning_tokens"]),
        "forced_exits": int(snapshot["forced_exits"]),
        "cost_usd": float(snapshot["adapter_cost_usd"]),
        "prompt_tokens": int(snapshot["adapter_prompt_tokens"]),
        "completion_tokens": int(snapshot["adapter_completion_tokens"]),
    }


def run_smoke(
    config: Config,
    *,
    raw_path: Path,
    model: Any | None = None,
) -> dict[str, Any]:
    active_model = model or build_model(config)
    raw: dict[str, Any] = {"preflight": None, "initial": None, "branches": []}
    schema = response_format()
    preflight_text = active_model.chat_complete_messages_batched_structured(
        [initial_messages(preflight=True)],
        temperature=0.0,
        block_size=1,
        response_format=schema,
        max_new_tokens=MAX_NEW_TOKENS,
    )[0]
    raw["preflight"] = preflight_text
    parse_support(preflight_text)

    initial_text = active_model.chat_complete_messages_batched_structured(
        [initial_messages()],
        temperature=0.0,
        block_size=1,
        response_format=schema,
        max_new_tokens=MAX_NEW_TOKENS,
    )[0]
    raw["initial"] = initial_text
    initial = parse_support(initial_text)
    initial_means = transformed_means(initial)
    initial_probabilities = support_probabilities(initial)
    representatives, branch_masses = representative_branches(
        initial_means,
        initial_probabilities,
    )
    branch_prompts = [
        branch_messages(radius, representatives[root_index, branch_index])
        for root_index, radius in enumerate(FROZEN_CANDIDATE.action_radii)
        for branch_index in range(NUM_BRANCHES)
    ]
    branch_texts = active_model.chat_complete_messages_batched_structured(
        branch_prompts,
        temperature=0.0,
        block_size=config.batched_block_size,
        response_format=schema,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    raw["branches"] = branch_texts
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    parsed_branches = [parse_support(text) for text in branch_texts]
    refreshes = [
        parsed_branches[index : index + NUM_BRANCHES]
        for index in range(0, len(parsed_branches), NUM_BRANCHES)
    ]
    scores = dynamic_scores(
        initial,
        refreshes,
        representatives,
        branch_masses,
    )
    immediate = scores["initial"]["immediate_eig_nats"]
    fixed = scores["fixed_depth_two_scores"]
    dynamic = scores["dynamic_depth_two_scores"]
    myopic_index = int(np.argmax(immediate))
    fixed_index = int(np.argmax(fixed))
    dynamic_index = int(np.argmax(dynamic))
    sorted_dynamic = np.sort(dynamic)
    usage = usage_summary(active_model)
    signatures = [
        {tuple(np.round(row, 6)) for row in force_features(support)}
        for support in [initial, *parsed_branches]
    ]
    branch_changed = [
        len(signatures[0].symmetric_difference(signature)) > 0
        for signature in signatures[1:]
    ]
    gates = {
        "exact_10_physical_requests": (
            usage["physical_requests"] == EXPECTED_REQUESTS
        ),
        "exact_10_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_transport_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_75": usage["cost_usd"] <= RUN_BUDGET_USD,
        "all_supports_compile": True,
        "all_eight_refreshes_change_support": all(branch_changed),
        "myopic_radius_5_5": (
            FROZEN_CANDIDATE.action_radii[myopic_index] == 5.5
        ),
        "dynamic_depth_two_radius_2_4": (
            FROZEN_CANDIDATE.action_radii[dynamic_index] == 2.4
        ),
        "dynamic_immediate_sacrifice_at_least_0_03": (
            immediate[myopic_index] - immediate[dynamic_index] >= 0.03
        ),
        "dynamic_margin_at_least_0_02": (
            dynamic[dynamic_index] - sorted_dynamic[-2] >= 0.02
        ),
        "refresh_is_root_load_bearing": (
            dynamic_index != fixed_index and dynamic_index == 0
        ),
    }
    return {
        "interface_version": INTERFACE_VERSION,
        "model": MODEL_ID,
        "protocol": {
            "expected_requests": EXPECTED_REQUESTS,
            "reasoning_requested": False,
            "response_format": "chat_strict_json_schema",
            "refresh_mass": REFRESH_MASS,
            "action_radii": list(FROZEN_CANDIDATE.action_radii),
            "scientific_endpoint_evaluated": False,
        },
        "representative_observations": representatives.tolist(),
        "branch_masses": branch_masses.tolist(),
        "scores": {
            "immediate_eig_nats": immediate.tolist(),
            "fixed_depth_two_eig_nats": fixed.tolist(),
            "dynamic_depth_two_score_nats": dynamic.tolist(),
            "branch_best_eig_nats": scores["branch_best_eig"].tolist(),
            "myopic_index": myopic_index,
            "fixed_depth_two_index": fixed_index,
            "dynamic_depth_two_index": dynamic_index,
        },
        "support_hashes": {
            "initial": sha256_text(initial_text),
            "branches": [sha256_text(text) for text in branch_texts],
            "preflight": sha256_text(preflight_text),
        },
        "usage": usage,
        "gates": gates,
        "passed": all(gates.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-raw", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = RUN_BUDGET_USD
    config.log_path = args.output.with_suffix(".log")
    result = run_smoke(config, raw_path=args.private_raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
