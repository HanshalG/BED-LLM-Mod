#!/usr/bin/env python3
"""Plain-record transport smoke for adaptive force-law BED."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from helpers import Config, load_config
from scripts.discoverphysics_extra_dimensions_llm_support_smoke import (
    EXPECTED_REQUESTS,
    FROZEN_CANDIDATE,
    MAX_NEW_TOKENS,
    MODEL_ID,
    NUM_BRANCHES,
    NUM_HYPOTHESES,
    OBSERVATION_NOISE_STD,
    PARAMETER_BOUNDS,
    PROJECTED_COST_USD,
    REFRESH_MASS,
    RUN_BUDGET_USD,
    build_model,
    dynamic_scores,
    force_features,
    formula_prompt,
    representative_branches,
    support_probabilities,
    transformed_means,
    usage_summary,
)


INTERFACE_VERSION = "discoverphysics-extra-dimensions-llm-support-smoke-2"
RECORD_FIELDS = (
    "record_id",
    "label",
    "weight",
    "log_amplitude",
    "long_exponent",
    "short_exponent",
    "transition_radius",
    "transition_width",
    "screening_rate",
)
LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,79}")
INTEGER_PATTERN = re.compile(r"(?:[1-9][0-9]?|100)")
DECIMAL_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def record_grammar() -> str:
    return "\n".join(
        [
            "Return exactly 12 nonempty lines and nothing else.",
            "Each line has exactly nine pipe-separated fields:",
            "H01|label|weight|log_amplitude|long_exponent|short_exponent|"
            "transition_radius|transition_width|screening_rate",
            "Use record IDs H01 through H12 in order.",
            "Labels use only ASCII letters, digits, spaces, underscores, or hyphens.",
            "Weight is an integer from 1 through 100.",
            "Every parameter is a plain decimal number: no exponent notation, units,"
            " brackets, comments, header, markdown fence, or blank lines.",
            "Parameter bounds: "
            + ", ".join(
                f"{name}=[{bounds[0]},{bounds[1]}]"
                for name, bounds in PARAMETER_BOUNDS.items()
            )
            + ".",
        ]
    )


def parse_support_records(text: str) -> list[dict[str, Any]]:
    if text != text.strip():
        raise ValueError("support response has leading or trailing whitespace")
    lines = text.splitlines()
    if len(lines) != NUM_HYPOTHESES or any(not line for line in lines):
        raise ValueError("support response must contain exactly 12 nonempty lines")

    parsed: list[dict[str, Any]] = []
    labels: set[str] = set()
    weights: list[float] = []
    parameter_names = RECORD_FIELDS[3:]
    for index, line in enumerate(lines):
        fields = line.split("|")
        if len(fields) != len(RECORD_FIELDS):
            raise ValueError(f"support line {index + 1} must have nine fields")
        expected_id = f"H{index + 1:02d}"
        if fields[0] != expected_id:
            raise ValueError(f"support line {index + 1} must use ID {expected_id}")
        label = fields[1]
        if LABEL_PATTERN.fullmatch(label) is None:
            raise ValueError(f"support line {index + 1} has an invalid label")
        normalized_label = label.casefold()
        if normalized_label in labels:
            raise ValueError("hypothesis labels must be unique")
        labels.add(normalized_label)
        if INTEGER_PATTERN.fullmatch(fields[2]) is None:
            raise ValueError("weights must be plain integers in [1,100]")
        weight = float(int(fields[2]))
        item: dict[str, Any] = {"label": label, "weight": weight}
        for name, field in zip(parameter_names, fields[3:], strict=True):
            if DECIMAL_PATTERN.fullmatch(field) is None:
                raise ValueError(f"{name} must be a plain decimal number")
            value = float(field)
            bounds = PARAMETER_BOUNDS[name]
            if not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"{name} is outside its frozen bounds")
            item[name] = value
        parsed.append(item)
        weights.append(weight)

    total = sum(weights)
    for item in parsed:
        item["probability"] = item.pop("weight") / total
    signatures = force_features(parsed)
    if len({tuple(np.round(row, 7)) for row in signatures}) < 10:
        raise ValueError("support compiles to fewer than ten distinct curves")
    transformed_means(parsed)
    return parsed


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
                "experimental-design system. Emit only the requested fixed records. "
                "Do not reason about which action the planner should select."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [context, formula_prompt(), record_grammar()]
            ),
        },
    ]


def branch_messages(radius: float, observation: float) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You regenerate executable force-law hypotheses after a hypothetical "
                "measurement. Emit only the requested fixed records. Preserve "
                "multiple plausible mechanisms while concentrating on laws compatible "
                "with the complete history."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    (
                        f"Complete history: at experiment radius {radius:g}, the "
                        f"observed scaled log displacement was {observation:.8f}; "
                        f"observation noise standard deviation is "
                        f"{OBSERVATION_NOISE_STD}. Available next radii are "
                        "2.4, 3.6, 5.5, and 8.0."
                    ),
                    formula_prompt(),
                    record_grammar(),
                ]
            ),
        },
    ]


def run_smoke(
    config: Config,
    *,
    raw_path: Path,
    model: Any | None = None,
) -> dict[str, Any]:
    active_model = model or build_model(config)
    raw: dict[str, Any] = {"preflight": None, "initial": None, "branches": []}

    preflight_text = active_model.chat_complete_messages_batched(
        [initial_messages(preflight=True)],
        temperature=0.0,
        block_size=1,
        max_new_tokens=MAX_NEW_TOKENS,
    )[0]
    raw["preflight"] = preflight_text
    checkpoint(raw_path, raw)
    parse_support_records(preflight_text)

    initial_text = active_model.chat_complete_messages_batched(
        [initial_messages()],
        temperature=0.0,
        block_size=1,
        max_new_tokens=MAX_NEW_TOKENS,
    )[0]
    raw["initial"] = initial_text
    checkpoint(raw_path, raw)
    initial = parse_support_records(initial_text)
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
    branch_texts = active_model.chat_complete_messages_batched(
        branch_prompts,
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    raw["branches"] = branch_texts
    checkpoint(raw_path, raw)
    parsed_branches = [parse_support_records(text) for text in branch_texts]
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
            "response_format": "chat_fixed_pipe_records",
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
