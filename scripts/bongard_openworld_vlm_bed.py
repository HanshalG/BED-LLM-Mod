#!/usr/bin/env python3
"""Core multimodal semantic-belief machinery for Bongard-OpenWorld BED."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_source_protocol_audit as source_audit


NUM_HYPOTHESES = 10
NUM_IMAGES = 14
IMAGE_MAX_SIDE = 512
IMAGE_JPEG_QUALITY = 88
IMAGE_DETAIL = "high"
MIN_RULE_LENGTH = 5
MAX_RULE_LENGTH = 240
PROBABILITY_FLOOR = 0.01
PROBABILITY_CEILING = 0.99
HYPOTHESIS_IDS = tuple(f"H{index:02d}" for index in range(1, 11))
LABELS = {True: "positive", False: "negative"}


@dataclass(frozen=True)
class SemanticHypothesis:
    hypothesis_id: str
    rule: str
    prior_weight: float
    positive_probabilities: tuple[float, ...]


@dataclass(frozen=True)
class SemanticBelief:
    image_ids: tuple[str, ...]
    history: tuple[tuple[str, bool], ...]
    hypotheses: tuple[SemanticHypothesis, ...]
    prior_weights: tuple[float, ...]
    posterior_weights: tuple[float, ...]


@dataclass(frozen=True)
class VisualTask:
    task_id: str
    image_ids: tuple[str, ...]
    initial_history: tuple[tuple[str, bool], ...]
    candidate_ids: tuple[str, ...]
    endpoint_ids: tuple[str, ...]
    image_bytes: Mapping[str, bytes]
    actual_labels: Mapping[str, bool]
    hidden_values: tuple[str, ...] = ()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_rule(value: str) -> str:
    return " ".join(value.casefold().split())


def strict_json_object(response: str) -> dict[str, Any]:
    stripped = response.strip()
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"belief response is not exact JSON: {exc}") from exc
    if stripped[end:].strip():
        raise ValueError("belief response has trailing content")
    if not isinstance(value, dict):
        raise ValueError("belief response must be one JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys are {sorted(value)}, expected {sorted(expected)}"
        )


def _validate_history(
    history: Sequence[tuple[str, bool]], image_ids: Sequence[str]
) -> tuple[tuple[str, bool], ...]:
    image_set = set(image_ids)
    observed: dict[str, bool] = {}
    for image_id, label in history:
        if image_id not in image_set:
            raise ValueError(f"history contains unknown image ID {image_id}")
        if not isinstance(label, bool):
            raise ValueError("history labels must be booleans")
        if image_id in observed and observed[image_id] != label:
            raise ValueError(f"history has conflicting labels for {image_id}")
        observed[image_id] = label
    return tuple(sorted(observed.items()))


def normalize_weights(values: Sequence[float]) -> tuple[float, ...]:
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("weights must be finite and positive")
    total = sum(values)
    return tuple(value / total for value in values)


def normalize_log_weights(values: Sequence[float]) -> tuple[float, ...]:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("log weights must be finite and nonempty")
    maximum = max(values)
    return normalize_weights([math.exp(value - maximum) for value in values])


def posterior_weights(
    hypotheses: Sequence[SemanticHypothesis],
    image_ids: Sequence[str],
    history: Sequence[tuple[str, bool]],
    *,
    starting_weights: Sequence[float] | None = None,
) -> tuple[float, ...]:
    if len(hypotheses) != NUM_HYPOTHESES:
        raise ValueError("posterior requires exactly ten hypotheses")
    index_by_id = {image_id: index for index, image_id in enumerate(image_ids)}
    checked_history = _validate_history(history, image_ids)
    initial = normalize_weights(
        starting_weights
        if starting_weights is not None
        else [hypothesis.prior_weight for hypothesis in hypotheses]
    )
    log_weights = [math.log(weight) for weight in initial]
    for hypothesis_index, hypothesis in enumerate(hypotheses):
        for image_id, label in checked_history:
            probability = hypothesis.positive_probabilities[index_by_id[image_id]]
            likelihood = probability if label else 1.0 - probability
            log_weights[hypothesis_index] += math.log(likelihood)
    return normalize_log_weights(log_weights)


def parse_belief_response(
    response: str,
    *,
    image_ids: Sequence[str],
    history: Sequence[tuple[str, bool]],
) -> SemanticBelief:
    image_ids = tuple(image_ids)
    if len(image_ids) != NUM_IMAGES or len(set(image_ids)) != NUM_IMAGES:
        raise ValueError("belief image order must contain fourteen unique IDs")
    value = strict_json_object(response)
    _exact_keys(value, {"hypotheses"}, "belief response")
    rows = value["hypotheses"]
    if not isinstance(rows, list) or len(rows) != NUM_HYPOTHESES:
        raise ValueError("belief response must contain exactly ten hypotheses")

    hypotheses = []
    canonical_rules = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"hypothesis {index + 1} is not an object")
        _exact_keys(
            row,
            {
                "hypothesis_id",
                "rule",
                "prior_weight",
                "positive_probabilities",
            },
            f"hypothesis {index + 1}",
        )
        expected_id = HYPOTHESIS_IDS[index]
        if row["hypothesis_id"] != expected_id:
            raise ValueError(
                f"hypothesis {index + 1} ID must be {expected_id}"
            )
        rule = row["rule"]
        if not isinstance(rule, str):
            raise ValueError("hypothesis rule must be a string")
        rule = " ".join(rule.split())
        if not MIN_RULE_LENGTH <= len(rule) <= MAX_RULE_LENGTH:
            raise ValueError("hypothesis rule has invalid length")
        normalized_rule = canonical_rule(rule)
        if normalized_rule in canonical_rules:
            raise ValueError("hypothesis rules must be unique")
        canonical_rules.add(normalized_rule)

        prior_weight = row["prior_weight"]
        if (
            isinstance(prior_weight, bool)
            or not isinstance(prior_weight, int)
            or not 1 <= prior_weight <= 100
        ):
            raise ValueError("prior weight must be an integer in [1,100]")
        raw_probabilities = row["positive_probabilities"]
        if not isinstance(raw_probabilities, list) or len(raw_probabilities) != NUM_IMAGES:
            raise ValueError("each hypothesis must contain fourteen probabilities")
        if not all(
            not isinstance(probability, bool)
            and isinstance(probability, int)
            and 1 <= probability <= 99
            for probability in raw_probabilities
        ):
            raise ValueError("positive probabilities must be integers in [1,99]")
        hypotheses.append(
            SemanticHypothesis(
                hypothesis_id=expected_id,
                rule=rule,
                prior_weight=float(prior_weight),
                positive_probabilities=tuple(
                    probability / 100.0 for probability in raw_probabilities
                ),
            )
        )

    hypotheses_tuple = tuple(hypotheses)
    checked_history = _validate_history(history, image_ids)
    prior = normalize_weights(
        [hypothesis.prior_weight for hypothesis in hypotheses_tuple]
    )
    posterior = posterior_weights(
        hypotheses_tuple,
        image_ids,
        checked_history,
        starting_weights=prior,
    )
    return SemanticBelief(
        image_ids=image_ids,
        history=checked_history,
        hypotheses=hypotheses_tuple,
        prior_weights=prior,
        posterior_weights=posterior,
    )


def entropy(weights: Sequence[float]) -> float:
    return -sum(weight * math.log(weight) for weight in weights if weight > 0)


def predictive_probability(
    belief: SemanticBelief,
    image_id: str,
    *,
    weights: Sequence[float] | None = None,
) -> float:
    try:
        image_index = belief.image_ids.index(image_id)
    except ValueError as exc:
        raise ValueError(f"unknown image ID {image_id}") from exc
    selected_weights = (
        tuple(weights) if weights is not None else belief.posterior_weights
    )
    if len(selected_weights) != len(belief.hypotheses):
        raise ValueError("weight count does not match hypotheses")
    return sum(
        weight * hypothesis.positive_probabilities[image_index]
        for weight, hypothesis in zip(
            selected_weights, belief.hypotheses, strict=True
        )
    )


def updated_weights_for_label(
    belief: SemanticBelief,
    image_id: str,
    label: bool,
    *,
    weights: Sequence[float] | None = None,
) -> tuple[float, ...]:
    image_index = belief.image_ids.index(image_id)
    selected = tuple(weights) if weights is not None else belief.posterior_weights
    likelihoods = [
        hypothesis.positive_probabilities[image_index]
        if label
        else 1.0 - hypothesis.positive_probabilities[image_index]
        for hypothesis in belief.hypotheses
    ]
    return normalize_weights(
        [weight * likelihood for weight, likelihood in zip(selected, likelihoods, strict=True)]
    )


def expected_information_gain(
    belief: SemanticBelief,
    image_id: str,
    *,
    weights: Sequence[float] | None = None,
) -> float:
    selected = tuple(weights) if weights is not None else belief.posterior_weights
    probability = predictive_probability(belief, image_id, weights=selected)
    positive = updated_weights_for_label(
        belief, image_id, True, weights=selected
    )
    negative = updated_weights_for_label(
        belief, image_id, False, weights=selected
    )
    value = entropy(selected) - (
        probability * entropy(positive)
        + (1.0 - probability) * entropy(negative)
    )
    return max(0.0, value)


def candidate_eigs(
    belief: SemanticBelief,
    candidate_ids: Sequence[str],
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    return {
        image_id: expected_information_gain(belief, image_id, weights=weights)
        for image_id in candidate_ids
    }


def fixed_support_depth_two_scores(
    root: SemanticBelief,
    candidate_ids: Sequence[str],
) -> dict[str, float]:
    candidates = tuple(candidate_ids)
    scores = {}
    for first in candidates:
        first_eig = expected_information_gain(root, first)
        probability = predictive_probability(root, first)
        future = 0.0
        for label, outcome_probability in (
            (True, probability),
            (False, 1.0 - probability),
        ):
            weights = updated_weights_for_label(root, first, label)
            remaining = [candidate for candidate in candidates if candidate != first]
            future += outcome_probability * max(
                expected_information_gain(root, second, weights=weights)
                for second in remaining
            )
        scores[first] = first_eig + future
    return scores


def dynamic_support_depth_two_scores(
    root: SemanticBelief,
    candidate_ids: Sequence[str],
    branches: Mapping[tuple[str, bool], SemanticBelief],
) -> dict[str, float]:
    candidates = tuple(candidate_ids)
    expected_keys = {
        (candidate, label) for candidate in candidates for label in (False, True)
    }
    if set(branches) != expected_keys:
        raise ValueError("dynamic branch map is incomplete or has extra branches")
    scores = {}
    for first in candidates:
        first_eig = expected_information_gain(root, first)
        probability = predictive_probability(root, first)
        future = 0.0
        for label, outcome_probability in (
            (True, probability),
            (False, 1.0 - probability),
        ):
            branch = branches[(first, label)]
            remaining = [candidate for candidate in candidates if candidate != first]
            future += outcome_probability * max(
                expected_information_gain(branch, second)
                for second in remaining
            )
        scores[first] = first_eig + future
    return scores


def select_best(scores: Mapping[str, float]) -> str:
    if not scores or not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("scores must be nonempty and finite")
    return min(scores, key=lambda key: (-scores[key], key))


def prior_history_log_loss(belief: SemanticBelief) -> float:
    if not belief.history:
        return 0.0
    losses = []
    for image_id, label in belief.history:
        probability = predictive_probability(
            belief, image_id, weights=belief.prior_weights
        )
        truth_probability = probability if label else 1.0 - probability
        losses.append(-math.log(truth_probability))
    return sum(losses) / len(losses)


def endpoint_metrics(
    belief: SemanticBelief,
    endpoint_labels: Mapping[str, bool],
) -> dict[str, Any]:
    rows = []
    for image_id in sorted(endpoint_labels):
        label = endpoint_labels[image_id]
        probability = predictive_probability(belief, image_id)
        truth_probability = probability if label else 1.0 - probability
        rows.append(
            {
                "image_id": image_id,
                "positive_probability": probability,
                "truth_probability": truth_probability,
                "brier": (probability - float(label)) ** 2,
                "log_loss": -math.log(truth_probability),
                "correct": (probability >= 0.5) is label,
            }
        )
    return {
        "mean_brier": sum(row["brier"] for row in rows) / len(rows),
        "mean_log_loss": sum(row["log_loss"] for row in rows) / len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "mean_truth_probability": sum(
            row["truth_probability"] for row in rows
        )
        / len(rows),
        "rows": rows,
    }


def belief_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "bongard_semantic_belief",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": NUM_HYPOTHESES,
                        "maxItems": NUM_HYPOTHESES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "hypothesis_id",
                                "rule",
                                "prior_weight",
                                "positive_probabilities",
                            ],
                            "properties": {
                                "hypothesis_id": {
                                    "type": "string",
                                    "enum": list(HYPOTHESIS_IDS),
                                },
                                "rule": {
                                    "type": "string",
                                    "minLength": MIN_RULE_LENGTH,
                                    "maxLength": MAX_RULE_LENGTH,
                                },
                                "prior_weight": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                },
                                "positive_probabilities": {
                                    "type": "array",
                                    "minItems": NUM_IMAGES,
                                    "maxItems": NUM_IMAGES,
                                    "items": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 99,
                                    },
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def normalized_image_bytes(data: bytes) -> bytes:
    with Image.open(BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(
            output,
            format="JPEG",
            quality=IMAGE_JPEG_QUALITY,
            optimize=True,
        )
    return output.getvalue()


def image_data_url(data: bytes) -> str:
    encoded = base64.b64encode(normalized_image_bytes(data)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_belief_messages(
    task: VisualTask,
    history: Sequence[tuple[str, bool]],
) -> list[dict[str, Any]]:
    checked_history = _validate_history(history, task.image_ids)
    request = {
        "task": (
            "Infer the hidden free-form visual concept from labelled images. "
            "Generate ten distinct plausible rules and calibrated image "
            "likelihoods for each rule."
        ),
        "task_id": task.task_id,
        "image_order": list(task.image_ids),
        "observed_labels": [
            {"image_id": image_id, "label": LABELS[label]}
            for image_id, label in checked_history
        ],
        "selectable_image_ids": list(task.candidate_ids),
        "endpoint_image_ids": list(task.endpoint_ids),
        "requirements": [
            "Use only the observed labels; every other label is unknown.",
            "Rules should jointly include broad, narrow, and compositional alternatives.",
            "Each rule must explain the labelled examples and remain visually testable.",
            "Probabilities follow image_order and estimate P(positive | rule, image).",
            "Do not infer labels from image IDs, role, order, or dataset balance.",
            "Return only the strict schema with H01 through H10 in order.",
        ],
    }
    content: list[dict[str, Any]] = [
        {"type": "text", "text": canonical_json(request)}
    ]
    for image_id in task.image_ids:
        content.extend(
            [
                {"type": "text", "text": f"IMAGE {image_id}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url(task.image_bytes[image_id]),
                        "detail": IMAGE_DETAIL,
                    },
                },
            ]
        )
    return [{"role": "user", "content": content}]


def request_text(messages: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        str(item.get("text", ""))
        for message in messages
        for item in message.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    )


def prompt_hidden_state_errors(
    task: VisualTask,
    history: Sequence[tuple[str, bool]],
    messages: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors = []
    text = request_text(messages)
    lowered = text.casefold()
    for value in task.hidden_values:
        if value and value in text:
            errors.append("hidden_source_value")
    if "pos__" in lowered or "neg__" in lowered or "images/" in lowered:
        errors.append("label_bearing_source_path")
    try:
        request = json.loads(
            next(
                item["text"]
                for item in messages[0]["content"]
                if item.get("type") == "text"
            )
        )
    except Exception:
        errors.append("invalid_request_json")
        return sorted(set(errors))
    expected_observed = [
        {"image_id": image_id, "label": LABELS[label]}
        for image_id, label in _validate_history(history, task.image_ids)
    ]
    if request.get("observed_labels") != expected_observed:
        errors.append("observed_label_mismatch")
    if set(request.get("image_order") or []) != set(task.image_ids):
        errors.append("image_order_mismatch")
    if set(request.get("selectable_image_ids") or []) != set(task.candidate_ids):
        errors.append("candidate_mismatch")
    if set(request.get("endpoint_image_ids") or []) != set(task.endpoint_ids):
        errors.append("endpoint_mismatch")
    return sorted(set(errors))


def load_mechanics_tasks() -> list[VisualTask]:
    mechanics, _, _, _ = source_audit.split_validation_rows(
        source_audit.load_rows("val")
    )
    tasks = []
    with ZipFile(source_audit.DATA_ROOT / "images.zip") as archive:
        for row in mechanics:
            layout = source_audit._task_layout(row)
            position_by_opaque = {
                opaque: position
                for position, opaque in layout["opaque_by_position"].items()
            }
            image_ids = tuple(sorted(position_by_opaque))
            actual_labels = {
                image_id: position_by_opaque[image_id] < 7
                for image_id in image_ids
            }
            initial_ids = {
                layout["opaque_by_position"][position]
                for position in layout["initial_positions"]
            }
            candidate_ids = tuple(
                sorted(
                    layout["opaque_by_position"][position]
                    for position in layout["candidate_positions"]
                )
            )
            endpoint_ids = tuple(
                sorted(
                    layout["opaque_by_position"][position]
                    for position in layout["endpoint_positions"]
                )
            )
            tasks.append(
                VisualTask(
                    task_id=layout["task_id"],
                    image_ids=image_ids,
                    initial_history=tuple(
                        sorted(
                            (image_id, actual_labels[image_id])
                            for image_id in initial_ids
                        )
                    ),
                    candidate_ids=candidate_ids,
                    endpoint_ids=endpoint_ids,
                    image_bytes={
                        image_id: archive.read(
                            row["imageFiles"][position_by_opaque[image_id]]
                        )
                        for image_id in image_ids
                    },
                    actual_labels=actual_labels,
                    hidden_values=(
                        row["concept"],
                        row["caption"],
                        *row["imageFiles"],
                    ),
                )
            )
    return sorted(tasks, key=lambda task: task.task_id)


def public_belief_summary(belief: SemanticBelief) -> dict[str, Any]:
    return {
        "history_size": len(belief.history),
        "prior_history_log_loss": prior_history_log_loss(belief),
        "prior_entropy": entropy(belief.prior_weights),
        "posterior_entropy": entropy(belief.posterior_weights),
        "hypotheses": [
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "rule": hypothesis.rule,
                "prior_weight": belief.prior_weights[index],
                "posterior_weight": belief.posterior_weights[index],
                "likelihood_sha256": hashlib.sha256(
                    canonical_json(hypothesis.positive_probabilities).encode()
                ).hexdigest(),
            }
            for index, hypothesis in enumerate(belief.hypotheses)
        ],
    }
