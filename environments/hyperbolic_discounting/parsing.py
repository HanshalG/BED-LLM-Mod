"""Parse LLM completions for hyperbolic temporal discounting."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from .runner import HyperbolicDesign, HyperbolicParams, normalize_design


def _clean_json_completion(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_first_json_value(text: str) -> str | None:
    cleaned = _clean_json_completion(text)
    if not cleaned:
        return None
    if cleaned[0] in "[{":
        return cleaned
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    return match.group(1) if match else None


def _loads_json_value(text: str) -> Any:
    return json.loads(_extract_first_json_value(text) or text)


def parse_hyperbolic_design(raw: object) -> HyperbolicDesign:
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return normalize_design(raw[0], raw[1], raw[2])
    if isinstance(raw, dict):
        keys = {str(key).lower(): value for key, value in raw.items()}
        for ir_key, dr_key, days_key in (
            ("ir", "dr", "days"),
            ("iR", "dR", "Days"),
            ("immediate_reward", "delayed_reward", "delay_days"),
        ):
            if ir_key in keys and dr_key in keys and days_key in keys:
                return normalize_design(keys[ir_key], keys[dr_key], keys[days_key])
    raise ValueError("design must be [iR, dR, days] or object with reward/delay fields")


def parse_candidate_designs(
    completion: str,
    *,
    ir_bounds: tuple[float, float],
    dr_bounds: tuple[float, float],
    days_bounds: tuple[float, float],
) -> list[HyperbolicDesign]:
    payload = _loads_json_value(completion)
    raw_designs: list[object]
    if isinstance(payload, dict) and "designs" in payload:
        raw_designs = list(payload["designs"])
    elif isinstance(payload, dict) and "locations" in payload:
        raw_designs = list(payload["locations"])
    elif isinstance(payload, list):
        raw_designs = payload
    else:
        raise ValueError("expected JSON list of designs or {\"designs\": [...]}")
    designs = [parse_hyperbolic_design(item) for item in raw_designs]
    clipped: list[HyperbolicDesign] = []
    for design in designs:
        clipped.append(
            normalize_design(
                min(max(design.immediate_reward, ir_bounds[0]), ir_bounds[1]),
                min(max(design.delayed_reward, dr_bounds[0]), dr_bounds[1]),
                int(min(max(design.days, days_bounds[0]), days_bounds[1])),
            )
        )
    return clipped


def parse_hyperbolic_hypothesis(raw: object) -> HyperbolicParams:
    if isinstance(raw, dict):
        keys = {str(key).lower(): value for key, value in raw.items()}
        if "k" in keys and "alpha" in keys:
            k = float(keys["k"])
            alpha = float(keys["alpha"])
            if k <= 0.0 or alpha <= 0.0 or not math.isfinite(k) or not math.isfinite(alpha):
                raise ValueError("k and alpha must be positive finite numbers")
            return HyperbolicParams(k=k, alpha=alpha)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        k = float(raw[0])
        alpha = float(raw[1])
        if k <= 0.0 or alpha <= 0.0:
            raise ValueError("k and alpha must be positive")
        return HyperbolicParams(k=k, alpha=alpha)
    raise ValueError("hypothesis must be {k, alpha} or [k, alpha]")


def parse_hyperbolic_hypotheses(completion: str) -> list[HyperbolicParams]:
    payload = _loads_json_value(completion)
    raw_items: list[object]
    if isinstance(payload, dict) and "hypotheses" in payload:
        raw_items = list(payload["hypotheses"])
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raise ValueError("expected JSON list of hypotheses or {\"hypotheses\": [...]}")
    seen: set[tuple[float, float]] = set()
    hypotheses: list[HyperbolicParams] = []
    for item in raw_items:
        params = parse_hyperbolic_hypothesis(item)
        key = (round(params.k, 8), round(params.alpha, 8))
        if key in seen:
            continue
        seen.add(key)
        hypotheses.append(params)
    return hypotheses
