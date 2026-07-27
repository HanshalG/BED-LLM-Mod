#!/usr/bin/env python3
"""Serve and validate executable LLM-generated dark-matter hypotheses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.discoverphysics_dark_matter_opportunity import (
    verify_discoverphysics,
)
from scripts.discoverphysics_dark_matter_semantic_smoke import (
    NonReasoningOpenRouterAdapter,
)
from scripts.discoverphysics_oscillator_belief_smoke import (
    SmokeExecutionError,
    canonical_text,
    checkpoint,
    probabilities,
    sha256_file,
    strict_json_object,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-executable-support-1"
DISCOVERPHYSICS_COMMIT = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
MODEL_ID = "openai/gpt-5.4"
SEED = 24507
NUM_HYPOTHESES = 8
EXPECTED_REQUESTS = 1
MAX_TOKENS = 3500
RUN_BUDGET_USD = 0.05
PROJECTED_COST_USD = 0.025
REGIONS = ("NE", "NW", "SW", "SE")
REGION_PRIOR = {"NE": 0.40, "NW": 0.30, "SW": 0.20, "SE": 0.10}
GEOMETRIES = ("compact", "radial", "tangential", "elliptical")
MIN_COMPILED_MAP_RMS = 0.20
MAX_REGION_MASS_L1_ERROR = 0.30


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _center(value: Any, *, region: str, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be a two-number list")
    center = [
        _number(value[0], label=f"{label}[0]"),
        _number(value[1], label=f"{label}[1]"),
    ]
    if any(abs(coordinate) < 0.5 or abs(coordinate) > 7.0 for coordinate in center):
        raise ValueError(f"{label} coordinates must have magnitude in [.5,7]")
    expected_signs = {
        "NE": (1, 1),
        "NW": (-1, 1),
        "SW": (-1, -1),
        "SE": (1, -1),
    }[region]
    signs = tuple(1 if coordinate > 0 else -1 for coordinate in center)
    if signs != expected_signs:
        raise ValueError(f"{label} signs do not match region {region}")
    return center


def parse_support(response: str) -> list[dict[str, Any]]:
    value = strict_json_object(response, label="executable support")
    if set(value) != {"hypotheses"}:
        raise ValueError("executable support has the wrong fields")
    hypotheses = value["hypotheses"]
    if not isinstance(hypotheses, list) or len(hypotheses) != NUM_HYPOTHESES:
        raise ValueError("support must contain exactly eight hypotheses")
    parsed = []
    masses = []
    descriptions = set()
    for index, hypothesis in enumerate(hypotheses):
        label = f"hypotheses[{index}]"
        expected_fields = {
            "description",
            "probability",
            "region",
            "center",
            "geometry",
            "major_spread",
            "minor_spread",
            "orientation_degrees",
        }
        if not isinstance(hypothesis, dict) or set(hypothesis) != expected_fields:
            raise ValueError(f"{label} has the wrong fields")
        description = hypothesis["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{label}.description is empty")
        description = description.strip()
        normalized_description = canonical_text(description)
        if normalized_description in descriptions:
            raise ValueError(f"{label}.description is duplicated")
        descriptions.add(normalized_description)
        region = hypothesis["region"]
        if region not in REGIONS:
            raise ValueError(f"{label}.region is invalid")
        geometry = hypothesis["geometry"]
        if geometry not in GEOMETRIES:
            raise ValueError(f"{label}.geometry is invalid")
        major = _number(
            hypothesis["major_spread"],
            label=f"{label}.major_spread",
        )
        minor = _number(
            hypothesis["minor_spread"],
            label=f"{label}.minor_spread",
        )
        if not 0.15 <= minor <= major <= 2.5:
            raise ValueError(
                f"{label} spreads must satisfy .15 <= minor <= major <= 2.5"
            )
        orientation = _number(
            hypothesis["orientation_degrees"],
            label=f"{label}.orientation_degrees",
        )
        if not 0.0 <= orientation < 180.0:
            raise ValueError(f"{label}.orientation_degrees is out of range")
        masses.append(hypothesis["probability"])
        parsed.append(
            {
                "description": description,
                "probability": 0.0,
                "region": region,
                "center": _center(
                    hypothesis["center"],
                    region=region,
                    label=f"{label}.center",
                ),
                "geometry": geometry,
                "major_spread": major,
                "minor_spread": minor,
                "orientation_degrees": orientation,
            }
        )
    normalized_masses = probabilities(
        masses,
        count=NUM_HYPOTHESES,
        label="hypothesis probabilities",
    )
    for hypothesis, probability in zip(parsed, normalized_masses, strict=True):
        hypothesis["probability"] = probability
    compiled = [compile_hypothesis(hypothesis) for hypothesis in parsed]
    compiled_keys = {
        tuple(np.round(source_map.reshape(-1), 6)) for source_map in compiled
    }
    if len(compiled_keys) != NUM_HYPOTHESES:
        raise ValueError("support compiles to duplicate source maps")
    return parsed


def compile_hypothesis(hypothesis: dict[str, Any]) -> np.ndarray:
    center = np.asarray(hypothesis["center"], dtype=float)
    major = float(hypothesis["major_spread"])
    minor = float(hypothesis["minor_spread"])
    geometry = hypothesis["geometry"]
    if geometry == "radial":
        angle = math.atan2(center[1], center[0])
    elif geometry == "tangential":
        angle = math.atan2(center[1], center[0]) + math.pi / 2
    else:
        angle = math.radians(float(hypothesis["orientation_degrees"]))
    major_axis = np.array([math.cos(angle), math.sin(angle)])
    minor_axis = np.array([-major_axis[1], major_axis[0]])

    if geometry in {"radial", "tangential"}:
        coordinates = np.linspace(-1.0, 1.0, 10)
        offsets = (
            major * np.outer(coordinates, major_axis)
            + 0.35
            * minor
            * np.outer(np.sin(np.arange(10) * 1.7), minor_axis)
        )
    else:
        phases = np.linspace(0.0, 2.0 * math.pi, 10, endpoint=False)
        effective_minor = (
            0.5 * (major + minor) if geometry == "compact" else minor
        )
        effective_major = (
            0.5 * (major + minor) if geometry == "compact" else major
        )
        offsets = (
            effective_major * np.outer(np.cos(phases), major_axis)
            + effective_minor * np.outer(np.sin(phases), minor_axis)
        )
    source_map = center + offsets
    if source_map.shape != (10, 2) or not np.all(np.isfinite(source_map)):
        raise ValueError("compiled source map is invalid")
    if np.max(np.abs(source_map)) > 9.5:
        raise ValueError("compiled source map leaves the supported arena")
    return source_map


def support_diagnostics(
    hypotheses: list[dict[str, Any]],
) -> dict[str, Any]:
    region_counts = {
        region: sum(hypothesis["region"] == region for hypothesis in hypotheses)
        for region in REGIONS
    }
    region_masses = {
        region: sum(
            hypothesis["probability"]
            for hypothesis in hypotheses
            if hypothesis["region"] == region
        )
        for region in REGIONS
    }
    geometries = sorted(
        {hypothesis["geometry"] for hypothesis in hypotheses}
    )
    compiled = [compile_hypothesis(hypothesis) for hypothesis in hypotheses]
    pairwise_rms = [
        float(np.sqrt(np.mean((compiled[left] - compiled[right]) ** 2)))
        for left in range(len(compiled))
        for right in range(left + 1, len(compiled))
    ]
    return {
        "region_counts": region_counts,
        "region_masses": region_masses,
        "region_mass_l1_error": sum(
            abs(region_masses[region] - REGION_PRIOR[region])
            for region in REGIONS
        ),
        "geometries": geometries,
        "minimum_compiled_map_rms": min(pairwise_rms),
        "maximum_absolute_source_coordinate": float(
            max(np.max(np.abs(source_map)) for source_map in compiled)
        ),
    }


def support_messages() -> list[dict[str, str]]:
    schema = {
        "hypotheses": [
            {
                "description": "free-form semantic hidden-halo map",
                "probability": 0.125,
                "region": "NE|NW|SW|SE",
                "center": [3.5, 3.5],
                "geometry": "compact|radial|tangential|elliptical",
                "major_spread": 1.2,
                "minor_spread": 0.4,
                "orientation_degrees": 45.0,
            }
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "You generate executable scientific hypotheses. Return one "
                "exact JSON object only, without markdown, comments, "
                "reasoning, or extra fields."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    "An unknown 2D field contains ten concealed positive",
                    "sources forming one compact or elongated halo.",
                    "Region prior: NE .40, NW .30, SW .20, SE .10.",
                    "Within a region, center, spread, and orientation are",
                    "unknown. Coordinates use +x east and +y north.",
                    "",
                    "Generate exactly eight distinct, plausible halo-map",
                    "hypotheses. Cover every region, use at least three",
                    "geometry types, assign at least two hypotheses to NE,",
                    "and make summed region probabilities approximately",
                    "match the disclosed prior.",
                    "Every center coordinate must have magnitude .5 to 7 and",
                    "signs matching region. Spreads must satisfy",
                    ".15 <= minor_spread <= major_spread <= 2.5.",
                    "orientation_degrees must be in [0,180).",
                    "The deterministic compiler interprets radial/tangential",
                    "relative to the center direction, compact as near-round,",
                    "and elliptical using orientation_degrees.",
                    "Do not assume access to an enumerated map bank.",
                    "Return exactly this structure, expanding the list:",
                    json.dumps(schema, separators=(",", ":")),
                ]
            ),
        },
    ]


def _adapter(
    *,
    run_id: str,
    output_dir: Path,
) -> NonReasoningOpenRouterAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=140.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=1,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    spec = ModelSpec(
        model=MODEL_ID,
        backend="openrouter",
        max_model_len=65536,
    )
    return NonReasoningOpenRouterAdapter(spec, config)


def run_serving_gate(
    *,
    discoverphysics_root: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    commit = verify_discoverphysics(discoverphysics_root)
    if commit != DISCOVERPHYSICS_COMMIT:
        raise ValueError("DiscoverPhysics commit changed after verification")
    adapter = _adapter(run_id=run_id, output_dir=output_dir)
    raw_path = output_dir / "RAW_RESPONSE.json"
    try:
        response = adapter.chat_complete_messages_batched(
            [support_messages()],
            temperature=0.0,
            block_size=1,
            max_new_tokens=MAX_TOKENS,
        )[0]
        checkpoint(raw_path, {"response": response})
        hypotheses = parse_support(response)
    except Exception as exc:
        raise SmokeExecutionError(
            f"{type(exc).__name__}: {exc}",
            adapter.usage_snapshot(),
        ) from exc
    diagnostics = support_diagnostics(hypotheses)
    usage = adapter.usage_snapshot()
    gates = {
        "exact_one_request": usage["adapter_requests"] == EXPECTED_REQUESTS,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "all_four_regions_present": all(
            diagnostics["region_counts"][region] >= 1 for region in REGIONS
        ),
        "at_least_two_northeast_hypotheses": (
            diagnostics["region_counts"]["NE"] >= 2
        ),
        "at_least_three_geometry_types": (
            len(diagnostics["geometries"]) >= 3
        ),
        "region_mass_l1_error_at_most_0_30": (
            diagnostics["region_mass_l1_error"]
            <= MAX_REGION_MASS_L1_ERROR
        ),
        "minimum_compiled_map_rms_at_least_0_20": (
            diagnostics["minimum_compiled_map_rms"]
            >= MIN_COMPILED_MAP_RMS
        ),
        "compiled_maps_stay_in_arena": (
            diagnostics["maximum_absolute_source_coordinate"] <= 9.5
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all(gates.values()) else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "discoverphysics_commit": commit,
            "model": MODEL_ID,
            "reasoning_enabled": False,
            "seed": SEED,
            "num_hypotheses": NUM_HYPOTHESES,
            "expected_requests": EXPECTED_REQUESTS,
            "run_budget_usd": RUN_BUDGET_USD,
            "projected_cost_usd": PROJECTED_COST_USD,
            "region_prior": REGION_PRIOR,
            "minimum_compiled_map_rms": MIN_COMPILED_MAP_RMS,
            "maximum_region_mass_l1_error": MAX_REGION_MASS_L1_ERROR,
        },
        "hypotheses": hypotheses,
        "diagnostics": diagnostics,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "simulator_calls": 0,
        "policy_endpoint_exists": False,
        "usage": usage,
        "raw_response_sha256": sha256_file(raw_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discoverphysics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "SERVING.json"
    failure_path = args.output_dir / "FAILURE.json"
    try:
        payload = run_serving_gate(
            discoverphysics_root=args.discoverphysics_root.resolve(),
            output_dir=args.output_dir.resolve(),
            run_id=args.run_id,
        )
    except SmokeExecutionError as exc:
        checkpoint(
            failure_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed",
                "interface_version": INTERFACE_VERSION,
                "error": str(exc),
                "usage": exc.usage,
                "simulator_calls": 0,
                "policy_endpoint_exists": False,
            },
        )
        raise
    checkpoint(output_path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "diagnostics": payload["diagnostics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
