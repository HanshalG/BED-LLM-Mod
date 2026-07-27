#!/usr/bin/env python3
"""Final integer-weight serving gate for executable dark-matter supports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_dark_matter_executable_support import (
    DISCOVERPHYSICS_COMMIT,
    EXPECTED_REQUESTS,
    GEOMETRIES,
    MAX_REGION_MASS_L1_ERROR,
    MIN_COMPILED_MAP_RMS,
    MODEL_ID,
    NUM_HYPOTHESES,
    PROJECTED_COST_USD,
    REGIONS,
    REGION_PRIOR,
    RUN_BUDGET_USD,
    _adapter,
    parse_support,
    support_diagnostics,
)
from scripts.discoverphysics_dark_matter_opportunity import (
    verify_discoverphysics,
)
from scripts.discoverphysics_oscillator_belief_smoke import (
    SmokeExecutionError,
    checkpoint,
    sha256_file,
    strict_json_object,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-executable-support-2"
SEED = 24508
MAX_TOKENS = 3500


def parse_weighted_support(response: str) -> list[dict[str, Any]]:
    value = strict_json_object(response, label="weighted executable support")
    if set(value) != {"hypotheses"}:
        raise ValueError("weighted executable support has the wrong fields")
    hypotheses = value["hypotheses"]
    if not isinstance(hypotheses, list) or len(hypotheses) != NUM_HYPOTHESES:
        raise ValueError("support must contain exactly eight hypotheses")
    weights = []
    converted = []
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, dict):
            raise ValueError(f"hypotheses[{index}] must be an object")
        if "probability" in hypothesis or "weight" not in hypothesis:
            raise ValueError(f"hypotheses[{index}] must use weight only")
        weight = hypothesis["weight"]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, int)
            or not 1 <= weight <= 100
        ):
            raise ValueError(
                f"hypotheses[{index}].weight must be an integer in [1,100]"
            )
        weights.append(weight)
        converted.append(
            {
                key: item
                for key, item in hypothesis.items()
                if key != "weight"
            }
        )
    total = sum(weights)
    for hypothesis, weight in zip(converted, weights, strict=True):
        hypothesis["probability"] = weight / total
    return parse_support(
        json.dumps({"hypotheses": converted}, separators=(",", ":"))
    )


def support_messages_v2() -> list[dict[str, str]]:
    schema = {
        "hypotheses": [
            {
                "description": "free-form semantic hidden-halo map",
                "weight": 40,
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
                    "geometry types, and assign at least two hypotheses to NE.",
                    "Use a positive integer weight from 1 to 100 for every",
                    "hypothesis. Weights need not sum to any particular value;",
                    "exact code will normalize them. Their relative totals by",
                    "region should approximate the disclosed region prior.",
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
            [support_messages_v2()],
            temperature=0.0,
            block_size=1,
            max_new_tokens=MAX_TOKENS,
        )[0]
        checkpoint(raw_path, {"response": response})
        hypotheses = parse_weighted_support(response)
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
            "geometries": GEOMETRIES,
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
