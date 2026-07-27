from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.discoverphysics_dark_matter_executable_support import (
    compile_hypothesis,
    parse_support,
    support_diagnostics,
)


def _hypothesis(
    index: int,
    region: str,
    geometry: str,
    probability: float,
) -> dict:
    signs = {
        "NE": (1, 1),
        "NW": (-1, 1),
        "SW": (-1, -1),
        "SE": (1, -1),
    }[region]
    return {
        "description": f"{region} {geometry} map {index}",
        "probability": probability,
        "region": region,
        "center": [signs[0] * (3.0 + 0.1 * index), signs[1] * 3.5],
        "geometry": geometry,
        "major_spread": 1.0 + 0.05 * index,
        "minor_spread": 0.35,
        "orientation_degrees": float(index * 17),
    }


def _valid_payload() -> dict:
    specifications = [
        ("NE", "compact", 0.20),
        ("NE", "radial", 0.20),
        ("NW", "tangential", 0.15),
        ("NW", "elliptical", 0.15),
        ("SW", "compact", 0.10),
        ("SW", "radial", 0.10),
        ("SE", "tangential", 0.05),
        ("SE", "elliptical", 0.05),
    ]
    return {
        "hypotheses": [
            _hypothesis(index, region, geometry, probability)
            for index, (region, geometry, probability) in enumerate(
                specifications
            )
        ]
    }


def test_parse_and_compile_executable_support():
    support = parse_support(json.dumps(_valid_payload()))
    compiled = [compile_hypothesis(hypothesis) for hypothesis in support]
    diagnostics = support_diagnostics(support)

    assert len(support) == 8
    assert all(source_map.shape == (10, 2) for source_map in compiled)
    assert all(np.all(np.isfinite(source_map)) for source_map in compiled)
    assert diagnostics["region_counts"] == {
        "NE": 2,
        "NW": 2,
        "SW": 2,
        "SE": 2,
    }
    assert np.isclose(diagnostics["region_mass_l1_error"], 0.0)
    assert len(diagnostics["geometries"]) == 4


def test_parser_rejects_region_center_sign_mismatch():
    payload = _valid_payload()
    payload["hypotheses"][0]["center"] = [-3.0, 3.0]

    with pytest.raises(ValueError, match="signs do not match"):
        parse_support(json.dumps(payload))


def test_parser_rejects_invalid_spread_order():
    payload = _valid_payload()
    payload["hypotheses"][0]["minor_spread"] = 1.5
    payload["hypotheses"][0]["major_spread"] = 1.0

    with pytest.raises(ValueError, match="spreads must satisfy"):
        parse_support(json.dumps(payload))
