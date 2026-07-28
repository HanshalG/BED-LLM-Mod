from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from scripts.analyze_discoverphysics_dark_matter_structured_replication import (
    weighted_correlation,
)
from scripts.discoverphysics_dark_matter_balanced_modular_replication import (
    MAP_SEEDS as BALANCED_MAP_SEEDS,
    balance_support_regions,
    balanced_refresh_messages,
    hidden_map_family as balanced_hidden_map_family,
    modular_component_posteriors,
)
from scripts.discoverphysics_dark_matter_executable_support_v2 import (
    parse_weighted_support,
)
from scripts.discoverphysics_dark_matter_grounded_policy import (
    LOOKAHEAD_ROOT_ID,
    MYOPIC_ROOT_ID,
    ROOTS,
)
from scripts.discoverphysics_dark_matter_structured_replication import (
    EXPECTED_REQUESTS,
    REPLICATION_MAP_SEEDS,
    phase_a_mechanics,
    phase_a_selection_gates,
    refresh_response_format,
    replication_hidden_map_family,
    stratified_bootstrap_interval,
    support_response_format,
)
from scripts.discoverphysics_dark_matter_structured_replication_v3 import (
    use_default_structured_routing,
)
from tests.test_discoverphysics_dark_matter_executable_support import (
    _valid_payload,
)


def _parsed_weighted_support() -> list[dict]:
    payload = _valid_payload()
    for hypothesis in payload["hypotheses"]:
        probability = hypothesis.pop("probability")
        hypothesis["weight"] = int(round(probability * 100))
    return parse_weighted_support(json.dumps(payload))


def _fresh_support(
    initial_support: list[dict],
    *,
    root_id: str,
    branch_index: int,
) -> list[dict]:
    support = deepcopy(initial_support)
    for index, hypothesis in enumerate(support):
        hypothesis["description"] = (
            f"{hypothesis['description']} {root_id}{branch_index}-{index}"
        )
    return support


def test_structured_schemas_are_strict_and_closed():
    initial_format = support_response_format()
    initial_schema = initial_format["json_schema"]["schema"]
    hypothesis = initial_schema["properties"]["hypotheses"]

    assert initial_format["type"] == "json_schema"
    assert initial_format["json_schema"]["strict"] is True
    assert initial_schema["additionalProperties"] is False
    assert hypothesis["minItems"] == hypothesis["maxItems"] == 8
    assert hypothesis["items"]["additionalProperties"] is False
    assert hypothesis["items"]["properties"]["weight"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 100,
    }

    refresh_schema = refresh_response_format()["json_schema"]["schema"]
    assert refresh_schema["additionalProperties"] is False
    assert set(refresh_schema["required"]) == {
        "hypotheses",
        "continuation_action",
    }
    assert set(
        refresh_schema["properties"]["continuation_action"]["enum"]
    )


def test_structured_weighted_payload_reuses_semantic_compiler():
    hypotheses = _parsed_weighted_support()

    assert len(hypotheses) == 8
    assert np.isclose(
        sum(hypothesis["probability"] for hypothesis in hypotheses),
        1.0,
    )
    assert all(
        set(hypothesis)
        == {
            "description",
            "probability",
            "region",
            "center",
            "geometry",
            "major_spread",
            "minor_spread",
            "orientation_degrees",
        }
        for hypothesis in hypotheses
    )


def test_v3_changes_only_provider_filter_in_structured_payload():
    payload = {
        "model": "openai/gpt-5.4",
        "messages": [{"role": "user", "content": "test"}],
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 3500,
        "n": 1,
        "reasoning": {"enabled": False, "exclude": True},
        "response_format": support_response_format(),
        "provider": {"require_parameters": True},
    }

    routed = use_default_structured_routing(payload)

    assert routed["provider"] == {"require_parameters": False}
    assert {
        key: value for key, value in routed.items() if key != "provider"
    } == {
        key: value for key, value in payload.items() if key != "provider"
    }
    assert payload["provider"] == {"require_parameters": True}


def test_balanced_support_projects_region_mass_and_preserves_ratios():
    hypotheses = [
        {
            "region": region,
            "probability": probability,
            "description": f"{region}-{index}",
        }
        for region, probabilities in {
            "NE": (0.03, 0.07),
            "NW": (0.08, 0.02),
            "SW": (0.01, 0.09),
            "SE": (0.04, 0.06),
        }.items()
        for index, probability in enumerate(probabilities)
    ]

    balanced = balance_support_regions(hypotheses)

    regional_mass = {
        region: sum(
            item["probability"]
            for item in balanced
            if item["region"] == region
        )
        for region in ("NE", "NW", "SW", "SE")
    }
    assert all(
        np.isclose(regional_mass[region], expected)
        for region, expected in {
            "NE": 0.4,
            "NW": 0.3,
            "SW": 0.2,
            "SE": 0.1,
        }.items()
    )
    assert np.isclose(
        balanced[0]["probability"] / balanced[1]["probability"],
        3.0 / 7.0,
    )
    assert hypotheses[0]["probability"] == 0.03


def test_balanced_support_rejects_missing_region_breadth():
    hypotheses = [
        {"region": "NE", "probability": 0.125}
        for _ in range(8)
    ]

    with pytest.raises(ValueError, match="exactly two maps per region"):
        balance_support_regions(hypotheses)


def test_balanced_refresh_prompt_freezes_two_maps_per_region():
    messages = balanced_refresh_messages(
        [],
        {
            "B": [
                {
                    "representative_final_coordinate": [0.0, 0.0],
                    "posterior_probabilities": [0.5, 0.5],
                }
            ]
        },
        root_id="B",
        branch_index=0,
    )

    assert "FROZEN_BREADTH_CONSTRAINT" in messages[-1]["content"]
    assert "exactly two hypotheses for each of NE, NW, SW, and SE" in (
        messages[-1]["content"]
    )


def test_modular_component_posteriors_normalize_independently():
    initial, refresh = modular_component_posteriors(
        initial_branch_prior=np.array([0.6, 0.4]),
        refresh_branch_prior=np.array([0.3, 0.7]),
        representative_observation=np.array([0.0, 0.0]),
        actual_root_observation=np.array([0.1, -0.1]),
        initial_root_means=np.array([[-0.2, 0.0], [0.2, 0.0]]),
        refresh_root_means=np.array([[0.0, -0.2], [0.0, 0.2]]),
        continuation_observations=np.array([[0.0, 0.1], [0.1, 0.0]]),
        initial_continuation_means=np.array(
            [[-0.1, 0.0], [0.1, 0.0]]
        ),
        refresh_continuation_means=np.array(
            [[0.0, -0.1], [0.0, 0.1]]
        ),
    )

    assert initial.shape == refresh.shape == (2, 2)
    assert np.allclose(initial.sum(axis=1), 1.0)
    assert np.allclose(refresh.sum(axis=1), 1.0)


def test_replication_endpoint_is_fresh_stratified_384_map_family():
    maps, regions, prior = replication_hidden_map_family()

    assert REPLICATION_MAP_SEEDS == tuple(range(24700, 24716))
    assert maps.shape == (384, 10, 2)
    assert len(regions) == 384
    assert np.isclose(prior.sum(), 1.0)
    regional_mass = {
        region: float(prior[np.asarray(regions) == region].sum())
        for region in ("NE", "NW", "SW", "SE")
    }
    assert all(
        np.isclose(regional_mass[region], expected)
        for region, expected in {
            "NE": 0.4,
            "NW": 0.3,
            "SW": 0.2,
            "SE": 0.1,
        }.items()
    )


def test_balanced_endpoint_uses_new_stratified_map_family():
    maps, regions, prior = balanced_hidden_map_family()

    assert BALANCED_MAP_SEEDS == tuple(range(24720, 24736))
    assert maps.shape == (384, 10, 2)
    assert len(regions) == 384
    assert np.isclose(prior.sum(), 1.0)
    assert set(BALANCED_MAP_SEEDS).isdisjoint(REPLICATION_MAP_SEEDS)


def test_stratified_bootstrap_preserves_constant_difference():
    regions = [
        region
        for region in ("NE", "NW", "SW", "SE")
        for _ in range(4)
    ]

    lower, upper = stratified_bootstrap_interval(
        np.full(len(regions), 0.25),
        regions,
    )

    assert np.isclose(lower, 0.25)
    assert np.isclose(upper, 0.25)


def test_weighted_correlation_handles_alignment_and_constant_input():
    values = np.array([-2.0, -1.0, 1.0, 2.0])
    weights = np.array([0.1, 0.2, 0.3, 0.4])

    assert np.isclose(
        weighted_correlation(values, values, weights),
        1.0,
    )
    assert np.isclose(
        weighted_correlation(values, -values, weights),
        -1.0,
    )
    assert weighted_correlation(
        values,
        np.ones_like(values),
        weights,
    ) == 0.0


def test_phase_a_mechanics_enforces_accounting_and_support_diversity():
    initial_support = _parsed_weighted_support()
    continuations = ["r2.5_a0", "r2.5_a1"]
    refreshes = {
        root["id"]: [
            {
                "hypotheses": _fresh_support(
                    initial_support,
                    root_id=root["id"],
                    branch_index=branch_index,
                ),
                "continuation_action": continuations[branch_index],
            }
            for branch_index in range(2)
        ]
        for root in ROOTS
    }
    usage = {
        "adapter_requests": EXPECTED_REQUESTS,
        "http_attempts": EXPECTED_REQUESTS + 2,
        "retry_count": 2,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "forced_final_requests": 0,
        "run_cost_usd": 0.1,
    }

    gates, mechanism = phase_a_mechanics(
        initial_support=initial_support,
        refreshes=refreshes,
        usage=usage,
    )

    assert all(gates.values())
    assert mechanism["refreshes_changed_from_initial"] == 8
    assert mechanism["roots_with_branch_distinct_supports"] == 4
    assert mechanism["center_continuations"] == continuations


def test_phase_a_selection_requires_frozen_d_to_b_result():
    selection = {
        "myopic_root": MYOPIC_ROOT_ID,
        "lookahead_root": LOOKAHEAD_ROOT_ID,
        "internal_risk_reduction": 0.10,
    }

    assert all(phase_a_selection_gates(selection).values())

    selection["internal_risk_reduction"] = np.nextafter(0.10, 0.0)
    assert not phase_a_selection_gates(selection)[
        "internal_risk_reduction_at_least_10_percent"
    ]
