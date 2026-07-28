from __future__ import annotations

from copy import deepcopy
import json

import numpy as np

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
