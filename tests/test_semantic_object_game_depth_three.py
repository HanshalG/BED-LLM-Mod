from __future__ import annotations

import json
import math

from scripts import semantic_object_game_depth_three as semantic


def _response() -> str:
    concepts = []
    for index in range(semantic.NUM_PROPOSALS):
        concepts.append(
            {
                "name": f"concept {index}",
                "description": f"coherent semantic property {index}",
                "membership_bits": (
                    "1" * (3 + index)
                    + "0" * (len(semantic.OBJECT_IDS) - 3 - index)
                ),
            }
        )
    return json.dumps({"concepts": concepts})


def _hypothesis(
    name: str,
    positives: set[int],
) -> semantic.SemanticHypothesis:
    return semantic.SemanticHypothesis(
        name=name,
        description=name,
        extension=tuple(
            index in positives for index in range(len(semantic.OBJECT_IDS))
        ),
    )


def test_universe_and_request_count_are_frozen() -> None:
    assert len(semantic.OBJECT_IDS) == 32
    assert len(set(semantic.OBJECT_IDS)) == 32
    assert semantic.NUM_PROPOSALS == 16
    assert semantic.NUM_ROOTS == 6
    assert semantic.EXPECTED_REQUESTS == (
        1
        + 2 * semantic.NUM_ROOTS
        + 4 * semantic.NUM_ROOTS
        + len(semantic.VALIDATION_SEEDS)
        + len(semantic.ENDPOINT_SEEDS)
    )
    membership_schema = semantic.proposal_response_format()["json_schema"][
        "schema"
    ]["properties"]["concepts"]["items"]["properties"]["membership_bits"]
    assert membership_schema == {
        "type": "string",
        "minLength": 32,
        "maxLength": 32,
    }


def test_parse_proposals_accepts_exact_memberships() -> None:
    support, diagnostic = semantic.parse_proposals(
        _response(),
        observations=((0, True), (31, False)),
    )
    assert len(support) == semantic.NUM_PROPOSALS
    assert diagnostic["valid_unique_count"] == semantic.NUM_PROPOSALS
    assert support[0].extension[:4] == (True, True, True, False)
    assert support[-1].extension[:19] == (True,) * 18 + (False,)


def test_parse_proposals_filters_inconsistent_and_duplicate_extensions() -> None:
    payload = json.loads(_response())
    payload["concepts"][0]["membership_bits"] = "0111" + "0" * 28
    payload["concepts"][-1]["membership_bits"] = payload["concepts"][1][
        "membership_bits"
    ]
    support, diagnostic = semantic.parse_proposals(
        json.dumps(payload),
        observations=((0, True),),
    )
    assert len(support) == semantic.NUM_PROPOSALS - 2
    assert diagnostic["rejected"]["inconsistent"] == 1
    assert diagnostic["rejected"]["duplicate_extension"] == 1


def test_retained_rejuvenation_keeps_consistent_parent_without_duplicates() -> None:
    parent_true = _hypothesis("parent true", {0, 1, 2})
    parent_false = _hypothesis("parent false", {3, 4, 5})
    generated = [
        _hypothesis("generated", {0, 6, 7}),
        _hypothesis("duplicate parent", {0, 1, 2}),
    ]
    merged, diagnostic = semantic.retain_parent_hypotheses(
        parent_support=[parent_true, parent_false],
        generated_support=generated,
        query=0,
        label=True,
    )
    assert [item.name for item in merged] == [
        "generated",
        "duplicate parent",
    ]
    assert diagnostic == {
        "generated_unique_count": 2,
        "retained_parent_consistent_count": 1,
        "retained_parent_novel_count": 0,
        "merged_unique_count": 2,
    }


def test_depth_three_uses_second_refresh_and_third_query() -> None:
    target = _hypothesis("target", {0, 2, 4, 6})
    first = {
        (0, True): [
            _hypothesis("a", {0, 1, 2}),
            _hypothesis("b", {0, 2, 4}),
        ]
    }
    second_query, _ = semantic.best_query(first[(0, True)], excluded=(0,))
    assert second_query == 1
    second = {
        (0, True, 1, False): [
            _hypothesis("b", {0, 2, 4}),
            _hypothesis("c", {0, 2, 4, 6}),
        ]
    }
    row = semantic.evaluate_depth_three_root(
        root=0,
        target=target,
        first_branches=first,
        second_branches=second,
    )
    assert row["second_query"] == 1
    assert row["third_query"] == 6
    assert row["posterior_predictive_brier"] == 0.0
    assert row["truth_extension_covered"] is True


def test_mechanics_gate_requires_active_depth_comparison() -> None:
    first = {
        "x": {"valid_unique_count": semantic.MIN_FIRST_VALID},
    }
    second = {
        "y": {"valid_unique_count": semantic.MIN_SECOND_VALID},
    }
    support = [
        _hypothesis(str(index), {0, 1, 2 + index})
        for index in range(semantic.MIN_TARGET_VALID)
    ]
    initial = support[: semantic.MIN_INITIAL_VALID]
    roots = list(range(semantic.NUM_ROOTS))
    depth_two = {root: 0.2 + root / 100 for root in roots}
    depth_three = {root: 0.2 + root / 100 for root in roots}
    depth_three[1] = 0.1
    gates = semantic.mechanics_gates(
        usage={
            "adapter_requests": semantic.EXPECTED_REQUESTS,
            "http_attempts": semantic.EXPECTED_REQUESTS,
            "retry_count": 0,
            "reasoning_tokens": 0,
            "forced_exits": 0,
            "run_cost_usd": semantic.RUN_BUDGET_USD,
        },
        initial=initial,
        first_diagnostics=first,
        second_diagnostics=second,
        validation_supports=[support] * len(semantic.VALIDATION_SEEDS),
        endpoint_supports=[support] * len(semantic.ENDPOINT_SEEDS),
        roots=roots,
        depth_two_scores=depth_two,
        depth_three_scores=depth_three,
    )
    assert all(gates.values())
    assert math.isfinite(min(depth_three.values()))
