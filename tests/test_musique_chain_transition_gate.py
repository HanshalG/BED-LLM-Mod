import json

import pytest

from scripts.musique_chain_transition_gate import (
    ACTION_COUNT,
    CHAIN_COUNT,
    binary_entropy,
    candidate_actions,
    immediate_eig_values,
    parse_chains,
    summarize,
    truth_coverage,
)


def _chains(titles):
    firsts = [0, 0, 1, 1, 2, 3, 4, 5]
    return [
        {
            "id": f"c{index + 1}",
            "first_doc_id": f"d{first + 1:02d}",
            "second_doc_id": f"d{6 + index + 1:02d}",
            "first_title": titles[first],
            "second_title": titles[6 + index],
            "bridge_entity_hypothesis": f"bridge {index}",
        }
        for index, first in enumerate(firsts)
    ]


def _payload(chains):
    roots = []
    offset = 0
    for root_index, count in enumerate((2, 2, 1, 1, 1, 1)):
        root_chains = chains[offset : offset + count]
        roots.append(
            {
                "id": f"r{root_index + 1}",
                "first_doc_id": root_chains[0]["first_doc_id"],
                "continuations": [
                    {
                        "id": f"r{root_index + 1}{chr(ord('a') + local_index)}",
                        "second_doc_id": chain["second_doc_id"],
                        "bridge_entity_hypothesis": chain[
                            "bridge_entity_hypothesis"
                        ],
                    }
                    for local_index, chain in enumerate(root_chains)
                ],
            }
        )
        offset += count
    return {"roots": roots}


def test_parse_chains_and_immediate_eig():
    titles = [f"title {index}" for index in range(20)]
    chains = _chains(titles)
    parsed = parse_chains(
        json.dumps(_payload(chains)),
        available_titles=titles,
    )
    actions = candidate_actions(parsed)
    assert len(parsed) == CHAIN_COUNT
    assert len(actions) == ACTION_COUNT
    values = immediate_eig_values(parsed, actions)
    assert values[0] == pytest.approx(binary_entropy(0.25))
    assert values[1] == pytest.approx(binary_entropy(0.25))
    assert values[2] == pytest.approx(binary_entropy(0.125))


def test_parse_chains_rejects_non_available_document_id():
    titles = [f"title {index}" for index in range(20)]
    chains = _chains(titles)
    payload = _payload(chains)
    payload["roots"][0]["first_doc_id"] = "d99"
    with pytest.raises(ValueError, match="not available"):
        parse_chains(
            json.dumps(payload),
            available_titles=titles,
        )


def test_parse_chains_requires_six_distinct_first_titles():
    titles = [f"title {index}" for index in range(20)]
    chains = _chains(titles)
    payload = _payload(chains)
    payload["roots"][-1]["first_doc_id"] = "d05"
    with pytest.raises(ValueError, match="must be unique"):
        parse_chains(
            json.dumps(payload),
            available_titles=titles,
        )


def test_parse_chains_allows_distinct_documents_with_same_title():
    titles = [f"title {index}" for index in range(20)]
    titles[6] = titles[0]
    chains = _chains(titles)
    parsed = parse_chains(
        json.dumps(_payload(chains)),
        available_titles=titles,
    )
    assert len(parsed) == CHAIN_COUNT


def test_truth_coverage_matches_ordered_pair():
    titles = [f"title {index}" for index in range(20)]
    chains = _chains(titles)
    assert truth_coverage(chains, ("d01", "d07")) == 1
    assert truth_coverage(chains, ("d07", "d01")) == 0


def test_formal_summary_applies_frozen_gates():
    records = []
    for index in range(12):
        initial = 0 if index < 6 else 1
        oracle = 1
        regret = 1 if index < 4 else 0
        records.append(
            {
                "initial_truth_coverage": initial,
                "omission_recovered": index < 4,
                "oracle_truth_coverage_gain": oracle - initial,
                "branch_truth_coverage_spread": 1 if index < 6 else 0,
                "immediate_eig_truth_coverage_regret": regret,
                "gold_root_is_candidate": index < 9,
                "branches": [{} for _ in range(ACTION_COUNT)],
            }
        )
    summary = summarize(
        records,
        {
            "physical_requests": 12 * (1 + ACTION_COUNT),
            "reasoning_tokens": 0,
        },
        stage="formal",
    )
    assert summary["gates"]["all_pass"]


def test_smoke_summary_requires_exact_requests():
    records = [{"branches": [{} for _ in range(ACTION_COUNT)]} for _ in range(2)]
    summary = summarize(
        records,
        {
            "physical_requests": 2 * (1 + ACTION_COUNT),
            "reasoning_tokens": 0,
        },
        stage="serving_smoke",
    )
    assert summary["gates"]["all_pass"]
