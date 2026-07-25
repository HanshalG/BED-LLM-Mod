#!/usr/bin/env python3
"""Run frozen Bamboogle mechanics through a strict flat line protocol."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import bamboogle_cached_search_mechanics as mechanics


INTERFACE_VERSION = "bamboogle-cached-search-mechanics-flat-3"


def _parse_lines(text: str, expected: Sequence[str]) -> dict[str, list[str]]:
    lines = text.splitlines()
    if len(lines) != len(expected):
        raise ValueError(
            f"response must contain exactly {len(expected)} lines"
        )
    parsed: dict[str, list[str]] = {}
    for line, identifier in zip(lines, expected):
        fields = line.split("|")
        if not fields or fields[0] != identifier:
            raise ValueError(f"expected ordered line {identifier}")
        if any(not field.strip() for field in fields):
            raise ValueError(f"line {identifier} contains an empty field")
        parsed[identifier] = [field.strip() for field in fields[1:]]
    return parsed


def _parse_belief(
    parsed: dict[str, list[str]],
    identifiers: Sequence[str],
) -> mechanics.Belief:
    payload: dict[str, object] = {}
    for index, identifier in enumerate(identifiers, start=1):
        fields = parsed[identifier]
        if len(fields) != 2:
            raise ValueError(f"line {identifier} must have three fields")
        weight, hypothesis = fields
        payload[f"hypothesis_{index}"] = hypothesis
        payload[f"weight_{index}"] = weight
    return mechanics.parse_belief_payload(payload)


def parse_initial(text: str) -> tuple[mechanics.Belief, list[str], list[str]]:
    hypothesis_ids = [f"H{index:02d}" for index in range(1, 9)]
    root_ids = [f"R{index:02d}" for index in range(1, 5)]
    fixed_ids = [f"F{index:02d}" for index in range(1, 5)]
    parsed = _parse_lines(text, hypothesis_ids + root_ids + fixed_ids)
    belief = _parse_belief(parsed, hypothesis_ids)
    roots = [mechanics._clean_query(parsed[key][0]) for key in root_ids]
    fixed = [mechanics._clean_query(parsed[key][0]) for key in fixed_ids]
    if any(len(parsed[key]) != 1 for key in root_ids + fixed_ids):
        raise ValueError("query lines must have exactly two fields")
    normalized = [
        mechanics.normalize_query(query) for query in roots + fixed
    ]
    if len(set(normalized)) != mechanics.ROOT_COUNT * 2:
        raise ValueError("all initial root and fixed queries must be distinct")
    return belief, roots, fixed


def parse_refresh(text: str) -> tuple[mechanics.Belief, str]:
    hypothesis_ids = [f"H{index:02d}" for index in range(1, 9)]
    parsed = _parse_lines(text, hypothesis_ids + ["A01"])
    belief = _parse_belief(parsed, hypothesis_ids)
    if len(parsed["A01"]) != 1:
        raise ValueError("adaptive query line must have exactly two fields")
    return belief, mechanics._clean_query(parsed["A01"][0])


def parse_terminal(text: str) -> mechanics.Belief:
    hypothesis_ids = [f"H{index:02d}" for index in range(1, 9)]
    parsed = _parse_lines(text, hypothesis_ids)
    return _parse_belief(parsed, hypothesis_ids)


def _belief_protocol() -> str:
    return "\n".join(
        f"H{index:02d}|<integer weight>|<short candidate answer>"
        for index in range(1, 9)
    )


def initial_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Maintain an open-world categorical belief over short final "
                "answers to a two-hop factual question. Emit eight distinct "
                "candidate answers with positive integer weights summing "
                "exactly to 100. Also design four diverse first Wikipedia "
                "search queries and, for each first query, a second query "
                "fixed now before any result is seen. All eight queries must "
                "be distinct and expose different intermediate entities or "
                "relations. Return exactly 16 ordered lines in this grammar, "
                "with no JSON, markdown, explanation, citations, blank lines, "
                "or reasoning:\n"
                f"{_belief_protocol()}\n"
                "R01|<first query>\nR02|<first query>\n"
                "R03|<first query>\nR04|<first query>\n"
                "F01|<precommitted second query>\n"
                "F02|<precommitted second query>\n"
                "F03|<precommitted second query>\n"
                "F04|<precommitted second query>"
            ),
        },
        {"role": "user", "content": json.dumps({"question": question})},
    ]


def refresh_messages(
    question: str,
    initial_belief: mechanics.Belief,
    root_query: str,
    root_documents: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    payload = {
        "question": question,
        "prior_belief": mechanics._belief_payload(initial_belief),
        "root_search": {
            "query": root_query,
            "results": list(root_documents),
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "Update the categorical answer belief using only the supplied "
                "question, prior belief, and Wikipedia extracts. Regenerate "
                "eight distinct short answers with positive integer weights "
                "summing exactly to 100, then choose the most useful second "
                "Wikipedia query conditional on the root result. Return "
                "exactly nine ordered lines in this grammar, with no JSON, "
                "markdown, explanation, citations, blank lines, or reasoning:\n"
                f"{_belief_protocol()}\nA01|<adaptive second query>"
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def terminal_messages(
    question: str,
    root_belief: mechanics.Belief,
    root_query: str,
    root_documents: Sequence[dict[str, str]],
    second_query: str,
    second_documents: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    payload = {
        "question": question,
        "prior_belief_after_root": mechanics._belief_payload(root_belief),
        "evidence": [
            {"query": root_query, "results": list(root_documents)},
            {"query": second_query, "results": list(second_documents)},
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Regenerate the categorical belief over short final answers "
                "after both Wikipedia searches. Use the supplied extracts as "
                "evidence, retain genuine alternatives when uncertainty "
                "remains, and assign eight positive integer weights summing "
                "exactly to 100. Return exactly eight ordered lines in this "
                "grammar, with no JSON, markdown, explanation, citations, "
                "blank lines, or reasoning:\n"
                f"{_belief_protocol()}"
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


FLAT_CODEC = mechanics.MechanicsCodec(
    initial_messages=initial_messages,
    parse_initial=parse_initial,
    refresh_messages=refresh_messages,
    parse_refresh=parse_refresh,
    terminal_messages=terminal_messages,
    parse_terminal=parse_terminal,
    response_format_name="strict_flat_lines",
)


if __name__ == "__main__":
    mechanics.run_cli(
        interface_version=INTERFACE_VERSION,
        codec=FLAT_CODEC,
    )
