#!/usr/bin/env python3
"""Strict schemas and codecs for HiddenBench dynamic-belief V3."""

from __future__ import annotations

import json
import math
from typing import Any, Sequence

from scripts.hiddenbench_dynamic_belief_v3_math import CHANNEL_IDS, QUERY_IDS


DIRECT_ANSWER_PHRASES = ("correct answer", "which option", "choose the answer")


def strict_object(response: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def normalized_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def probability_items(
    value: Any, *, ids: Sequence[str], id_key: str, label: str
) -> list[float]:
    if not isinstance(value, list) or len(value) != len(ids):
        raise ValueError(f"{label} has the wrong length")
    found: dict[str, float] = {}
    for item in value:
        if not isinstance(item, dict) or set(item) != {id_key, "probability"}:
            raise ValueError(f"{label} has the wrong fields")
        item_id = item[id_key]
        probability = item["probability"]
        if item_id not in ids or item_id in found:
            raise ValueError(f"{label} has invalid or duplicate IDs")
        if (
            not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or probability < 0
            or probability > 1
        ):
            raise ValueError(f"{label} has an invalid probability")
        found[item_id] = float(probability)
    if set(found) != set(ids) or abs(sum(found.values()) - 1.0) > 1e-6:
        raise ValueError(f"{label} is not normalized")
    return [found[item_id] for item_id in ids]


def probability_array_schema(ids: Sequence[str], id_key: str) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": len(ids),
        "maxItems": len(ids),
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [id_key, "probability"],
            "properties": {
                id_key: {"type": "string", "enum": list(ids)},
                "probability": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
    }


def root_response_format(option_ids: Sequence[str]) -> dict[str, Any]:
    channel_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["channel_id", "description"],
        "properties": {
            "channel_id": {"type": "string", "enum": list(CHANNEL_IDS)},
            "description": {"type": "string", "minLength": 1},
        },
    }
    likelihood_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["option_id", "channels"],
        "properties": {
            "option_id": {"type": "string", "enum": list(option_ids)},
            "channels": probability_array_schema(CHANNEL_IDS, "channel_id"),
        },
    }
    query_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "query_id", "request", "target_dimension", "channels", "likelihoods"
        ],
        "properties": {
            "query_id": {"type": "string", "enum": list(QUERY_IDS)},
            "request": {"type": "string", "minLength": 1},
            "target_dimension": {"type": "string", "minLength": 1},
            "channels": {
                "type": "array", "minItems": 3, "maxItems": 3, "items": channel_schema
            },
            "likelihoods": {
                "type": "array",
                "minItems": len(option_ids),
                "maxItems": len(option_ids),
                "items": likelihood_schema,
            },
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hiddenbench_dynamic_root",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["prior", "queries"],
                "properties": {
                    "prior": probability_array_schema(option_ids, "option_id"),
                    "queries": {
                        "type": "array", "minItems": 4, "maxItems": 4, "items": query_schema
                    },
                },
            },
        },
    }


def refresh_response_format(option_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hiddenbench_dynamic_refresh",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["branches"],
                "properties": {
                    "branches": {
                        "type": "array",
                        "minItems": 12,
                        "maxItems": 12,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["query_id", "channel_id", "belief"],
                            "properties": {
                                "query_id": {"type": "string", "enum": list(QUERY_IDS)},
                                "channel_id": {"type": "string", "enum": list(CHANNEL_IDS)},
                                "belief": probability_array_schema(option_ids, "option_id"),
                            },
                        },
                    }
                },
            },
        },
    }


def routing_response_format() -> dict[str, Any]:
    mapping = {
        "type": "object",
        "additionalProperties": False,
        "required": ["query_id", "fact_id", "channel_id"],
        "properties": {
            "query_id": {"type": "string", "enum": list(QUERY_IDS)},
            "fact_id": {"type": "string", "enum": ["F1", "F2", "F3", "F4"]},
            "channel_id": {"type": "string", "enum": list(CHANNEL_IDS)},
        },
    }
    task = {
        "type": "object",
        "additionalProperties": False,
        "required": ["slot", "mappings"],
        "properties": {
            "slot": {"type": "string", "enum": ["T1", "T2", "T3", "T4"]},
            "mappings": {
                "type": "array", "minItems": 4, "maxItems": 4, "items": mapping
            },
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hiddenbench_dynamic_routing",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tasks"],
                "properties": {
                    "tasks": {"type": "array", "minItems": 4, "maxItems": 4, "items": task}
                },
            },
        },
    }


def parse_root(response: str, option_ids: Sequence[str]) -> dict[str, Any]:
    value = strict_object(response, label="root response")
    if set(value) != {"prior", "queries"}:
        raise ValueError("root response has the wrong fields")
    prior = probability_items(
        value["prior"], ids=option_ids, id_key="option_id", label="prior"
    )
    queries = value["queries"]
    if not isinstance(queries, list) or len(queries) != 4:
        raise ValueError("root response must have four queries")
    parsed: dict[str, Any] = {}
    requests: set[str] = set()
    dimensions: set[str] = set()
    for query in queries:
        if not isinstance(query, dict) or set(query) != {
            "query_id", "request", "target_dimension", "channels", "likelihoods"
        }:
            raise ValueError("root query has the wrong fields")
        query_id = query["query_id"]
        request = query["request"]
        dimension = query["target_dimension"]
        if (
            query_id not in QUERY_IDS
            or query_id in parsed
            or not isinstance(request, str)
            or not request.strip()
            or not isinstance(dimension, str)
            or not dimension.strip()
        ):
            raise ValueError("root query is invalid")
        normalized_request = normalized_text(request)
        normalized_dimension = normalized_text(dimension)
        if (
            normalized_request in requests
            or normalized_dimension in dimensions
            or any(phrase in normalized_request for phrase in DIRECT_ANSWER_PHRASES)
        ):
            raise ValueError("root query is duplicated or asks for the answer")
        requests.add(normalized_request)
        dimensions.add(normalized_dimension)
        channels = query["channels"]
        if not isinstance(channels, list) or len(channels) != 3:
            raise ValueError("query channels have the wrong length")
        channel_descriptions: dict[str, str] = {}
        normalized_channels: set[str] = set()
        for channel in channels:
            if not isinstance(channel, dict) or set(channel) != {"channel_id", "description"}:
                raise ValueError("channel has the wrong fields")
            channel_id = channel["channel_id"]
            description = channel["description"]
            if (
                channel_id not in CHANNEL_IDS
                or channel_id in channel_descriptions
                or not isinstance(description, str)
                or not description.strip()
                or normalized_text(description) in normalized_channels
            ):
                raise ValueError("channel is invalid or duplicated")
            channel_descriptions[channel_id] = description.strip()
            normalized_channels.add(normalized_text(description))
        if set(channel_descriptions) != set(CHANNEL_IDS):
            raise ValueError("channel IDs are incomplete")
        likelihood_rows: dict[str, list[float]] = {}
        likelihoods = query["likelihoods"]
        if not isinstance(likelihoods, list) or len(likelihoods) != len(option_ids):
            raise ValueError("likelihood rows have the wrong length")
        for row in likelihoods:
            if not isinstance(row, dict) or set(row) != {"option_id", "channels"}:
                raise ValueError("likelihood row has the wrong fields")
            option_id = row["option_id"]
            if option_id not in option_ids or option_id in likelihood_rows:
                raise ValueError("likelihood option ID is invalid or duplicated")
            likelihood_rows[option_id] = probability_items(
                row["channels"], ids=CHANNEL_IDS, id_key="channel_id", label=f"{query_id}.{option_id}"
            )
        if set(likelihood_rows) != set(option_ids):
            raise ValueError("likelihood option coverage is incomplete")
        parsed[query_id] = {
            "request": request.strip(),
            "target_dimension": dimension.strip(),
            "channels": channel_descriptions,
            "likelihoods": likelihood_rows,
        }
    if set(parsed) != set(QUERY_IDS):
        raise ValueError("root query coverage is incomplete")
    return {"prior": prior, "queries": parsed}


def parse_refresh(response: str, option_ids: Sequence[str]) -> dict[str, dict[str, list[float]]]:
    value = strict_object(response, label="refresh response")
    if set(value) != {"branches"} or not isinstance(value["branches"], list) or len(value["branches"]) != 12:
        raise ValueError("refresh response has the wrong fields or length")
    result: dict[str, dict[str, list[float]]] = {query_id: {} for query_id in QUERY_IDS}
    for branch in value["branches"]:
        if not isinstance(branch, dict) or set(branch) != {"query_id", "channel_id", "belief"}:
            raise ValueError("refresh branch has the wrong fields")
        query_id = branch["query_id"]
        channel_id = branch["channel_id"]
        if query_id not in QUERY_IDS or channel_id not in CHANNEL_IDS or channel_id in result[query_id]:
            raise ValueError("refresh branch ID is invalid or duplicated")
        result[query_id][channel_id] = probability_items(
            branch["belief"], ids=option_ids, id_key="option_id", label=f"{query_id}.{channel_id}"
        )
    if any(set(result[query_id]) != set(CHANNEL_IDS) for query_id in QUERY_IDS):
        raise ValueError("refresh branch coverage is incomplete")
    return result


def parse_routing(
    response: str, fact_ids_by_slot: dict[str, Sequence[str]]
) -> dict[str, dict[str, dict[str, str]]]:
    value = strict_object(response, label="routing response")
    if set(value) != {"tasks"} or not isinstance(value["tasks"], list) or len(value["tasks"]) != 4:
        raise ValueError("routing response has the wrong fields or length")
    result: dict[str, dict[str, dict[str, str]]] = {}
    for task in value["tasks"]:
        if not isinstance(task, dict) or set(task) != {"slot", "mappings"}:
            raise ValueError("routing task has the wrong fields")
        slot = task["slot"]
        if slot not in fact_ids_by_slot or slot in result or not isinstance(task["mappings"], list) or len(task["mappings"]) != 4:
            raise ValueError("routing task is invalid")
        mappings: dict[str, dict[str, str]] = {}
        for mapping in task["mappings"]:
            if not isinstance(mapping, dict) or set(mapping) != {"query_id", "fact_id", "channel_id"}:
                raise ValueError("routing mapping has the wrong fields")
            query_id = mapping["query_id"]
            fact_id = mapping["fact_id"]
            channel_id = mapping["channel_id"]
            if query_id not in QUERY_IDS or query_id in mappings or fact_id not in fact_ids_by_slot[slot] or channel_id not in CHANNEL_IDS:
                raise ValueError("routing mapping is invalid")
            mappings[query_id] = {"fact_id": fact_id, "channel_id": channel_id}
        if set(mappings) != set(QUERY_IDS):
            raise ValueError("routing query coverage is incomplete")
        result[slot] = mappings
    if set(result) != set(fact_ids_by_slot):
        raise ValueError("routing task coverage is incomplete")
    return result
