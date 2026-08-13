#!/usr/bin/env python3
"""Pure codec for factorized Number Game semantic support."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Sequence

from scripts.number_game_extension_native_semantic_codec import (
    audit_response_format,
    canonical_json,
    description_lexically_valid,
    parse_audit,
)


INTERFACE_VERSION = "number-game-factorized-semantic-support-1"
HISTORIES = (
    ((11, True),),
    ((11, True), (33, True)),
    ((1, False), (17, True)),
    ((16, True), (17, False)),
    ((19, False), (38, True)),
)
PROPOSAL_SEEDS = tuple(range(202608133000, 202608133030))
TRANSLATION_SEEDS = tuple(range(202608133100, 202608133130))
AUDIT_SEEDS = tuple(range(202608133200, 202608133210))
SHARDS_PER_DRAW = 3
ITEMS_PER_SHARD = 8


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def proposal_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "number_game_semantic_description_shard",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": 8,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["hypothesis_id", "name", "description"],
                            "properties": {
                                "hypothesis_id": {"type": "string", "pattern": "^H[1-8]$"},
                                "name": {"type": "string", "minLength": 1, "maxLength": 60},
                                "description": {"type": "string", "minLength": 1, "maxLength": 180},
                            },
                        },
                    }
                },
            },
        },
    }


def proposal_messages(history: Sequence[tuple[int, bool]], shard: int) -> list[dict[str, str]]:
    observations = [{"number": n, "answer": "YES" if y else "NO"} for n, y in history]
    system = (
        "Propose exactly 8 distinct general semantic rules for subsets of integers 0 through 100. "
        "Return only required JSON with IDs H1 through H8 exactly once. Each description states one "
        "coherent reusable ordinary-language rule. Do not use code, explicit member lists, lookup tables, "
        "exception lists, masks, or restate observed answers. Every rule must agree with all observations. "
        "The shard nonce requests diversity only."
    )
    payload = {"observations": observations, "domain": "integers 0 through 100 inclusive", "diversity_shard_nonce": shard}
    return [{"role": "system", "content": system}, {"role": "user", "content": canonical_json(payload)}]


def parse_proposal(raw: str) -> list[dict[str, str]]:
    value = json.loads(raw)
    rows = value.get("hypotheses") if isinstance(value, dict) and set(value) == {"hypotheses"} else None
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("proposal shard shape changed")
    found: dict[str, dict[str, str]] = {}
    names: set[str] = set()
    descriptions: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"hypothesis_id", "name", "description"}:
            raise ValueError("proposal item shape changed")
        hypothesis_id = row["hypothesis_id"]
        name = " ".join(str(row["name"]).strip().split())
        description = " ".join(str(row["description"]).strip().split())
        if hypothesis_id not in {f"H{i}" for i in range(1, 9)} or hypothesis_id in found:
            raise ValueError("proposal ID coverage changed")
        if not name or len(name) > 60 or name.casefold() in names:
            raise ValueError("proposal name is invalid")
        if (
            not description_lexically_valid(description)
            or description.casefold() in descriptions
            or "mask" in description.casefold()
            or re.search(r"\b(?:yes|no|observed|observation|answer)\b", description, re.I)
        ):
            raise ValueError("proposal description is invalid")
        names.add(name.casefold())
        descriptions.add(description.casefold())
        found[hypothesis_id] = {"hypothesis_id": hypothesis_id, "name": name, "description": description, "description_hash": sha256_text(description)}
    expected = [f"H{i}" for i in range(1, 9)]
    if set(found) != set(expected):
        raise ValueError("proposal ID coverage is incomplete")
    return [found[hypothesis_id] for hypothesis_id in expected]


def translation_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "number_game_description_extensions",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["extensions"],
                "properties": {
                    "extensions": {
                        "type": "array",
                        "minItems": 8,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["hypothesis_id", "membership_mask"],
                            "properties": {
                                "hypothesis_id": {"type": "string", "pattern": "^H[1-8]$"},
                                "membership_mask": {"type": "string", "pattern": "^[01]{101}$"},
                            },
                        },
                    }
                },
            },
        },
    }


def translation_messages(proposals: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    rules = [{"hypothesis_id": row["hypothesis_id"], "name": row["name"], "description": row["description"]} for row in proposals]
    system = (
        "Translate each ordinary-language subset rule into its exact extension over integers 0 through 100. "
        "Return only required JSON. membership_mask has exactly 101 bits; bit i is 1 exactly when integer i "
        "satisfies the description. Use only the supplied descriptions and domain. Do not infer any hidden "
        "observations, desired answers, or external representation."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": canonical_json({"domain": "integers 0 through 100 inclusive", "rules": rules})}]


def parse_translation(raw: str, proposals: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    value = json.loads(raw)
    rows = value.get("extensions") if isinstance(value, dict) and set(value) == {"extensions"} else None
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("translation shape changed")
    expected = {row["hypothesis_id"]: row for row in proposals}
    found: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"hypothesis_id", "membership_mask"}:
            raise ValueError("translation item shape changed")
        hypothesis_id, mask = row["hypothesis_id"], row["membership_mask"]
        if hypothesis_id not in expected or hypothesis_id in found:
            raise ValueError("translation ID coverage changed")
        if not isinstance(mask, str) or len(mask) != 101 or set(mask) - {"0", "1"} or mask in {"0" * 101, "1" * 101}:
            raise ValueError("translation mask is invalid")
        found[hypothesis_id] = mask
    if set(found) != set(expected):
        raise ValueError("translation ID coverage is incomplete")
    return [{**row, "mask": found[row["hypothesis_id"]], "extension_hash": sha256_text(found[row["hypothesis_id"]])} for row in proposals]


def merge_draw(shards: Sequence[Sequence[dict[str, str]]], history: Sequence[tuple[int, bool]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len(shards) != 3 or any(len(shard) != 8 for shard in shards):
        raise ValueError("draw shard coverage changed")
    rows: list[dict[str, Any]] = []
    for shard_index, shard in enumerate(shards):
        for row in shard:
            rows.append({**row, "global_index": 8 * shard_index + int(row["hypothesis_id"][1:]) - 1, "history_consistent": all((row["mask"][n] == "1") is y for n, y in history)})
    unique_extensions = len({row["mask"] for row in rows})
    if unique_extensions != 24:
        raise ValueError("translated draw contains duplicate extensions")
    consistent = [row for row in rows if row["history_consistent"]]
    return rows, {"translated_unique_count": unique_extensions, "history_consistent_count": len(consistent)}


def probes_for(draw_index: int, hypothesis_index: int, history: Sequence[tuple[int, bool]]) -> tuple[int, ...]:
    seed = TRANSLATION_SEEDS[3 * draw_index + hypothesis_index // 8]
    probes: list[int] = []
    for probe_index in range(8):
        raw = f"{INTERFACE_VERSION}|{seed}|{hypothesis_index}|{probe_index}".encode()
        candidate = int.from_bytes(hashlib.sha256(raw).digest(), "big") % 101
        while candidate in probes:
            candidate = (candidate + 1) % 101
        probes.append(candidate)
    observed = [n for n, _ in history]
    return tuple([value for value in probes if value not in observed][: 8 - len(observed)] + observed)


def audit_messages(draw: Sequence[dict[str, Any]], draw_index: int, history: Sequence[tuple[int, bool]]) -> list[dict[str, str]]:
    rules = [{"hypothesis_index": index, "name": row["name"], "description": row["description"], "probe_integers": list(probes_for(draw_index, index, history))} for index, row in enumerate(draw)]
    system = (
        "Independently interpret each ordinary-language subset rule. Return only required JSON. For each probe "
        "integer report true exactly when it satisfies the description. Use only names, descriptions, and probe "
        "integers. Do not infer hidden observations, desired answers, masks, or translated extensions."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": canonical_json({"rules": rules})}]


def diagnostics(draws: Sequence[Sequence[dict[str, Any]]], audits: Sequence[Sequence[Sequence[bool]]]) -> tuple[dict[str, Any], dict[str, bool]]:
    if len(draws) != 10 or len(audits) != 10:
        raise ValueError("ten complete draws are required")
    total_agreement = 0
    valid_by_draw: list[set[str]] = []
    draw_rows = []
    consistent_by_draw: list[set[str]] = []
    for draw_index, (draw, judgments) in enumerate(zip(draws, audits, strict=True)):
        history = HISTORIES[draw_index // 2]
        scores = []
        valid: set[str] = set()
        consistent = {row["extension_hash"] for row in draw if row["history_consistent"]}
        for index, (row, answers) in enumerate(zip(draw, judgments, strict=True)):
            probes = probes_for(draw_index, index, history)
            score = sum(answer is (row["mask"][probe] == "1") for answer, probe in zip(answers, probes, strict=True))
            scores.append(score)
            total_agreement += score
            if row["history_consistent"] and score >= 7:
                valid.add(row["extension_hash"])
        consistent_by_draw.append(consistent)
        valid_by_draw.append(valid)
        draw_rows.append({"draw_index": draw_index, "history_index": draw_index // 2, "translated_unique_count": len({row["extension_hash"] for row in draw}), "history_consistent_count": len(consistent), "semantic_valid_consistent_count": len(valid), "extension_hashes": [row["extension_hash"] for row in draw], "agreement_count": sum(scores), "agreement_histogram": {str(score): scores.count(score) for score in range(9)}})
    pools = []
    for history_index in range(5):
        left, right = consistent_by_draw[2 * history_index : 2 * history_index + 2]
        valid_left, valid_right = valid_by_draw[2 * history_index : 2 * history_index + 2]
        pools.append({"history_index": history_index, "history_consistent_unique_count": len(left | right), "second_draw_novel_count": len(right - left), "semantic_valid_unique_count": len(valid_left | valid_right)})
    baseline = valid_by_draw[0] | valid_by_draw[1]
    novel = [len((valid_by_draw[2 * h] | valid_by_draw[2 * h + 1]) - baseline) for h in range(1, 5)]
    gates = {
        "every_draw_exactly_24_unique_translations": all(row["translated_unique_count"] == 24 for row in draw_rows),
        "every_draw_at_least_20_history_consistent": all(row["history_consistent_count"] >= 20 for row in draw_rows),
        "every_pool_at_least_28_history_consistent_unique": all(row["history_consistent_unique_count"] >= 28 for row in pools),
        "every_second_draw_at_least_4_novel": all(row["second_draw_novel_count"] >= 4 for row in pools),
        "pooled_agreement_at_least_90_percent": total_agreement >= 1728,
        "every_draw_at_least_18_semantic_valid_consistent": all(row["semantic_valid_consistent_count"] >= 18 for row in draw_rows),
        "every_pool_at_least_26_semantic_valid": all(row["semantic_valid_unique_count"] >= 26 for row in pools),
        "every_later_history_at_least_12_novel_semantic_valid": all(value >= 12 for value in novel),
        "accepted_extensions_obey_history_exactly": all(all((row["mask"][n] == "1") is y for n, y in HISTORIES[d // 2]) for d, draw in enumerate(draws) for row in draw if row["history_consistent"]),
    }
    return {"total_judgments": 1920, "agreement_count": total_agreement, "agreement_rate": total_agreement / 1920, "draws": draw_rows, "pools": pools, "observed_history_novel_counts": novel}, gates


def assert_translation_blind(messages: Sequence[dict[str, str]]) -> None:
    if len(messages) != 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError("translation message envelope changed")
    payload = json.loads(messages[1]["content"])
    if not isinstance(payload, dict) or set(payload) != {"domain", "rules"} or payload["domain"] != "integers 0 through 100 inclusive":
        raise ValueError("translation payload leaks history or extension data")
    if not isinstance(payload["rules"], list) or len(payload["rules"]) != 8 or any(not isinstance(row, dict) or set(row) != {"hypothesis_id", "name", "description"} for row in payload["rules"]):
        raise ValueError("translation payload leaks history or extension data")


def assert_audit_blind(messages: Sequence[dict[str, str]]) -> None:
    if len(messages) != 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError("audit message envelope changed")
    payload = json.loads(messages[1]["content"])
    if not isinstance(payload, dict) or set(payload) != {"rules"} or not isinstance(payload["rules"], list) or len(payload["rules"]) != 24:
        raise ValueError("audit payload leaks history or extension data")
    if any(not isinstance(row, dict) or set(row) != {"hypothesis_index", "name", "description", "probe_integers"} for row in payload["rules"]):
        raise ValueError("audit payload leaks history or extension data")
