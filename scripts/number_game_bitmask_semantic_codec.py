#!/usr/bin/env python3
"""Pure codec for the compact Number Game semantic bitmask gate."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

from scripts.number_game_extension_native_semantic_codec import (
    audit_response_format,
    canonical_json,
    description_lexically_valid,
    digest_bytes,
    parse_audit,
)


INTERFACE_VERSION = "number-game-bitmask-semantic-support-1"
HISTORIES: tuple[tuple[tuple[int, bool], ...], ...] = (
    ((9, True),),
    ((9, True), (18, True)),
    ((5, False), (25, True)),
    ((12, True), (13, False)),
    ((6, False), (30, True)),
)
PROPOSAL_SEEDS = tuple(range(202608132600, 202608132610))
AUDIT_SEEDS = tuple(range(202608132700, 202608132710))
HYPOTHESES_PER_DRAW = 24
PROBES_PER_HYPOTHESIS = 8


def mask_hash(mask: str) -> str:
    return digest_bytes(mask.encode("ascii"))


def proposal_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "number_game_semantic_bitmasks",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": 24,
                        "maxItems": 24,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "description", "membership_mask"],
                            "properties": {
                                "name": {"type": "string", "minLength": 1, "maxLength": 60},
                                "description": {"type": "string", "minLength": 1, "maxLength": 180},
                                "membership_mask": {"type": "string", "pattern": "^[01]{101}$"},
                            },
                        },
                    }
                },
            },
        },
    }


def proposal_messages(history: Sequence[tuple[int, bool]]) -> list[dict[str, str]]:
    observations = [{"number": number, "answer": "YES" if answer else "NO"} for number, answer in history]
    system = (
        "Propose exactly 24 distinct general semantic rules for subsets of integers 0 through 100. "
        "Return only the required JSON. Each description states one coherent reusable ordinary-language "
        "rule, without code, explicit member lists, lookup tables, exception lists, or references to the "
        "mask. membership_mask must contain exactly 101 bits: bit i is 1 exactly when integer i satisfies "
        "the description. Every rule and mask must agree with every observation."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": canonical_json({"observations": observations, "domain": "integers 0 through 100 inclusive"})}]


def parse_proposal(raw: str, history: Sequence[tuple[int, bool]]) -> list[dict[str, Any]]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses"}:
        raise ValueError("proposal top-level shape changed")
    rows = value["hypotheses"]
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError("proposal must contain exactly 24 hypotheses")
    names: set[str] = set()
    masks: set[str] = set()
    parsed = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"name", "description", "membership_mask"}:
            raise ValueError(f"hypothesis {index} shape changed")
        name = " ".join(str(row["name"]).strip().split())
        description = " ".join(str(row["description"]).strip().split())
        mask = row["membership_mask"]
        normalized_name = name.casefold()
        if not name or len(name) > 60 or normalized_name in names:
            raise ValueError(f"hypothesis {index} name is invalid")
        if not description_lexically_valid(description) or "mask" in description.casefold():
            raise ValueError(f"hypothesis {index} description fails lexical gate")
        if not isinstance(mask, str) or len(mask) != 101 or set(mask) - {"0", "1"} or mask in {"0" * 101, "1" * 101}:
            raise ValueError(f"hypothesis {index} mask is invalid")
        if mask in masks:
            raise ValueError(f"hypothesis {index} duplicates a mask")
        if any((mask[number] == "1") is not answer for number, answer in history):
            raise ValueError(f"hypothesis {index} contradicts history")
        names.add(normalized_name)
        masks.add(mask)
        parsed.append({"name": name, "description": description, "mask": mask, "mask_hash": mask_hash(mask)})
    return parsed


def probes_for(proposal_seed: int, hypothesis_index: int, history: Sequence[tuple[int, bool]]) -> tuple[int, ...]:
    probes: list[int] = []
    for probe_index in range(8):
        raw = f"{INTERFACE_VERSION}|{proposal_seed}|{hypothesis_index}|{probe_index}".encode()
        candidate = int.from_bytes(hashlib.sha256(raw).digest(), "big") % 101
        while candidate in probes:
            candidate = (candidate + 1) % 101
        probes.append(candidate)
    observed = [number for number, _ in history]
    return tuple([value for value in probes if value not in observed][: 8 - len(observed)] + observed)


def audit_messages(proposals: Sequence[dict[str, Any]], proposal_seed: int, history: Sequence[tuple[int, bool]]) -> list[dict[str, str]]:
    rules = [{"hypothesis_index": index, "name": row["name"], "description": row["description"], "probe_integers": list(probes_for(proposal_seed, index, history))} for index, row in enumerate(proposals)]
    system = (
        "Independently interpret each ordinary-language subset rule. Return only the required JSON. "
        "For each probe integer, report true exactly when it satisfies the description. Use only the "
        "name, description, and probes; do not infer a hidden representation or revise the rule."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": canonical_json({"rules": rules})}]


def diagnostics(proposals_by_draw: Sequence[Sequence[dict[str, Any]]], audits_by_draw: Sequence[Sequence[Sequence[bool]]]) -> tuple[dict[str, Any], dict[str, bool]]:
    if len(proposals_by_draw) != 10 or len(audits_by_draw) != 10:
        raise ValueError("ten complete draws are required")
    total_agreement = 0
    draws = []
    valid_by_draw: list[set[str]] = []
    for draw_index, (proposals, audits) in enumerate(zip(proposals_by_draw, audits_by_draw, strict=True)):
        scores = []
        valid: set[str] = set()
        for hypothesis_index, (proposal, judgments) in enumerate(zip(proposals, audits, strict=True)):
            probes = probes_for(PROPOSAL_SEEDS[draw_index], hypothesis_index, HISTORIES[draw_index // 2])
            score = sum(judgment is (proposal["mask"][probe] == "1") for judgment, probe in zip(judgments, probes, strict=True))
            scores.append(score)
            total_agreement += score
            if score >= 7:
                valid.add(proposal["mask_hash"])
        valid_by_draw.append(valid)
        draws.append({"draw_index": draw_index, "history_index": draw_index // 2, "unique_mask_count": len({row["mask_hash"] for row in proposals}), "mask_hashes": [row["mask_hash"] for row in proposals], "agreement_count": sum(scores), "semantic_valid_count": len(valid), "agreement_histogram": {str(score): scores.count(score) for score in range(9)}})
    pools = []
    for history_index in range(5):
        left, right = valid_by_draw[2 * history_index : 2 * history_index + 2]
        all_left = {row["mask_hash"] for row in proposals_by_draw[2 * history_index]}
        all_right = {row["mask_hash"] for row in proposals_by_draw[2 * history_index + 1]}
        pools.append({"history_index": history_index, "unique_mask_count": len(all_left | all_right), "second_draw_novel_count": len(all_right - all_left), "semantic_valid_unique_count": len(left | right)})
    baseline = valid_by_draw[0] | valid_by_draw[1]
    history_novel = [len((valid_by_draw[2*i] | valid_by_draw[2*i+1]) - baseline) for i in range(1, 5)]
    gates = {
        "every_draw_exactly_24_valid_unique": all(row["unique_mask_count"] == 24 for row in draws),
        "every_pool_at_least_28_unique": all(row["unique_mask_count"] >= 28 for row in pools),
        "every_second_draw_at_least_4_novel": all(row["second_draw_novel_count"] >= 4 for row in pools),
        "pooled_agreement_at_least_90_percent": total_agreement >= 1728,
        "every_draw_at_least_18_semantic_valid": all(row["semantic_valid_count"] >= 18 for row in draws),
        "every_pool_at_least_26_semantic_valid": all(row["semantic_valid_unique_count"] >= 26 for row in pools),
        "every_later_history_at_least_12_novel_semantic_valid": all(value >= 12 for value in history_novel),
        "observed_answer_obedience_exact": all(all((row["mask"][number] == "1") is answer for number, answer in HISTORIES[index // 2]) for index, rows in enumerate(proposals_by_draw) for row in rows),
    }
    return {"total_judgments": 1920, "agreement_count": total_agreement, "agreement_rate": total_agreement / 1920, "draws": draws, "pools": pools, "observed_history_novel_counts": history_novel}, gates


__all__ = ["AUDIT_SEEDS", "HISTORIES", "INTERFACE_VERSION", "PROPOSAL_SEEDS", "audit_messages", "audit_response_format", "diagnostics", "parse_audit", "parse_proposal", "proposal_messages", "proposal_response_format", "probes_for"]
