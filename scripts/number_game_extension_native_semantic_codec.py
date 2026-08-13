#!/usr/bin/env python3
"""Pure codec and gate math for extension-native Number Game support."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Sequence


INTERFACE_VERSION = "number-game-extension-native-semantic-support-1"
DOMAIN = tuple(range(101))
HYPOTHESES_PER_DRAW = 24
PROBES_PER_HYPOTHESIS = 8
HISTORIES: tuple[tuple[tuple[int, bool], ...], ...] = (
    (),
    ((8, True),),
    ((8, True), (16, True)),
    ((7, False), (21, True)),
    ((10, True), (11, False)),
)
PROPOSAL_SEEDS = tuple(range(202608132400, 202608132410))
AUDIT_SEEDS = tuple(range(202608132500, 202608132510))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_json(value: Any) -> str:
    return digest_bytes(canonical_json(value).encode("utf-8"))


def extension_hash(members: Sequence[int]) -> str:
    return digest_json(list(members))


def proposal_response_format() -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "description", "members"],
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 60},
            "description": {"type": "string", "minLength": 1, "maxLength": 180},
            "members": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 100},
                "minItems": 1,
                "maxItems": 100,
            },
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "number_game_semantic_extensions",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "items": item,
                        "minItems": HYPOTHESES_PER_DRAW,
                        "maxItems": HYPOTHESES_PER_DRAW,
                    }
                },
            },
        },
    }


def history_payload(history: Sequence[tuple[int, bool]]) -> list[dict[str, Any]]:
    return [{"number": number, "answer": "YES" if answer else "NO"} for number, answer in history]


def proposal_messages(history: Sequence[tuple[int, bool]]) -> list[dict[str, str]]:
    system = (
        "Propose exactly 24 distinct general semantic rules for subsets of the integers 0 through 100. "
        "Return only the required JSON. Each description must state one coherent reusable rule in ordinary "
        "language. Do not use code, formulas copied as code, explicit member lists, lookup tables, exception "
        "lists, or references to the members field. The members field must be the exact sorted extension of "
        "the description over 0..100. Every rule must agree with every observation."
    )
    user = canonical_json({"observations": history_payload(history), "domain": {"minimum": 0, "maximum": 100}})
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_CODE = re.compile(r"(?:==|!=|<=|>=|\b(?:lambda|def|return|python|javascript|regex|bitmask|lookup table)\b|[%{}\[\];])", re.I)
_ENUMERATION = re.compile(r"(?:\bmembers?\s+(?:are|is|:)\b|\b(?:exactly|only)\s*[:{\[]|(?:\b\d{1,3}\b\s*[,;]\s*){3,}\b\d{1,3}\b)", re.I)


def description_lexically_valid(description: str) -> bool:
    text = " ".join(description.strip().split())
    return bool(text) and len(text) <= 180 and _CODE.search(text) is None and _ENUMERATION.search(text) is None


def parse_proposal(raw: str, history: Sequence[tuple[int, bool]]) -> list[dict[str, Any]]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses"}:
        raise ValueError("proposal top-level shape changed")
    rows = value["hypotheses"]
    if not isinstance(rows, list) or len(rows) != HYPOTHESES_PER_DRAW:
        raise ValueError("proposal must contain exactly 24 hypotheses")
    parsed: list[dict[str, Any]] = []
    names: set[str] = set()
    extensions: set[tuple[int, ...]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"name", "description", "members"}:
            raise ValueError(f"proposal hypothesis {index} shape changed")
        name = " ".join(str(row["name"]).strip().split())
        description = " ".join(str(row["description"]).strip().split())
        members = row["members"]
        normalized_name = name.casefold()
        if not name or len(name) > 60 or normalized_name in names:
            raise ValueError(f"proposal hypothesis {index} name is invalid")
        if not description_lexically_valid(description):
            raise ValueError(f"proposal hypothesis {index} description fails lexical gate")
        if (
            not isinstance(members, list)
            or not members
            or len(members) >= len(DOMAIN)
            or any(isinstance(x, bool) or not isinstance(x, int) or x not in DOMAIN for x in members)
            or members != sorted(set(members))
        ):
            raise ValueError(f"proposal hypothesis {index} extension is invalid")
        extension = tuple(members)
        if extension in extensions:
            raise ValueError(f"proposal hypothesis {index} duplicates an extension")
        member_set = set(extension)
        if any((number in member_set) is not answer for number, answer in history):
            raise ValueError(f"proposal hypothesis {index} contradicts the history")
        names.add(normalized_name)
        extensions.add(extension)
        parsed.append({"name": name, "description": description, "members": extension, "extension_hash": extension_hash(extension)})
    return parsed


def probes_for(proposal_seed: int, hypothesis_index: int, history: Sequence[tuple[int, bool]]) -> tuple[int, ...]:
    probes: list[int] = []
    for probe_index in range(PROBES_PER_HYPOTHESIS):
        raw = f"{INTERFACE_VERSION}|{proposal_seed}|{hypothesis_index}|{probe_index}".encode()
        candidate = int.from_bytes(hashlib.sha256(raw).digest(), "big") % len(DOMAIN)
        while candidate in probes:
            candidate = (candidate + 1) % len(DOMAIN)
        probes.append(candidate)
    observed = [number for number, _ in history]
    keep = PROBES_PER_HYPOTHESIS - len(observed)
    prefix = [value for value in probes if value not in observed][:keep]
    return tuple(prefix + observed)


def audit_response_format() -> dict[str, Any]:
    row = {
        "type": "object",
        "additionalProperties": False,
        "required": ["hypothesis_index", "memberships"],
        "properties": {
            "hypothesis_index": {"type": "integer", "minimum": 0, "maximum": 23},
            "memberships": {
                "type": "array",
                "items": {"type": "boolean"},
                "minItems": PROBES_PER_HYPOTHESIS,
                "maxItems": PROBES_PER_HYPOTHESIS,
            },
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "number_game_semantic_membership_audit",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["judgments"],
                "properties": {
                    "judgments": {
                        "type": "array",
                        "items": row,
                        "minItems": HYPOTHESES_PER_DRAW,
                        "maxItems": HYPOTHESES_PER_DRAW,
                    }
                },
            },
        },
    }


def audit_messages(proposals: Sequence[dict[str, Any]], proposal_seed: int, history: Sequence[tuple[int, bool]]) -> list[dict[str, str]]:
    rows = []
    for index, proposal in enumerate(proposals):
        rows.append({
            "hypothesis_index": index,
            "name": proposal["name"],
            "description": proposal["description"],
            "probe_integers": list(probes_for(proposal_seed, index, history)),
        })
    system = (
        "Independently interpret each ordinary-language subset rule. Return only the required JSON. "
        "For each probe integer, report true exactly when that integer satisfies the description. "
        "Use only the name, description, and probes shown; do not infer a hidden list or revise the rule."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": canonical_json({"rules": rows})}]


def parse_audit(raw: str) -> list[tuple[bool, ...]]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"judgments"}:
        raise ValueError("audit top-level shape changed")
    rows = value["judgments"]
    if not isinstance(rows, list) or len(rows) != HYPOTHESES_PER_DRAW:
        raise ValueError("audit must contain exactly 24 judgments")
    found: dict[int, tuple[bool, ...]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"hypothesis_index", "memberships"}:
            raise ValueError("audit judgment shape changed")
        index, memberships = row["hypothesis_index"], row["memberships"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 24 or index in found:
            raise ValueError("audit hypothesis index is invalid")
        if not isinstance(memberships, list) or len(memberships) != 8 or any(not isinstance(x, bool) for x in memberships):
            raise ValueError("audit memberships are invalid")
        found[index] = tuple(memberships)
    if set(found) != set(range(24)):
        raise ValueError("audit coverage is incomplete")
    return [found[index] for index in range(24)]


def semantic_diagnostics(proposals_by_draw: Sequence[Sequence[dict[str, Any]]], audits_by_draw: Sequence[Sequence[Sequence[bool]]]) -> tuple[dict[str, Any], dict[str, bool]]:
    if len(proposals_by_draw) != 10 or len(audits_by_draw) != 10:
        raise ValueError("semantic diagnostics require ten complete draws")
    draw_rows = []
    total_agree = 0
    valid_extensions_by_draw: list[set[str]] = []
    for draw_index, (proposals, audits) in enumerate(zip(proposals_by_draw, audits_by_draw, strict=True)):
        history = HISTORIES[draw_index // 2]
        agreements = []
        valid: set[str] = set()
        for hypothesis_index, (proposal, judgments) in enumerate(zip(proposals, audits, strict=True)):
            probes = probes_for(PROPOSAL_SEEDS[draw_index], hypothesis_index, history)
            members = set(proposal["members"])
            agreement = sum(judgment is (probe in members) for judgment, probe in zip(judgments, probes, strict=True))
            agreements.append(agreement)
            total_agree += agreement
            if agreement >= 7:
                valid.add(proposal["extension_hash"])
        valid_extensions_by_draw.append(valid)
        draw_rows.append({"draw_index": draw_index, "history_index": draw_index // 2, "agreement_count": sum(agreements), "semantic_valid_count": len(valid), "agreement_histogram": {str(score): agreements.count(score) for score in range(9)}})
    pools = []
    for history_index in range(5):
        left, right = valid_extensions_by_draw[2 * history_index : 2 * history_index + 2]
        pools.append({"history_index": history_index, "semantic_valid_unique_count": len(left | right), "second_draw_novel_count": len(right - left)})
    no_observation = valid_extensions_by_draw[0] | valid_extensions_by_draw[1]
    novel_counts = [len((valid_extensions_by_draw[2*i] | valid_extensions_by_draw[2*i+1]) - no_observation) for i in range(1, 5)]
    gates = {
        "pooled_auditor_agreement_at_least_90_percent": total_agree >= 1728,
        "every_draw_has_at_least_18_semantic_valid": all(row["semantic_valid_count"] >= 18 for row in draw_rows),
        "every_pool_has_at_least_26_semantic_valid": all(row["semantic_valid_unique_count"] >= 26 for row in pools),
        "every_observed_history_has_at_least_12_novel_semantic_valid": all(value >= 12 for value in novel_counts),
        "observed_answer_obedience_is_exact": all(all((number in set(proposal["members"])) is answer for number, answer in HISTORIES[draw_index // 2]) for draw_index, proposals in enumerate(proposals_by_draw) for proposal in proposals),
    }
    summary = {"total_judgments": 1920, "agreement_count": total_agree, "agreement_rate": total_agree / 1920, "draws": draw_rows, "pools": pools, "observed_history_novel_counts": novel_counts}
    return summary, gates
