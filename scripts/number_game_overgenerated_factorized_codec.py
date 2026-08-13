#!/usr/bin/env python3
"""Pure codec for overgenerated factorized Number Game support."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Any, Sequence

from scripts.number_game_extension_native_semantic_codec import (
    audit_response_format,
    canonical_json,
    description_lexically_valid,
    parse_audit,
)
from scripts.number_game_factorized_semantic_codec import (
    assert_audit_blind,
    assert_translation_blind,
    parse_translation,
    translation_messages,
    translation_response_format,
)


INTERFACE_VERSION = "number-game-overgenerated-factorized-support-1"
HISTORIES = (
    ((13, True),),
    ((13, True), (39, True)),
    ((7, False), (35, True)),
    ((18, True), (19, False)),
    ((23, False), (46, True)),
)
PROPOSAL_SEEDS = tuple(range(202608133300, 202608133340))
TRANSLATION_SEEDS = tuple(range(202608133400, 202608133430))
AUDIT_SEEDS = tuple(range(202608133500, 202608133510))
SHARDS_PER_DRAW = 4
ITEMS_PER_SHARD = 8
SELECTED_PER_DRAW = 24
MIN_VALID_PER_SHARD = 7
MIN_VALID_PER_DRAW = 28


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def proposal_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "number_game_overgenerated_description_shard",
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
    payload = {
        "observations": observations,
        "domain": "integers 0 through 100 inclusive",
        "diversity_shard_nonce": shard,
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": canonical_json(payload)}]


def _item_reason(
    row: Any,
    id_counts: Counter[str],
    name_counts: Counter[str],
    description_counts: Counter[str],
) -> tuple[str | None, dict[str, str] | None]:
    if not isinstance(row, dict) or set(row) != {"hypothesis_id", "name", "description"}:
        return "shape", None
    hypothesis_id = row["hypothesis_id"]
    name = " ".join(str(row["name"]).strip().split())
    description = " ".join(str(row["description"]).strip().split())
    folded_name, folded_description = name.casefold(), description.casefold()
    if hypothesis_id not in {f"H{i}" for i in range(1, 9)} or id_counts[hypothesis_id] != 1:
        return "id", None
    if not name or len(name) > 60 or name_counts[folded_name] != 1:
        return "name", None
    if not description_lexically_valid(description):
        return "lexical", None
    if description_counts[folded_description] != 1:
        return "duplicate_description", None
    if "mask" in folded_description:
        return "mask_language", None
    if re.search(r"\b(?:yes|no|observed|observation|answer)\b", description, re.I):
        return "observed_answer_language", None
    parsed = {
        "hypothesis_id": hypothesis_id,
        "name": name,
        "description": description,
        "description_hash": sha256_text(description),
    }
    return None, parsed


def parse_proposal_shard(raw: str, *, shard_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value = json.loads(raw)
    rows = value.get("hypotheses") if isinstance(value, dict) and set(value) == {"hypotheses"} else None
    if not isinstance(rows, list) or len(rows) != ITEMS_PER_SHARD:
        raise ValueError("proposal shard shape changed")
    accepted: list[dict[str, Any]] = []
    shaped = [row for row in rows if isinstance(row, dict) and set(row) == {"hypothesis_id", "name", "description"}]
    id_counts = Counter(str(row["hypothesis_id"]) for row in shaped)
    name_counts = Counter(" ".join(str(row["name"]).strip().split()).casefold() for row in shaped)
    description_counts = Counter(" ".join(str(row["description"]).strip().split()).casefold() for row in shaped)
    rejected: dict[str, int] = {}
    for row in rows:
        reason, parsed = _item_reason(row, id_counts, name_counts, description_counts)
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        assert parsed is not None
        accepted.append({
            **parsed,
            "source_hypothesis_id": f"S{shard_index + 1}{parsed['hypothesis_id']}",
            "shard_index": shard_index,
            "item_index": int(parsed["hypothesis_id"][1:]) - 1,
        })
    accepted.sort(key=lambda row: row["item_index"])
    diagnostic = {
        "shard_index": shard_index,
        "valid_count": len(accepted),
        "invalid_count": ITEMS_PER_SHARD - len(accepted),
        "rejection_counts": rejected,
    }
    if len(accepted) < MIN_VALID_PER_SHARD:
        raise ValueError("proposal shard retained fewer than seven valid items")
    return accepted, diagnostic


def select_draw(
    shards: Sequence[Sequence[dict[str, Any]]],
    diagnostics: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(shards) != SHARDS_PER_DRAW or len(diagnostics) != SHARDS_PER_DRAW:
        raise ValueError("proposal draw shard coverage changed")
    ordered = sorted(
        (row for shard in shards for row in shard),
        key=lambda row: (row["shard_index"], row["item_index"]),
    )
    if len(ordered) < MIN_VALID_PER_DRAW:
        raise ValueError("proposal draw retained fewer than 28 valid items")
    selected = [{**row, "selection_index": index} for index, row in enumerate(ordered[:SELECTED_PER_DRAW])]
    return selected, {
        "valid_count": len(ordered),
        "invalid_count": SHARDS_PER_DRAW * ITEMS_PER_SHARD - len(ordered),
        "selected_count": len(selected),
        "selected_source_ids": [row["source_hypothesis_id"] for row in selected],
        "shards": list(diagnostics),
    }


def translation_chunks(selected: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if len(selected) != SELECTED_PER_DRAW:
        raise ValueError("selected proposal count changed")
    chunks: list[list[dict[str, Any]]] = []
    for start in range(0, SELECTED_PER_DRAW, ITEMS_PER_SHARD):
        chunks.append([
            {**row, "hypothesis_id": f"H{offset + 1}"}
            for offset, row in enumerate(selected[start : start + ITEMS_PER_SHARD])
        ])
    return chunks


def parse_and_select_proposals(raw_responses: Sequence[str]) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    if len(raw_responses) != 40:
        raise ValueError("proposal response count changed")
    selected_draws: list[list[dict[str, Any]]] = []
    proposal_draw_diagnostics: list[dict[str, Any]] = []
    for draw in range(10):
        shards, shard_diagnostics = [], []
        for shard_index, response in enumerate(raw_responses[4 * draw : 4 * draw + 4]):
            parsed, diagnostic = parse_proposal_shard(response, shard_index=shard_index)
            shards.append(parsed)
            shard_diagnostics.append(diagnostic)
        selected, draw_diagnostic = select_draw(shards, shard_diagnostics)
        selected_draws.append(selected)
        proposal_draw_diagnostics.append(draw_diagnostic)
    return selected_draws, proposal_draw_diagnostics


def merge_draw(
    translated_chunks: Sequence[Sequence[dict[str, Any]]],
    history: Sequence[tuple[int, bool]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len(translated_chunks) != 3 or any(len(chunk) != 8 for chunk in translated_chunks):
        raise ValueError("translation chunk coverage changed")
    rows = []
    for chunk in translated_chunks:
        for row in chunk:
            rows.append({
                **row,
                "history_consistent": all((row["mask"][number] == "1") is answer for number, answer in history),
            })
    unique_extensions = len({row["mask"] for row in rows})
    if unique_extensions != SELECTED_PER_DRAW:
        raise ValueError("translated draw contains duplicate extensions")
    return rows, {
        "translated_unique_count": unique_extensions,
        "history_consistent_count": sum(row["history_consistent"] for row in rows),
    }


def probes_for(draw_index: int, hypothesis_index: int, history: Sequence[tuple[int, bool]]) -> tuple[int, ...]:
    seed = TRANSLATION_SEEDS[3 * draw_index + hypothesis_index // 8]
    probes: list[int] = []
    for probe_index in range(8):
        raw = f"{INTERFACE_VERSION}|{seed}|{hypothesis_index}|{probe_index}".encode()
        candidate = int.from_bytes(hashlib.sha256(raw).digest(), "big") % 101
        while candidate in probes:
            candidate = (candidate + 1) % 101
        probes.append(candidate)
    observed = [number for number, _ in history]
    return tuple([value for value in probes if value not in observed][: 8 - len(observed)] + observed)


def audit_messages(draw: Sequence[dict[str, Any]], draw_index: int, history: Sequence[tuple[int, bool]]) -> list[dict[str, str]]:
    rules = [
        {
            "hypothesis_index": index,
            "name": row["name"],
            "description": row["description"],
            "probe_integers": list(probes_for(draw_index, index, history)),
        }
        for index, row in enumerate(draw)
    ]
    system = (
        "Independently interpret each ordinary-language subset rule. Return only required JSON. For each probe "
        "integer report true exactly when it satisfies the description. Use only names, descriptions, and probe "
        "integers. Do not infer hidden observations, desired answers, masks, or translated extensions."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": canonical_json({"rules": rules})}]


def diagnostics(
    draws: Sequence[Sequence[dict[str, Any]]],
    audits: Sequence[Sequence[Sequence[bool]]],
    proposal_draws: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    if len(draws) != 10 or len(audits) != 10 or len(proposal_draws) != 10:
        raise ValueError("ten complete draws are required")
    total_agreement = 0
    valid_by_draw: list[set[str]] = []
    consistent_by_draw: list[set[str]] = []
    draw_rows = []
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
        draw_rows.append({
            "draw_index": draw_index,
            "history_index": draw_index // 2,
            "proposal_valid_count": proposal_draws[draw_index]["valid_count"],
            "proposal_invalid_count": proposal_draws[draw_index]["invalid_count"],
            "proposal_shard_valid_counts": [row["valid_count"] for row in proposal_draws[draw_index]["shards"]],
            "selected_source_ids": proposal_draws[draw_index]["selected_source_ids"],
            "translated_unique_count": len({row["extension_hash"] for row in draw}),
            "history_consistent_count": len(consistent),
            "semantic_valid_consistent_count": len(valid),
            "extension_hashes": [row["extension_hash"] for row in draw],
            "agreement_count": sum(scores),
            "agreement_histogram": {str(score): scores.count(score) for score in range(9)},
        })
    pools = []
    for history_index in range(5):
        left, right = consistent_by_draw[2 * history_index : 2 * history_index + 2]
        valid_left, valid_right = valid_by_draw[2 * history_index : 2 * history_index + 2]
        pools.append({
            "history_index": history_index,
            "history_consistent_unique_count": len(left | right),
            "second_draw_novel_count": len(right - left),
            "semantic_valid_unique_count": len(valid_left | valid_right),
        })
    baseline = valid_by_draw[0] | valid_by_draw[1]
    novel = [len((valid_by_draw[2 * h] | valid_by_draw[2 * h + 1]) - baseline) for h in range(1, 5)]
    gates = {
        "every_proposal_shard_at_least_7_valid": all(all(value >= 7 for value in row["proposal_shard_valid_counts"]) for row in draw_rows),
        "every_proposal_draw_at_least_28_valid": all(row["proposal_valid_count"] >= 28 for row in draw_rows),
        "every_draw_exactly_24_unique_translations": all(row["translated_unique_count"] == 24 for row in draw_rows),
        "every_draw_at_least_20_history_consistent": all(row["history_consistent_count"] >= 20 for row in draw_rows),
        "every_pool_at_least_28_history_consistent_unique": all(row["history_consistent_unique_count"] >= 28 for row in pools),
        "every_second_draw_at_least_4_novel": all(row["second_draw_novel_count"] >= 4 for row in pools),
        "pooled_agreement_at_least_90_percent": total_agreement >= 1728,
        "every_draw_at_least_18_semantic_valid_consistent": all(row["semantic_valid_consistent_count"] >= 18 for row in draw_rows),
        "every_pool_at_least_26_semantic_valid": all(row["semantic_valid_unique_count"] >= 26 for row in pools),
        "every_later_history_at_least_12_novel_semantic_valid": all(value >= 12 for value in novel),
        "accepted_extensions_obey_history_exactly": all(
            all((row["mask"][n] == "1") is y for n, y in HISTORIES[d // 2])
            for d, draw in enumerate(draws)
            for row in draw
            if row["history_consistent"]
        ),
    }
    return {
        "total_judgments": 1920,
        "agreement_count": total_agreement,
        "agreement_rate": total_agreement / 1920,
        "draws": draw_rows,
        "pools": pools,
        "observed_history_novel_counts": novel,
    }, gates
