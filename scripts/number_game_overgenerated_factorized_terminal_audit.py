#!/usr/bin/env python3
"""Audit terminal closure of overgenerated factorized proposal mechanics."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from scripts.number_game_overgenerated_factorized_codec import _item_reason


BINDING_SHA256 = "8f6bcc82c702d0ab44afa56dd54e558640517c43526c4d5a5659fcd1b63c0d60"
OPENING = 220.134128880


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proposal_signature(raw_responses: list[str]) -> dict[str, object]:
    valid_counts: list[int] = []
    rejection_counts: Counter[str] = Counter()
    for response in raw_responses:
        rows = json.loads(response)["hypotheses"]
        shaped = [row for row in rows if isinstance(row, dict) and set(row) == {"hypothesis_id", "name", "description"}]
        id_counts = Counter(str(row["hypothesis_id"]) for row in shaped)
        name_counts = Counter(" ".join(str(row["name"]).strip().split()).casefold() for row in shaped)
        description_counts = Counter(" ".join(str(row["description"]).strip().split()).casefold() for row in shaped)
        valid = 0
        for row in rows:
            reason, parsed = _item_reason(row, id_counts, name_counts, description_counts)
            if reason is None:
                assert parsed is not None
                valid += 1
            else:
                rejection_counts[reason] += 1
        valid_counts.append(valid)
    per_draw = [sum(valid_counts[4 * draw : 4 * draw + 4]) for draw in range(10)]
    return {
        "requests": len(raw_responses),
        "items": 8 * len(raw_responses),
        "valid_items": sum(valid_counts),
        "invalid_items": 8 * len(raw_responses) - sum(valid_counts),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "per_shard_valid_counts": valid_counts,
        "per_draw_valid_counts": per_draw,
        "below_seven_shards": sum(value < 7 for value in valid_counts),
        "minimum_shard_valid": min(valid_counts),
        "minimum_draw_valid": min(per_draw),
    }


def audit(root: Path, *, output: Path | None = None):
    run = root / "mechanics-20260813"
    raw = json.loads((run / "private/RAW_RESPONSES.json").read_text())
    failure = json.loads((root / "DAILY_FAILURE_20260813.json").read_text())
    ledger_path = root.parent / "openrouter_daily_budget/2026-08-13-number-game-overgenerated-factorized.json"
    ledger = json.loads(ledger_path.read_text())
    events = [
        json.loads(line)
        for line in (run / "run.log").read_text().splitlines()
        if '"event": "llm_token_usage"' in line
    ]
    signature = proposal_signature(raw.get("proposal", []))
    local_cost = sum(float(event.get("cost_usd", 0.0)) for event in events)
    recorded = max(
        float(ledger["closing_total_usage_usd"]) - OPENING,
        float(ledger["execution_opening_total_usage_usd"]) - OPENING + local_cost,
    )
    gates = {
        "binding_ledger": digest(root / "EXECUTION_BINDING.json") == BINDING_SHA256 and failure.get("ledger_sha256") == digest(ledger_path),
        "exact_clean_transport": len(events) == 40 and all(event.get("finish_reasons") == ["stop"] and int(event.get("reasoning_tokens", -1)) == 0 for event in events),
        "proposal_only": len(raw.get("proposal", [])) == 40 and raw.get("translation") == [] and raw.get("audit") == [],
        "aggregate_signature": signature == {
            "requests": 40,
            "items": 320,
            "valid_items": 302,
            "invalid_items": 18,
            "rejection_counts": {"lexical": 4, "observed_answer_language": 14},
            "per_shard_valid_counts": [8, 7, 7, 8, 8, 8, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 7, 7, 8, 8, 7, 7, 8, 0, 8, 8],
            "per_draw_valid_counts": [30, 30, 32, 32, 31, 32, 32, 29, 30, 24],
            "below_seven_shards": 1,
            "minimum_shard_valid": 0,
            "minimum_draw_valid": 24,
        },
        "cost_reconciles": abs(local_cost - float(failure["actual_cost_usd"])) < 1e-12 and abs(recorded - float(ledger["recorded_actual_spend_usd"])) < 1e-12,
        "authority_closed": failure.get("status") == "failed_closed" and failure.get("authorizes") == "nothing" and failure.get("targets_opened") is False and failure.get("endpoints_opened") is False,
    }
    result = {
        "schema_version": 1,
        "interface_version": "number-game-overgenerated-factorized-terminal-audit-1",
        "status": "terminal_audit_pass" if all(gates.values()) else "terminal_audit_failed",
        "decision": "close_exact_overgenerated_factorized_interface",
        "authorizes": "nothing",
        "gates": gates,
        "proposal": {**signature, "translation_requests": 0, "audit_requests": 0, "actual_cost_usd": local_cost},
        "targets_opened": False,
        "endpoints_opened": False,
        "model_calls_made": 0,
    }
    if output:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "terminal_audit_pass":
        raise RuntimeError("overgenerated factorized terminal audit failed")
    return result
