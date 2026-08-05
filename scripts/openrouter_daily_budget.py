#!/usr/bin/env python3
"""Enforce a calendar-day OpenRouter spend cap from cumulative usage."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


CREDITS_URL = "https://openrouter.ai/api/v1/credits"


def read_live_credits() -> dict[str, float]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    request = Request(
        CREDITS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)["data"]
    total_credits = float(payload["total_credits"])
    total_usage = float(payload["total_usage"])
    return {
        "total_credits_usd": total_credits,
        "total_usage_usd": total_usage,
        "balance_usd": total_credits - total_usage,
    }


def budget_status(
    ledger: dict[str, Any],
    *,
    total_usage_usd: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    timezone = ZoneInfo(str(ledger["timezone"]))
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    ledger_date = str(ledger["date"])
    if local_now.date().isoformat() != ledger_date:
        raise RuntimeError(
            f"ledger date {ledger_date} is not current in {timezone.key}; "
            "initialize a new daily ledger"
        )
    opening_usage = float(ledger["opening_total_usage_usd"])
    cap = float(ledger["daily_cap_usd"])
    posted_spend = max(0.0, total_usage_usd - opening_usage)
    recorded_spend = float(ledger.get("recorded_actual_spend_usd", 0.0))
    spent = max(posted_spend, recorded_spend)
    return {
        "date": ledger_date,
        "timezone": timezone.key,
        "daily_cap_usd": cap,
        "opening_total_usage_usd": opening_usage,
        "current_total_usage_usd": total_usage_usd,
        "posted_spend_today_usd": posted_spend,
        "recorded_spend_today_usd": recorded_spend,
        "spent_today_usd": spent,
        "remaining_today_usd": max(0.0, cap - spent),
    }


def require_budget(
    ledger: dict[str, Any],
    *,
    projected_cost_usd: float,
    total_usage_usd: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    if projected_cost_usd < 0:
        raise ValueError("projected cost must be non-negative")
    status = budget_status(ledger, total_usage_usd=total_usage_usd, now=now)
    if projected_cost_usd > status["remaining_today_usd"] + 1e-12:
        raise RuntimeError(
            f"projected ${projected_cost_usd:.6f} exceeds today's remaining "
            f"OpenRouter allowance ${status['remaining_today_usd']:.6f}"
        )
    status["projected_cost_usd"] = projected_cost_usd
    status["authorized"] = True
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--projected-cost-usd", type=float, required=True)
    args = parser.parse_args()
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    live = read_live_credits()
    status = require_budget(
        ledger,
        projected_cost_usd=args.projected_cost_usd,
        total_usage_usd=live["total_usage_usd"],
    )
    status.update(live)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
