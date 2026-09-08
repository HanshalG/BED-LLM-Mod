#!/usr/bin/env python3
"""Enforce a calendar-day OpenRouter spend cap from cumulative usage."""

from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


CREDITS_URL = "https://openrouter.ai/api/v1/credits"


def _money(value: Any, name: str) -> Decimal:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid {name}") from exc
    if not number.is_finite() or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


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
    total_credits = _money(payload["total_credits"], "total credits")
    total_usage = _money(payload["total_usage"], "total usage")
    return {
        "total_credits_usd": float(total_credits),
        "total_usage_usd": float(total_usage),
        "balance_usd": float(total_credits - total_usage),
    }


def budget_status(
    ledger: dict[str, Any],
    *,
    total_usage_usd: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    if ledger["timezone"] != "Europe/London":
        raise ValueError("daily budget boundary must use Europe/London")
    if now is not None and (now.tzinfo is None or now.utcoffset() is None):
        raise ValueError("budget clock must be timezone aware")
    timezone = ZoneInfo("Europe/London")
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    ledger_date = str(ledger["date"])
    if local_now.date().isoformat() != ledger_date:
        raise RuntimeError(
            f"ledger date {ledger_date} is not current in {timezone.key}; "
            "initialize a new daily ledger"
        )
    opening_usage = _money(ledger["opening_total_usage_usd"], "opening usage")
    current_usage = _money(total_usage_usd, "current usage")
    cap = _money(ledger["daily_cap_usd"], "daily cap")
    if not 0 < cap <= 5:
        raise ValueError("daily cap must be positive and no more than $5")
    if current_usage < opening_usage:
        raise RuntimeError(
            "current usage is below frozen opening; revalidate account boundary"
        )
    posted_spend = current_usage - opening_usage
    recorded_spend = _money(
        ledger.get("recorded_actual_spend_usd", 0.0), "recorded spend"
    )
    spent = max(posted_spend, recorded_spend)
    return {
        "date": ledger_date,
        "timezone": timezone.key,
        "daily_cap_usd": float(cap),
        "opening_total_usage_usd": float(opening_usage),
        "current_total_usage_usd": float(current_usage),
        "posted_spend_today_usd": float(posted_spend),
        "recorded_spend_today_usd": float(recorded_spend),
        "spent_today_usd": float(spent),
        "remaining_today_usd": float(max(Decimal(0), cap - spent)),
    }


def require_budget(
    ledger: dict[str, Any],
    *,
    projected_cost_usd: float,
    total_usage_usd: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    projected = _money(projected_cost_usd, "projected cost")
    status = budget_status(ledger, total_usage_usd=total_usage_usd, now=now)
    # Compare in decimal before display conversion; no epsilon may expand the cap.
    posted = _money(total_usage_usd, "current usage") - _money(
        ledger["opening_total_usage_usd"], "opening usage"
    )
    spent = max(
        posted, _money(ledger.get("recorded_actual_spend_usd", 0.0), "recorded spend")
    )
    remaining = max(Decimal(0), _money(ledger["daily_cap_usd"], "daily cap") - spent)
    if projected > remaining:
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
