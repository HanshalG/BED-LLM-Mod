from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from scripts.openrouter_daily_budget import budget_status, require_budget


LEDGER = {
    "date": "2026-08-05",
    "timezone": "Europe/London",
    "daily_cap_usd": 5.0,
    "opening_total_usage_usd": 212.5,
}
NOW = datetime(2026, 8, 5, 12, tzinfo=ZoneInfo("Europe/London"))


def test_budget_status_uses_cumulative_account_usage() -> None:
    status = budget_status(LEDGER, total_usage_usd=214.0, now=NOW)

    assert status["spent_today_usd"] == pytest.approx(1.5)
    assert status["remaining_today_usd"] == pytest.approx(3.5)


def test_budget_authorization_fails_closed_above_remaining() -> None:
    with pytest.raises(RuntimeError, match="exceeds today's remaining"):
        require_budget(
            LEDGER,
            projected_cost_usd=3.51,
            total_usage_usd=214.0,
            now=NOW,
        )


def test_local_accepted_cost_wins_when_provider_posting_lags() -> None:
    ledger = {**LEDGER, "recorded_actual_spend_usd": 2.25}

    status = budget_status(ledger, total_usage_usd=213.0, now=NOW)

    assert status["posted_spend_today_usd"] == pytest.approx(0.5)
    assert status["spent_today_usd"] == pytest.approx(2.25)
    assert status["remaining_today_usd"] == pytest.approx(2.75)


def test_budget_ledger_expires_at_local_midnight() -> None:
    tomorrow = datetime(
        2026, 8, 6, 0, 1, tzinfo=ZoneInfo("Europe/London")
    )
    with pytest.raises(RuntimeError, match="initialize a new daily ledger"):
        budget_status(LEDGER, total_usage_usd=214.0, now=tomorrow)
