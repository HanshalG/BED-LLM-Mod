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
    tomorrow = datetime(2026, 8, 6, 0, 1, tzinfo=ZoneInfo("Europe/London"))
    with pytest.raises(RuntimeError, match="initialize a new daily ledger"):
        budget_status(LEDGER, total_usage_usd=214.0, now=tomorrow)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "0", None])
def test_malformed_exposure_cannot_authorize(bad):
    with pytest.raises(ValueError):
        require_budget(LEDGER, projected_cost_usd=bad, total_usage_usd=213.0, now=NOW)


@pytest.mark.parametrize(
    "field", ["daily_cap_usd", "opening_total_usage_usd", "recorded_actual_spend_usd"]
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "0", None])
def test_malformed_ledger_money_fails_closed(field, bad):
    with pytest.raises(ValueError):
        budget_status({**LEDGER, field: bad}, total_usage_usd=213.0, now=NOW)


def test_negative_usage_delta_is_not_zero_spend():
    with pytest.raises(RuntimeError, match="below frozen opening"):
        budget_status(LEDGER, total_usage_usd=212.49, now=NOW)


def test_no_budget_or_timezone_override():
    for update in ({"daily_cap_usd": 5.01}, {"daily_cap_usd": 0}, {"timezone": "UTC"}):
        with pytest.raises(ValueError):
            budget_status({**LEDGER, **update}, total_usage_usd=213.0, now=NOW)
    with pytest.raises(ValueError, match="timezone aware"):
        budget_status(LEDGER, total_usage_usd=213.0, now=NOW.replace(tzinfo=None))


def test_exact_boundary_and_no_epsilon_overspend():
    ledger = {**LEDGER, "opening_total_usage_usd": 0.1}
    assert require_budget(ledger, projected_cost_usd=4.8, total_usage_usd=0.3, now=NOW)[
        "authorized"
    ]
    with pytest.raises(RuntimeError, match="exceeds"):
        require_budget(
            ledger, projected_cost_usd=4.8000000000001, total_usage_usd=0.3, now=NOW
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, None])
def test_malformed_live_usage(bad):
    with pytest.raises(ValueError):
        budget_status(LEDGER, total_usage_usd=bad, now=NOW)
