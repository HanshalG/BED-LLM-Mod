from __future__ import annotations

import pytest

from helpers import load_config
from scripts.clindiag_deanchored_deterministic_support_gate import (
    SELECTION_SEED as DETERMINISTIC_SELECTION_SEED,
)
from scripts.clindiag_deanchored_deterministic_support_gate import (
    SMOKE_IDS as DETERMINISTIC_SMOKE_IDS,
)
from scripts.clindiag_deanchored_support_gate import SELECTION_SEED, SMOKE_IDS


def test_deanchored_smoke_split_and_config_are_frozen() -> None:
    assert SELECTION_SEED == 24297
    assert SMOKE_IDS == ("21991897", "rare70")
    config = load_config(
        "configs/config_clindiag_deanchored_support_gate_openrouter.yaml"
    )
    assert config.openrouter_budget_usd == pytest.approx(70.38480269545715)
    assert config.openrouter_run_budget_usd == pytest.approx(0.5)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.15)
    assert config.openrouter_concurrency == 24


def test_deterministic_deanchored_smoke_is_fresh_and_temperature_zero() -> None:
    assert DETERMINISTIC_SELECTION_SEED == 24298
    assert DETERMINISTIC_SMOKE_IDS == ("23697517", "rare216")
    assert not set(DETERMINISTIC_SMOKE_IDS).intersection(SMOKE_IDS)
    config = load_config(
        "configs/config_clindiag_deanchored_deterministic_support_gate_openrouter.yaml"
    )
    assert config.generation_temperature_diverse == pytest.approx(0.0)
    assert config.openrouter_run_budget_usd == pytest.approx(0.5)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.15)
