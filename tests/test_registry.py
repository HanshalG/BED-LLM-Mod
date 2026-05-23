"""Tests for the environment/method registry."""

from __future__ import annotations

import pytest

from core import (
    Environment,
    Method,
    build_environment,
    build_method,
    list_environments,
    list_methods,
    register_environment,
    register_method,
)
from core.registry import clear_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with a fresh registry and restores defaults after."""
    clear_registry()
    yield
    # Re-install defaults (force=True wipes + reinstalls) so subsequent test
    # modules see the built-in environments and methods.
    import core.defaults as defaults_module
    defaults_module.register_defaults(force=True)


def test_register_and_build_environment_round_trip():
    sentinel = object()

    def builder(config, questioner, answerer):
        return ("env", config, questioner, answerer, sentinel)

    register_environment("dummy", builder)

    result = build_environment("dummy", "cfg", "Q", "A")

    assert result == ("env", "cfg", "Q", "A", sentinel)


def test_register_method_keyed_by_env_and_method_name():
    register_environment("envA", lambda cfg, q, a: "envA-instance")
    register_environment("envB", lambda cfg, q, a: "envB-instance")
    register_method("envA", "M", lambda cfg: ("envA-M", cfg))
    register_method("envB", "M", lambda cfg: ("envB-M", cfg))

    assert build_method("envA", "M", "cfg-A") == ("envA-M", "cfg-A")
    assert build_method("envB", "M", "cfg-B") == ("envB-M", "cfg-B")


def test_duplicate_environment_registration_raises():
    register_environment("dummy", lambda *args: None)
    with pytest.raises(ValueError, match="already registered"):
        register_environment("dummy", lambda *args: None)


def test_duplicate_method_registration_raises():
    register_environment("dummy", lambda *args: None)
    register_method("dummy", "M", lambda cfg: None)
    with pytest.raises(ValueError, match="already registered"):
        register_method("dummy", "M", lambda cfg: None)


def test_build_environment_raises_for_unknown_env():
    with pytest.raises(KeyError, match="no environment"):
        build_environment("missing", None, None, None)


def test_build_method_raises_for_unknown_env_method_combo():
    register_environment("envA", lambda *args: None)
    register_method("envA", "X", lambda cfg: None)
    with pytest.raises(KeyError, match="envA"):
        build_method("envA", "Y", None)


def test_list_environments_and_methods_returns_sorted():
    register_environment("envA", lambda *args: None)
    register_environment("envB", lambda *args: None)
    register_method("envB", "Naive", lambda cfg: None)
    register_method("envB", "EIG", lambda cfg: None)

    assert list_environments() == ["envA", "envB"]
    assert list_methods("envB") == ["EIG", "Naive"]


def test_defaults_register_both_built_in_environments():
    # Use force=True so we know we're testing a clean install.
    import core.defaults as defaults_module
    defaults_module.register_defaults(force=True)

    assert "animals" in list_environments()
    assert "location_finding" in list_environments()
    assert "hyperbolic_discounting" in list_environments()
    assert list_methods("animals") == [
        "EIG",
        "Entropy",
        "Naive",
        "StrategyEIG",
        "StrategyEIG+root",
        "naive",
        "naive+belief",
        "split",
    ]
    assert list_methods("location_finding") == [
        "EIG",
        "Naive",
        "StrategyEIG",
        "StrategyEIG+root",
        "naive",
        "naive+belief",
    ]
    assert list_methods("hyperbolic_discounting") == [
        "EIG",
        "Naive",
        "StrategyEIG",
        "StrategyEIG+root",
        "naive",
        "naive+belief",
    ]
