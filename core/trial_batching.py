"""Cross-trial batching configuration and dispatch helpers."""

from __future__ import annotations

from typing import Any


def trial_batch_size(config: Any, env_name: str) -> int:
    """Return configured cross-trial batch size for ``env_name``."""
    environment = getattr(config, "environment", {}) or {}
    if isinstance(environment, dict) and "trial_batch_size" in environment:
        return int(environment["trial_batch_size"] or 1)
    if env_name == "location_finding":
        return int(getattr(config, "location_trial_batch_size", 1) or 1)
    if env_name == "paprika_customer_service":
        return int(getattr(config, "paprika_trial_batch_size", 1) or 1)
    return int(getattr(config, "trial_batch_size", 1) or 1)


def supports_trial_batching(environment: Any, batch_size: int) -> bool:
    """Whether ``environment`` can run with ``batch_size > 1``."""
    return batch_size >= 1
