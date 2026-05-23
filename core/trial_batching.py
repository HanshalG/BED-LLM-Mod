"""Cross-trial batching configuration and dispatch helpers."""

from __future__ import annotations

from typing import Any


def trial_batch_size(config: Any, env_name: str) -> int:
    """Return configured cross-trial batch size for ``env_name``."""
    if env_name == "location_finding":
        return int(getattr(config, "location_trial_batch_size", 1) or 1)
    if env_name == "hyperbolic_discounting":
        return int(getattr(config, "htd_trial_batch_size", 1) or 1)
    return int(getattr(config, "trial_batch_size", 1) or 1)


def supports_trial_batching(environment: Any, batch_size: int) -> bool:
    """Whether ``environment`` can run with ``batch_size > 1``."""
    if batch_size <= 1:
        return True
    return callable(getattr(environment, "run_batched_experiment", None))
