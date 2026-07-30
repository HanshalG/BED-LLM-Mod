"""Deterministic seed fallback for zero-cost OpenRouter provider errors."""

from __future__ import annotations

from threading import Lock
from typing import Any, Callable, MutableSequence, Sequence


PROVIDER_ERROR_MESSAGE = "provider-error response persisted after retries"
DEFAULT_FALLBACK_OFFSETS = (10_000_000, 20_000_000)


def install_provider_seed_fallback(
    adapter: Any,
    *,
    events: MutableSequence[dict[str, Any]],
    event_lock: Lock,
    fallback_offsets: Sequence[int] = DEFAULT_FALLBACK_OFFSETS,
) -> Any:
    """Retry an exhausted zero-cost provider error with frozen seed offsets."""
    if not fallback_offsets:
        raise ValueError("fallback_offsets must be non-empty")
    if any(int(offset) <= 0 for offset in fallback_offsets):
        raise ValueError("fallback offsets must be positive")
    if len(set(int(offset) for offset in fallback_offsets)) != len(
        fallback_offsets
    ):
        raise ValueError("fallback offsets must be unique")

    original_post = adapter._post

    def fallback_post(payload: dict[str, Any]) -> dict[str, Any]:
        if "seed" not in payload:
            return original_post(payload)
        original_seed = int(payload["seed"])
        seeds = [original_seed] + [
            original_seed + int(offset) for offset in fallback_offsets
        ]
        for group_index, seed in enumerate(seeds):
            request_payload = {**payload, "seed": seed}
            try:
                return original_post(request_payload)
            except RuntimeError as exc:
                is_exhausted_provider_error = (
                    str(exc) == PROVIDER_ERROR_MESSAGE
                )
                if not is_exhausted_provider_error or group_index >= (
                    len(seeds) - 1
                ):
                    raise
                next_seed = seeds[group_index + 1]
                # The transition to a new seed is one more retry after the
                # exhausted group's final zero-cost HTTP response.
                with adapter._usage_lock:
                    adapter.retry_count += 1
                    adapter.provider_error_retries += 1
                event = {
                    "model": adapter.model_name,
                    "original_seed": original_seed,
                    "exhausted_seed": seed,
                    "fallback_seed": next_seed,
                    "fallback_group": group_index + 1,
                }
                with event_lock:
                    events.append(event)
        raise AssertionError("unreachable")

    adapter._post = fallback_post
    return adapter


def fallback_adapter_factory(
    original_factory: Callable[..., Any],
    *,
    events: MutableSequence[dict[str, Any]],
    event_lock: Lock,
    fallback_offsets: Sequence[int] = DEFAULT_FALLBACK_OFFSETS,
) -> Callable[..., Any]:
    """Wrap every adapter produced by ``original_factory``."""

    def factory(**kwargs: Any) -> Any:
        adapter = original_factory(**kwargs)
        return install_provider_seed_fallback(
            adapter,
            events=events,
            event_lock=event_lock,
            fallback_offsets=fallback_offsets,
        )

    return factory
