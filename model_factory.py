"""Lazy model-backend factory so API inference does not import the GPU stack."""

from __future__ import annotations

from typing import Any

from helpers import Config, ModelSpec


def build_model_adapter(
    spec: ModelSpec,
    config: Config,
    tensor_parallel_size: int | None = None,
    dtype: str = "bfloat16",
) -> Any:
    if spec.backend == "openrouter":
        from openrouter_model import OpenRouterAdapter

        return OpenRouterAdapter(spec=spec, config=config)
    from model import build_model_adapter as build_vllm_adapter

    if tensor_parallel_size is None and dtype == "bfloat16":
        return build_vllm_adapter(spec, config=config)
    return build_vllm_adapter(spec, config=config, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
