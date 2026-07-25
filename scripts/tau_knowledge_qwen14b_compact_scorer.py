#!/usr/bin/env python3
"""Rescore frozen tau-Knowledge trees with compact thinking-Qwen outputs."""

from __future__ import annotations

from scripts.tau_knowledge_gemma26b_compact_scorer import main


MODEL_ID = "qwen/qwen3-14b"
INTERFACE_VERSION = "qwen3-14b-thinking-compact-1"
THINKING_MAX_NEW_TOKENS = 7680
THINKING_FINAL_MAX_NEW_TOKENS = 512


if __name__ == "__main__":
    main(
        model_id=MODEL_ID,
        interface_version=INTERFACE_VERSION,
        thinking_max_new_tokens=THINKING_MAX_NEW_TOKENS,
        thinking_final_max_new_tokens=THINKING_FINAL_MAX_NEW_TOKENS,
    )
