#!/usr/bin/env python3
"""Verify Zendo JSON Schema through OpenRouter's Responses API."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.zendo_structured_output_serving_smoke import run_cli


if __name__ == "__main__":
    run_cli(
        interface_version="zendo-responses-structured-output-serving-1",
        use_responses_api=True,
    )
