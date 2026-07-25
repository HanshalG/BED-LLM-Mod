#!/usr/bin/env python3
"""Run Bamboogle mechanics with provider-enforced strict JSON Schema."""

from scripts.bamboogle_cached_search_mechanics import run_cli


if __name__ == "__main__":
    run_cli(
        interface_version="bamboogle-cached-search-mechanics-2",
        structured_outputs=True,
    )
