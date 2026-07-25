#!/usr/bin/env python3
"""Run the prospectively filtered-particle Zendo confirmation."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.zendo_particle_multiset_confirmation import run_cli


if __name__ == "__main__":
    run_cli(
        interface_version="zendo-filtered-particle-confirmation-1",
        filter_invalid_particles=True,
    )
