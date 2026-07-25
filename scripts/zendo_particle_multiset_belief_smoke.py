#!/usr/bin/env python3
"""Run the fresh-task Zendo particle-multiset final-readiness smoke."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.zendo_final_readiness_belief_smoke import run_cli


INTERFACE_VERSION = "zendo-particle-multiset-belief-1"
TASK_NAME = "mu"
SELECTION_SEED = 24370


if __name__ == "__main__":
    run_cli(
        interface_version=INTERFACE_VERSION,
        task_name=TASK_NAME,
        selection_seed=SELECTION_SEED,
        allow_duplicate_particles=True,
        audit_mode="random_only",
    )
