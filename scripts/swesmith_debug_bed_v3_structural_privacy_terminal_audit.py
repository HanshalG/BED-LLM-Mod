#!/usr/bin/env python3
"""Audit terminal closure of the invalid V3 structural payload projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(root: Path) -> dict[str, bool]:
    result = json.loads((root / "results/nonmyopic/swesmith_debug_bed_v3_structural_privacy_terminal/TERMINAL_RESULT.json").read_text())
    implementation = (root / "scripts/swesmith_debug_bed_v3_structural_screen.py").read_text()
    return {
        "exact_terminal_identity": result.get("status") == "privacy_ordering_failed_closed"
        and result.get("decision") == "close_exact_swesmith_debug_bed_v3_structural_cohort"
        and result.get("authorizes") == "nothing",
        "violation_present": "pq.read_table(data_root / rel)" in implementation
        and "columns=" not in implementation.split("pq.read_table(data_root / rel)", 1)[0].rsplit("\n", 1)[-1],
        "exact_accounting": result.get("accounting") == {
            "oatml_cluster_use": 0, "openrouter_calls": 0, "openrouter_cost_usd": 0.0
        },
        "execution_never_opened": result.get("ordering") == {
            "mechanics_workflow_pushed": False, "model_calls_opened": False,
            "native_task_execution_opened": False, "patch_endpoints_opened": False,
            "planner_opened": False,
        },
        "no_mechanics_result": not (root / "results/nonmyopic/swesmith_debug_bed_v3_mechanics/MECHANICS_RESULT.json").exists(),
        "source_admission_preserved": json.loads((root / "results/nonmyopic/swesmith_debug_bed_source_v3/SOURCE_AUDIT.json").read_text()).get("status") == "source_pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    gates = audit(args.root)
    print(json.dumps(gates, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
