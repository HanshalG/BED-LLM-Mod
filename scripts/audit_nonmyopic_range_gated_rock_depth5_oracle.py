"""Independent replay of the focused-prior range-gated h5 qualification."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_nonmyopic_range_gated_rock_depth4_fixed_tail import _close
from scripts.nonmyopic_range_gated_rock_depth5_oracle import (
    FocusedDepth5Config,
    run_qualification,
)
from scripts.nonmyopic_rock_depth_oracle import _comparison


def _without_ci(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in comparison.items()
        if not key.endswith("_ci95")
    }


def _json_normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def audit_result(
    result: dict[str, Any],
    *,
    audit_bootstrap_seed: int = 24_236,
) -> dict[str, Any]:
    config_keys = {field.name for field in fields(FocusedDepth5Config)}
    raw_config = {
        key: value
        for key, value in result["config"].items()
        if key in config_keys
    }
    raw_config["start_position"] = tuple(raw_config["start_position"])
    config = FocusedDepth5Config(**raw_config)
    config.validate()
    replay = run_qualification(config)
    audit_comparison = _comparison(
        replay["traces"]["5"],
        replay["traces"]["4"],
        config=replace(config, seed=audit_bootstrap_seed),
        label="focused-range-gated-d5-minus-d4",
    )
    endpoint_gate = {
        "entropy_auc_lower_bound_positive": audit_comparison[
            "entropy_auc_gain_ci95"
        ][0]
        > 0.0,
        "truth_log_auc_lower_bound_positive": audit_comparison[
            "truth_log_probability_auc_gain_ci95"
        ][0]
        > 0.0,
    }
    mechanics = {
        "all_traces_replayed": _close(replay["traces"], result["traces"]),
        "all_truth_indices_replayed": replay["truth_indices"]
        == result["truth_indices"],
        "source_prior_recomputed": _close(
            _json_normalize(replay["source"]),
            _json_normalize(result["source"]),
        ),
        "initial_values_recomputed": _close(
            replay["initial_action_values"],
            result["initial_action_values"],
        ),
        "producer_comparison_recomputed": _close(
            replay["comparison"],
            result["comparison"],
        ),
        "audit_values_match_without_ci": _close(
            _without_ci(audit_comparison),
            _without_ci(result["comparison"]),
        ),
        "producer_mechanics_recomputed": replay["mechanics"]
        == result["mechanics"],
        "producer_primary_gate_passed": bool(result["primary_gate_passed"]),
        "producer_truth_gate_passed": bool(
            result["truth_log_corroboration_passed"]
        ),
    }
    return {
        "schema_version": 1,
        "stage": "focused_range_gated_rock_depth5_exact_qualification_audit",
        "source_stage": result["stage"],
        "audit_bootstrap_seed": audit_bootstrap_seed,
        "mechanics": mechanics,
        "comparison": audit_comparison,
        "endpoint_gate": endpoint_gate,
        "gate": {
            "passed": all(mechanics.values()) and all(endpoint_gate.values())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_json", type=Path)
    parser.add_argument("--audit-bootstrap-seed", type=int, default=24_236)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.report_json.read_text(encoding="utf-8"))
    audit = audit_result(
        result,
        audit_bootstrap_seed=args.audit_bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate": audit["gate"],
                "endpoint_gate": audit["endpoint_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
