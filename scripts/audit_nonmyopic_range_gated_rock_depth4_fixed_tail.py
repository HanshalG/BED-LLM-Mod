"""Independent replay of the corner-start range-gated h4 proposal gate."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.nonmyopic_range_gated_rock_depth4_fixed_tail import (
    RangeGatedDepth4ProposalConfig,
    _score_cell,
    _summary,
    build_depth4_model,
)
from scripts.nonmyopic_range_gated_rock_strategy import enumerate_legal_plans


def _close(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isclose(left, right, atol=atol, rtol=0.0))
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(a, b, atol=atol) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _close(left[key], right[key], atol=atol) for key in left
        )
    return left == right


def _comparison_payload(
    records: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    kwargs = {
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
    }
    return {
        "llm_minus_matched_random": _summary(
            [row["llm_value"] - row["matched_random_value"] for row in records],
            label="llm-minus-matched-random",
            **kwargs,
        ),
        "llm_minus_shared_h3": _summary(
            [
                row["llm_value"] - row["shared_h3_selected_full_value"]
                for row in records
            ],
            label="llm-minus-shared-h3",
            **kwargs,
        ),
        "llm_minus_strong_d3": _summary(
            [row["llm_value"] - row["strong_d3_h4_value"] for row in records],
            label="llm-minus-strong-d3",
            **kwargs,
        ),
        "recovery_fraction": _summary(
            [row["recovery_fraction"] for row in records],
            label="recovery-fraction",
            **kwargs,
        ),
        "exact_h4_root_selection_rate": float(
            np.mean(
                [
                    row["llm_selected_plan"][0] == row["exact_h4_plan"][0]
                    for row in records
                ]
            )
        ),
        "exact_h4_route_selection_rate": float(
            np.mean(
                [row["llm_selected_plan"] == row["exact_h4_plan"] for row in records]
            )
        ),
    }


def _without_ci(comparisons: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            {field: value for field, value in row.items() if field != "ci95"}
            if isinstance(row, dict)
            else row
        )
        for key, row in comparisons.items()
    }


def audit_result(
    result: dict[str, Any],
    *,
    audit_bootstrap_seed: int = 24_211,
) -> dict[str, Any]:
    config_keys = {field.name for field in fields(RangeGatedDepth4ProposalConfig)}
    config = RangeGatedDepth4ProposalConfig(
        **{
            key: value
            for key, value in result["config"].items()
            if key in config_keys
        }
    )
    config.validate()
    model = build_depth4_model()
    position = model.map_spec.start_position
    exhaustive_plans = enumerate_legal_plans(
        model, position=position, horizon=4
    )
    replayed: list[dict[str, Any]] = []
    for source in result["records"]:
        belief = model.initial_belief.copy()
        history = tuple(tuple(item) for item in source["history"])
        for action, outcome in history:
            belief = model.posterior(position, belief, action, outcome)
        replayed.append(
            _score_cell(
                model,
                cell_index=int(source["cell_index"]),
                belief=belief,
                history=history,
                llm_plans=tuple(tuple(plan) for plan in source["llm_plans"]),
                config=config,
                exhaustive_plans=exhaustive_plans,
            )
        )
    producer_recomputed = _comparison_payload(
        replayed,
        bootstrap_seed=config.bootstrap_seed,
        bootstrap_replicates=config.bootstrap_replicates,
    )
    audit_comparisons = _comparison_payload(
        replayed,
        bootstrap_seed=audit_bootstrap_seed,
        bootstrap_replicates=config.bootstrap_replicates,
    )
    request_plans = {
        request["cell_index"]: request["compiled_plans"]
        for request in result["candidate_requests"]
    }
    endpoint_gate = {
        "matched_random_lower_bound_positive": audit_comparisons[
            "llm_minus_matched_random"
        ]["ci95"][0]
        > 0.0,
        "shared_h3_lower_bound_positive": audit_comparisons[
            "llm_minus_shared_h3"
        ]["ci95"][0]
        > 0.0,
        "strong_d3_lower_bound_positive": audit_comparisons[
            "llm_minus_strong_d3"
        ]["ci95"][0]
        > 0.0,
        "route_selection_at_least_threshold": audit_comparisons[
            "exact_h4_route_selection_rate"
        ]
        >= config.route_selection_threshold,
        "mean_recovery_at_least_threshold": audit_comparisons[
            "recovery_fraction"
        ]["mean"]
        >= config.recovery_threshold,
    }
    mechanics = {
        "all_records_replayed": len(replayed) == config.num_cells,
        "serialized_requests_match_records": all(
            request_plans[row["cell_index"]] == row["llm_plans"]
            for row in replayed
        ),
        "all_record_fields_match": _close(replayed, result["records"]),
        "producer_comparisons_recomputed": _close(
            producer_recomputed, result["comparisons"]
        ),
        "producer_comparison_values_match_without_ci": _close(
            _without_ci(audit_comparisons),
            _without_ci(result["comparisons"]),
        ),
        "producer_gate_passed": bool(result["gate"]["passed"]),
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_depth4_fixed_tail_proposal_audit",
        "source_stage": result["stage"],
        "audit_bootstrap_seed": audit_bootstrap_seed,
        "mechanics": mechanics,
        "comparisons": audit_comparisons,
        "endpoint_gate": endpoint_gate,
        "gate": {
            "passed": all(mechanics.values()) and all(endpoint_gate.values())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal_json", type=Path)
    parser.add_argument("--audit-bootstrap-seed", type=int, default=24_211)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.proposal_json.read_text(encoding="utf-8"))
    audit = audit_result(
        result, audit_bootstrap_seed=args.audit_bootstrap_seed
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
