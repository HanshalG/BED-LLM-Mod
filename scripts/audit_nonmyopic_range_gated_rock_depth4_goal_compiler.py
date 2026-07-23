"""Independent replay of the hierarchical range-gated h4 proposal gate."""

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

from scripts.audit_nonmyopic_range_gated_rock_depth4_fixed_tail import _close
from scripts.nonmyopic_range_gated_rock_depth4_fixed_tail import (
    RangeGatedDepth4ProposalConfig,
)
from scripts.nonmyopic_range_gated_rock_depth4_goal_compiler import (
    _comparison_payload,
    _score_goal_cell,
    build_depth4_model,
    compile_target_plan,
)
from scripts.nonmyopic_range_gated_rock_strategy import enumerate_legal_plans


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
    audit_bootstrap_seed: int = 24_231,
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
    request_by_cell = {
        int(request["cell_index"]): request
        for request in result["candidate_requests"]
    }
    replayed: list[dict[str, Any]] = []
    compiler_matches: list[bool] = []
    for source in result["records"]:
        cell_index = int(source["cell_index"])
        request = request_by_cell[cell_index]
        belief = model.initial_belief.copy()
        history = tuple(tuple(item) for item in source["history"])
        for action, outcome in history:
            belief = model.posterior(position, belief, action, outcome)
        roots = tuple(request["roots"])
        targets = tuple(int(value) for value in request["target_assignments"])
        plans = tuple(
            compile_target_plan(
                model,
                position=position,
                root=root,
                target_rock=target,
            )
            for root, target in zip(roots, targets, strict=True)
        )
        compiler_matches.append(
            [list(plan) for plan in plans] == request["compiled_plans"]
        )
        replayed.append(
            _score_goal_cell(
                model,
                cell_index=cell_index,
                belief=belief,
                history=history,
                llm_targets=targets,
                llm_plans=plans,
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
    endpoint_gate = {
        "matched_random_goal_lower_bound_positive": audit_comparisons[
            "llm_minus_matched_random_goals"
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
        "all_target_assignments_recompiled_independently": all(compiler_matches),
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
        "stage": "range_gated_rock_depth4_goal_compiler_proposal_audit",
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
    parser.add_argument("--audit-bootstrap-seed", type=int, default=24_231)
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
