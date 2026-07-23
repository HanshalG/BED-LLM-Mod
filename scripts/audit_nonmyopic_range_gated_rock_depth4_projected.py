"""Independent replay of the bounded-projection range-gated h4 proposal gate."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_nonmyopic_range_gated_rock_depth4_fixed_tail import (
    _close,
    _comparison_payload,
    _without_ci,
)
from scripts.nonmyopic_range_gated_rock_depth4_fixed_tail import (
    CRITICAL_ROUTE,
    RangeGatedDepth4ProposalConfig,
    RangeGatedDepth4StrategyConfig,
    _score_cell,
    build_depth4_model,
    canonical_nonrouting_projection,
    extract_valid_depth4_branches,
)
from scripts.nonmyopic_range_gated_rock_strategy import enumerate_legal_plans


def audit_projected_result(
    result: dict[str, Any],
    *,
    audit_bootstrap_seed: int = 24_219,
) -> dict[str, Any]:
    config_keys = {field.name for field in fields(RangeGatedDepth4ProposalConfig)}
    config = RangeGatedDepth4ProposalConfig(
        **{
            key: value
            for key, value in result["config"].items()
            if key in config_keys
        }
    )
    strategy_keys = {
        field.name for field in fields(RangeGatedDepth4StrategyConfig)
    }
    strategy_config = RangeGatedDepth4StrategyConfig(
        **{
            key: value
            for key, value in result["strategy_config"].items()
            if key in strategy_keys
        }
    )
    config.validate()
    strategy_config.validate()
    model = build_depth4_model()
    position = model.map_spec.start_position
    exhaustive_plans = enumerate_legal_plans(
        model, position=position, horizon=4
    )
    requests = {
        int(request["cell_index"]): request
        for request in result["candidate_requests"]
    }
    replayed: list[dict[str, Any]] = []
    source_checks: list[bool] = []
    projection_checks: list[bool] = []
    for source in result["records"]:
        cell_index = int(source["cell_index"])
        request = requests[cell_index]
        belief = model.initial_belief.copy()
        history = tuple(tuple(item) for item in source["history"])
        for action, outcome in history:
            belief = model.posterior(position, belief, action, outcome)
        plans = tuple(tuple(plan) for plan in source["llm_plans"])
        replay = _score_cell(
            model,
            cell_index=cell_index,
            belief=belief,
            history=history,
            llm_plans=plans,
            config=config,
            exhaustive_plans=exhaustive_plans,
        )
        selected_index = plans.index(tuple(replay["llm_selected_plan"]))
        replay.update(
            {
                "branch_sources": request["branch_sources"],
                "projected_indices": request["projected_indices"],
                "selected_branch_source": request["branch_sources"][
                    selected_index
                ],
            }
        )
        replayed.append(replay)

        valid_by_attempt = []
        roots = tuple(request["roots"])
        for response in request["attempt_responses"]:
            valid, _errors = extract_valid_depth4_branches(
                response,
                model=model,
                position=position,
                roots=roots,
                config=strategy_config,
                accept_json_prefix=True,
            )
            valid_by_attempt.append(valid)
        for index, (plan, branch_source) in enumerate(
            zip(plans, request["branch_sources"], strict=True)
        ):
            if branch_source == "projected":
                projection_checks.append(
                    plan == canonical_nonrouting_projection(roots[index])
                    and plan != CRITICAL_ROUTE
                )
                continue
            attempt = int(branch_source.split("_", 1)[1])
            source_checks.append(
                attempt < len(valid_by_attempt)
                and valid_by_attempt[attempt].get(index) == plan
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
            requests[row["cell_index"]]["compiled_plans"] == row["llm_plans"]
            for row in replayed
        ),
        "all_record_fields_match": _close(replayed, result["records"]),
        "all_llm_branch_sources_recovered": bool(source_checks)
        and all(source_checks),
        "all_projections_recomputed": all(projection_checks),
        "all_selected_plans_are_llm_authored": all(
            row["selected_branch_source"] != "projected"
            for row in replayed
        ),
        "producer_comparisons_recomputed": _close(
            producer_recomputed, result["comparisons"]
        ),
        "producer_values_match_without_ci": _close(
            _without_ci(audit_comparisons),
            _without_ci(result["comparisons"]),
        ),
        "producer_gate_passed": bool(result["gate"]["passed"]),
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_depth4_projected_proposal_audit",
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
    parser.add_argument("--audit-bootstrap-seed", type=int, default=24_219)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.proposal_json.read_text(encoding="utf-8"))
    audit = audit_projected_result(
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
