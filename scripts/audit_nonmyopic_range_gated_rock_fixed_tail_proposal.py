"""Independent mechanical replay of the fixed-tail range-gated proposal gate."""

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

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel, get_paper_map
from scripts.nonmyopic_range_gated_rock_fixed_tail import fixed_roots
from scripts.nonmyopic_range_gated_rock_fixed_tail_proposal_gate import (
    matched_random_fixed_root_plans,
)
from scripts.nonmyopic_range_gated_rock_proposal_gate import (
    RangeGatedProposalGateConfig,
    _best_plan,
    _stable_seed,
    _summary,
)
from scripts.nonmyopic_range_gated_rock_strategy import (
    enumerate_legal_plans,
    plan_value,
    validate_plan,
)
from scripts.nonmyopic_rock_depth_oracle import exhaustive_action_values


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


def audit_result(result: dict[str, Any]) -> dict[str, Any]:
    config_keys = {field.name for field in fields(RangeGatedProposalGateConfig)}
    config = RangeGatedProposalGateConfig(
        **{key: value for key, value in result["config"].items() if key in config_keys}
    )
    config.validate()
    model = RangeGatedRockDiagnosisModel(
        get_paper_map("7-8"), remote_accuracy=0.55, onsite_accuracy=0.95
    )
    position = model.map_spec.start_position
    exhaustive_plans = enumerate_legal_plans(model, position=position, horizon=3)
    replayed: list[dict[str, Any]] = []
    legal = model.legal_actions(position)
    for row in result["records"]:
        belief = model.initial_belief.copy()
        for action, outcome in row["history"]:
            belief = model.posterior(position, belief, action, outcome)
        roots = fixed_roots(model, position=position, belief=belief)
        plans = tuple(tuple(plan) for plan in row["llm_plans"])
        for plan in plans:
            validate_plan(model, position=position, plan=plan, horizon=3)
        llm_plan, llm_value = _best_plan(
            model, position=position, belief=belief, plans=plans
        )
        random_plans = matched_random_fixed_root_plans(
            model,
            position=position,
            belief=belief,
            seed=_stable_seed(config.seed, "fixed-root-random", row["cell_index"]),
        )
        random_plan, random_value = _best_plan(
            model, position=position, belief=belief, plans=random_plans
        )
        truncated = tuple(plan[:2] for plan in plans)
        shared_plan, _ = _best_plan(
            model, position=position, belief=belief, plans=truncated
        )
        shared_full = plans[truncated.index(shared_plan)]
        shared_value = plan_value(
            model, position=position, belief=belief, plan=shared_full
        )
        d2_values, _ = exhaustive_action_values(
            model, position=position, belief=belief, depth=2
        )
        d2_root = max(
            legal, key=lambda action: (d2_values[action], -legal.index(action))
        )
        strong_plans = tuple(
            plan for plan in exhaustive_plans if plan[0] == d2_root
        )
        strong_plan, strong_value = _best_plan(
            model, position=position, belief=belief, plans=strong_plans
        )
        exact_plan, exact_value = _best_plan(
            model, position=position, belief=belief, plans=exhaustive_plans
        )
        opportunity = exact_value - strong_value
        replayed.append(
            {
                "cell_index": row["cell_index"],
                "history": row["history"],
                "fixed_roots": list(roots),
                "llm_plans": [list(plan) for plan in plans],
                "llm_selected_plan": list(llm_plan),
                "llm_value": llm_value,
                "matched_random_plans": [list(plan) for plan in random_plans],
                "matched_random_selected_plan": list(random_plan),
                "matched_random_value": random_value,
                "shared_d2_selected_full_plan": list(shared_full),
                "shared_d2_selected_full_value": shared_value,
                "strong_d2_root": d2_root,
                "strong_d2_h3_plan": list(strong_plan),
                "strong_d2_h3_value": strong_value,
                "exact_h3_plan": list(exact_plan),
                "exact_h3_value": exact_value,
                "d3_opportunity": opportunity,
                "recovery_fraction": (llm_value - strong_value) / opportunity,
            }
        )
    comparisons = {
        "llm_minus_matched_random": _summary(
            [row["llm_value"] - row["matched_random_value"] for row in replayed],
            config=config,
            label="fixed-tail-llm-minus-random",
        ),
        "llm_minus_shared_d2": _summary(
            [
                row["llm_value"] - row["shared_d2_selected_full_value"]
                for row in replayed
            ],
            config=config,
            label="fixed-tail-llm-minus-shared-d2",
        ),
        "llm_minus_strong_d2": _summary(
            [row["llm_value"] - row["strong_d2_h3_value"] for row in replayed],
            config=config,
            label="fixed-tail-llm-minus-strong-d2",
        ),
        "recovery_fraction": _summary(
            [row["recovery_fraction"] for row in replayed],
            config=config,
            label="fixed-tail-recovery",
        ),
        "exact_d3_root_selection_rate": float(
            np.mean(
                [
                    row["llm_selected_plan"][0] == row["exact_h3_plan"][0]
                    for row in replayed
                ]
            )
        ),
    }
    request_plans = {
        request["cell_index"]: request["compiled_plans"]
        for request in result["candidate_requests"]
    }
    mechanics = {
        "all_records_replayed": len(replayed) == config.num_cells,
        "serialized_requests_match_records": all(
            request_plans[row["cell_index"]] == row["llm_plans"]
            for row in replayed
        ),
        "all_roots_recomputed": all(
            [plan[0] for plan in row["llm_plans"]] == row["fixed_roots"]
            for row in replayed
        ),
        "all_record_fields_match": _close(replayed, result["records"]),
        "all_comparisons_match": _close(comparisons, result["comparisons"]),
        "producer_gate_passed": bool(result["gate"]["passed"]),
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_successor_grounded_proposal_audit",
        "source_stage": result["stage"],
        "mechanics": mechanics,
        "replayed_comparisons": comparisons,
        "gate": {"passed": all(mechanics.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.proposal_json.read_text(encoding="utf-8"))
    audit = audit_result(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["gate"], indent=2))


if __name__ == "__main__":
    main()
