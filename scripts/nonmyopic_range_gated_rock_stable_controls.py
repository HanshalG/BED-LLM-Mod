"""Cross-platform qualification for range-gated Rock control selection."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

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
    _stable_seed,
)
from scripts.nonmyopic_range_gated_rock_strategy import (
    Plan,
    enumerate_legal_plans,
    plan_value,
    validate_plan,
)
from scripts.nonmyopic_rock_depth_oracle import exhaustive_action_values


VALUE_TIE_ATOL = 1e-12
VALUE_DIGITS = 12


def stable_best_index(
    values: Sequence[float], *, atol: float = VALUE_TIE_ATOL
) -> int:
    """Choose the first value within ``atol`` of the numerical maximum."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("stable selection requires a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError("stable selection requires finite values")
    if atol < 0.0:
        raise ValueError("stable selection tolerance must be non-negative")
    maximum = float(np.max(array))
    return int(np.flatnonzero(array >= maximum - atol)[0])


def stable_best_plan(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    plans: tuple[Plan, ...],
) -> tuple[Plan, float, list[float]]:
    values = [
        plan_value(model, position=position, belief=belief, plan=plan)
        for plan in plans
    ]
    index = stable_best_index(values)
    return plans[index], float(values[index]), values


def _quantize(value: float) -> float:
    return round(float(value), VALUE_DIGITS)


def _selection_gap(values: Sequence[float]) -> float | None:
    maximum = max(values)
    outside = [maximum - value for value in values if maximum - value > VALUE_TIE_ATOL]
    return min(outside) if outside else None


def _selection_is_perturbation_stable(values: Sequence[float]) -> bool:
    expected = stable_best_index(values)
    scale = VALUE_TIE_ATOL / 16.0
    perturbations = (
        [scale if index % 2 else -scale for index in range(len(values))],
        [-scale if index % 2 else scale for index in range(len(values))],
    )
    return all(
        stable_best_index(
            [value + perturbation for value, perturbation in zip(values, delta, strict=True)]
        )
        == expected
        for delta in perturbations
    )


def qualify_stable_controls(proposal: dict[str, Any]) -> dict[str, Any]:
    config_keys = {field.name for field in fields(RangeGatedProposalGateConfig)}
    config = RangeGatedProposalGateConfig(
        **{
            key: value
            for key, value in proposal["config"].items()
            if key in config_keys
        }
    )
    config.validate()
    model = RangeGatedRockDiagnosisModel(
        get_paper_map("7-8"), remote_accuracy=0.55, onsite_accuracy=0.95
    )
    position = model.map_spec.start_position
    legal = model.legal_actions(position)
    exhaustive_plans = enumerate_legal_plans(model, position=position, horizon=3)
    records: list[dict[str, Any]] = []
    all_value_vectors: list[list[float]] = []

    for source in proposal["records"]:
        belief = model.initial_belief.copy()
        for action, outcome in source["history"]:
            belief = model.posterior(position, belief, action, outcome)

        llm_plans = tuple(tuple(plan) for plan in source["llm_plans"])
        for plan in llm_plans:
            validate_plan(model, position=position, plan=plan, horizon=3)
        llm_plan, llm_value, llm_values = stable_best_plan(
            model, position=position, belief=belief, plans=llm_plans
        )

        random_plans = matched_random_fixed_root_plans(
            model,
            position=position,
            belief=belief,
            seed=_stable_seed(
                config.seed, "fixed-root-random", source["cell_index"]
            ),
        )
        random_plan, random_value, random_values = stable_best_plan(
            model, position=position, belief=belief, plans=random_plans
        )

        truncated = tuple(plan[:2] for plan in llm_plans)
        shared_plan, _, shared_values = stable_best_plan(
            model, position=position, belief=belief, plans=truncated
        )
        shared_full = llm_plans[truncated.index(shared_plan)]
        shared_value = plan_value(
            model, position=position, belief=belief, plan=shared_full
        )

        d2_values, _ = exhaustive_action_values(
            model, position=position, belief=belief, depth=2
        )
        d2_vector = [d2_values[action] for action in legal]
        d2_root = legal[stable_best_index(d2_vector)]
        strong_plans = tuple(
            plan for plan in exhaustive_plans if plan[0] == d2_root
        )
        strong_plan, strong_value, strong_values = stable_best_plan(
            model, position=position, belief=belief, plans=strong_plans
        )
        exact_plan, exact_value, exact_values = stable_best_plan(
            model, position=position, belief=belief, plans=exhaustive_plans
        )

        vectors = [
            llm_values,
            random_values,
            shared_values,
            d2_vector,
            strong_values,
            exact_values,
        ]
        all_value_vectors.extend(vectors)
        gaps = [_selection_gap(values) for values in vectors]
        records.append(
            {
                "cell_index": int(source["cell_index"]),
                "fixed_roots": list(
                    fixed_roots(model, position=position, belief=belief)
                ),
                "llm_selected_plan": list(llm_plan),
                "llm_value": _quantize(llm_value),
                "matched_random_selected_plan": list(random_plan),
                "matched_random_value": _quantize(random_value),
                "shared_d2_selected_full_plan": list(shared_full),
                "shared_d2_selected_full_value": _quantize(shared_value),
                "strong_d2_root": d2_root,
                "strong_d2_h3_plan": list(strong_plan),
                "strong_d2_h3_value": _quantize(strong_value),
                "exact_h3_plan": list(exact_plan),
                "exact_h3_value": _quantize(exact_value),
                "minimum_non_tie_gap": _quantize(
                    min(gap for gap in gaps if gap is not None)
                ),
            }
        )

    canonical_json = json.dumps(records, sort_keys=True, separators=(",", ":"))
    mechanics = {
        "all_records_canonicalized": len(records) == config.num_cells,
        "all_selection_gaps_exceed_100_tolerances": all(
            gap is None or gap > 100.0 * VALUE_TIE_ATOL
            for values in all_value_vectors
            for gap in [_selection_gap(values)]
        ),
        "all_selections_stable_under_sub_tolerance_perturbations": all(
            _selection_is_perturbation_stable(values)
            for values in all_value_vectors
        ),
        "all_exact_h3_routes_are_delayed_onsite": all(
            row["exact_h3_plan"] == ["move-SOUTH", "move-SOUTH", "check-5"]
            for row in records
        ),
        "no_llm_calls": True,
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_stable_control_qualification",
        "source_stage": proposal["stage"],
        "value_tie_atol": VALUE_TIE_ATOL,
        "value_digits": VALUE_DIGITS,
        "canonical_digest": hashlib.sha256(canonical_json.encode()).hexdigest(),
        "mechanics": mechanics,
        "records": records,
        "gate": {"passed": all(mechanics.values())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualification = qualify_stable_controls(
        json.loads(args.proposal_json.read_text(encoding="utf-8"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate": qualification["gate"],
                "canonical_digest": qualification["canonical_digest"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
