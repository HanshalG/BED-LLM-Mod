"""Measure exact second-action closure on banked Rock strategy root sets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
from environments.rock_diagnosis.core import EPSILON
from scripts.nonmyopic_rock_strategy_prior import _exhaustive_action_values


EXPECTED_CONFIG = {
    "map_names": ["7-8"],
    "num_trials_per_map": 30,
    "num_rounds": 10,
    "num_strategies": 6,
    "planning_horizon": 2,
    "seed": 24072,
    "bootstrap_replicates": 10_000,
    "temperature": 0.0,
    "validation_retries": 1,
    "trial_concurrency": 32,
    "strategy_schema": "branch_policy_v2",
    "primary_endpoint": "entropy_auc",
}


def _rows_for_arm(payload: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    model = RockDiagnosisModel(get_paper_map("7-8"))
    rows: list[dict[str, Any]] = []
    for trace in payload["traces"]["7-8"][arm]:
        belief = model.initial_belief.copy()
        position = model.map_spec.start_position
        for round_index, step in enumerate(trace["steps"]):
            assert tuple(step["position_before"]) == position
            if round_index < EXPECTED_CONFIG["num_rounds"] - 1:
                values, _units = _exhaustive_action_values(
                    model, position=position, belief=belief, horizon=2
                )
                roots = tuple(dict.fromkeys(step["candidate_roots"]))
                assert roots
                assert all(root in values for root in roots)
                exhaustive_value = max(values.values())
                closed_value = max(values[root] for root in roots)
                original_value = float(step["planning_score"])
                original_fraction = (
                    original_value / exhaustive_value if exhaustive_value > EPSILON else 1.0
                )
                closed_fraction = (
                    closed_value / exhaustive_value if exhaustive_value > EPSILON else 1.0
                )
                rows.append(
                    {
                        "arm": arm,
                        "trial_index": trace["trial_index"],
                        "round": round_index + 1,
                        "position": list(position),
                        "candidate_roots": list(roots),
                        "original_value": original_value,
                        "closed_value": closed_value,
                        "exhaustive_value": exhaustive_value,
                        "original_fraction": original_fraction,
                        "closed_fraction": closed_fraction,
                        "contains_exhaustive_optimal_root": math.isclose(
                            closed_value, exhaustive_value, abs_tol=1e-12, rel_tol=0.0
                        ),
                    }
                )
            belief = model.posterior(position, belief, step["action"], step["observation"])
            position = model.next_position(position, step["action"])
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    original = np.asarray([row["original_fraction"] for row in rows], dtype=float)
    closed = np.asarray([row["closed_fraction"] for row in rows], dtype=float)
    return {
        "num_states": len(rows),
        "mean_original_exhaustive_fraction": float(np.mean(original)),
        "mean_closed_exhaustive_fraction": float(np.mean(closed)),
        "mean_fraction_gain": float(np.mean(closed - original)),
        "median_fraction_gain": float(np.median(closed - original)),
        "optimal_root_coverage": float(
            np.mean([row["contains_exhaustive_optimal_root"] for row in rows])
        ),
        "closed_improves_states": int(np.count_nonzero(closed > original + 1e-12)),
        "closed_ties_states": int(np.count_nonzero(np.abs(closed - original) <= 1e-12)),
        "closed_worsens_states": int(np.count_nonzero(closed < original - 1e-12)),
        "round_closed_fraction_mean": [
            float(np.mean([row["closed_fraction"] for row in rows if row["round"] == round_index]))
            for round_index in range(1, 10)
        ],
    }


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    assert payload["schema_version"] == 1
    assert payload["stage"] == "L1"
    assert payload["run_id"] == "nonmyopic-rocksample-7-8-scale-20260721"
    assert payload["config"] == EXPECTED_CONFIG
    assert payload["gate_passed"] is True
    assert payload["mechanics"]["rollout_scoring_llm_calls"] == 0

    strategy_rows = _rows_for_arm(payload, "strategy_eig")
    random_rows = _rows_for_arm(payload, "random_strategy")
    strategy = _summary(strategy_rows)
    random = _summary(random_rows)
    assert strategy["num_states"] == 270
    assert random["num_states"] == 270
    llm_random_gap = (
        strategy["mean_closed_exhaustive_fraction"]
        - random["mean_closed_exhaustive_fraction"]
    )
    conditions = {
        "closed_fraction_at_least_0_97": strategy[
            "mean_closed_exhaustive_fraction"
        ]
        >= 0.97,
        "fraction_gain_at_least_0_08": strategy["mean_fraction_gain"] >= 0.08,
        "optimal_root_coverage_at_least_0_90": strategy["optimal_root_coverage"] >= 0.90,
        "llm_minus_random_closed_fraction_at_least_0_03": llm_random_gap >= 0.03,
    }
    return {
        "schema_version": 1,
        "stage": "followup_closure_screen",
        "source_run_id": payload["run_id"],
        "no_llm_calls": True,
        "strategy_eig": strategy,
        "random_strategy": random,
        "llm_minus_random_closed_fraction": llm_random_gap,
        "gate_conditions": conditions,
        "gate_passed": all(conditions.values()),
        "rows": {
            "strategy_eig": strategy_rows,
            "random_strategy": random_rows,
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    strategy = summary["strategy_eig"]
    random = summary["random_strategy"]
    lines = [
        "# RockSample[7,8] Exact Follow-Up Closure Screen",
        "",
        "This is a zero-LLM-call proposal-fidelity diagnostic, not a policy endpoint.",
        "",
        "| Root source | States | Original fraction | Closed fraction | Gain | Optimal-root coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Gemma StrategyEIG | {strategy['num_states']} | "
        f"{strategy['mean_original_exhaustive_fraction']:.4f} | "
        f"{strategy['mean_closed_exhaustive_fraction']:.4f} | "
        f"{strategy['mean_fraction_gain']:+.4f} | {strategy['optimal_root_coverage']:.4f} |",
        f"| Matched random | {random['num_states']} | "
        f"{random['mean_original_exhaustive_fraction']:.4f} | "
        f"{random['mean_closed_exhaustive_fraction']:.4f} | "
        f"{random['mean_fraction_gain']:+.4f} | {random['optimal_root_coverage']:.4f} |",
        "",
        f"LLM-minus-random closed fraction: `{summary['llm_minus_random_closed_fraction']:+.4f}`.",
        "",
    ]
    for condition, passed in summary["gate_conditions"].items():
        lines.append(f"- {condition}: **{passed}**.")
    lines.extend(["", f"**Gate passed: {summary['gate_passed']}.**", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(json.loads(args.result.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.summary_output.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"gate_passed": summary["gate_passed"]}, indent=2))


if __name__ == "__main__":
    main()
