#!/usr/bin/env python3
"""Run constructed zero-call horizon checks, NOT a ChemBench efficacy study."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.chembench_mopen.horizon import FiniteBeliefModel, HorizonPlanner


def xor_model() -> FiniteBeliefModel:
    """Two complementary exact bits versus a 75%-accurate target measurement."""
    worlds = list(product((0, 1), repeat=2))
    likelihoods = np.zeros((4, 3, 2))
    targets = []
    for index, (u, v) in enumerate(worlds):
        truth = u ^ v
        likelihoods[index, 0, u] = 1
        likelihoods[index, 1, v] = 1
        likelihoods[index, 2, truth] = 0.75
        likelihoods[index, 2, 1 - truth] = 0.25
        targets.append([truth])
    return FiniteBeliefModel(likelihoods, np.array(targets), np.full(4, 0.25))


def adaptive_model() -> FiniteBeliefModel:
    """A regime measurement determines which of two target assays is reliable."""
    worlds = list(product((0, 1), repeat=2))
    likelihoods = np.full((4, 3, 2), 0.5)
    targets = []
    for index, (regime, target) in enumerate(worlds):
        likelihoods[index, 0] = (1 - regime, regime)
        likelihoods[index, 1 + regime] = (1 - target, target)
        targets.append([target])
    return FiniteBeliefModel(likelihoods, np.array(targets), np.full(4, 0.25))


def receding_reference_tree(
    model, state, *, horizon, mode, remaining=2, available=(0, 1, 2)
):
    """Exact fixed-budget replay for these tiny fixtures, not a live run driver."""
    if not remaining:
        return {
            "remaining_measurements": 0,
            "terminal_expected_risk": model.risk(state),
        }
    plan = HorizonPlanner(model).plan(
        state, min(horizon, remaining), available=available, mode=mode
    )
    action = plan.root.action
    children = [
        {
            "observation": row.observation,
            "probability": row.probability,
            "child": receding_reference_tree(
                model,
                row.state,
                horizon=horizon,
                mode=mode,
                remaining=remaining - 1,
                available=tuple(a for a in available if a != action),
            ),
        }
        for row in model.branches(state, action)
    ]
    return {
        "remaining_measurements": remaining,
        "optimized_horizon": plan.effective_horizon,
        "action": action,
        "planned_risk": plan.root.expected_risk,
        "terminal_expected_risk": sum(
            row["probability"] * row["child"]["terminal_expected_risk"]
            for row in children
        ),
        "branches": children,
    }


def build_report() -> dict:
    cases = {}
    receding = {}
    for name, model in (
        ("xor_complementarity", xor_model()),
        ("adaptive_assay", adaptive_model()),
    ):
        planner = HorizonPlanner(model)
        cases[name] = {
            mode: {
                str(h): asdict(planner.plan(model.initial_state, h, mode=mode))
                for h in (1, 2, 3)
            }
            for mode in ("adaptive", "open_loop")
        }
        receding[name] = {
            mode: {
                str(h): receding_reference_tree(
                    model, model.initial_state, horizon=h, mode=mode
                )
                for h in (1, 2, 3)
            }
            for mode in ("adaptive", "open_loop")
        }
    xor = cases["xor_complementarity"]["adaptive"]
    adaptive = cases["adaptive_assay"]

    def close(left, right):
        return abs(left - right) <= 1e-12

    gates = {
        "xor_h1_prefers_noisy_target": xor["1"]["root"]["action"] == 2,
        "xor_h1_risk_matches_3_over_16": close(
            xor["1"]["root"]["expected_risk"], 3 / 16
        ),
        "xor_h2_resolves_target": close(xor["2"]["root"]["expected_risk"], 0),
        "xor_h3_legitimate_plateau": close(xor["3"]["root"]["expected_risk"], 0),
        "adaptive_h2_resolves_target": close(
            adaptive["adaptive"]["2"]["root"]["expected_risk"], 0
        ),
        "open_loop_h2_risk_matches_1_over_8": close(
            adaptive["open_loop"]["2"]["root"]["expected_risk"], 1 / 8
        ),
        "adaptive_second_action_depends_on_observation": {
            edge["child"]["action"]
            for edge in adaptive["adaptive"]["2"]["root"]["branches"]
        }
        == {1, 2},
        "fixed_budget_xor_receding_h1_risk": close(
            receding["xor_complementarity"]["adaptive"]["1"]["terminal_expected_risk"],
            3 / 16,
        ),
        "fixed_budget_xor_receding_h2_risk": close(
            receding["xor_complementarity"]["adaptive"]["2"]["terminal_expected_risk"],
            0,
        ),
        "open_loop_replanning_can_close_planned_adaptivity_gap": close(
            receding["adaptive_assay"]["open_loop"]["2"]["terminal_expected_risk"], 0
        ),
    }
    paths = (
        Path(__file__).resolve(),
        REPO_ROOT / "environments/chembench_mopen/horizon.py",
    )
    return {
        "schema_version": 1,
        "status": "reference_checks_passed"
        if all(gates.values())
        else "reference_checks_failed",
        "scope": "constructed mathematical checks only; not an LLM or ChemBench efficacy result",
        "model_calls": 0,
        "paid_cost_usd": 0,
        "chemistry_endpoints_opened": False,
        "gates": gates,
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        },
        "cases": cases,
        "fixed_budget_receding": {"measurement_budget": 2, "cases": receding},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        report = build_report()
        filename = "RESULT.json"
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "reference_execution_failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "model_calls": 0,
            "paid_cost_usd": 0,
            "chemistry_endpoints_opened": False,
            "gates": {},
        }
        filename = "FAILURE.json"
    temporary = args.output_dir / (filename + ".tmp")
    with temporary.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(args.output_dir / filename)
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2))
    if report["status"] != "reference_checks_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
