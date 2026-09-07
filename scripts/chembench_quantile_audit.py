"""Prespecified bounded synthetic numerical panel, not chemistry efficacy."""

import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np

from environments.chembench_mopen.horizon import HorizonPlanner, SearchLimits
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.chembench_mopen.raw_integration import predictive_expectation


def run(output_dir):
    root = Path(__file__).resolve().parents[1]
    raw = (
        root / "results/nonmyopic/chembench_raw_integration/20260908-v1/RESULT.json"
    ).read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != "3a3352b8087fb3d95be99fa03eda625e7684b7979df95a77549d7e87120dab23"
    ):
        raise ValueError("reference binding changed")
    started = monotonic()
    records = []

    def checkpoint(name, value):
        temp = output_dir / (name + ".tmp")
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temp.replace(output_dir / (name + ".json"))

    def plan(model, horizon, mode="adaptive"):
        left = 120 - (monotonic() - started)
        if left <= 0:
            raise TimeoutError("panel time cap")
        return HorizonPlanner(
            model, limits=SearchLimits(max_nodes=250_000, max_seconds=min(30, left))
        ).plan(model.initial_state, horizon, mode=mode)

    for count in (16, 32):
        one_step = []
        for case in json.loads(raw)["cases"]:
            s = case["separation"]
            sigmas = [[0.3, 1], [2, 1]] if case["unequal_noise"] else 1
            model = QuantileGaussianModel(
                [[-s, -0.5], [s, 0.5]],
                sigmas,
                [[0], [1]],
                [0.3, 0.7],
                branch_count=count,
            )
            result = plan(model, 1)
            values = [v for _, v in result.root_action_values]
            refs = case["reference_risks"]
            one_step.append(
                {
                    "separation": s,
                    "unequal_noise": case["unequal_noise"],
                    "max_error": max(abs(v - r) for v, r in zip(values, refs)),
                    "regret": refs[result.root.action] - min(refs),
                }
            )
        checkpoint(f"order{count}_one_step", one_step)
        model = QuantileGaussianModel(
            [[-0.3] * 3, [0.3] * 3], 1, [[0], [1]], [0.5, 0.5], branch_count=count
        )
        three = plan(model, 3)
        ref = QuantileGaussianModel(
            [[-0.3], [0.3]], 1 / np.sqrt(3), [[0], [1]], [0.5, 0.5]
        )
        checkpoint(
            f"order{count}_three_step",
            {
                "value": three.root.expected_risk,
                "seconds": three.elapsed_seconds,
                "nodes": three.expanded_nodes,
            },
        )
        exact = predictive_expectation(
            ref, ref.initial_state, 0, ref.risk, value_bound=0.25
        )
        # Public constructed latent regime/target; source truth is never used.
        means, sigmas, targets = [], [], []
        for regime in (0, 1):
            for target in (0, 1):
                means.append(
                    [
                        2 * regime - 1,
                        2 * target - 1 if regime == 0 else 0,
                        2 * target - 1 if regime == 1 else 0,
                    ]
                )
                sigmas.append(
                    [0.3, 0.3 if regime == 0 else 2, 0.3 if regime == 1 else 2]
                )
                targets.append([target])
        adaptive = QuantileGaussianModel(
            means, sigmas, targets, [0.25] * 4, branch_count=count
        )
        tree = plan(adaptive, 2)
        fixed = plan(adaptive, 2, "open_loop")
        records.append(
            {
                "branch_count": count,
                "one_step": one_step,
                "three_step": {
                    "value": three.root.expected_risk,
                    "reference": exact.value,
                    "absolute_error": abs(three.root.expected_risk - exact.value),
                    "expanded_nodes": three.expanded_nodes,
                    "seconds": three.elapsed_seconds,
                },
                "adaptivity": {
                    "tree_value": tree.root.expected_risk,
                    "fixed_value": fixed.root.expected_risk,
                    "gap": fixed.root.expected_risk - tree.root.expected_risk,
                    "root_action": tree.root.action,
                    "continuations": [
                        {"observation": edge.observation, "action": edge.child.action}
                        for edge in tree.root.branches
                    ],
                },
            }
        )
        checkpoint(f"order{count}_complete", records[-1])
    passed = all(
        max(c["max_error"] for c in r["one_step"]) <= 0.001
        and max(c["regret"] for c in r["one_step"]) <= 0.001
        and r["three_step"]["absolute_error"] <= 0.001
        for r in records
    )
    stable = (
        abs(
            records[0]["adaptivity"]["tree_value"]
            - records[1]["adaptivity"]["tree_value"]
        )
        <= 0.001
        and abs(
            records[0]["adaptivity"]["fixed_value"]
            - records[1]["adaptivity"]["fixed_value"]
        )
        <= 0.001
    )
    positive = all(
        r["adaptivity"]["gap"] >= 0.001
        and len({c["action"] for c in r["adaptivity"]["continuations"]}) > 1
        for r in records
    )
    return {
        "status": "synthetic_checks_passed"
        if passed and stable and positive
        else "synthetic_checks_failed",
        "checks": {
            "reference_accuracy": passed,
            "adaptivity_refinement": stable,
            "positive_adaptivity": positive,
        },
        "records": records,
        "settings": {
            "orders": [16, 32],
            "absolute_error_cap": 0.001,
            "regret_cap": 0.001,
            "adaptivity_floor": 0.001,
            "max_panel_seconds": 120,
            "max_plan_seconds": 30,
            "max_plan_nodes": 250000,
        },
        "source_hashes": {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in [
                "scripts/chembench_quantile_audit.py",
                "environments/chembench_mopen/quantile_belief.py",
                "environments/chembench_mopen/horizon.py",
                "environments/chembench_mopen/raw_belief.py",
            ]
        },
        "model_calls": 0,
        "cost_usd": 0,
        "chemistry_outcomes_opened": False,
        "authorizes_paid_calls": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args.output_dir)
    except Exception as exc:
        result = {
            "status": "execution_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "authorizes_paid_calls": False,
        }
    temp = args.output_dir / "RESULT.tmp"
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output_dir / "RESULT.json")
    print(result["status"])


if __name__ == "__main__":
    main()
