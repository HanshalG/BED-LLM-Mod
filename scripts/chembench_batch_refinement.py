"""Prospective 32/64-branch refinement on opened synthetic reference fixtures."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np

from environments.chembench_mopen.batch_horizon import plan_batched
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.chembench_mopen.raw_integration import predictive_expectation


def regime_model(count, model_type=QuantileGaussianModel):
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
            sigmas.append([0.3, 0.3 if regime == 0 else 2, 0.3 if regime == 1 else 2])
            targets.append([target])
    return model_type(means, sigmas, targets, [0.25] * 4, branch_count=count)


def execute(output, model_type=QuantileGaussianModel):
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

    def plan(m, h, **kwargs):
        left = 180 - (monotonic() - started)
        if left <= 0:
            raise TimeoutError("panel cap")
        return plan_batched(
            m,
            m.initial_state,
            h,
            max_seconds=min(60, left),
            max_states=5_000_000,
            **kwargs,
        )

    def save(name, value):
        temp = output / (name + ".tmp")
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temp.replace(output / (name + ".json"))

    records = []
    for count in (32, 64):
        one = []
        for case in json.loads(raw)["cases"]:
            s = case["separation"]
            noise = [[0.3, 1], [2, 1]] if case["unequal_noise"] else 1
            m = model_type(
                [[-s, -0.5], [s, 0.5]],
                noise,
                [[0], [1]],
                [0.3, 0.7],
                branch_count=count,
            )
            result = plan(m, 1)
            refs = case["reference_risks"]
            one.append(
                {
                    "separation": s,
                    "unequal_noise": case["unequal_noise"],
                    "error": max(
                        abs(v - r) for (_, v), r in zip(result.root_values, refs)
                    ),
                    "regret": refs[result.action] - min(refs),
                }
            )
        save(f"order{count}_one", one)
        m = model_type(
            [[-0.3] * 3, [0.3] * 3], 1, [[0], [1]], [0.5, 0.5], branch_count=count
        )
        three = plan(m, 3)
        ref = QuantileGaussianModel(
            [[-0.3], [0.3]], 1 / np.sqrt(3), [[0], [1]], [0.5, 0.5]
        )
        exact = predictive_expectation(
            ref, ref.initial_state, 0, ref.risk, value_bound=0.25
        )
        save(f"order{count}_three", asdict(three))
        m = regime_model(count, model_type)
        tree = plan(m, 2)
        fixed = plan(m, 2, mode="open_loop")
        branches = m.branches(m.initial_state, tree.action)
        # Two preselected quadrature positions, not cherry-picked after results.
        continuations = []
        for i in (len(branches) // 4, 3 * len(branches) // 4):
            b = branches[i]
            remaining = tuple(a for a in range(3) if a != tree.action)
            child = plan_batched(m, b.state, 1, available=remaining)
            continuations.append(
                {
                    "observation": b.observation,
                    "action": child.action,
                    "root_values": child.root_values,
                }
            )
        record = {
            "order": count,
            "one_step": one,
            "three_step": asdict(three),
            "three_reference": exact.value,
            "three_error": abs(three.value - exact.value),
            "tree": asdict(tree),
            "open_loop": asdict(fixed),
            "gap": fixed.value - tree.value,
            "continuations": continuations,
        }
        records.append(record)
        save(f"order{count}_complete", record)
    checks = {
        "one_step": all(
            c["error"] <= 0.001 and c["regret"] <= 0.001
            for r in records
            for c in r["one_step"]
        ),
        "three_step": all(r["three_error"] <= 0.001 for r in records),
        "refinement": all(
            abs(records[0][key]["value"] - records[1][key]["value"]) <= 0.001
            for key in ("tree", "open_loop", "three_step")
        ),
        "adaptivity": all(
            r["gap"] >= 0.001 and len({c["action"] for c in r["continuations"]}) == 2
            for r in records
        ),
    }
    return {
        "status": "synthetic_refinement_passed"
        if all(checks.values())
        else "synthetic_refinement_failed",
        "checks": checks,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    hashes = {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest()
        for p in [
            "scripts/chembench_batch_refinement.py",
            "environments/chembench_mopen/batch_horizon.py",
            "environments/chembench_mopen/quantile_belief.py",
            "environments/chembench_mopen/raw_belief.py",
        ]
    }
    try:
        result = execute(args.output_dir)
    except Exception as exc:
        result = {
            "status": "execution_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    result.update(
        {
            "source_hashes": hashes,
            "settings": {
                "orders": [32, 64],
                "error_cap": 0.001,
                "regret_cap": 0.001,
                "adaptivity_floor": 0.001,
                "max_plan_states": 5_000_000,
                "max_plan_seconds": 60,
                "max_panel_seconds": 180,
                "workspace_bytes": 64 * 1024 * 1024,
            },
            "model_calls": 0,
            "cost_usd": 0,
            "chemistry_outcomes_opened": False,
            "authorizes_paid_calls": False,
        }
    )
    temp = args.output_dir / "RESULT.tmp"
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output_dir / "RESULT.json")
    print(result["status"])


if __name__ == "__main__":
    main()
