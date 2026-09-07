"""Frozen four-context conditional source audit; no hidden-world outcomes."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np

from environments.chembench_mopen.batch_horizon import plan_batched
from environments.chembench_mopen.native_belief import NativeEnvelopeGaussianModel
from environments.chembench_mopen.pilot_data import (
    build_public_pilot,
    predict,
    read_protocol,
)
from environments.chembench_mopen.policy_value import evaluate_myopic_policy
from scripts.chembench_mopen_nonmyopic_opportunity import load_source
from scripts.chembench_pilot_preflight import preflight


PROTOCOL = Path(
    "results/nonmyopic/CHEMBENCH_CALIBRATED_CONTEXT_OPPORTUNITY_PROTOCOL_20260908.json"
)


def execute(output, source_root):
    config = json.loads(PROTOCOL.read_text())
    physics, physics_sha = read_protocol()
    if physics_sha != config["physics_sha256"]:
        raise ValueError("physics changed")
    started = monotonic()
    if preflight(source_root, "native")["status"] != "public_preflight_passed":
        raise ValueError("numerical predecessor failed")
    source = load_source(source_root)
    public = build_public_pilot(source, model_type=NativeEnvelopeGaussianModel)
    prefix_means = predict(
        source, public.candidate_parameters, np.array([config["common_history_design"]])
    )
    prefix = NativeEnvelopeGaussianModel(
        prefix_means,
        physics["observation_sigma"],
        public.model.targets,
        np.exp(public.model.initial_state),
        branch_count=64,
    )
    observations = prefix._invert_quantiles_many(
        np.exp(prefix.initial_state)[None, :],
        prefix.means[:, 0],
        prefix.sigmas[:, 0],
        np.array([config["common_history_observation_quantiles"]]),
    )[0]
    rows = []

    def limits():
        remaining = config["max_panel_seconds"] - (monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("context panel time cap")
        return {
            "max_states": config["max_states"],
            "max_seconds": min(remaining, config["max_seconds_per_evaluation"]),
            "max_workspace_bytes": config["max_workspace_bytes"],
        }

    for context, observation in enumerate(observations):
        state = prefix.condition(prefix.initial_state, 0, float(observation))
        for order in config["orders"]:
            model = NativeEnvelopeGaussianModel(
                public.model.means,
                public.model.sigmas,
                public.model.targets,
                np.exp(public.model.initial_state),
                branch_count=order,
            )
            one = evaluate_myopic_policy(
                model, state, config["new_measurement_budget"], **limits()
            )
            two = plan_batched(model, state, 2, **limits())
            three = plan_batched(model, state, 3, **limits())
            row = {
                "context": context,
                "quantile": config["common_history_observation_quantiles"][context],
                "common_observation": float(observation),
                "order": order,
                "h1_full_budget": asdict(one),
                "h2_plan": asdict(two),
                "h3_plan": asdict(three),
                "full_budget_values": [
                    one.value,
                    dict(three.root_values)[two.action],
                    three.value,
                ],
            }
            rows.append(row)
            (output / f"context{context}_order{order}.json").write_text(
                json.dumps(row, indent=2, allow_nan=False) + "\n"
            )
    limits()
    aggregate = {
        str(order): np.array(config["context_weights"])
        @ np.array([r["full_budget_values"] for r in rows if r["order"] == order])
        for order in config["orders"]
    }
    coarse, fine = [aggregate[str(order)] for order in config["orders"]]
    context_deltas = np.abs(
        np.array(
            [r["full_budget_values"] for r in rows if r["order"] == config["orders"][0]]
        )
        - np.array(
            [r["full_budget_values"] for r in rows if r["order"] == config["orders"][1]]
        )
    )
    gains = {
        order: [1 - v[1] / v[0], 1 - v[2] / v[1]] for order, v in aggregate.items()
    }
    gates = {
        "complete": len(rows) == 8,
        "context_refinement": float(context_deltas.max())
        <= config["absolute_refinement_tolerance"],
        "adjacent_gains_at_both_orders": all(
            gain >= config["minimum_adjacent_aggregate_gain_fraction"]
            for values in gains.values()
            for gain in values
        ),
    }
    return {
        "status": "conditional_opportunity_pass"
        if all(gates.values())
        else "conditional_opportunity_null",
        "rows": rows,
        "aggregate_full_budget_values": {k: v.tolist() for k, v in aggregate.items()},
        "adjacent_fractional_gains": gains,
        "maximum_context_refinement_delta": float(context_deltas.max()),
        "aggregate_refinement_delta": np.abs(coarse - fine).tolist(),
        "gates": gates,
        "elapsed_seconds": monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bindings = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            PROTOCOL,
            Path(__file__),
            Path("environments/chembench_mopen/policy_value.py"),
        ]
    }
    try:
        report = execute(args.output_dir, args.source_root)
    except Exception as exc:
        report = {
            "status": "execution_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    report.update(
        {
            "source_hashes": bindings,
            "hidden_worlds_constructed": False,
            "model_calls": 0,
            "cost_usd": 0,
            "paid_calls_authorized": False,
        }
    )
    temporary = args.output_dir / "RESULT.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output_dir / "RESULT.json")
    print(report["status"])


if __name__ == "__main__":
    main()
