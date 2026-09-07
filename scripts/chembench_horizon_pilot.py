"""Frozen zero-call source engineering pilot, with paired complete-world controls."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np

from environments.chembench_mopen.batch_horizon import plan_batched
from environments.chembench_mopen.envelope_belief import EnvelopeGaussianModel
from environments.chembench_mopen.pilot_data import (
    build_public_pilot,
    build_hidden_worlds,
    read_protocol,
)
from environments.chembench_mopen.raw_integration import predictive_expectation
from scripts.chembench_mopen_nonmyopic_opportunity import load_source
from scripts.chembench_pilot_preflight import preflight


def decide(model, state, available, remaining, arm, config, seconds):
    """Policy interface contains no realized hidden world or target labels."""
    if seconds <= 0:
        raise TimeoutError("decision allowance exhausted")
    if arm == "refined_myopic":
        started = monotonic()
        bound = float(np.ptp(model.targets, axis=0) ** 2 @ model.target_weights / 4)
        values = []
        for action in available:
            left = seconds - (monotonic() - started)
            if left <= 0:
                raise TimeoutError("myopic time cap")
            estimate = predictive_expectation(
                model,
                state,
                action,
                model.risk,
                value_bound=max(bound, 1e-15),
                tolerance=1e-6,
                max_evaluations=100_000,
                max_seconds=left,
            )
            values.append((action, estimate.value))
        _, action = min((v, a) for a, v in values)
        return {
            "action": action,
            "root_values": values,
            "effective_horizon": 1,
            "mode": "adaptive_reference",
            "elapsed_seconds": monotonic() - started,
        }
    mode = "open_loop" if arm == "open_loop_h3" else "adaptive"
    horizon = 3 if arm == "open_loop_h3" else int(arm[1:])
    return asdict(
        plan_batched(
            model,
            state,
            min(horizon, remaining),
            available=available,
            mode=mode,
            max_seconds=seconds,
            max_states=config["max_planner_states"],
            max_workspace_bytes=config["tensor_workspace_bytes"],
        )
    )


def episode(
    model,
    world,
    truth_observations,
    truth_targets,
    noise,
    arm,
    config,
    root_cache,
    deadline,
):
    state = model.initial_state
    available = tuple(range(model.num_actions))
    prior_mse = float(np.mean((model.forecast(state) - truth_targets) ** 2))
    rng = np.random.default_rng(config["random_policy_seed"] + world)
    history = []
    for round_index in range(config["measurement_budget"]):
        left = min(config["max_decision_seconds"], deadline - monotonic())
        if left <= 0:
            raise TimeoutError("complete world time cap")
        if arm == "random":
            choice = {
                "action": int(rng.choice(available)),
                "effective_horizon": 0,
                "mode": "random",
            }
        elif round_index == 0:
            choice = root_cache[arm]
        else:
            choice = decide(
                model,
                state,
                available,
                config["measurement_budget"] - round_index,
                arm,
                config,
                left,
            )
        action = choice["action"]
        if action not in available:
            raise ValueError("invalid policy action")
        risk_before = model.risk(state)
        observation = float(truth_observations[action] + noise[round_index, action])
        state = model.condition(state, action, observation)
        forecast = model.forecast(state)
        history.append(
            {
                "round": round_index + 1,
                "choice": choice,
                "observation": observation,
                "posterior_log_weights": state,
                "forecast": forecast.tolist(),
                "target_mse": float(np.mean((forecast - truth_targets) ** 2)),
                "model_risk_before": risk_before,
                "model_risk_after": model.risk(state),
                "reused_initial_plan": round_index == 0 and arm != "random",
            }
        )
        available = tuple(a for a in available if a != action)
    if monotonic() > deadline:
        raise TimeoutError("complete world time cap")
    return {
        "world": world,
        "arm": arm,
        "prior_mse": prior_mse,
        "history": history,
        "final_mse": history[-1]["target_mse"],
    }


def run(output, source_root):
    config, digest = read_protocol()
    started = monotonic()
    phase = "public_preflight"
    opened = False
    records = []
    written = 0

    def save(name, value):
        nonlocal written
        data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        if written + len(data) > config["max_output_bytes"]:
            raise RuntimeError("output cap")
        temp = output / (name + ".tmp")
        temp.write_bytes(data)
        temp.replace(output / (name + ".json"))
        written += len(data)

    def roots(model, label, arms=None):
        plans = {}
        for arm in config["core_arms"] if arms is None else arms:
            if arm == "random":
                continue
            left = min(
                config["max_decision_seconds"],
                config["max_panel_seconds"] - (monotonic() - started),
            )
            plans[arm] = decide(
                model,
                model.initial_state,
                tuple(range(model.num_actions)),
                config["measurement_budget"],
                arm,
                config,
                left,
            )
            save(label + "_" + arm, plans[arm])
        return plans

    try:
        check = preflight(source_root, "envelope")
        save("preflight", check)
        if check["status"] != "public_preflight_passed":
            raise RuntimeError("public predecessor not ready")
        source = load_source(source_root)
        public = build_public_pilot(source, model_type=EnvelopeGaussianModel)
        phase = "public_root_planning"
        public_roots = roots(public.model, "public_root")
        # All initial core decisions must exist before this endpoint boundary.
        phase = "hidden_world_construction"
        opened = True
        bank, observations, targets, noise = build_hidden_worlds(source)
        hidden_digest = hashlib.sha256(
            observations.tobytes() + targets.tobytes() + noise.tobytes()
        ).hexdigest()
        save(
            "hidden_binding",
            {
                "sha256": hidden_digest,
                "world_count": len(bank),
                "world_parameters": bank,
            },
        )
        oracle = EnvelopeGaussianModel(
            observations,
            config["observation_sigma"],
            targets,
            np.full(len(bank), 1 / len(bank)),
            branch_count=config["candidate_branch_count"],
        )
        phase = "oracle_root_planning"
        oracle_roots = roots(oracle, "oracle_root", ("h1", "h2", "h3"))
        for world in range(len(bank)):
            phase = f"world_{world}"
            deadline = min(
                started + config["max_panel_seconds"],
                monotonic() + config["max_world_seconds"],
            )
            core = []
            for arm in config["core_arms"]:
                result = episode(
                    public.model,
                    world,
                    observations[world],
                    targets[world],
                    noise[world],
                    arm,
                    config,
                    public_roots,
                    deadline,
                )
                core.append(result)
                save(f"world{world}_{arm}", result)
            oracle_results = []
            for arm in ("h1", "h2", "h3"):
                result = episode(
                    oracle,
                    world,
                    observations[world],
                    targets[world],
                    noise[world],
                    arm,
                    config,
                    oracle_roots,
                    deadline,
                )
                oracle_results.append(result)
                save(f"world{world}_oracle_{arm}", result)
            row = {"world": world, "core": core, "population_oracle": oracle_results}
            records.append(row)
            save(f"world{world}_complete", row)
        means = {
            arm: float(
                np.mean(
                    [
                        next(r["final_mse"] for r in w["core"] if r["arm"] == arm)
                        for w in records
                    ]
                )
            )
            for arm in config["core_arms"]
        }
        prior = float(np.mean([w["core"][0]["prior_mse"] for w in records]))
        changed = sum(
            [
                r["choice"]["action"]
                for r in next(a for a in w["core"] if a["arm"] == "h1")["history"]
            ]
            != [
                r["choice"]["action"]
                for r in next(a for a in w["core"] if a["arm"] == "h3")["history"]
            ]
            for w in records
        )
        gates = {
            "complete": len(records) == 8,
            "forecast_improvement": means["h3"]
            <= prior * (1 - config["minimum_forecast_improvement_fraction"]),
            "h3_vs_h1": means["h3"]
            <= means["h1"] * (1 - config["minimum_h3_vs_h1_gain_fraction"]),
            "h3_nonworse_h2": means["h3"] <= means["h2"],
            "changed_worlds": changed >= config["minimum_h1_h3_changed_worlds"],
        }
        result = {
            "status": "engineering_pass" if all(gates.values()) else "engineering_null",
            "gates": gates,
            "mean_final_mse": means,
            "prior_mse": prior,
            "changed_worlds": changed,
            "records": records,
        }
    except Exception as exc:
        result = {
            "status": "execution_failed",
            "phase": phase,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "completed_worlds": len(records),
        }
    result.update(
        {
            "protocol_sha256": digest,
            "hidden_worlds_opened": opened,
            "model_calls": 0,
            "cost_usd": 0,
            "paid_calls_authorized": False,
            "elapsed_seconds": monotonic() - started,
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = run(args.output_dir, args.source_root)
    root = Path(__file__).resolve().parents[1]
    result["source_hashes"] = {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest()
        for p in [
            "scripts/chembench_horizon_pilot.py",
            "scripts/chembench_pilot_preflight.py",
            "environments/chembench_mopen/pilot_data.py",
        ]
    }
    temp = args.output_dir / "RESULT.tmp"
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(args.output_dir / "RESULT.json")
    print(result["status"], result.get("phase", ""))


if __name__ == "__main__":
    main()
