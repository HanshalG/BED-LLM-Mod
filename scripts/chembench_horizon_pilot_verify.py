"""Replay a complete frozen pilot: independent physics and qualified policy code."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from environments.chembench_mopen.pilot_data import (
    build_hidden_worlds,
    build_public_pilot,
    read_protocol,
)
from environments.chembench_mopen.native_belief import NativeEnvelopeGaussianModel
from environments.chembench_mopen.envelope_belief import EnvelopeGaussianModel
from scripts.chembench_horizon_pilot import decide
from scripts.chembench_mopen_nonmyopic_opportunity import load_source, verify_source
from scripts.chembench_pilot_preflight import preflight


def close(actual, expected, label):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (
        a.shape != b.shape
        or not np.isfinite(a).all()
        or not np.allclose(a, b, atol=1e-9, rtol=1e-10)
    ):
        raise ValueError(f"replay mismatch: {label}")


def replay_episode(
    record, model, truth_observations, truth_targets, noise, config, roots, world, arm
):
    if (
        record["world"] != world
        or record["arm"] != arm
        or len(record["history"]) != config["measurement_budget"]
    ):
        raise ValueError("episode identity or coverage mismatch")
    state = np.asarray(model.initial_state)
    available = tuple(range(model.num_actions))
    rng = np.random.default_rng(config["random_policy_seed"] + world)
    weights = np.exp(state)
    close(
        record["prior_mse"],
        np.mean((weights @ model.targets - truth_targets) ** 2),
        "prior MSE",
    )
    for step, row in enumerate(record["history"]):
        if row["round"] != step + 1 or row["reused_initial_plan"] != (
            step == 0 and arm != "random"
        ):
            raise ValueError("round identity mismatch")
        choice = row["choice"]
        if arm == "random":
            if choice["action"] != int(rng.choice(available)):
                raise ValueError("random action mismatch")
        else:
            expected = (
                roots[arm]
                if step == 0
                else decide(
                    model,
                    tuple(state),
                    available,
                    config["measurement_budget"] - step,
                    arm,
                    config,
                    config["max_decision_seconds"],
                )
            )
            expected = json.loads(json.dumps(expected))
            for key in ("action", "effective_horizon", "mode"):
                if choice[key] != expected[key]:
                    raise ValueError(f"policy mismatch: {key}")
            close(choice["root_values"], expected["root_values"], "policy root values")
            for key in (
                "value",
                "requested_horizon",
                "processed_states",
                "fixed_sequence",
            ):
                if key in expected and choice.get(key) != expected[key]:
                    if key == "value":
                        close(choice.get(key), expected[key], "policy value")
                    else:
                        raise ValueError(f"policy mismatch: {key}")
        action = choice["action"]
        if action not in available:
            raise ValueError("repeated or invalid action")
        observation = float(truth_observations[action] + noise[step, action])
        close(row["observation"], observation, "paired observation")
        forecast = weights @ model.targets
        close(
            row["model_risk_before"],
            weights @ ((model.targets - forecast) ** 2 @ model.target_weights),
            "risk before",
        )
        likelihood = np.array(
            [
                -0.5 * ((observation - mu) / sigma) ** 2
                - math.log(sigma)
                - 0.5 * math.log(2 * math.pi)
                for mu, sigma in zip(model.means[:, action], model.sigmas[:, action])
            ]
        )
        state = state + likelihood
        state -= np.max(state)
        state -= math.log(math.fsum(math.exp(float(x)) for x in state))
        weights = np.exp(state)
        forecast = weights @ model.targets
        close(row["posterior_log_weights"], state, "full-likelihood posterior")
        close(row["forecast"], forecast, "fixed-target forecast")
        close(row["target_mse"], np.mean((forecast - truth_targets) ** 2), "target MSE")
        close(
            row["model_risk_after"],
            weights @ ((model.targets - forecast) ** 2 @ model.target_weights),
            "risk after",
        )
        available = tuple(a for a in available if a != action)
    close(record["final_mse"], record["history"][-1]["target_mse"], "final MSE")


def verify(run_dir, source_root):
    result = json.loads((run_dir / "RESULT.json").read_text())
    if (
        result.get("status") not in {"engineering_pass", "engineering_null"}
        or len(result.get("records", [])) != 8
    ):
        raise ValueError("complete eight-world terminal result required before replay")
    config, digest = read_protocol()
    if (
        result["protocol_sha256"] != digest
        or result["model_calls"] != 0
        or result["cost_usd"] != 0
        or result["paid_calls_authorized"] is not False
        or result["hidden_worlds_opened"] is not True
    ):
        raise ValueError("protocol or execution accounting mismatch")
    root = Path(__file__).resolve().parents[1]
    for relative, expected in result["source_hashes"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("runner source binding changed")
    engine = result["engine"]
    checked = preflight(source_root, engine)
    if json.loads((run_dir / "preflight.json").read_text()) != checked:
        raise ValueError("preflight replay mismatch")
    verify_source(source_root)
    source = load_source(source_root)
    model_type = (
        NativeEnvelopeGaussianModel if engine == "native" else EnvelopeGaussianModel
    )
    public = build_public_pilot(source, model_type=model_type).model
    bank, observations, targets, noise = build_hidden_worlds(source)
    hidden = json.loads((run_dir / "hidden_binding.json").read_text())
    if (
        hidden["sha256"]
        != hashlib.sha256(
            observations.tobytes() + targets.tobytes() + noise.tobytes()
        ).hexdigest()
        or hidden["world_count"] != len(bank)
        or hidden["world_parameters"] != json.loads(json.dumps(bank))
    ):
        raise ValueError("hidden/CRN reconstruction mismatch")
    oracle = model_type(
        observations,
        config["observation_sigma"],
        targets,
        np.full(len(bank), 1 / len(bank)),
        branch_count=config["candidate_branch_count"],
    )
    all_roots = {}
    for label, model, arms in [
        ("public", public, config["core_arms"]),
        ("oracle", oracle, ["h1", "h2", "h3"]),
    ]:
        all_roots[label] = {}
        for arm in arms:
            if arm == "random":
                continue
            replayed = decide(
                model,
                model.initial_state,
                tuple(range(model.num_actions)),
                config["measurement_budget"],
                arm,
                config,
                config["max_decision_seconds"],
            )
            saved = json.loads((run_dir / f"{label}_root_{arm}.json").read_text())
            if replayed["action"] != saved["action"]:
                raise ValueError("initial action replay mismatch")
            close(saved["root_values"], replayed["root_values"], "initial root values")
            all_roots[label][arm] = replayed
    for world, row in enumerate(result["records"]):
        if row["world"] != world or row != json.loads(
            (run_dir / f"world{world}_complete.json").read_text()
        ):
            raise ValueError("ordered world checkpoint mismatch")
        for key, model, arms, label in [
            ("core", public, config["core_arms"], "public"),
            ("population_oracle", oracle, ["h1", "h2", "h3"], "oracle"),
        ]:
            if [r["arm"] for r in row[key]] != arms:
                raise ValueError("arm coverage mismatch")
            for record in row[key]:
                arm = record["arm"]
                name = (
                    f"world{world}_{'oracle_' if label == 'oracle' else ''}{arm}.json"
                )
                if record != json.loads((run_dir / name).read_text()):
                    raise ValueError("episode checkpoint mismatch")
                replay_episode(
                    record,
                    model,
                    observations[world],
                    targets[world],
                    noise[world],
                    config,
                    all_roots[label],
                    world,
                    arm,
                )
    means = {
        arm: float(
            np.mean(
                [
                    next(r["final_mse"] for r in w["core"] if r["arm"] == arm)
                    for w in result["records"]
                ]
            )
        )
        for arm in config["core_arms"]
    }
    for arm in means:
        close(result["mean_final_mse"][arm], means[arm], "aggregate MSE")
    prior = float(np.mean([w["core"][0]["prior_mse"] for w in result["records"]]))
    close(result["prior_mse"], prior, "aggregate prior MSE")
    changed = sum(
        [r["choice"]["action"] for r in w["core"][0]["history"]]
        != [r["choice"]["action"] for r in w["core"][2]["history"]]
        for w in result["records"]
    )
    gates = {
        "complete": True,
        "forecast_improvement": means["h3"]
        <= prior * (1 - config["minimum_forecast_improvement_fraction"]),
        "h3_vs_h1": means["h3"]
        <= means["h1"] * (1 - config["minimum_h3_vs_h1_gain_fraction"]),
        "h3_nonworse_h2": means["h3"] <= means["h2"],
        "changed_worlds": changed >= config["minimum_h1_h3_changed_worlds"],
    }
    if (
        result["changed_worlds"] != changed
        or result["gates"] != gates
        or (result["status"] == "engineering_pass") != all(gates.values())
    ):
        raise ValueError("frozen gate mismatch")
    return {
        "status": "replay_verified",
        "result_sha256": hashlib.sha256(
            (run_dir / "RESULT.json").read_bytes()
        ).hexdigest(),
        "worlds": 8,
        "episodes": 72,
        "physics": "independent_full_likelihood_and_CRN_reconstruction",
        "policies": "same_qualified_planner_reexecuted",
        "gates": gates,
        "mean_final_mse": means,
        "model_calls": 0,
        "cost_usd": 0,
        "paid_calls_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        report = verify(args.run_dir, args.source_root)
    except Exception as exc:
        report = {
            "status": "replay_failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "paid_calls_authorized": False,
        }
    report["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(report["status"])


if __name__ == "__main__":
    main()
