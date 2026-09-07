"""Seal complete matched proposal forecasts before loading test outcomes.

Hashes detect changes, not when data were first inspected. Prospective execution
and separate outcome access remain protocol requirements. This module issues no
calls, supplies no scientific pass thresholds and grants no paid authorization.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from .executable_belief import ExecutableSnapshot


ARMS = ("history_aware", "history_blind", "symbolic_search")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _matrix(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid {name}")
    return array


def _validate(panel):
    if not isinstance(panel, dict) or set(panel) != {
        "schema_version",
        "arms",
        "cases",
        "interpretation",
    }:
        raise ValueError("invalid forecast panel schema")
    if (
        type(panel["schema_version"]) is not int
        or panel["schema_version"] != 1
        or panel["arms"] != list(ARMS)
    ):
        raise ValueError("invalid forecast schema version or control arms")
    if panel["interpretation"] != "descriptive_heldout_prediction_not_an_efficacy_gate":
        raise ValueError("invalid interpretation")
    if not isinstance(panel["cases"], list) or not panel["cases"]:
        raise ValueError("empty forecast panel")
    ids = []
    for case in panel["cases"]:
        if set(case) != {
            "case_id",
            "history_sha256",
            "target_inputs",
            "sigma",
            "forecasts",
        }:
            raise ValueError("invalid case schema")
        if not isinstance(case["case_id"], str) or not case["case_id"]:
            raise ValueError("invalid case ID")
        ids.append(case["case_id"])
        history = case["history_sha256"]
        if (
            not isinstance(history, str)
            or len(history) != 64
            or any(x not in "0123456789abcdef" for x in history)
        ):
            raise ValueError("invalid history binding")
        targets = np.asarray(case["target_inputs"], dtype=float)
        if (
            targets.ndim != 2
            or targets.shape[1] != 7
            or len(targets) == 0
            or not np.isfinite(targets).all()
        ):
            raise ValueError("invalid fixed targets")
        if (
            isinstance(case["sigma"], bool)
            or not isinstance(case["sigma"], (int, float))
            or not math.isfinite(case["sigma"])
            or case["sigma"] <= 0
        ):
            raise ValueError("invalid predictive noise")
        if set(case["forecasts"]) != set(ARMS):
            raise ValueError("every case requires all three matched arms")
        for forecast in case["forecasts"].values():
            if set(forecast) != {
                "log_weights",
                "target_means",
                "law_keys",
                "evaluated_scalar_nodes",
            }:
                raise ValueError("invalid forecast schema")
            logs = np.asarray(forecast["log_weights"], dtype=float)
            if logs.ndim != 1 or not len(logs) or not np.isfinite(logs).all():
                raise ValueError("invalid forecast log weights")
            shifted = logs - logs.max()
            normalizer = logs.max() + np.log(np.exp(shifted).sum())
            if abs(normalizer) > 1e-12:
                raise ValueError("forecast log weights must be normalized")
            means = _matrix(
                forecast["target_means"], (len(logs), len(targets)), "target means"
            )
            if np.any(means < 0):
                raise ValueError("predicted noiseless log1p rates must be nonnegative")
            keys = forecast["law_keys"]
            if (
                not isinstance(keys, list)
                or not keys
                or any(not isinstance(x, str) for x in keys)
                or len(set(keys)) != len(keys)
            ):
                raise ValueError("invalid law keys")
            work = forecast["evaluated_scalar_nodes"]
            if isinstance(work, bool) or not isinstance(work, int) or work <= 0:
                raise ValueError("invalid work count")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")


def seal_panel(
    path: Path, cases: dict[str, dict[str, ExecutableSnapshot]], *, sigma: float
) -> str:
    """Write once, with all arms present and identical fitting histories/targets.

    History-blind refers only to the proposer. Every numerical fitter receives
    the same real history; otherwise this would confound proposal and inference.
    """
    panel = {
        "schema_version": 1,
        "arms": list(ARMS),
        "cases": [],
        "interpretation": "descriptive_heldout_prediction_not_an_efficacy_gate",
    }
    for case_id, snapshots in cases.items():
        if set(snapshots) != set(ARMS):
            raise ValueError("every case requires all three matched arms")
        reference = snapshots[ARMS[0]]
        if not isinstance(reference, ExecutableSnapshot):
            raise ValueError("incomplete snapshot")
        forecasts = {}
        for arm in ARMS:
            snapshot = snapshots[arm]
            if not isinstance(snapshot, ExecutableSnapshot):
                raise ValueError("incomplete snapshot")
            if (
                snapshot.history_sha256 != reference.history_sha256
                or snapshot.target_inputs != reference.target_inputs
            ):
                raise ValueError(
                    "arms must share the fitting history and fixed target inputs"
                )
            if not np.all(snapshot.model.sigmas == sigma):
                raise ValueError("predictive sigma must match the fitted noise model")
            forecasts[arm] = {
                "log_weights": list(snapshot.state),
                "target_means": snapshot.model.targets.tolist(),
                "law_keys": list(snapshot.law_keys),
                "evaluated_scalar_nodes": snapshot.evaluated_scalar_nodes,
            }
        panel["cases"].append(
            {
                "case_id": case_id,
                "history_sha256": reference.history_sha256,
                "target_inputs": reference.target_inputs,
                "sigma": sigma,
                "forecasts": forecasts,
            }
        )
    _validate(panel)
    data = _bytes(panel)
    with Path(path).open("xb") as file:
        file.write(data)
    return _digest(data)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in forecast seal")
        result[key] = value
    return result


def score_sealed_panel(path: Path, expected_sha256: str, load_outcomes):
    """Validate every saved prediction before the sole test-outcome access.

    Outcomes contain matched noiseless log1p rates for MSE and separate noisy
    observations for proper predictive density and probability-integral checks.
    No model fitting or proposal callbacks occur after that access.
    """
    data = Path(path).read_bytes()
    if _digest(data) != expected_sha256:
        raise ValueError("sealed predictions changed")
    panel = json.loads(data, object_pairs_hook=_unique_object)
    _validate(panel)
    outcomes = load_outcomes()
    ids = {case["case_id"] for case in panel["cases"]}
    if not isinstance(outcomes, dict) or set(outcomes) != ids:
        raise ValueError("outcomes must cover every sealed case exactly")
    rows = []
    for case in panel["cases"]:
        target_count = len(case["target_inputs"])
        observed = outcomes[case["case_id"]]
        if not isinstance(observed, dict) or set(observed) != {
            "target_inputs",
            "true_log_rates",
            "noisy_log_rates",
        }:
            raise ValueError("invalid outcome schema")
        if not np.array_equal(
            _matrix(observed["target_inputs"], (target_count, 7), "outcome inputs"),
            case["target_inputs"],
        ):
            raise ValueError("outcome target identities differ from the frozen targets")
        truth = _matrix(observed["true_log_rates"], (target_count,), "true log rates")
        if np.any(truth < 0):
            raise ValueError("noiseless log1p rates must be nonnegative")
        noisy = _matrix(observed["noisy_log_rates"], (target_count,), "noisy log rates")
        scores = {}
        for arm, forecast in case["forecasts"].items():
            logs = np.asarray(forecast["log_weights"])
            weights = np.exp(logs)
            means = np.asarray(forecast["target_means"])
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                prediction = weights @ means
                mse = (prediction - truth) ** 2
                z = (noisy[None, :] - means) / case["sigma"]
                mixture_logs = (
                    logs[:, None]
                    - 0.5 * z**2
                    - math.log(case["sigma"])
                    - 0.5 * math.log(2 * math.pi)
                )
                maximum = mixture_logs.max(axis=0)
                log_density = maximum + np.log(
                    np.exp(mixture_logs - maximum).sum(axis=0)
                )
                pit = weights @ ndtr(z)
            scores[arm] = {
                "mean_squared_error": float(mse.mean()),
                "mean_negative_log_density": float(-log_density.mean()),
                "predictive_90pct_coverage": float(
                    ((pit >= 0.05) & (pit <= 0.95)).mean()
                ),
                "per_target_prediction": prediction.tolist(),
                "per_target_squared_error": mse.tolist(),
                "per_target_negative_log_density": (-log_density).tolist(),
                "per_target_pit": pit.tolist(),
                "evaluated_scalar_nodes": forecast["evaluated_scalar_nodes"],
            }
        rows.append({"case_id": case["case_id"], "scores": scores})
    paired = {
        control: [
            r["scores"]["history_aware"]["mean_squared_error"]
            - r["scores"][control]["mean_squared_error"]
            for r in rows
        ]
        for control in ARMS[1:]
    }
    result = {
        "status": "heldout_scores_complete",
        "forecast_sha256": expected_sha256,
        "outcome_sha256": _digest(_bytes(outcomes)),
        "rows": rows,
        "paired_mse_differences_aware_minus_control": paired,
        "mean_paired_mse_differences": {
            k: float(np.mean(v)) for k, v in paired.items()
        },
        "scientific_pass_authorized": False,
        "paid_calls_authorized": False,
    }
    _bytes(result)
    return result
