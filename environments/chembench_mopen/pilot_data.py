"""Strict public-prior versus hidden-world boundaries for the source pilot."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .crossing_belief import CrossingGaussianModel


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "results/nonmyopic/CHEMBENCH_HORIZON_PILOT_PROTOCOL_20260908.json"
)


def read_protocol():
    raw = PROTOCOL_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != "8e1fc9df41d177fa80b2e500c22c663e3a78a980ecedaa355c667116cc2e1d36":
        raise ValueError("frozen pilot protocol changed")
    return json.loads(raw), digest


@dataclass(frozen=True)
class PublicPilot:
    model: CrossingGaussianModel
    designs: np.ndarray
    target_inputs: np.ndarray
    candidate_parameters: tuple
    protocol_sha256: str


def draw_parameters(source, config, count, seed):
    banks = []
    rng = np.random.default_rng(seed)
    for family in config["families"]:
        versions = [
            source._PARAMS[family][config["difficulty"]][version]
            for version in config["public_parameter_versions"]
        ]
        keys = sorted(versions[0])
        if any(sorted(v) != keys for v in versions):
            raise ValueError("parameter keys differ")
        values = np.array([[v[k] for k in keys] for v in versions], dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("expected finite positive parameters")
        lo, hi = np.log(values.min(axis=0)), np.log(values.max(axis=0))
        draws = np.exp(rng.uniform(lo, hi, size=(count, len(keys))))
        banks.extend((family, dict(zip(keys, map(float, row)))) for row in draws)
    return tuple(banks)


def target_inputs(config):
    rng = np.random.default_rng(config["target_seed"])
    result = np.tile([1, 0, 1, 0, 1, 310, 7], (config["target_count"], 1)).astype(float)
    for name, (low, high) in config["target_bounds"].items():
        column = config["input_order"].index(name)
        if name in config["target_log_uniform_inputs"]:
            result[:, column] = np.exp(
                rng.uniform(np.log(low), np.log(high), len(result))
            )
        else:
            result[:, column] = rng.uniform(low, high, len(result))
    result.setflags(write=False)
    return result


def predict(source, bank, inputs):
    values = np.array(
        [
            [source._RATE_FNS[family](parameters, *map(float, x)) for x in inputs]
            for family, parameters in bank
        ],
        dtype=float,
    )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("invalid source rates")
    return np.log1p(values)


def build_public_pilot(source):
    """Never sample hidden worlds or evaluate their observations/targets here."""
    config, digest = read_protocol()
    bank = draw_parameters(
        source, config, config["prior_particles_per_family"], config["prior_seed"]
    )
    designs = np.array(config["designs"], dtype=float)
    targets = target_inputs(config)
    model = CrossingGaussianModel(
        predict(source, bank, designs),
        config["observation_sigma"],
        predict(source, bank, targets),
        np.full(len(bank), 1 / len(bank)),
        branch_count=config["candidate_branch_count"],
    )
    designs.setflags(write=False)
    return PublicPilot(model, designs, targets, bank, digest)


def build_hidden_worlds(source):
    """Endpoint boundary: call only after the public preflight has passed."""
    config, _ = read_protocol()
    bank = draw_parameters(
        source, config, config["worlds_per_family"], config["world_seed"]
    )
    observations = predict(source, bank, np.array(config["designs"]))
    targets = predict(source, bank, target_inputs(config))
    noise = np.random.default_rng(config["noise_seed"]).normal(
        0,
        config["observation_sigma"],
        size=(len(bank), config["measurement_budget"], len(config["designs"])),
    )
    return bank, observations, targets, noise
