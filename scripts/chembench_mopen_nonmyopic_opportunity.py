#!/usr/bin/env python3
"""Exact fixed-support horizon-opportunity audit for ActiveSciBench-Chem.

This script implements the frozen V2 protocol in
results/nonmyopic/CHEMBENCH_MOPEN_NONMYOPIC_OPPORTUNITY_V2_PROTOCOL_20260814.md.
It makes no model or network calls. The upstream source checkout is supplied by
path and verified before any validation response matrix is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


SOURCE_COMMIT = "acf160eb6c96897748dd92b152703b59b74efc05"
SOURCE_TREE = "e042d418fc30c6c70f6d1c6b0636d43f0c1c0f7a"
SOURCE_HASHES = {
    "autoscilab/oracle/chembench.py": (
        "eba9514d68573b4aa8c6a427606f1d0d7a9c431d8396e69ef4ae0774d2a449de"
    ),
    "autoscilab/oracle/compound_domains.py": (
        "0dfd47c1858efacb732f0fbceace3ebd61bb30f864ec777d628fb70f876343f3"
    ),
    "autoscilab/oracle/chembench_excluded.py": (
        "defc6c0c5edafe75dffaa366a61298856f413bd465a6e4e3636e9b0c57124003"
    ),
}
NOISE_LEVEL = 0.01
EXECUTION_BUDGET = 4
NUM_QUERY_ASSAYS = 1_000
NUM_DOMAINS = 57
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ValidationSlice:
    difficulty: str
    version: str
    query_seed: int

    @property
    def name(self) -> str:
        return f"{self.difficulty}/{self.version}"


VALIDATION_SLICES = (
    ValidationSlice("easy", "v2", 2026081502),
    ValidationSlice("medium", "v0", 2026081503),
    ValidationSlice("medium", "v1", 2026081504),
    ValidationSlice("medium", "v2", 2026081505),
    ValidationSlice("hard", "v0", 2026081506),
    ValidationSlice("hard", "v1", 2026081507),
    ValidationSlice("hard", "v2", 2026081508),
)


@dataclass(frozen=True)
class Assay:
    name: str
    values: tuple[float, float, float, float, float, float, float]


def frozen_assays() -> tuple[Assay, ...]:
    base = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0])
    names = {"C_A": 0, "C_I": 1, "C_B": 2, "C_P": 3, "Enz": 4, "T": 5, "pH": 6}
    assays: list[Assay] = []

    def add(label: str, **changes: float) -> None:
        values = base.copy()
        for name, value in changes.items():
            values[names[name]] = value
        assays.append(Assay(label, tuple(float(value) for value in values)))

    add("baseline")
    for value in (0.02, 0.1, 10.0, 100.0):
        add(f"C_A={value:g}", C_A=value)
    for value in (0.1, 1.0, 10.0, 100.0):
        add(f"C_I=50,C_A={value:g}", C_I=50.0, C_A=value)
    for value in (0.01, 0.1, 100.0):
        add(f"C_B={value:g}", C_B=value)
    for value in (10.0, 20.0):
        add(f"C_P={value:g}", C_P=value)
    for value in (278.0, 368.0):
        add(f"T={value:g}", T=value)
    for value in (4.0, 10.0):
        add(f"pH={value:g}", pH=value)
    if len(assays) != 18:
        raise AssertionError(f"expected 18 frozen assays, got {len(assays)}")
    return tuple(assays)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(source_root: Path, revision: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", revision],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_source(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise ValueError(f"source root does not exist: {source_root}")
    commit = _git_value(source_root, "HEAD")
    tree = _git_value(source_root, "HEAD^{tree}")
    if commit != SOURCE_COMMIT:
        raise ValueError(f"source commit mismatch: {commit} != {SOURCE_COMMIT}")
    if tree != SOURCE_TREE:
        raise ValueError(f"source tree mismatch: {tree} != {SOURCE_TREE}")
    actual_hashes = {}
    for relative, expected in SOURCE_HASHES.items():
        actual = _sha256(source_root / relative)
        if actual != expected:
            raise ValueError(f"source hash mismatch for {relative}: {actual} != {expected}")
        actual_hashes[relative] = actual
    return {
        "root": str(source_root),
        "commit": commit,
        "tree": tree,
        "file_sha256": actual_hashes,
    }


def load_source(source_root: Path) -> Any:
    source_root = source_root.resolve()
    verify_source(source_root)
    root_text = str(source_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module_name = "autoscilab.oracle.chembench"
    prior = sys.modules.get(module_name)
    if prior is not None:
        module_path = Path(prior.__file__).resolve()
        if source_root not in module_path.parents:
            raise RuntimeError(f"{module_name} was already imported from {module_path}")
        return prior
    return importlib.import_module(module_name)


def active_domains(source: Any) -> tuple[str, ...]:
    domains = tuple(
        name
        for name in source.CHEM_DOMAIN_REGISTRY
        if name not in source.CHEM_EXCLUDED_DOMAINS
    )
    if len(domains) != NUM_DOMAINS:
        raise ValueError(f"expected {NUM_DOMAINS} active domains, got {len(domains)}")
    for domain in domains:
        for item in VALIDATION_SLICES:
            if item.difficulty not in source._PARAMS[domain]:
                raise ValueError(f"missing difficulty {item.difficulty} for {domain}")
            if item.version not in source._PARAMS[domain][item.difficulty]:
                raise ValueError(f"missing version {item.name} for {domain}")
    return domains


def query_assays(
    seed: int,
    bounds: dict[str, tuple[float, float]],
    log_variables: Iterable[str],
    count: int = NUM_QUERY_ASSAYS,
) -> np.ndarray:
    ordered = ("C_A", "C_I", "C_B", "C_P", "Enz", "T", "pH")
    log_variables = set(log_variables)
    rng = np.random.default_rng(seed)
    values = np.empty((count, len(ordered)), dtype=float)
    for column, name in enumerate(ordered):
        low, high = bounds[name]
        if name in log_variables:
            if low <= 0:
                raise ValueError(f"log-uniform lower bound is not positive for {name}")
            values[:, column] = np.exp(rng.uniform(math.log(low), math.log(high), count))
        else:
            values[:, column] = rng.uniform(low, high, count)
    return values


def response_matrices(
    source: Any,
    domains: Sequence[str],
    validation_slice: ValidationSlice,
    assays: Sequence[Assay],
) -> tuple[np.ndarray, np.ndarray]:
    query_values = query_assays(
        validation_slice.query_seed,
        source.CHEM_INPUT_BOUNDS,
        source.CHEM_LOG_VARS,
    )
    observation_means = np.empty((len(domains), len(assays)), dtype=float)
    target_log_rates = np.empty((len(domains), len(query_values)), dtype=float)
    for domain_index, domain in enumerate(domains):
        params = source._PARAMS[domain][validation_slice.difficulty][validation_slice.version]
        rate_fn: Callable[..., float] = source._RATE_FNS[domain]
        for action_index, assay in enumerate(assays):
            value = float(rate_fn(params, *assay.values))
            value *= float(source._secondary_effects(assay.values[5], assay.values[6]))
            observation_means[domain_index, action_index] = value
        for query_index, query in enumerate(query_values):
            target_log_rates[domain_index, query_index] = math.log1p(
                max(float(rate_fn(params, *query)), 0.0)
            )
    if not np.isfinite(observation_means).all() or not np.isfinite(target_log_rates).all():
        raise ValueError(f"non-finite response in {validation_slice.name}")
    return observation_means, target_log_rates


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def categorical_likelihoods(means: np.ndarray, noise_level: float = NOISE_LEVEL) -> np.ndarray:
    if means.ndim != 2:
        raise ValueError("means must have shape (hypotheses, actions)")
    if noise_level <= 0:
        raise ValueError("noise_level must be positive")
    num_hypotheses, num_actions = means.shape
    likelihoods = np.zeros((num_hypotheses, num_actions, 3), dtype=float)
    for action in range(num_actions):
        low, high = np.quantile(means[:, action], (1.0 / 3.0, 2.0 / 3.0))
        for hypothesis, mean in enumerate(means[:, action]):
            scale = noise_level * abs(float(mean))
            if scale <= np.finfo(float).tiny:
                outcome = int(np.digitize(float(mean), (low, high), right=True))
                likelihoods[hypothesis, action, outcome] = 1.0
                continue
            low_mass = _normal_cdf((float(low) - float(mean)) / scale)
            high_mass = 1.0 - _normal_cdf((float(high) - float(mean)) / scale)
            likelihoods[hypothesis, action] = (
                low_mass,
                max(0.0, 1.0 - low_mass - high_mass),
                high_mass,
            )
            likelihoods[hypothesis, action] /= likelihoods[hypothesis, action].sum()
    if not np.allclose(likelihoods.sum(axis=2), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("categorical likelihoods do not sum to one")
    return likelihoods


class ExactPlanner:
    def __init__(self, likelihoods: np.ndarray, target_features: np.ndarray):
        if likelihoods.ndim != 3 or likelihoods.shape[2] != 3:
            raise ValueError("likelihoods must have shape (hypotheses, actions, 3)")
        if target_features.ndim != 2 or target_features.shape[0] != likelihoods.shape[0]:
            raise ValueError("target feature hypotheses must match likelihoods")
        self.likelihoods = np.asarray(likelihoods, dtype=float)
        self.features = np.asarray(target_features, dtype=float)
        self.num_hypotheses = likelihoods.shape[0]
        self.all_actions = tuple(range(likelihoods.shape[1]))
        self.initial_belief = self.belief_key(np.full(self.num_hypotheses, 1.0 / self.num_hypotheses))

    @staticmethod
    def belief_key(weights: Sequence[float]) -> tuple[float, ...]:
        result = np.maximum(np.asarray(weights, dtype=float), 0.0)
        total = float(result.sum())
        if total <= 0:
            raise ValueError("belief has no mass")
        result /= total
        return tuple(float(value) for value in result)

    @staticmethod
    def normalized_posterior(weights: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
        posterior = weights * likelihood
        total = float(posterior.sum())
        if total <= 0 or not math.isfinite(total):
            raise FloatingPointError("reachable observation has zero represented posterior mass")
        posterior /= total
        return posterior

    @lru_cache(maxsize=None)
    def bayes_risk(self, belief: tuple[float, ...]) -> float:
        weights = np.asarray(belief)
        forecast = weights @ self.features
        return float(np.mean(np.sum(weights[:, None] * (self.features - forecast) ** 2, axis=0)))

    @lru_cache(maxsize=None)
    def truth_loss(self, belief: tuple[float, ...], truth: int) -> float:
        weights = np.asarray(belief)
        forecast = weights @ self.features
        return float(np.mean((forecast - self.features[truth]) ** 2))

    @lru_cache(maxsize=None)
    def plan(
        self,
        belief: tuple[float, ...],
        available: tuple[int, ...],
        depth: int,
    ) -> tuple[float, int]:
        if depth <= 0 or not available:
            return self.bayes_risk(belief), -1
        weights = np.asarray(belief)
        best_value = math.inf
        best_action = -1
        for action in available:
            outcome_mass = weights @ self.likelihoods[:, action, :]
            remainder = tuple(item for item in available if item != action)
            value = 0.0
            for outcome, probability in enumerate(outcome_mass):
                probability = float(probability)
                if probability <= 1e-14:
                    continue
                posterior = self.normalized_posterior(
                    weights, self.likelihoods[:, action, outcome]
                )
                child_value, _ = self.plan(self.belief_key(posterior), remainder, depth - 1)
                value += probability * child_value
            if value < best_value - TIE_TOLERANCE:
                best_value = value
                best_action = action
        if best_action < 0:
            raise AssertionError("planner failed to select an action")
        return best_value, best_action

    @lru_cache(maxsize=None)
    def expected_policy_risk(
        self,
        belief: tuple[float, ...],
        available: tuple[int, ...],
        remaining: int,
        horizon: int,
    ) -> float:
        if remaining == 0:
            return self.bayes_risk(belief)
        _, action = self.plan(belief, available, min(remaining, horizon))
        weights = np.asarray(belief)
        outcome_mass = weights @ self.likelihoods[:, action, :]
        remainder = tuple(item for item in available if item != action)
        result = 0.0
        for outcome, probability in enumerate(outcome_mass):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            posterior = self.normalized_posterior(
                weights, self.likelihoods[:, action, outcome]
            )
            result += probability * self.expected_policy_risk(
                self.belief_key(posterior), remainder, remaining - 1, horizon
            )
        return result

    @lru_cache(maxsize=None)
    def expected_truth_loss(
        self,
        belief: tuple[float, ...],
        available: tuple[int, ...],
        remaining: int,
        horizon: int,
        truth: int,
    ) -> float:
        if remaining == 0:
            return self.truth_loss(belief, truth)
        _, action = self.plan(belief, available, min(remaining, horizon))
        remainder = tuple(item for item in available if item != action)
        weights = np.asarray(belief)
        result = 0.0
        for outcome, probability in enumerate(self.likelihoods[truth, action, :]):
            probability = float(probability)
            if probability <= 1e-14:
                continue
            posterior = self.normalized_posterior(
                weights, self.likelihoods[:, action, outcome]
            )
            result += probability * self.expected_truth_loss(
                self.belief_key(posterior), remainder, remaining - 1, horizon, truth
            )
        return result

    def evaluate_horizon(self, horizon: int, budget: int = EXECUTION_BUDGET) -> dict[str, Any]:
        planned_value, root_action = self.plan(
            self.initial_belief, self.all_actions, min(horizon, budget)
        )
        prior_risk = self.expected_policy_risk(
            self.initial_belief, self.all_actions, budget, horizon
        )
        truth_losses = [
            self.expected_truth_loss(
                self.initial_belief, self.all_actions, budget, horizon, truth
            )
            for truth in range(self.num_hypotheses)
        ]
        truth_mean = float(np.mean(truth_losses))
        if not math.isclose(prior_risk, truth_mean, abs_tol=1e-9, rel_tol=0.0):
            raise AssertionError(
                f"prior/truth replay mismatch for d{horizon}: {prior_risk} vs {truth_mean}"
            )
        return {
            "horizon": horizon,
            "root_action_index": root_action,
            "planned_value": planned_value,
            "expected_terminal_mse": prior_risk,
            "expected_terminal_rmsle": math.sqrt(max(prior_risk, 0.0)),
            "truth_losses": truth_losses,
        }


def comparison(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    """Compare lower-is-better `right` against `left`."""
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    difference = left_values - right_values
    wins = int(np.sum(difference > TIE_TOLERANCE))
    losses = int(np.sum(difference < -TIE_TOLERANCE))
    ties = int(len(difference) - wins - losses)
    left_mean = float(np.mean(left_values))
    right_mean = float(np.mean(right_values))
    return {
        "left_mean": left_mean,
        "right_mean": right_mean,
        "absolute_reduction": left_mean - right_mean,
        "relative_reduction": ((left_mean - right_mean) / left_mean if left_mean > 0 else 0.0),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def apply_gate(slice_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    horizons = (1, 2, 3)
    aggregate = {
        horizon: [
            loss
            for item in slice_results
            for loss in item["horizons"][f"d{horizon}"]["truth_losses"]
        ]
        for horizon in horizons
    }
    d2_vs_d1 = comparison(aggregate[1], aggregate[2])
    d3_vs_d2 = comparison(aggregate[2], aggregate[3])
    d3_vs_d1 = comparison(aggregate[1], aggregate[3])
    slice_d2_wins = 0
    slice_d3_wins = 0
    slice_d3_over_d1 = 0
    per_slice = []
    for item in slice_results:
        risks = {
            horizon: float(item["horizons"][f"d{horizon}"]["expected_terminal_mse"])
            for horizon in horizons
        }
        slice_d2_wins += int(risks[2] < risks[1] - TIE_TOLERANCE)
        slice_d3_wins += int(risks[3] < risks[2] - TIE_TOLERANCE)
        slice_d3_over_d1 += int(risks[3] < risks[1] - TIE_TOLERANCE)
        per_slice.append({"slice": item["slice"], "risks": risks})
    required_slice_wins = 6
    num_slices = len(VALIDATION_SLICES)
    conditions = {
        "d2_mean_reduction_at_least_5pct": d2_vs_d1["relative_reduction"] >= 0.05,
        "d3_mean_reduction_at_least_5pct": d3_vs_d2["relative_reduction"] >= 0.05,
        "d2_slice_wins_at_least_6_of_7": slice_d2_wins >= required_slice_wins,
        "d3_slice_wins_at_least_6_of_7": slice_d3_wins >= required_slice_wins,
        "d3_beats_d1_on_all_7_slices": slice_d3_over_d1 == num_slices,
        "d2_truth_cell_majority": d2_vs_d1["wins"] > d2_vs_d1["losses"],
        "d3_truth_cell_majority": d3_vs_d2["wins"] > d3_vs_d2["losses"],
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "d2_vs_d1": d2_vs_d1,
        "d3_vs_d2": d3_vs_d2,
        "d3_vs_d1": d3_vs_d1,
        "slice_d2_wins": slice_d2_wins,
        "slice_d3_wins": slice_d3_wins,
        "slice_d3_over_d1": slice_d3_over_d1,
        "per_slice": per_slice,
    }


def run_validation(source_root: Path) -> dict[str, Any]:
    source_binding = verify_source(source_root)
    source = load_source(source_root)
    domains = active_domains(source)
    assays = frozen_assays()
    slice_results = []
    for item in VALIDATION_SLICES:
        means, target_features = response_matrices(source, domains, item, assays)
        planner = ExactPlanner(categorical_likelihoods(means), target_features)
        horizons = {}
        for horizon in (1, 2, 3):
            result = planner.evaluate_horizon(horizon)
            result["root_action"] = assays[result["root_action_index"]].name
            horizons[f"d{horizon}"] = result
        slice_results.append(
            {
                "slice": item.name,
                "query_seed": item.query_seed,
                "num_domains": len(domains),
                "num_queries": target_features.shape[1],
                "horizons": horizons,
            }
        )
    gate = apply_gate(slice_results)
    return {
        "schema_version": "chembench-mopen-nonmyopic-opportunity-v2",
        "status": "passed" if gate["passed"] else "failed_closed",
        "model_calls": 0,
        "cost_usd": 0.0,
        "source": source_binding,
        "active_domains": list(domains),
        "active_domains_sha256": hashlib.sha256(
            json.dumps(list(domains), separators=(",", ":")).encode()
        ).hexdigest(),
        "assays": [{"name": assay.name, "values": assay.values} for assay in assays],
        "noise_level": NOISE_LEVEL,
        "execution_budget": EXECUTION_BUDGET,
        "validation_slices": [item.__dict__ for item in VALIDATION_SLICES],
        "slice_results": slice_results,
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.output}")
    result = run_validation(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": result["status"],
        "gate": result["gate"],
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
