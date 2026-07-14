"""Exact restricted-pool 20 Questions oracle control with no LLM calls."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


UCI_ZOO_SHA256 = "cddc71c26ab9bc82795b8f4ff114cade41885d92720c6af29ffb69bcf73f0315"
BOOLEAN_TRAITS = (
    "hair",
    "feathers",
    "eggs",
    "milk",
    "airborne",
    "aquatic",
    "predator",
    "toothed",
    "backbone",
    "breathes",
    "venomous",
    "fins",
    "tail",
    "domestic",
    "catsize",
)
LEG_VALUES = (0, 2, 4, 5, 6, 8)
TRAITS = BOOLEAN_TRAITS + tuple(f"legs_eq_{value}" for value in LEG_VALUES)


@dataclass(frozen=True)
class FrozenMatrix:
    names: tuple[str, ...]
    traits: tuple[str, ...]
    values: np.ndarray
    source_sha256: str


@dataclass(frozen=True)
class OracleControlConfig:
    num_trials: int = 2000
    num_rounds: int = 8
    seed: int = 1304
    bootstrap_replicates: int = 10_000
    restricted_widths: tuple[int, ...] = (2, 3, 4, 6, 8)
    primary_width: int = 3
    score_noise_sds: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.20, 0.40)


@dataclass(frozen=True)
class Selection:
    action: int
    scores: dict[int, float]
    candidate_pool: tuple[int, ...]


@dataclass(frozen=True)
class PolicyTrace:
    accuracy: tuple[float, ...]
    entropy: tuple[float, ...]
    actions: tuple[int, ...]


def _stable_seed(*parts: Any) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def _entropy(support: np.ndarray) -> float:
    return math.log(len(support)) if len(support) else 0.0


def load_frozen_matrix(path: Path) -> FrozenMatrix:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != UCI_ZOO_SHA256:
        raise ValueError(f"Unexpected UCI Zoo SHA-256: {digest}")

    rows = list(csv.reader(raw.decode("utf-8").splitlines()))
    if len(rows) != 101 or any(len(row) != 18 for row in rows):
        raise ValueError("Expected the 101-row, 18-column UCI Zoo release")

    seen_names: dict[str, int] = {}
    names: list[str] = []
    values: list[list[bool]] = []
    raw_rows: dict[str, list[str]] = {}
    for row in rows:
        raw_name = row[0]
        seen_names[raw_name] = seen_names.get(raw_name, 0) + 1
        name = raw_name if seen_names[raw_name] == 1 else f"{raw_name}_{seen_names[raw_name]}"
        boolean_values = [value == "1" for value in row[1:13]] + [value == "1" for value in row[14:17]]
        legs = int(row[13])
        names.append(name)
        values.append(boolean_values + [legs == value for value in LEG_VALUES])
        raw_rows[name] = row

    matrix = np.asarray(values, dtype=bool)
    if matrix.shape != (101, len(TRAITS)):
        raise ValueError(f"Unexpected frozen matrix shape: {matrix.shape}")

    # These released rows make accidental column reordering visible immediately.
    expected_checks = {
        "aardvark": {"hair": True, "milk": True, "eggs": False, "legs_eq_4": True},
        "bass": {"aquatic": True, "fins": True, "milk": False, "legs_eq_0": True},
        "honeybee": {"eggs": True, "airborne": True, "legs_eq_6": True},
    }
    trait_index = {trait: index for index, trait in enumerate(TRAITS)}
    name_index = {name: index for index, name in enumerate(names)}
    for animal, checks in expected_checks.items():
        for trait, expected in checks.items():
            if bool(matrix[name_index[animal], trait_index[trait]]) != expected:
                raise ValueError(f"Spot check failed for {animal} / {trait}")

    return FrozenMatrix(tuple(names), TRAITS, matrix, digest)


def _available_actions(history: tuple[tuple[int, bool], ...], num_traits: int) -> tuple[int, ...]:
    asked = {action for action, _answer in history}
    return tuple(action for action in range(num_traits) if action not in asked)


def candidate_pool(
    history: tuple[tuple[int, bool], ...],
    *,
    width: int,
    num_traits: int,
    seed: int,
    trial_index: int,
) -> tuple[int, ...]:
    if width <= 0:
        raise ValueError("candidate width must be positive")
    available = _available_actions(history, num_traits)
    if not available:
        return ()
    rng = np.random.default_rng(_stable_seed(seed, "candidate-pool", trial_index, history))
    order = np.asarray(available, dtype=int)[rng.permutation(len(available))]
    return tuple(int(action) for action in order[: min(width, len(order))])


def _posterior_after(matrix: FrozenMatrix, support: np.ndarray, action: int, answer: bool) -> np.ndarray:
    return support[matrix.values[support, action] == answer]


def expected_information_gain(matrix: FrozenMatrix, support: np.ndarray, action: int) -> float:
    count = len(support)
    if count <= 1:
        return 0.0
    yes_count = int(np.count_nonzero(matrix.values[support, action]))
    no_count = count - yes_count
    expected_entropy = 0.0
    for branch_count in (yes_count, no_count):
        if branch_count:
            expected_entropy += (branch_count / count) * math.log(branch_count)
    return math.log(count) - expected_entropy


def _score_noise(
    config: OracleControlConfig,
    *,
    trial_index: int,
    history: tuple[tuple[int, bool], ...],
    action: int,
    stage: str,
    noise_sd: float,
) -> float:
    if noise_sd == 0.0:
        return 0.0
    rng = np.random.default_rng(_stable_seed(config.seed, "score-noise", trial_index, history, action, stage))
    return float(rng.normal(0.0, noise_sd))


def _select_action(
    matrix: FrozenMatrix,
    support: np.ndarray,
    history: tuple[tuple[int, bool], ...],
    *,
    width: int,
    depth: int,
    config: OracleControlConfig,
    trial_index: int,
    noise_sd: float,
    remaining_rounds: int,
) -> Selection:
    if depth not in (1, 2):
        raise ValueError("oracle control only supports depths 1 and 2")
    if remaining_rounds <= 0:
        raise ValueError("remaining_rounds must be positive")
    # A finite-horizon planner cannot score a continuation after the final action.
    depth = min(depth, remaining_rounds)
    pool = candidate_pool(
        history,
        width=width,
        num_traits=len(matrix.traits),
        seed=config.seed,
        trial_index=trial_index,
    )
    if not pool:
        raise ValueError("cannot select from an empty candidate pool")

    scores: dict[int, float] = {}
    support_size = len(support)
    for action in pool:
        immediate = expected_information_gain(matrix, support, action)
        if depth == 1:
            scores[action] = immediate + _score_noise(
                config,
                trial_index=trial_index,
                history=history,
                action=action,
                stage="one-step",
                noise_sd=noise_sd,
            )
            continue

        continuation = 0.0
        for answer in (False, True):
            next_support = _posterior_after(matrix, support, action, answer)
            if not len(next_support):
                continue
            next_history = history + ((action, answer),)
            future_pool = candidate_pool(
                next_history,
                width=width,
                num_traits=len(matrix.traits),
                seed=config.seed,
                trial_index=trial_index,
            )
            if future_pool:
                future_scores = [
                    expected_information_gain(matrix, next_support, future_action)
                    + _score_noise(
                        config,
                        trial_index=trial_index,
                        history=next_history,
                        action=future_action,
                        stage="two-step-future",
                        noise_sd=noise_sd,
                    )
                    for future_action in future_pool
                ]
                continuation += (len(next_support) / support_size) * max(future_scores)
        scores[action] = immediate + continuation + _score_noise(
            config,
            trial_index=trial_index,
            history=history,
            action=action,
            stage="two-step-root",
            noise_sd=noise_sd,
        )

    action = max(pool, key=lambda candidate: (scores[candidate], -candidate))
    return Selection(action=action, scores=scores, candidate_pool=pool)


def run_policy(
    matrix: FrozenMatrix,
    target: int,
    *,
    width: int,
    depth: int,
    config: OracleControlConfig,
    trial_index: int,
    noise_sd: float,
) -> PolicyTrace:
    support = np.arange(len(matrix.names), dtype=int)
    history: tuple[tuple[int, bool], ...] = ()
    accuracy: list[float] = []
    entropy: list[float] = []
    actions: list[int] = []
    for round_index in range(config.num_rounds):
        selection = _select_action(
            matrix,
            support,
            history,
            width=width,
            depth=depth,
            config=config,
            trial_index=trial_index,
            noise_sd=noise_sd,
            remaining_rounds=config.num_rounds - round_index,
        )
        answer = bool(matrix.values[target, selection.action])
        support = _posterior_after(matrix, support, selection.action, answer)
        history = history + ((selection.action, answer),)
        actions.append(selection.action)
        accuracy.append(float(int(support[0]) == target))
        entropy.append(_entropy(support))
    return PolicyTrace(tuple(accuracy), tuple(entropy), tuple(actions))


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    return float(np.mean(array)) if len(array) else 0.0


def _bootstrap_mean_ci(values: np.ndarray, *, replicates: int, seed: int) -> tuple[float, float]:
    if not len(values):
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    for _start in range(0, replicates, 512):
        batch = min(512, replicates - _start)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        means.append(np.mean(values[indices], axis=1))
    samples = np.concatenate(means)
    return (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))


def _policy_summary(traces: list[PolicyTrace]) -> dict[str, Any]:
    accuracy = np.asarray([trace.accuracy for trace in traces], dtype=float)
    entropy = np.asarray([trace.entropy for trace in traces], dtype=float)
    return {
        "accuracy_auc_mean": float(np.mean(accuracy)),
        "final_accuracy_mean": float(np.mean(accuracy[:, -1])),
        "final_entropy_mean": float(np.mean(entropy[:, -1])),
        "round_accuracy_mean": [float(value) for value in np.mean(accuracy, axis=0)],
        "round_entropy_mean": [float(value) for value in np.mean(entropy, axis=0)],
    }


def _paired_summary(
    first: list[PolicyTrace], second: list[PolicyTrace], *, config: OracleControlConfig, label: str
) -> dict[str, Any]:
    first_accuracy = np.asarray([trace.accuracy for trace in first], dtype=float)
    second_accuracy = np.asarray([trace.accuracy for trace in second], dtype=float)
    first_entropy = np.asarray([trace.entropy for trace in first], dtype=float)
    second_entropy = np.asarray([trace.entropy for trace in second], dtype=float)
    auc_delta = np.mean(first_accuracy, axis=1) - np.mean(second_accuracy, axis=1)
    final_accuracy_delta = first_accuracy[:, -1] - second_accuracy[:, -1]
    final_entropy_delta = first_entropy[:, -1] - second_entropy[:, -1]
    ci = _bootstrap_mean_ci(
        auc_delta,
        replicates=config.bootstrap_replicates,
        seed=_stable_seed(config.seed, "bootstrap", label),
    )
    return {
        "accuracy_auc_delta_mean": float(np.mean(auc_delta)),
        "accuracy_auc_delta_ci95": [ci[0], ci[1]],
        "final_accuracy_delta_mean": float(np.mean(final_accuracy_delta)),
        "final_entropy_delta_mean": float(np.mean(final_entropy_delta)),
        "round_accuracy_delta_mean": [float(value) for value in np.mean(first_accuracy - second_accuracy, axis=0)],
        "wins_ties_losses": [
            int(np.count_nonzero(auc_delta > 0.0)),
            int(np.count_nonzero(auc_delta == 0.0)),
            int(np.count_nonzero(auc_delta < 0.0)),
        ],
    }


def _run_condition(
    matrix: FrozenMatrix,
    targets: np.ndarray,
    *,
    width: int,
    noise_sd: float,
    config: OracleControlConfig,
    include_width_controls: bool,
) -> dict[str, Any]:
    one_step: list[PolicyTrace] = []
    two_step: list[PolicyTrace] = []
    wider: dict[int, list[PolicyTrace]] = {}
    if include_width_controls:
        for candidate_width in (min(2 * width, len(matrix.traits)), min(4 * width, len(matrix.traits))):
            if candidate_width > width:
                wider[candidate_width] = []
    disagreement: list[float] = []
    q2_regret: list[float] = []

    for trial_index, target in enumerate(targets):
        support = np.arange(len(matrix.names), dtype=int)
        history: tuple[tuple[int, bool], ...] = ()
        first = _select_action(
            matrix,
            support,
            history,
            width=width,
            depth=1,
            config=config,
            trial_index=trial_index,
            noise_sd=noise_sd,
            remaining_rounds=config.num_rounds,
        )
        second = _select_action(
            matrix,
            support,
            history,
            width=width,
            depth=2,
            config=config,
            trial_index=trial_index,
            noise_sd=noise_sd,
            remaining_rounds=config.num_rounds,
        )
        disagreement.append(float(first.action != second.action))
        q2_regret.append(max(second.scores.values()) - second.scores[first.action])
        one_step.append(run_policy(matrix, int(target), width=width, depth=1, config=config, trial_index=trial_index, noise_sd=noise_sd))
        two_step.append(run_policy(matrix, int(target), width=width, depth=2, config=config, trial_index=trial_index, noise_sd=noise_sd))
        for candidate_width, traces in wider.items():
            traces.append(run_policy(matrix, int(target), width=candidate_width, depth=1, config=config, trial_index=trial_index, noise_sd=noise_sd))

    result: dict[str, Any] = {
        "candidate_width": width,
        "score_noise_sd": noise_sd,
        "one_step": _policy_summary(one_step),
        "two_step": _policy_summary(two_step),
        "depth_two_minus_one": _paired_summary(two_step, one_step, config=config, label=f"d2-d1-k{width}-n{noise_sd}"),
        "first_decision_disagreement_rate": _mean(disagreement),
        "mean_q2_regret_of_one_step_first_action": _mean(q2_regret),
    }
    if wider:
        result["wider_one_step_controls"] = {
            str(candidate_width): {
                "summary": _policy_summary(traces),
                "width_minus_base_one_step": _paired_summary(
                    traces, one_step, config=config, label=f"wide{candidate_width}-base{width}-n{noise_sd}"
                ),
                "depth_two_minus_width": _paired_summary(
                    two_step, traces, config=config, label=f"d2-wide{candidate_width}-k{width}-n{noise_sd}"
                ),
            }
            for candidate_width, traces in wider.items()
        }
    return result


def run_oracle_control(matrix: FrozenMatrix, config: OracleControlConfig) -> dict[str, Any]:
    if config.primary_width not in config.restricted_widths:
        raise ValueError("primary_width must appear in restricted_widths")
    rng = np.random.default_rng(config.seed)
    targets = rng.integers(0, len(matrix.names), size=config.num_trials)
    exhaustive = _run_condition(
        matrix, targets, width=len(matrix.traits), noise_sd=0.0, config=config, include_width_controls=False
    )
    restricted = [
        _run_condition(matrix, targets, width=width, noise_sd=0.0, config=config, include_width_controls=True)
        for width in config.restricted_widths
    ]
    noise_frontier = [
        _run_condition(
            matrix, targets, width=config.primary_width, noise_sd=noise_sd, config=config, include_width_controls=False
        )
        for noise_sd in config.score_noise_sds
    ]

    restricted_gap = any(
        row["depth_two_minus_one"]["accuracy_auc_delta_ci95"][0] > 0.0 for row in restricted
    )
    zero_noise_gain = noise_frontier[0]["depth_two_minus_one"]["accuracy_auc_delta_mean"]
    high_noise_gain = noise_frontier[-1]["depth_two_minus_one"]["accuracy_auc_delta_mean"]
    noise_headwind = high_noise_gain < zero_noise_gain
    boundary = next(
        (
            row["score_noise_sd"]
            for row in noise_frontier
            if row["depth_two_minus_one"]["accuracy_auc_delta_ci95"][0] <= 0.0
            <= row["depth_two_minus_one"]["accuracy_auc_delta_ci95"][1]
        ),
        None,
    )
    decision = "proceed_to_llm_exploration" if restricted_gap and noise_headwind else "stop_and_discuss"
    return {
        "schema_version": 1,
        "no_llm_calls": True,
        "data": {
            "num_entities": len(matrix.names),
            "num_traits": len(matrix.traits),
            "traits": list(matrix.traits),
            "source_sha256": matrix.source_sha256,
        },
        "config": {
            "num_trials": config.num_trials,
            "num_rounds": config.num_rounds,
            "seed": config.seed,
            "bootstrap_replicates": config.bootstrap_replicates,
            "restricted_widths": list(config.restricted_widths),
            "primary_width": config.primary_width,
            "score_noise_sds": list(config.score_noise_sds),
        },
        "exhaustive": exhaustive,
        "restriction": restricted,
        "noise_frontier": noise_frontier,
        "decision": {
            "restricted_gap_with_ci_excluding_zero": restricted_gap,
            "zero_noise_primary_gap": zero_noise_gain,
            "high_noise_primary_gap": high_noise_gain,
            "noise_is_headwind": noise_headwind,
            "planning_viability_boundary_noise_sd": boundary,
            "verdict": decision,
        },
    }


def _fmt(value: float) -> str:
    return f"{value:+.4f}"


def render_report(summary: dict[str, Any]) -> str:
    exhaustive = summary["exhaustive"]
    lines = [
        "# Restricted-Pool Oracle Control",
        "",
        "## Scope",
        "",
        "This is an exact scripted 20 Questions control with no LLM calls. Animal identity is the target, the answerer is the frozen UCI Zoo trait matrix, and the endpoint is deterministic MAP identity decoding. The preregistration is `ORACLE_CONTROL_PREREGISTRATION.md`.",
        "",
        "## Exhaustive Pool",
        "",
        "| Candidate pool | d1 AUC | d2 AUC | d2 - d1 AUC | 95% CI | Round accuracy delta |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        "| all traits | "
        f"{exhaustive['one_step']['accuracy_auc_mean']:.4f} | "
        f"{exhaustive['two_step']['accuracy_auc_mean']:.4f} | "
        f"{_fmt(exhaustive['depth_two_minus_one']['accuracy_auc_delta_mean'])} | "
        f"[{exhaustive['depth_two_minus_one']['accuracy_auc_delta_ci95'][0]:+.4f}, "
        f"{exhaustive['depth_two_minus_one']['accuracy_auc_delta_ci95'][1]:+.4f}] | "
        + ", ".join(_fmt(value) for value in exhaustive['depth_two_minus_one']['round_accuracy_delta_mean'])
        + " |",
        "",
        "## Restriction and Width",
        "",
        "| K | d1 AUC | d2 AUC | d2 - d1 AUC | 95% CI | first-action disagreement | d1 width control(s) |",
        "| ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in summary["restriction"]:
        pair = row["depth_two_minus_one"]
        controls = []
        for width, control in row.get("wider_one_step_controls", {}).items():
            control_pair = control["depth_two_minus_width"]
            controls.append(f"d2-d1(K={width}) {_fmt(control_pair['accuracy_auc_delta_mean'])}")
        lines.append(
            f"| {row['candidate_width']} | {row['one_step']['accuracy_auc_mean']:.4f} | "
            f"{row['two_step']['accuracy_auc_mean']:.4f} | {_fmt(pair['accuracy_auc_delta_mean'])} | "
            f"[{pair['accuracy_auc_delta_ci95'][0]:+.4f}, {pair['accuracy_auc_delta_ci95'][1]:+.4f}] | "
            f"{row['first_decision_disagreement_rate']:.3f} | {'; '.join(controls) or '-'} |"
        )
    lines.extend(["", "## Noise Frontier", "", "| Noise SD (nats) | d2 - d1 AUC | 95% CI | First-action disagreement |", "| ---: | ---: | --- | ---: |"])
    for row in summary["noise_frontier"]:
        pair = row["depth_two_minus_one"]
        lines.append(
            f"| {row['score_noise_sd']:.3f} | {_fmt(pair['accuracy_auc_delta_mean'])} | "
            f"[{pair['accuracy_auc_delta_ci95'][0]:+.4f}, {pair['accuracy_auc_delta_ci95'][1]:+.4f}] | "
            f"{row['first_decision_disagreement_rate']:.3f} |"
        )
    decision = summary["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Restricted gap with a paired 95% CI excluding zero: `{decision['restricted_gap_with_ci_excluding_zero']}`.",
            f"- Primary K=3 depth gain: `{decision['zero_noise_primary_gap']:+.4f}` at zero noise and `{decision['high_noise_primary_gap']:+.4f}` at the largest injected noise.",
            f"- Noise acts as the preregistered headwind: `{decision['noise_is_headwind']}`.",
            f"- First noise level with a CI including zero: `{decision['planning_viability_boundary_noise_sd']}`.",
            f"- **Verdict: `{decision['verdict']}`.**",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python scripts/nonmyopic_oracle_control.py",
            "pytest -q tests/test_nonmyopic_oracle_control.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_numbers(text: str, cast: type[int] | type[float]) -> tuple[int, ...] | tuple[float, ...]:
    values = tuple(cast(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/nonmyopic/uci_zoo.data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/nonmyopic"))
    parser.add_argument("--num-trials", type=int, default=2000)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1304)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--restricted-widths", default="2,3,4,6,8")
    parser.add_argument("--primary-width", type=int, default=3)
    parser.add_argument("--score-noise-sds", default="0,0.025,0.05,0.10,0.20,0.40")
    args = parser.parse_args()
    config = OracleControlConfig(
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        restricted_widths=tuple(_parse_numbers(args.restricted_widths, int)),
        primary_width=args.primary_width,
        score_noise_sds=tuple(_parse_numbers(args.score_noise_sds, float)),
    )
    matrix = load_frozen_matrix(args.data)
    summary = run_oracle_control(matrix, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ORACLE_CONTROL.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "ORACLE_CONTROL.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
