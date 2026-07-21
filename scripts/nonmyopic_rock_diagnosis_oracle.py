"""Exact depth-versus-width oracle for the Rock Diagnosis information task.

This script evaluates a held-out paper map with no LLM calls.  Candidate cells
are deterministic legal-action samples that stand in for restricted proposal
sets; all target sampling, sensor outcomes, likelihoods, posterior updates, and
decodes are exact and paired across policy arms.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Literal

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
from environments.rock_diagnosis.core import EPSILON


ArmName = Literal["d1_shared", "d2", "d1_call_matched_width"]


@dataclass(frozen=True)
class OracleConfig:
    """Frozen configuration for the exact Rock Diagnosis confirmation."""

    map_name: str = "3-6"
    num_trials: int = 2_000
    num_rounds: int = 8
    candidate_widths: tuple[int, ...] = (2, 3, 4)
    seed: int = 2304
    trial_offset: int = 0
    bootstrap_replicates: int = 10_000
    half_efficiency_distance: float = math.log(2.0)

    def validate(self) -> None:
        get_paper_map(self.map_name)
        if self.num_trials <= 0 or self.trial_offset < 0 or self.num_rounds < 2:
            raise ValueError("num_trials must be positive, trial_offset non-negative, and num_rounds at least two")
        if not self.candidate_widths or any(width <= 0 for width in self.candidate_widths):
            raise ValueError("candidate_widths must contain positive values")
        if len(set(self.candidate_widths)) != len(self.candidate_widths):
            raise ValueError("candidate_widths must be distinct")
        if self.bootstrap_replicates <= 0 or self.half_efficiency_distance <= 0.0:
            raise ValueError("bootstrap_replicates and half_efficiency_distance must be positive")


@dataclass(frozen=True)
class Selection:
    action: str
    candidate_pool: tuple[str, ...]
    base_candidate_pool: tuple[str, ...]
    scores: dict[str, float]
    candidate_call_budget: int
    virtual_future_cells: int
    width_contains_base: bool
    call_budget_matches_virtual_depth_two: bool


@dataclass(frozen=True)
class StepTrace:
    action: str
    position_before: tuple[int, int]
    entropy: float
    selected_eig: float
    candidate_pool: tuple[str, ...]
    base_candidate_pool: tuple[str, ...]
    candidate_call_budget: int
    virtual_future_cells: int
    width_contains_base: bool
    call_budget_matches_virtual_depth_two: bool


@dataclass(frozen=True)
class PolicyTrace:
    arm: ArmName
    trial_index: int
    truth_index: int
    final_entropy: float
    final_map_accuracy: float
    final_truth_log_probability: float
    steps: tuple[StepTrace, ...]


def _stable_seed(*parts: Any) -> int:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def _uniform(*parts: Any) -> float:
    return _stable_seed(*parts) / 2**64


def _candidate_cell(
    model: RockDiagnosisModel,
    *,
    position: tuple[int, int],
    history: tuple[tuple[str, str | None], ...],
    trial_index: int,
    width: int,
    config: OracleConfig,
    label: Any,
) -> tuple[str, ...]:
    """Return a legal proposal cell; prefixes are nested as K increases."""

    legal = model.legal_actions(position)
    rng = np.random.default_rng(
        _stable_seed(config.seed, "rock-diagnosis-candidate-cell", trial_index, position, history, label)
    )
    order = rng.permutation(len(legal))
    return tuple(legal[index] for index in order[: min(width, len(legal))])


def _nonzero_future_cells(
    model: RockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    base_candidate_pool: tuple[str, ...],
) -> tuple[tuple[str, str | None], ...]:
    return tuple(
        (action, outcome)
        for action in base_candidate_pool
        for outcome in model.outcomes(action)
        if model.outcome_probability(position, belief, action, outcome) > EPSILON
    )


def _depth_two_value(
    model: RockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    history: tuple[tuple[str, str | None], ...],
    trial_index: int,
    action: str,
    width: int,
    config: OracleConfig,
) -> float:
    """Two-step sum of incremental EIG, equivalent to expected final information."""

    value = model.expected_information_gain(position, belief, action)
    next_position = model.next_position(position, action)
    for outcome in model.outcomes(action):
        probability = model.outcome_probability(position, belief, action, outcome)
        if probability <= EPSILON:
            continue
        posterior = model.posterior(position, belief, action, outcome)
        future_history = history + ((action, outcome),)
        future_candidates = _candidate_cell(
            model,
            position=next_position,
            history=future_history,
            trial_index=trial_index,
            width=width,
            config=config,
            label=("future", action, outcome),
        )
        value += probability * max(
            model.expected_information_gain(next_position, posterior, future_action)
            for future_action in future_candidates
        )
    return value


def _call_matched_width_pool(
    model: RockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    history: tuple[tuple[str, str | None], ...],
    trial_index: int,
    width: int,
    config: OracleConfig,
    base_candidate_pool: tuple[str, ...],
) -> tuple[tuple[str, ...], int]:
    """Spend one current-state cell per virtual d2 outcome cell, then dedupe."""

    future_cells = _nonzero_future_cells(
        model,
        position=position,
        belief=belief,
        base_candidate_pool=base_candidate_pool,
    )
    union = list(base_candidate_pool)
    for cell_index, _future_cell in enumerate(future_cells):
        extra = _candidate_cell(
            model,
            position=position,
            history=history,
            trial_index=trial_index,
            width=width,
            config=config,
            label=("current-width", cell_index),
        )
        union.extend(candidate for candidate in extra if candidate not in union)
    return tuple(union), len(future_cells)


def select_action(
    model: RockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    history: tuple[tuple[str, str | None], ...],
    trial_index: int,
    width: int,
    arm: ArmName,
    planning_depth: int,
    config: OracleConfig,
) -> Selection:
    """Choose a legal action for one arm at one exact belief state."""

    if planning_depth not in (1, 2):
        raise ValueError("planning_depth must be one or two")
    base_candidate_pool = _candidate_cell(
        model,
        position=position,
        history=history,
        trial_index=trial_index,
        width=width,
        config=config,
        label="base",
    )
    future_cells = _nonzero_future_cells(
        model,
        position=position,
        belief=belief,
        base_candidate_pool=base_candidate_pool,
    )

    if arm == "d1_call_matched_width" and planning_depth == 2:
        candidate_pool, virtual_future_cells = _call_matched_width_pool(
            model,
            position=position,
            belief=belief,
            history=history,
            trial_index=trial_index,
            width=width,
            config=config,
            base_candidate_pool=base_candidate_pool,
        )
        scores = {
            action: model.expected_information_gain(position, belief, action) for action in candidate_pool
        }
        candidate_call_budget = 1 + virtual_future_cells
        width_contains_base = set(base_candidate_pool).issubset(candidate_pool)
        call_budget_matches = candidate_call_budget == 1 + len(future_cells)
    elif arm == "d2" and planning_depth == 2:
        candidate_pool = base_candidate_pool
        scores = {
            action: _depth_two_value(
                model,
                position=position,
                belief=belief,
                history=history,
                trial_index=trial_index,
                action=action,
                width=width,
                config=config,
            )
            for action in candidate_pool
        }
        virtual_future_cells = len(future_cells)
        candidate_call_budget = 1 + virtual_future_cells
        width_contains_base = True
        call_budget_matches = True
    else:
        candidate_pool = base_candidate_pool
        scores = {
            action: model.expected_information_gain(position, belief, action) for action in candidate_pool
        }
        virtual_future_cells = 0
        candidate_call_budget = 1
        width_contains_base = True
        call_budget_matches = True

    action = max(candidate_pool, key=lambda candidate: (scores[candidate], -candidate_pool.index(candidate)))
    return Selection(
        action=action,
        candidate_pool=candidate_pool,
        base_candidate_pool=base_candidate_pool,
        scores=scores,
        candidate_call_budget=candidate_call_budget,
        virtual_future_cells=virtual_future_cells,
        width_contains_base=width_contains_base,
        call_budget_matches_virtual_depth_two=call_budget_matches,
    )


def run_policy(
    model: RockDiagnosisModel,
    *,
    trial_index: int,
    width: int,
    arm: ArmName,
    config: OracleConfig,
) -> PolicyTrace:
    truth_rng = np.random.default_rng(_stable_seed(config.seed, "rock-diagnosis-truth", trial_index))
    truth_index = int(truth_rng.integers(len(model.hidden_states)))
    belief = model.initial_belief.copy()
    position = model.map_spec.start_position
    history: tuple[tuple[str, str | None], ...] = ()
    check_counts: dict[tuple[tuple[int, int], int], int] = {}
    steps: list[StepTrace] = []

    for round_index in range(config.num_rounds):
        planning_depth = min(2, config.num_rounds - round_index)
        selection = select_action(
            model,
            position=position,
            belief=belief,
            history=history,
            trial_index=trial_index,
            width=width,
            arm=arm,
            planning_depth=planning_depth,
            config=config,
        )
        check_id = model.check_id(selection.action)
        if check_id is None:
            outcome: str | None = None
        else:
            key = (position, check_id)
            repeat_index = check_counts.get(key, 0)
            check_counts[key] = repeat_index + 1
            probability_good = float(model.likelihood_vector(position, selection.action, "good")[truth_index])
            outcome = (
                "good"
                if _uniform(config.seed, "rock-diagnosis-observation", trial_index, position, check_id, repeat_index)
                < probability_good
                else "bad"
            )
        selected_eig = model.expected_information_gain(position, belief, selection.action)
        belief = model.posterior(position, belief, selection.action, outcome)
        steps.append(
            StepTrace(
                action=selection.action,
                position_before=position,
                entropy=model.entropy(belief),
                selected_eig=selected_eig,
                candidate_pool=selection.candidate_pool,
                base_candidate_pool=selection.base_candidate_pool,
                candidate_call_budget=selection.candidate_call_budget,
                virtual_future_cells=selection.virtual_future_cells,
                width_contains_base=selection.width_contains_base,
                call_budget_matches_virtual_depth_two=selection.call_budget_matches_virtual_depth_two,
            )
        )
        history = history + ((selection.action, outcome),)
        position = model.next_position(position, selection.action)

    map_index = model.decode_map_index(belief)
    return PolicyTrace(
        arm=arm,
        trial_index=trial_index,
        truth_index=truth_index,
        final_entropy=model.entropy(belief),
        final_map_accuracy=float(map_index == truth_index),
        final_truth_log_probability=float(math.log(max(float(belief[truth_index]), np.finfo(float).tiny))),
        steps=tuple(steps),
    )


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, replicates: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    for start in range(0, replicates, 512):
        batch = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        means.append(np.mean(values[indices], axis=1))
    samples = np.concatenate(means)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _mean(values: Iterable[float]) -> float:
    array = np.asarray(tuple(values), dtype=float)
    return float(np.mean(array)) if len(array) else 0.0


def _trace_summary(model: RockDiagnosisModel, traces: list[PolicyTrace]) -> dict[str, Any]:
    initial_moves = [model.is_move(trace.steps[0].action) for trace in traces]
    return {
        "final_entropy_mean": _mean(trace.final_entropy for trace in traces),
        "final_map_accuracy_mean": _mean(trace.final_map_accuracy for trace in traces),
        "final_truth_log_probability_mean": _mean(trace.final_truth_log_probability for trace in traces),
        "initial_move_rate": _mean(initial_moves),
        "mean_selected_eig": _mean(step.selected_eig for trace in traces for step in trace.steps),
        "mean_candidate_pool_size": _mean(len(step.candidate_pool) for trace in traces for step in trace.steps),
        "mean_candidate_call_budget": _mean(step.candidate_call_budget for trace in traces for step in trace.steps),
    }


def _paired_comparison(
    first: list[PolicyTrace],
    second: list[PolicyTrace],
    *,
    config: OracleConfig,
    label: str,
) -> dict[str, Any]:
    entropy_gain = np.asarray(
        [second[index].final_entropy - first[index].final_entropy for index in range(len(first))], dtype=float
    )
    map_accuracy_delta = np.asarray(
        [first[index].final_map_accuracy - second[index].final_map_accuracy for index in range(len(first))],
        dtype=float,
    )
    truth_log_delta = np.asarray(
        [
            first[index].final_truth_log_probability - second[index].final_truth_log_probability
            for index in range(len(first))
        ],
        dtype=float,
    )
    ci = _bootstrap_mean_ci(
        entropy_gain,
        seed=_stable_seed(config.seed, "rock-diagnosis-bootstrap", label),
        replicates=config.bootstrap_replicates,
    )
    return {
        "final_entropy_reduction_mean": float(np.mean(entropy_gain)),
        "final_entropy_reduction_ci95": [ci[0], ci[1]],
        "final_map_accuracy_delta_mean": float(np.mean(map_accuracy_delta)),
        "final_truth_log_probability_delta_mean": float(np.mean(truth_log_delta)),
        "wins_ties_losses": [
            int(np.count_nonzero(entropy_gain > EPSILON)),
            int(np.count_nonzero(np.abs(entropy_gain) <= EPSILON)),
            int(np.count_nonzero(entropy_gain < -EPSILON)),
        ],
    }


def _trace_payload(trace: PolicyTrace) -> dict[str, Any]:
    return {
        "arm": trace.arm,
        "trial_index": trace.trial_index,
        "truth_index": trace.truth_index,
        "final_entropy": trace.final_entropy,
        "final_map_accuracy": trace.final_map_accuracy,
        "final_truth_log_probability": trace.final_truth_log_probability,
        "steps": [
            {
                "action": step.action,
                "position_before": list(step.position_before),
                "entropy": step.entropy,
                "selected_eig": step.selected_eig,
                "candidate_pool": list(step.candidate_pool),
                "base_candidate_pool": list(step.base_candidate_pool),
                "candidate_call_budget": step.candidate_call_budget,
                "virtual_future_cells": step.virtual_future_cells,
                "width_contains_base": step.width_contains_base,
                "call_budget_matches_virtual_depth_two": step.call_budget_matches_virtual_depth_two,
            }
            for step in trace.steps
        ],
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Retain all inferential inputs without tracking the full per-step trace tree."""

    compact = {key: value for key, value in summary.items() if key != "widths"}
    compact_widths: dict[str, Any] = {}
    for width, row in summary["widths"].items():
        compact_row = {key: value for key, value in row.items() if key != "traces"}
        compact_row["trial_metrics"] = {
            arm: {
                "trial_indices": [trace["trial_index"] for trace in traces],
                "truth_indices": [trace["truth_index"] for trace in traces],
                "final_entropy": [trace["final_entropy"] for trace in traces],
                "final_map_accuracy": [trace["final_map_accuracy"] for trace in traces],
                "final_truth_log_probability": [trace["final_truth_log_probability"] for trace in traces],
                "initial_actions": [trace["steps"][0]["action"] for trace in traces],
            }
            for arm, traces in row["traces"].items()
        }
        compact_widths[width] = compact_row
    compact["widths"] = compact_widths
    return compact


def write_full_traces(summary: dict[str, Any], output_path: Path) -> None:
    """Write complete actions/candidates as a compressed, locally ignored audit log."""

    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for width, row in summary["widths"].items():
            for arm, traces in row["traces"].items():
                for trace in traces:
                    handle.write(json.dumps({"candidate_width": width, "arm": arm, **trace}, sort_keys=True))
                    handle.write("\n")


def run_oracle(config: OracleConfig) -> dict[str, Any]:
    """Run all paired arms and return a self-contained exact-oracle report."""

    config.validate()
    model = RockDiagnosisModel(get_paper_map(config.map_name), half_efficiency_distance=config.half_efficiency_distance)
    by_width: dict[str, Any] = {}
    any_pass = False

    for width in config.candidate_widths:
        traces: dict[ArmName, list[PolicyTrace]] = {
            arm: [
                run_policy(model, trial_index=trial_index, width=width, arm=arm, config=config)
                for trial_index in range(config.trial_offset, config.trial_offset + config.num_trials)
            ]
            for arm in ("d1_shared", "d2", "d1_call_matched_width")
        }
        d2_minus_d1 = _paired_comparison(
            traces["d2"], traces["d1_shared"], config=config, label=f"d2-d1-k{width}"
        )
        d2_minus_width = _paired_comparison(
            traces["d2"], traces["d1_call_matched_width"], config=config, label=f"d2-width-k{width}"
        )
        passed_width = (
            d2_minus_d1["final_entropy_reduction_ci95"][0] > 0.0
            and d2_minus_width["final_entropy_reduction_ci95"][0] > 0.0
        )
        any_pass = any_pass or passed_width
        d1_steps = traces["d1_shared"]
        d2_steps = traces["d2"]
        width_steps = traces["d1_call_matched_width"]
        by_width[str(width)] = {
            "summaries": {arm: _trace_summary(model, arm_traces) for arm, arm_traces in traces.items()},
            "comparisons": {
                "d2_minus_shared_d1": d2_minus_d1,
                "d2_minus_call_matched_width": d2_minus_width,
            },
            "mechanics": {
                "initial_base_candidate_cells_shared": all(
                    d1_steps[index].steps[0].base_candidate_pool == d2_steps[index].steps[0].base_candidate_pool
                    for index in range(config.num_trials)
                ),
                "initial_width_contains_shared_base": all(
                    set(d1_steps[index].steps[0].base_candidate_pool).issubset(
                        width_steps[index].steps[0].candidate_pool
                    )
                    for index in range(config.num_trials)
                ),
                "width_cells_match_virtual_depth_two_cells": all(
                    step.call_budget_matches_virtual_depth_two
                    for trace in width_steps
                    for step in trace.steps
                ),
                "all_selected_actions_legal": all(
                    step.action in model.legal_actions(step.position_before)
                    for arm_traces in traces.values()
                    for trace in arm_traces
                    for step in trace.steps
                ),
            },
            "passed_confirmation_gate": passed_width,
            "traces": {arm: [_trace_payload(trace) for trace in arm_traces] for arm, arm_traces in traces.items()},
        }

    return {
        "schema_version": 1,
        "no_llm_calls": True,
        "source": {
            "paper": model.map_spec.source_citation,
            "url": model.map_spec.source_url,
            "page": model.map_spec.source_page,
            "map": config.map_name,
            "map_spec": asdict(model.map_spec),
            "pomdp_py_version": "1.3.5.1",
        },
        "config": {
            **asdict(config),
            "candidate_widths": list(config.candidate_widths),
        },
        "widths": by_width,
        "decision": {
            "confirmation_passes": any_pass,
            "rule": "at least one K has d2 entropy superiority over shared d1 and call-matched width",
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    """Render a concise, tracked Markdown summary from the JSON report."""

    source = summary["source"]
    lines = [
        "# Rock Diagnosis Exact Confirmation",
        "",
        "## Scope",
        "",
        "This is a zero-LLM-call exact mechanism confirmation. The static target is the full rock-type vector; motion, likelihoods, exact Bayesian updates, and MAP decodes are deterministic/auditable. The held-out map is "
        f"`{source['map']}` from {source['paper']}, page {source['page']}.",
        "",
        "## Final Entropy",
        "",
        "Positive reductions favor depth two. Width uses the same number of current-state proposal cells as the full depth-two root tree's root plus outcome branches.",
        "",
        "| K | d2 - shared d1 | 95% CI | d2 - call-matched width | 95% CI | Gate |",
        "| ---: | ---: | --- | ---: | --- | --- |",
    ]
    for width, row in summary["widths"].items():
        d1 = row["comparisons"]["d2_minus_shared_d1"]
        wide = row["comparisons"]["d2_minus_call_matched_width"]
        lines.append(
            f"| {width} | {d1['final_entropy_reduction_mean']:+.4f} | "
            f"[{d1['final_entropy_reduction_ci95'][0]:+.4f}, {d1['final_entropy_reduction_ci95'][1]:+.4f}] | "
            f"{wide['final_entropy_reduction_mean']:+.4f} | "
            f"[{wide['final_entropy_reduction_ci95'][0]:+.4f}, {wide['final_entropy_reduction_ci95'][1]:+.4f}] | "
            f"{'pass' if row['passed_confirmation_gate'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Mechanics",
            "",
            "| K | Shared root cells | Width contains base | Width calls match virtual d2 cells | Legal actions |",
            "| ---: | --- | --- | --- | --- |",
        ]
    )
    for width, row in summary["widths"].items():
        mechanics = row["mechanics"]
        lines.append(
            f"| {width} | {mechanics['initial_base_candidate_cells_shared']} | "
            f"{mechanics['initial_width_contains_shared_base']} | "
            f"{mechanics['width_cells_match_virtual_depth_two_cells']} | "
            f"{mechanics['all_selected_actions_legal']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- **Confirmation passes:** `{summary['decision']['confirmation_passes']}`.",
            f"- Rule: {summary['decision']['rule']}.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python scripts/nonmyopic_rock_diagnosis_oracle.py",
            "pytest -q tests/test_nonmyopic_rock_diagnosis_oracle.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_widths(value: str) -> tuple[int, ...]:
    widths = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not widths:
        raise argparse.ArgumentTypeError("expected at least one candidate width")
    return widths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", dest="map_name", default="3-6", choices=("3-6", "5-7", "7-8"))
    parser.add_argument("--num-trials", type=int, default=2_000)
    parser.add_argument("--num-rounds", type=int, default=8)
    parser.add_argument("--candidate-widths", type=_parse_widths, default=(2, 3, 4))
    parser.add_argument("--seed", type=int, default=2304)
    parser.add_argument("--trial-offset", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/nonmyopic/rock_diagnosis_confirmation"))
    args = parser.parse_args()
    config = OracleConfig(
        map_name=args.map_name,
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        candidate_widths=args.candidate_widths,
        seed=args.seed,
        trial_offset=args.trial_offset,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    summary = run_oracle(config)
    compact = compact_summary(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "REPORT.md").write_text(render_report(compact), encoding="utf-8")
    write_full_traces(summary, args.output_dir / "TRACES.jsonl.gz")
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
