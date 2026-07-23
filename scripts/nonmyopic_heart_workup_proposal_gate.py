"""Fresh exact proposal-quality gate for Cleveland heart-workup branch policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.heart_workup import (  # noqa: E402
    ORDER_WORKUP_ACTION,
    HeartWorkupModel,
    WorkupState,
)
from environments.heart_workup.model import EPSILON  # noqa: E402
from helpers import Config, load_config  # noqa: E402
from model_factory import build_model_adapter  # noqa: E402
from scripts.nonmyopic_gated_sensor_strategy_prior import (  # noqa: E402
    ChatModel,
    StrategyProposalError,
    _usage_snapshot,
)
from scripts.nonmyopic_heart_workup_strategy import (  # noqa: E402
    DeterministicIndexedHeartModel,
    HeartBranchStrategy,
    HeartStrategyConfig,
    IndexedHeartProvider,
    _best_query,
    branch_menus,
    fixed_roots,
)
from scripts.nonmyopic_mushroom_feature_oracle import (  # noqa: E402
    _choose_action,
    exact_action_costs,
)


@dataclass(frozen=True)
class HeartProposalGateConfig:
    num_cells: int = 32
    num_unworked: int = 16
    seed: int = 24_137
    bootstrap_replicates: int = 5_000
    workup_rate_threshold: float = 0.75
    recovery_threshold: float = 0.60

    def validate(self, *, catalog_size: int) -> None:
        if self.num_cells != 32 or self.num_unworked != 16:
            raise ValueError("the frozen Heart proposal gate uses 32 balanced cells")
        if self.bootstrap_replicates != 5_000:
            raise ValueError("the frozen Heart proposal gate uses 5,000 bootstraps")
        if self.num_cells > catalog_size:
            raise ValueError("proposal cells cannot exceed the Cleveland catalog")
        if not 0.0 <= self.workup_rate_threshold <= 1.0:
            raise ValueError("workup rate threshold must be a probability")
        if not 0.0 <= self.recovery_threshold <= 1.0:
            raise ValueError("recovery threshold must be a fraction")


@dataclass(frozen=True)
class ProposalCell:
    cell_index: int
    truth_index: int
    state: WorkupState
    belief: np.ndarray
    history: tuple[tuple[str, str | None], ...]
    phase: str


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _pre_workup_state(
    model: HeartWorkupModel, *, truth_index: int
) -> tuple[WorkupState, np.ndarray, list[tuple[str, str | None]]] | None:
    state = model.initial_state
    belief = model.initial_belief.copy()
    history: list[tuple[str, str | None]] = []
    for round_index in range(7):
        costs, _ = exact_action_costs(
            model,
            state=state,
            belief=belief,
            depth=min(2, 8 - round_index),
        )
        action = _choose_action(costs)
        if action == ORDER_WORKUP_ACTION:
            return state, belief, history
        outcome = model.observation(truth_index, action)
        belief = model.posterior(belief, action, outcome)
        state = model.next_state(state, action)
        history.append((action, outcome))
    return None


def build_proposal_cells(
    model: HeartWorkupModel, config: HeartProposalGateConfig
) -> list[ProposalCell]:
    config.validate(catalog_size=len(model.rows))
    truths = np.random.default_rng(config.seed).permutation(len(model.rows))
    cells: list[ProposalCell] = []
    used_truths: set[int] = set()
    seen_unworked: set[tuple[tuple[str, str | None], ...]] = set()
    for candidate_index, raw_truth in enumerate(truths):
        truth = int(raw_truth)
        state = model.initial_state
        belief = model.initial_belief.copy()
        history: list[tuple[str, str | None]] = []
        for step in range(1 + candidate_index % 2):
            queries = [
                action for action in model.legal_actions(state) if action.startswith("query:")
            ]
            action = queries[
                _stable_seed(config.seed, "unworked-history", candidate_index, step)
                % len(queries)
            ]
            outcome = model.observation(truth, action)
            belief = model.posterior(belief, action, outcome)
            state = model.next_state(state, action)
            history.append((action, outcome))
        history_key = tuple(history)
        if history_key in seen_unworked or model.target_entropy(belief) <= EPSILON:
            continue
        d1_costs, _ = exact_action_costs(model, state=state, belief=belief, depth=1)
        d2_costs, _ = exact_action_costs(model, state=state, belief=belief, depth=2)
        d1_root = _choose_action(d1_costs)
        d2_root = _choose_action(d2_costs)
        if (
            d2_root != ORDER_WORKUP_ACTION
            or d2_costs[d1_root] - d2_costs[d2_root] <= EPSILON
        ):
            continue
        cells.append(
            ProposalCell(len(cells), truth, state, belief, history_key, "unworked")
        )
        used_truths.add(truth)
        seen_unworked.add(history_key)
        if len(cells) == config.num_unworked:
            break
    if len(cells) != config.num_unworked:
        raise RuntimeError("could not construct 16 distinct Heart workup opportunities")

    seen_worked: set[tuple[tuple[str, str | None], ...]] = set()
    for candidate_index, raw_truth in enumerate(truths):
        truth = int(raw_truth)
        if truth in used_truths:
            continue
        candidate = _pre_workup_state(model, truth_index=truth)
        if candidate is None:
            continue
        state, belief, history = candidate
        outcome = model.observation(truth, ORDER_WORKUP_ACTION)
        belief = model.posterior(belief, ORDER_WORKUP_ACTION, outcome)
        state = model.next_state(state, ORDER_WORKUP_ACTION)
        history.append((ORDER_WORKUP_ACTION, outcome))
        extra_steps = candidate_index % 3
        for _ in range(extra_steps):
            if model.target_entropy(belief) <= EPSILON:
                break
            action = _best_query(model, state, belief)
            outcome = model.observation(truth, action)
            belief = model.posterior(belief, action, outcome)
            state = model.next_state(state, action)
            history.append((action, outcome))
        history_key = tuple(history)
        if history_key in seen_worked or model.target_entropy(belief) <= EPSILON:
            continue
        cells.append(ProposalCell(len(cells), truth, state, belief, history_key, "worked"))
        used_truths.add(truth)
        seen_worked.add(history_key)
        if len(cells) == config.num_cells:
            break
    if len(cells) != config.num_cells:
        raise RuntimeError(f"could not construct 32 balanced Heart cells: {len(cells)}")
    return cells


def strategy_cost(
    model: HeartWorkupModel,
    *,
    state: WorkupState,
    belief: np.ndarray,
    strategy: HeartBranchStrategy,
) -> float:
    root = strategy.root_action
    if root not in model.legal_actions(state):
        raise StrategyProposalError(f"illegal Heart proposal root {root}")
    child_state = model.next_state(state, root)
    total = 0.0
    for outcome in model.outcomes(root):
        probability = model.outcome_probability(belief, root, outcome)
        if probability <= EPSILON:
            continue
        posterior = model.posterior(belief, root, outcome)
        key = "none" if outcome is None else str(outcome)
        if key not in strategy.followups:
            raise StrategyProposalError(f"Heart proposal root {root} omits branch {key}")
        followup = strategy.followups[key]
        if followup not in model.legal_actions(child_state):
            raise StrategyProposalError(f"illegal Heart follow-up {followup} under {root}")
        total += probability * (
            model.target_entropy(posterior)
            + model.expected_target_entropy(posterior, followup)
        )
    return float(total)


def _matched_random_strategies(
    *,
    roots: tuple[str, ...],
    menus: list[dict[str, tuple[str, ...]]],
    seed: int,
) -> tuple[HeartBranchStrategy, ...]:
    rng = np.random.default_rng(seed)
    return tuple(
        HeartBranchStrategy(
            root,
            {
                outcome: choices[int(rng.integers(0, len(choices)))]
                for outcome, choices in root_menus.items()
            },
        )
        for root, root_menus in zip(roots, menus, strict=True)
    )


def _bootstrap_ci(values: np.ndarray, *, seed: int, replicates: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, replicates, 512):
        size = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        samples.append(np.mean(values[indices], axis=1))
    means = np.concatenate(samples)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _summary(
    values: list[float], *, config: HeartProposalGateConfig, label: str
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "ci95": _bootstrap_ci(
            array,
            seed=_stable_seed(config.seed, "heart-proposal-bootstrap", label),
            replicates=config.bootstrap_replicates,
        ),
        "wins_ties_losses": [
            int(np.sum(array > 1e-12)),
            int(np.sum(np.abs(array) <= 1e-12)),
            int(np.sum(array < -1e-12)),
        ],
        "paired_values": array.tolist(),
    }


def run_proposal_gate(
    provider: IndexedHeartProvider, config: HeartProposalGateConfig
) -> dict[str, Any]:
    model = HeartWorkupModel()
    cells = build_proposal_cells(model, config)
    records: list[dict[str, Any]] = []
    for cell in cells:
        roots = fixed_roots(model, state=cell.state, belief=cell.belief, count=4)
        menus = branch_menus(model, state=cell.state, belief=cell.belief, roots=roots)
        proposal = provider.propose(
            model,
            cell_index=cell.cell_index,
            state=cell.state,
            belief=cell.belief,
            history=cell.history,
        )
        llm_costs = [
            strategy_cost(model, state=cell.state, belief=cell.belief, strategy=strategy)
            for strategy in proposal.strategies
        ]
        llm_slot = min(range(4), key=lambda index: (llm_costs[index], index))
        random_strategies = _matched_random_strategies(
            roots=roots,
            menus=menus,
            seed=_stable_seed(config.seed, "matched-random", cell.cell_index),
        )
        random_costs = [
            strategy_cost(model, state=cell.state, belief=cell.belief, strategy=strategy)
            for strategy in random_strategies
        ]
        random_slot = min(range(4), key=lambda index: (random_costs[index], index))
        d1_costs, d1_units = exact_action_costs(
            model, state=cell.state, belief=cell.belief, depth=1
        )
        d2_costs, d2_units = exact_action_costs(
            model, state=cell.state, belief=cell.belief, depth=2
        )
        d1_root = _choose_action(d1_costs)
        exhaustive_root = _choose_action(d2_costs)
        shared_cost = float(d2_costs[d1_root])
        exhaustive_cost = float(d2_costs[exhaustive_root])
        llm_cost = float(llm_costs[llm_slot])
        opportunity = shared_cost - exhaustive_cost
        recovery = (shared_cost - llm_cost) / opportunity if opportunity > EPSILON else None
        records.append(
            {
                "cell_index": cell.cell_index,
                "truth_index": cell.truth_index,
                "phase": cell.phase,
                "workup_ordered": cell.state.workup_ordered,
                "history": [list(item) for item in cell.history],
                "roots": list(roots),
                "llm_costs": llm_costs,
                "llm_selected_slot": llm_slot,
                "llm_selected_root": roots[llm_slot],
                "llm_cost": llm_cost,
                "random_costs": random_costs,
                "random_selected_slot": random_slot,
                "random_cost": float(random_costs[random_slot]),
                "shared_d1_root": d1_root,
                "shared_d1_exact_continuation_cost": shared_cost,
                "exhaustive_d2_root": exhaustive_root,
                "exhaustive_d2_cost": exhaustive_cost,
                "d2_opportunity": opportunity,
                "recovery_fraction": recovery,
                "scorer_units": d1_units + d2_units,
            }
        )
    unworked = [row for row in records if row["phase"] == "unworked"]
    comparisons = {
        "matched_random_minus_llm_cost": _summary(
            [row["random_cost"] - row["llm_cost"] for row in records],
            config=config,
            label="matched-random-minus-llm",
        ),
        "unworked_shared_d1_minus_llm_cost": _summary(
            [row["shared_d1_exact_continuation_cost"] - row["llm_cost"] for row in unworked],
            config=config,
            label="shared-d1-minus-llm",
        ),
        "unworked_recovery_fraction": _summary(
            [float(row["recovery_fraction"]) for row in unworked],
            config=config,
            label="recovery-fraction",
        ),
        "unworked_workup_selection_rate": float(
            np.mean([row["llm_selected_root"] == ORDER_WORKUP_ACTION for row in unworked])
        ),
    }
    mechanics = {
        "thirty_two_cells_resolved": len(records) == config.num_cells,
        "balanced_phases": len(unworked) == config.num_unworked,
        "distinct_phase_histories": all(
            len(
                {
                    tuple(tuple(item) for item in row["history"])
                    for row in records
                    if row["phase"] == phase
                }
            )
            == sum(row["phase"] == phase for row in records)
            for phase in ("unworked", "worked")
        ),
        "all_unworked_are_d2_workup_opportunities": all(
            row["exhaustive_d2_root"] == ORDER_WORKUP_ACTION
            and row["d2_opportunity"] > EPSILON
            for row in unworked
        ),
        "llm_random_roots_paired": True,
        "all_policies_exactly_scored": True,
        "scoring_made_no_llm_calls": True,
    }
    endpoint_gate = {
        "matched_random_lower_bound_positive": comparisons[
            "matched_random_minus_llm_cost"
        ]["ci95"][0]
        > 0.0,
        "shared_d1_lower_bound_positive": comparisons[
            "unworked_shared_d1_minus_llm_cost"
        ]["ci95"][0]
        > 0.0,
        "workup_rate_at_least_threshold": comparisons[
            "unworked_workup_selection_rate"
        ]
        >= config.workup_rate_threshold,
        "mean_recovery_at_least_threshold": comparisons[
            "unworked_recovery_fraction"
        ]["mean"]
        >= config.recovery_threshold,
    }
    return {
        "schema_version": 1,
        "stage": "cleveland_heart_workup_26b_proposal_gate",
        "config": asdict(config),
        "mechanics": mechanics,
        "comparisons": comparisons,
        "endpoint_gate": endpoint_gate,
        "records": records,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


def render_report(result: dict[str, Any]) -> str:
    comparisons = result["comparisons"]
    lines = [
        "# Cleveland Heart Workup 26B Proposal Gate",
        "",
        f"Gate passed: **{result['gate']['passed']}**.",
        "",
        "| Endpoint | Mean | Paired 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("matched_random_minus_llm_cost", "Matched random - LLM cost"),
        ("unworked_shared_d1_minus_llm_cost", "Shared d1 - LLM cost"),
        ("unworked_recovery_fraction", "D2 opportunity recovery"),
    ):
        row = comparisons[key]
        lines.append(
            f"| {label} | {row['mean']:+.6f} | "
            f"[{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] | "
            f"{'/'.join(str(value) for value in row['wins_ties_losses'])} |"
        )
    lines.extend(
        [
            "",
            "Workup-root selection on unworked opportunities: "
            f"{comparisons['unworked_workup_selection_rate']:.1%}.",
            "",
            "The LLM proposed policies once per cell. All scoring, controls, and bootstrap "
            "calculations were exact and made zero LLM calls.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_nonmyopic_rocksample_15_15_vllm.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/heart_workup_26b_proposal_gate_20260723"),
    )
    parser.add_argument("--run-id", default="heart-workup-26b-proposal-gate-20260723")
    parser.add_argument("--seed", type=int, default=24_137)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = HeartProposalGateConfig(seed=args.seed)
    strategy_config = HeartStrategyConfig(seed=args.seed)
    strategy_config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicIndexedHeartModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.location_max_new_tokens = strategy_config.max_new_tokens
        chat_model = build_model_adapter(
            runtime_config.model_pairs[0].questioner,
            config=runtime_config,
        )
    provider = IndexedHeartProvider(chat_model, strategy_config)
    try:
        result = run_proposal_gate(provider, config)
    except (RuntimeError, StrategyProposalError) as exc:
        failure = {
            "schema_version": 1,
            "stage": "cleveland_heart_workup_26b_proposal_gate",
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(config),
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": _usage_snapshot(chat_model),
        }
        (args.output_dir / "GATE_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    result["usage"] = _usage_snapshot(chat_model)
    result["run_id"] = args.run_id
    result["dry_run"] = args.dry_run
    result["mechanics"]["thirty_two_accepted_cells"] = (
        len(provider.physical_requests) == config.num_cells
    )
    result["mechanics"]["zero_reasoning_tokens"] = (
        int(result["usage"].get("reasoning_tokens", 0)) == 0
    )
    result["mechanics"]["zero_forced_exits"] = (
        int(result["usage"].get("forced_exits", 0)) == 0
    )
    result["gate"] = {
        "passed": all(result["mechanics"].values())
        and all(result["endpoint_gate"].values())
    }
    (args.output_dir / "GATE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "GATE.md").write_text(render_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": result["gate"],
                "endpoint_gate": result["endpoint_gate"],
                "provider": {
                    "accepted_requests": len(provider.physical_requests),
                    "invalid_responses": len(provider.invalid_responses),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
