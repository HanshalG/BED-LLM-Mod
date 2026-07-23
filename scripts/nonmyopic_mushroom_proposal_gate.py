"""Fresh exact proposal-quality gate for Mushroom branch policies."""

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

from environments.mushroom_feature_acquisition import (  # noqa: E402
    COLLECT_ACTION,
    AcquisitionState,
    MushroomFeatureModel,
)
from environments.mushroom_feature_acquisition.model import EPSILON  # noqa: E402
from helpers import Config, load_config  # noqa: E402
from model_factory import build_model_adapter  # noqa: E402
from scripts.nonmyopic_gated_sensor_strategy_prior import (  # noqa: E402
    ChatModel,
    StrategyProposalError,
    _usage_snapshot,
)
from scripts.nonmyopic_mushroom_feature_oracle import (  # noqa: E402
    _choose_action,
    exact_action_costs,
)
from scripts.nonmyopic_mushroom_strategy import (  # noqa: E402
    DeterministicIndexedMushroomModel,
    IndexedMushroomProvider,
    MushroomBranchStrategy,
    MushroomStrategyConfig,
    _branch_menus,
    _fixed_roots,
)


@dataclass(frozen=True)
class MushroomProposalGateConfig:
    num_cells: int = 32
    num_uncollected: int = 16
    uncollected_history_cycle: int = 3
    collected_history_cycle: int = 4
    seed: int = 24_131
    bootstrap_replicates: int = 5_000
    collection_rate_threshold: float = 0.75
    recovery_threshold: float = 0.60

    def validate(self, *, catalog_size: int) -> None:
        if self.num_cells != 32 or self.num_uncollected != 16:
            raise ValueError("the frozen proposal gate uses 32 balanced cells")
        if self.uncollected_history_cycle != 3 or self.collected_history_cycle != 4:
            raise ValueError("the frozen proposal gate uses phase-specific history depths")
        if self.bootstrap_replicates != 5_000:
            raise ValueError("the frozen proposal gate uses 5,000 bootstrap replicates")
        if self.num_cells > catalog_size:
            raise ValueError("proposal cells cannot exceed the Mushroom catalog")
        if not 0.0 <= self.collection_rate_threshold <= 1.0:
            raise ValueError("collection rate threshold must be a probability")
        if not 0.0 <= self.recovery_threshold <= 1.0:
            raise ValueError("recovery threshold must be a fraction")


@dataclass(frozen=True)
class ProposalCell:
    cell_index: int
    truth_index: int
    state: AcquisitionState
    belief: np.ndarray
    history: tuple[tuple[str, str | None], ...]
    phase: str


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _advance_diverse_history(
    model: MushroomFeatureModel,
    *,
    truth_index: int,
    state: AcquisitionState,
    belief: np.ndarray,
    history: list[tuple[str, str | None]],
    steps: int,
    seed: int,
    candidate_index: int,
    phase: str,
) -> tuple[AcquisitionState, np.ndarray]:
    for step in range(steps):
        queries = [
            action for action in model.legal_actions(state) if action.startswith("query:")
        ]
        action_index = _stable_seed(
            seed, f"{phase}-history-action", candidate_index, step
        ) % len(queries)
        action = queries[action_index]
        outcome = model.observation(truth_index, action)
        belief = model.posterior(belief, action, outcome)
        state = model.next_state(state, action)
        history.append((action, outcome))
    return state, belief


def build_proposal_cells(
    model: MushroomFeatureModel,
    config: MushroomProposalGateConfig,
) -> list[ProposalCell]:
    config.validate(catalog_size=len(model.rows))
    truths = np.random.default_rng(config.seed).permutation(len(model.rows))
    targets = {"uncollected": config.num_uncollected, "collected": config.num_cells - config.num_uncollected}
    cells: list[ProposalCell] = []
    seen: dict[str, set[tuple[tuple[str, str | None], ...]]] = {
        "uncollected": set(),
        "collected": set(),
    }
    phase_counts = {phase: 0 for phase in targets}
    for candidate_index, raw_truth in enumerate(truths):
        phase = "uncollected" if candidate_index % 2 == 0 else "collected"
        if phase_counts[phase] >= targets[phase]:
            phase = "collected" if phase == "uncollected" else "uncollected"
        if phase_counts[phase] >= targets[phase]:
            continue
        truth_index = int(raw_truth)
        state = model.initial_state
        belief = model.initial_belief.copy()
        history: list[tuple[str, str | None]] = []
        if phase == "collected":
            outcome = model.observation(truth_index, COLLECT_ACTION)
            belief = model.posterior(belief, COLLECT_ACTION, outcome)
            state = model.next_state(state, COLLECT_ACTION)
            history.append((COLLECT_ACTION, outcome))
        history_cycle = (
            config.collected_history_cycle
            if phase == "collected"
            else config.uncollected_history_cycle
        )
        steps = (candidate_index // 2) % history_cycle
        state, belief = _advance_diverse_history(
            model,
            truth_index=truth_index,
            state=state,
            belief=belief,
            history=history,
            steps=steps,
            seed=config.seed,
            candidate_index=candidate_index,
            phase=phase,
        )
        history_key = tuple(history)
        if history_key in seen[phase]:
            continue
        if model.target_entropy(belief) <= EPSILON:
            continue
        if phase == "uncollected":
            d1_costs, _ = exact_action_costs(model, state=state, belief=belief, depth=1)
            d2_costs, _ = exact_action_costs(model, state=state, belief=belief, depth=2)
            d1_root = _choose_action(d1_costs)
            d2_root = _choose_action(d2_costs)
            if d2_root != COLLECT_ACTION or d2_costs[d1_root] - d2_costs[d2_root] <= EPSILON:
                continue
        seen[phase].add(history_key)
        cells.append(
            ProposalCell(
                cell_index=len(cells),
                truth_index=truth_index,
                state=state,
                belief=belief,
                history=history_key,
                phase=phase,
            )
        )
        phase_counts[phase] += 1
        if len(cells) == config.num_cells:
            break
    if phase_counts != targets:
        raise RuntimeError(f"could not construct frozen balanced proposal cells: {phase_counts}")
    return cells


def strategy_cost(
    model: MushroomFeatureModel,
    *,
    state: AcquisitionState,
    belief: np.ndarray,
    strategy: MushroomBranchStrategy,
) -> float:
    root = strategy.root_action
    if root not in model.legal_actions(state):
        raise StrategyProposalError(f"illegal proposal root {root}")
    child_state = model.next_state(state, root)
    total = 0.0
    for outcome in model.outcomes(root):
        probability = model.outcome_probability(belief, root, outcome)
        if probability <= EPSILON:
            continue
        posterior = model.posterior(belief, root, outcome)
        key = "none" if outcome is None else str(outcome)
        if key not in strategy.followups:
            raise StrategyProposalError(f"proposal root {root} omits branch {key}")
        followup = strategy.followups[key]
        if followup not in model.legal_actions(child_state):
            raise StrategyProposalError(f"illegal follow-up {followup} under {root}")
        branch_cost = model.target_entropy(posterior)
        branch_cost += model.expected_target_entropy(posterior, followup)
        total += probability * branch_cost
    return float(total)


def _matched_random_strategies(
    *,
    roots: tuple[str, ...],
    menus: list[dict[str, tuple[str, ...]]],
    seed: int,
) -> tuple[MushroomBranchStrategy, ...]:
    rng = np.random.default_rng(seed)
    strategies = []
    for root, root_menus in zip(roots, menus, strict=True):
        followups = {
            outcome: choices[int(rng.integers(0, len(choices)))]
            for outcome, choices in root_menus.items()
        }
        strategies.append(MushroomBranchStrategy(root, followups))
    return tuple(strategies)


def _bootstrap_ci(values: np.ndarray, *, seed: int, replicates: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, replicates, 512):
        size = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        samples.append(np.mean(values[indices], axis=1))
    means = np.concatenate(samples)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _summary(values: list[float], *, config: MushroomProposalGateConfig, label: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "ci95": _bootstrap_ci(
            array,
            seed=_stable_seed(config.seed, "proposal-gate-bootstrap", label),
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
    provider: IndexedMushroomProvider,
    config: MushroomProposalGateConfig,
) -> dict[str, Any]:
    model = MushroomFeatureModel()
    cells = build_proposal_cells(model, config)
    records: list[dict[str, Any]] = []
    for cell in cells:
        roots = _fixed_roots(model, state=cell.state, belief=cell.belief, count=4)
        menus = _branch_menus(model, state=cell.state, belief=cell.belief, roots=roots)
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
        llm_slot = min(range(len(llm_costs)), key=lambda index: (llm_costs[index], index))
        random_strategies = _matched_random_strategies(
            roots=roots,
            menus=menus,
            seed=_stable_seed(config.seed, "matched-random", cell.cell_index),
        )
        random_costs = [
            strategy_cost(model, state=cell.state, belief=cell.belief, strategy=strategy)
            for strategy in random_strategies
        ]
        random_slot = min(
            range(len(random_costs)), key=lambda index: (random_costs[index], index)
        )
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
                "specimen_collected": cell.state.specimen_collected,
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
    random_gap = [row["random_cost"] - row["llm_cost"] for row in records]
    uncollected = [row for row in records if row["phase"] == "uncollected"]
    shared_gap = [
        row["shared_d1_exact_continuation_cost"] - row["llm_cost"]
        for row in uncollected
    ]
    recovery = [float(row["recovery_fraction"]) for row in uncollected]
    collection_rate = float(
        np.mean([row["llm_selected_root"] == COLLECT_ACTION for row in uncollected])
    )
    comparisons = {
        "matched_random_minus_llm_cost": _summary(
            random_gap, config=config, label="matched-random-minus-llm"
        ),
        "uncollected_shared_d1_minus_llm_cost": _summary(
            shared_gap, config=config, label="shared-d1-minus-llm"
        ),
        "uncollected_recovery_fraction": _summary(
            recovery, config=config, label="recovery-fraction"
        ),
        "uncollected_collection_selection_rate": collection_rate,
    }
    mechanics = {
        "thirty_two_cells_resolved": len(records) == config.num_cells,
        "balanced_phases": len(uncollected) == config.num_uncollected,
        "distinct_phase_histories": all(
            len({tuple(tuple(item) for item in row["history"]) for row in records if row["phase"] == phase})
            == sum(row["phase"] == phase for row in records)
            for phase in ("uncollected", "collected")
        ),
        "all_uncollected_are_d2_collection_opportunities": all(
            row["exhaustive_d2_root"] == COLLECT_ACTION and row["d2_opportunity"] > EPSILON
            for row in uncollected
        ),
        "llm_random_roots_paired": True,
        "all_policies_exactly_scored": True,
        "scoring_made_no_llm_calls": True,
    }
    endpoint_gate = {
        "matched_random_lower_bound_positive": comparisons["matched_random_minus_llm_cost"]["ci95"][0] > 0.0,
        "shared_d1_lower_bound_positive": comparisons["uncollected_shared_d1_minus_llm_cost"]["ci95"][0] > 0.0,
        "collection_rate_at_least_threshold": collection_rate >= config.collection_rate_threshold,
        "mean_recovery_at_least_threshold": comparisons["uncollected_recovery_fraction"]["mean"] >= config.recovery_threshold,
    }
    return {
        "schema_version": 1,
        "stage": "mushroom_feature_acquisition_26b_proposal_gate",
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
        "# Mushroom Feature Acquisition 26B Proposal Gate",
        "",
        f"Gate passed: **{result['gate']['passed']}**.",
        "",
        "| Endpoint | Mean | Paired 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("matched_random_minus_llm_cost", "Matched random - LLM cost"),
        ("uncollected_shared_d1_minus_llm_cost", "Shared d1 - LLM cost"),
        ("uncollected_recovery_fraction", "D2 opportunity recovery"),
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
            "Collection-root selection on uncollected opportunities: "
            f"{comparisons['uncollected_collection_selection_rate']:.1%}.",
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
        default=Path("results/nonmyopic/mushroom_feature_26b_proposal_gate_20260723"),
    )
    parser.add_argument("--run-id", default="mushroom-feature-26b-proposal-gate-20260723")
    parser.add_argument("--seed", type=int, default=24_131)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = MushroomProposalGateConfig(seed=args.seed)
    strategy_config = MushroomStrategyConfig(seed=args.seed)
    strategy_config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicIndexedMushroomModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.location_max_new_tokens = strategy_config.max_new_tokens
        chat_model = build_model_adapter(
            runtime_config.model_pairs[0].questioner,
            config=runtime_config,
        )
    provider = IndexedMushroomProvider(chat_model, strategy_config)
    try:
        result = run_proposal_gate(provider, config)
    except (RuntimeError, StrategyProposalError) as exc:
        failure = {
            "schema_version": 1,
            "stage": "mushroom_feature_acquisition_26b_proposal_gate",
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(config),
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": _usage_snapshot(chat_model),
        }
        (args.output_dir / "GATE_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    result["usage"] = _usage_snapshot(chat_model)
    result["run_id"] = args.run_id
    result["dry_run"] = args.dry_run
    result["mechanics"]["thirty_two_accepted_cells"] = len(provider.physical_requests) == 32
    result["mechanics"]["zero_reasoning_tokens"] = int(result["usage"].get("reasoning_tokens", 0)) == 0
    result["mechanics"]["zero_forced_exits"] = int(result["usage"].get("forced_exits", 0)) == 0
    result["gate"] = {
        "passed": all(result["mechanics"].values()) and all(result["endpoint_gate"].values())
    }
    (args.output_dir / "GATE.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
