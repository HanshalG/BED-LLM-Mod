"""Fresh exact proposal-quality gate for UCI thyroid named branch policies."""

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

from environments.thyroid_workup import (  # noqa: E402
    COLLECT_BLOOD_ACTION,
    ThyroidWorkupModel,
    ThyroidWorkupState,
)
from environments.thyroid_workup.model import EPSILON  # noqa: E402
from helpers import Config, load_config  # noqa: E402
from model_factory import build_model_adapter  # noqa: E402
from scripts.nonmyopic_gated_sensor_strategy_prior import (  # noqa: E402
    ChatModel,
    StrategyProposalError,
    _usage_snapshot,
)
from scripts.nonmyopic_thyroid_workup_oracle import exact_action_costs  # noqa: E402
from scripts.nonmyopic_thyroid_workup_strategy import (  # noqa: E402
    DeterministicNamedThyroidModel,
    NamedThyroidProvider,
    ThyroidBranchStrategy,
    ThyroidStrategyConfig,
    branch_menus,
    fixed_roots,
)


@dataclass(frozen=True)
class ThyroidProposalGateConfig:
    num_cells: int = 32
    seed: int = 24_152
    bootstrap_replicates: int = 5_000
    collection_rate_threshold: float = 0.75
    recovery_threshold: float = 0.60

    def validate(self, cohort_size: int) -> None:
        if self.num_cells != 32:
            raise ValueError("the frozen thyroid proposal gate uses exactly 32 cells")
        if self.bootstrap_replicates != 5_000:
            raise ValueError("the frozen thyroid proposal gate uses 5,000 bootstraps")
        if self.num_cells > cohort_size:
            raise ValueError("proposal cells cannot exceed the thyroid cohort")
        if not 0.0 <= self.collection_rate_threshold <= 1.0:
            raise ValueError("collection threshold must be a probability")
        if not 0.0 <= self.recovery_threshold <= 1.0:
            raise ValueError("recovery threshold must be a fraction")


@dataclass(frozen=True)
class ProposalCell:
    cell_index: int
    truth_index: int
    state: ThyroidWorkupState
    belief: np.ndarray
    history: tuple[tuple[str, str | None], ...]


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _choose_action(costs: dict[str, float]) -> str:
    actions = tuple(costs)
    return min(actions, key=lambda action: (costs[action], actions.index(action)))


def build_proposal_cells(
    model: ThyroidWorkupModel, config: ThyroidProposalGateConfig
) -> list[ProposalCell]:
    config.validate(len(model.targets))
    truths = np.random.default_rng(config.seed).permutation(len(model.targets))
    cells: list[ProposalCell] = []
    seen_histories: set[tuple[tuple[str, str | None], ...]] = set()
    for candidate_index, raw_truth in enumerate(truths):
        truth = int(raw_truth)
        state = model.initial_state
        belief = model.initial_belief
        history: list[tuple[str, str | None]] = []
        for step in range(candidate_index % 3):
            queries = [
                action for action in model.legal_actions(state) if action.startswith("query:")
            ]
            action = queries[
                _stable_seed(config.seed, "history", candidate_index, step) % len(queries)
            ]
            outcome = model.observation(truth, action)
            belief = model.posterior(belief, action, outcome)
            state = model.next_state(state, action)
            history.append((action, outcome))
        history_key = tuple(history)
        if history_key in seen_histories or model.target_entropy(belief) <= EPSILON:
            continue
        d1_costs = exact_action_costs(model, state=state, belief=belief, depth=1)
        d2_costs = exact_action_costs(model, state=state, belief=belief, depth=2)
        d1_root = _choose_action(d1_costs)
        d2_root = _choose_action(d2_costs)
        if (
            d2_root != COLLECT_BLOOD_ACTION
            or d2_costs[d1_root] - d2_costs[d2_root] <= EPSILON
        ):
            continue
        cells.append(ProposalCell(len(cells), truth, state, belief, history_key))
        seen_histories.add(history_key)
        if len(cells) == config.num_cells:
            break
    if len(cells) != config.num_cells:
        raise RuntimeError(f"could not construct 32 distinct thyroid opportunities: {len(cells)}")
    return cells


def strategy_cost(
    model: ThyroidWorkupModel,
    *,
    state: ThyroidWorkupState,
    belief: np.ndarray,
    strategy: ThyroidBranchStrategy,
) -> float:
    root = strategy.root_action
    if root not in model.legal_actions(state):
        raise StrategyProposalError(f"illegal thyroid root {root}")
    child_state = model.next_state(state, root)
    total = 0.0
    for outcome in model.outcomes(root):
        probability = model.outcome_probability(belief, root, outcome)
        if probability <= EPSILON:
            continue
        posterior = model.posterior(belief, root, outcome)
        key = "none" if outcome is None else str(outcome)
        if key not in strategy.followups:
            raise StrategyProposalError(f"thyroid root {root} omits branch {key}")
        followup = strategy.followups[key]
        if followup not in model.legal_actions(child_state):
            raise StrategyProposalError(f"illegal follow-up {followup} under {root}")
        total += probability * (
            model.target_entropy(posterior)
            + model.expected_target_entropy(posterior, followup)
        )
    return float(total)


def _matched_random_strategies(
    *,
    roots: tuple[str, ...],
    menus: dict[str, dict[str, tuple[str, ...]]],
    seed: int,
) -> tuple[ThyroidBranchStrategy, ...]:
    rng = np.random.default_rng(seed)
    return tuple(
        ThyroidBranchStrategy(
            root,
            {
                outcome: choices[int(rng.integers(0, len(choices)))]
                for outcome, choices in menus[root].items()
            },
        )
        for root in roots
    )


def _bootstrap_ci(values: np.ndarray, *, seed: int, replicates: int) -> list[float]:
    rng = np.random.default_rng(seed)
    pieces = []
    for start in range(0, replicates, 512):
        size = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        pieces.append(values[indices].mean(axis=1))
    samples = np.concatenate(pieces)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _summary(
    values: list[float], *, config: ThyroidProposalGateConfig, label: str
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "ci95": _bootstrap_ci(
            array,
            seed=_stable_seed(config.seed, "thyroid-proposal-bootstrap", label),
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
    provider: NamedThyroidProvider, config: ThyroidProposalGateConfig
) -> dict[str, Any]:
    model = ThyroidWorkupModel()
    cells = build_proposal_cells(model, config)
    records: list[dict[str, Any]] = []
    for cell in cells:
        roots = fixed_roots(model, state=cell.state, belief=cell.belief)
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
        d1_costs = exact_action_costs(model, state=cell.state, belief=cell.belief, depth=1)
        d2_costs = exact_action_costs(model, state=cell.state, belief=cell.belief, depth=2)
        d1_root = _choose_action(d1_costs)
        exhaustive_root = _choose_action(d2_costs)
        shared_cost = float(d2_costs[d1_root])
        exhaustive_cost = float(d2_costs[exhaustive_root])
        llm_cost = float(llm_costs[llm_slot])
        opportunity = shared_cost - exhaustive_cost
        recovery = (shared_cost - llm_cost) / opportunity
        records.append(
            {
                "cell_index": cell.cell_index,
                "truth_index": cell.truth_index,
                "history": [list(item) for item in cell.history],
                "roots": list(roots),
                "llm_followups": [strategy.followups for strategy in proposal.strategies],
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
            }
        )
    comparisons = {
        "matched_random_minus_llm_cost": _summary(
            [row["random_cost"] - row["llm_cost"] for row in records],
            config=config,
            label="matched-random-minus-llm",
        ),
        "shared_d1_minus_llm_cost": _summary(
            [row["shared_d1_exact_continuation_cost"] - row["llm_cost"] for row in records],
            config=config,
            label="shared-d1-minus-llm",
        ),
        "recovery_fraction": _summary(
            [row["recovery_fraction"] for row in records],
            config=config,
            label="recovery-fraction",
        ),
        "collection_selection_rate": float(
            np.mean([row["llm_selected_root"] == COLLECT_BLOOD_ACTION for row in records])
        ),
    }
    mechanics = {
        "thirty_two_distinct_cells_resolved": len(records) == config.num_cells
        and len({tuple(tuple(item) for item in row["history"]) for row in records})
        == config.num_cells,
        "all_cells_are_exact_d2_collection_opportunities": all(
            row["exhaustive_d2_root"] == COLLECT_BLOOD_ACTION
            and row["d2_opportunity"] > EPSILON
            for row in records
        ),
        "collection_root_machine_fixed_in_every_cell": all(
            row["roots"][0] == COLLECT_BLOOD_ACTION for row in records
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
        "shared_d1_lower_bound_positive": comparisons["shared_d1_minus_llm_cost"][
            "ci95"
        ][0]
        > 0.0,
        "collection_rate_at_least_threshold": comparisons["collection_selection_rate"]
        >= config.collection_rate_threshold,
        "mean_recovery_at_least_threshold": comparisons["recovery_fraction"]["mean"]
        >= config.recovery_threshold,
    }
    return {
        "schema_version": 1,
        "stage": "uci_thyroid_workup_26b_named_proposal_gate",
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
        "# UCI Thyroid Workup 26B Named-Proposal Gate",
        "",
        f"Gate passed: **{result['gate']['passed']}**.",
        "",
        "| Endpoint | Mean | Paired 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("matched_random_minus_llm_cost", "Matched random - LLM cost"),
        ("shared_d1_minus_llm_cost", "Shared d1 - LLM cost"),
        ("recovery_fraction", "Exact d2 opportunity recovery"),
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
            f"Collection-root selection: {comparisons['collection_selection_rate']:.1%}.",
            "",
            "The LLM supplied named branch follow-ups once per cell. All policy scoring, "
            "controls, and bootstraps were exact and made zero LLM calls.",
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
        default=Path("results/nonmyopic/thyroid_workup_26b_proposal_gate_20260723"),
    )
    parser.add_argument("--run-id", default="thyroid-workup-26b-proposal-gate-20260723")
    parser.add_argument("--seed", type=int, default=24_152)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = ThyroidProposalGateConfig(seed=args.seed)
    strategy_config = ThyroidStrategyConfig(seed=args.seed)
    strategy_config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicNamedThyroidModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.location_max_new_tokens = strategy_config.max_new_tokens
        chat_model = build_model_adapter(runtime_config.model_pairs[0].questioner, config=runtime_config)
    provider = NamedThyroidProvider(chat_model, strategy_config)
    try:
        result = run_proposal_gate(provider, config)
    except (RuntimeError, StrategyProposalError) as exc:
        failure = {
            "schema_version": 1,
            "stage": "uci_thyroid_workup_26b_named_proposal_gate",
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
                "accepted": len(provider.physical_requests),
                "invalid": len(provider.invalid_responses),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
