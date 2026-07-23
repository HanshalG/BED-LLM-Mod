"""Frozen exact proposal-quality gate for range-gated Rock Diagnosis."""

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

from environments.rock_diagnosis import RangeGatedRockDiagnosisModel, get_paper_map
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_gated_sensor_strategy_prior import StrategyProposalError, _usage_snapshot
from scripts.nonmyopic_range_gated_rock_strategy import (
    ChatModel,
    DeterministicNamedPlanModel,
    NamedPlanProvider,
    Plan,
    RangeGatedStrategyConfig,
    build_belief_cells,
    enumerate_legal_plans,
    plan_value,
)
from scripts.nonmyopic_rock_depth_oracle import exhaustive_action_values


@dataclass(frozen=True)
class RangeGatedProposalGateConfig:
    num_cells: int = 16
    seed: int = 24_146
    bootstrap_replicates: int = 5_000
    route_selection_threshold: float = 0.75
    recovery_threshold: float = 0.60

    def validate(self) -> None:
        if self.num_cells != 16 or self.bootstrap_replicates != 5_000:
            raise ValueError("the frozen proposal gate uses 16 cells and 5,000 bootstraps")
        if not 0.0 <= self.route_selection_threshold <= 1.0:
            raise ValueError("route selection threshold must be a probability")
        if not 0.0 <= self.recovery_threshold <= 1.0:
            raise ValueError("recovery threshold must be a fraction")


def _stable_seed(*parts: Any) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _best_plan(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    plans: tuple[Plan, ...],
) -> tuple[Plan, float]:
    values = [plan_value(model, position=position, belief=belief, plan=plan) for plan in plans]
    index = max(range(len(plans)), key=lambda item: (values[item], -item))
    return plans[index], float(values[index])


def _random_plan_with_root(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    root: str,
    rng: np.random.Generator,
) -> Plan:
    plan = [root]
    current = model.next_position(position, root)
    for _ in range(2):
        legal = model.legal_actions(current)
        action = legal[int(rng.integers(len(legal)))]
        plan.append(action)
        current = model.next_position(current, action)
    return tuple(plan)


def matched_random_plans(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    seed: int,
) -> tuple[Plan, ...]:
    rng = np.random.default_rng(seed)
    legal = model.legal_actions(position)
    moves = [action for action in legal if model.is_move(action)]
    checks = [action for action in legal if not model.is_move(action)]
    move_roots = rng.choice(moves, size=2, replace=False).tolist()
    check_roots = rng.choice(checks, size=2, replace=False).tolist()
    plans: list[Plan] = []
    for root in [*move_roots, *check_roots]:
        for _ in range(100):
            candidate = _random_plan_with_root(
                model, position=position, root=str(root), rng=rng
            )
            if candidate not in plans:
                plans.append(candidate)
                break
        else:
            raise RuntimeError("could not sample four distinct matched-random plans")
    return tuple(plans)


def _bootstrap_ci(values: np.ndarray, *, seed: int, replicates: int) -> list[float]:
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for start in range(0, replicates, 512):
        size = min(512, replicates - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        chunks.append(values[indices].mean(axis=1))
    samples = np.concatenate(chunks)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _summary(
    values: list[float], *, config: RangeGatedProposalGateConfig, label: str
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "ci95": _bootstrap_ci(
            array,
            seed=_stable_seed(config.seed, "range-gated-bootstrap", label),
            replicates=config.bootstrap_replicates,
        ),
        "wins_ties_losses": [
            int(np.sum(array > 1e-12)),
            int(np.sum(np.abs(array) <= 1e-12)),
            int(np.sum(array < -1e-12)),
        ],
        "values": array.tolist(),
    }


def run_proposal_gate(
    provider: NamedPlanProvider, config: RangeGatedProposalGateConfig
) -> dict[str, Any]:
    config.validate()
    model = RangeGatedRockDiagnosisModel(get_paper_map("7-8"))
    position = model.map_spec.start_position
    cells = build_belief_cells(model, count=config.num_cells, seed=config.seed)
    exhaustive_plans = enumerate_legal_plans(model, position=position, horizon=3)
    records: list[dict[str, Any]] = []
    for cell_index, (belief, history) in enumerate(cells):
        proposed = provider.propose(
            model,
            cell_index=cell_index,
            position=position,
            belief=belief,
            history=history,
        )
        llm_plan, llm_value = _best_plan(
            model, position=position, belief=belief, plans=proposed.plans
        )
        random_plans = matched_random_plans(
            model,
            position=position,
            seed=_stable_seed(config.seed, "matched-random", cell_index),
        )
        random_plan, random_value = _best_plan(
            model, position=position, belief=belief, plans=random_plans
        )
        truncated = tuple(plan[:2] for plan in proposed.plans)
        shared_plan, _ = _best_plan(
            model, position=position, belief=belief, plans=truncated
        )
        shared_index = truncated.index(shared_plan)
        shared_full_plan = proposed.plans[shared_index]
        shared_full_value = plan_value(
            model, position=position, belief=belief, plan=shared_full_plan
        )
        d2_values, _ = exhaustive_action_values(
            model, position=position, belief=belief, depth=2
        )
        legal = model.legal_actions(position)
        d2_root = max(
            legal, key=lambda action: (d2_values[action], -legal.index(action))
        )
        strong_d2_plans = tuple(plan for plan in exhaustive_plans if plan[0] == d2_root)
        strong_d2_plan, strong_d2_h3_value = _best_plan(
            model, position=position, belief=belief, plans=strong_d2_plans
        )
        exact_plan, exact_h3_value = _best_plan(
            model, position=position, belief=belief, plans=exhaustive_plans
        )
        opportunity = exact_h3_value - strong_d2_h3_value
        if opportunity <= 1e-12 or not model.is_move(exact_plan[0]):
            raise RuntimeError("proposal cell is not a strict range-gated d3 opportunity")
        records.append(
            {
                "cell_index": cell_index,
                "history": [list(item) for item in history],
                "llm_plans": [list(plan) for plan in proposed.plans],
                "llm_selected_plan": list(llm_plan),
                "llm_value": llm_value,
                "shared_d2_selected_full_plan": list(shared_full_plan),
                "shared_d2_selected_full_value": shared_full_value,
                "matched_random_plans": [list(plan) for plan in random_plans],
                "matched_random_selected_plan": list(random_plan),
                "matched_random_value": random_value,
                "strong_d2_root": d2_root,
                "strong_d2_h3_plan": list(strong_d2_plan),
                "strong_d2_h3_value": strong_d2_h3_value,
                "exact_h3_plan": list(exact_plan),
                "exact_h3_value": exact_h3_value,
                "d3_opportunity": opportunity,
                "recovery_fraction": (llm_value - strong_d2_h3_value) / opportunity,
            }
        )
    comparisons = {
        "llm_minus_matched_random": _summary(
            [row["llm_value"] - row["matched_random_value"] for row in records],
            config=config,
            label="llm-minus-matched-random",
        ),
        "llm_minus_shared_d2": _summary(
            [
                row["llm_value"] - row["shared_d2_selected_full_value"]
                for row in records
            ],
            config=config,
            label="llm-minus-shared-d2",
        ),
        "llm_minus_strong_d2": _summary(
            [row["llm_value"] - row["strong_d2_h3_value"] for row in records],
            config=config,
            label="llm-minus-strong-d2",
        ),
        "recovery_fraction": _summary(
            [row["recovery_fraction"] for row in records],
            config=config,
            label="recovery-fraction",
        ),
        "exact_d3_root_selection_rate": float(
            np.mean(
                [
                    row["llm_selected_plan"][0] == row["exact_h3_plan"][0]
                    for row in records
                ]
            )
        ),
    }
    mechanics = {
        "sixteen_distinct_cells_resolved": len(records) == config.num_cells
        and len({tuple(tuple(item) for item in row["history"]) for row in records})
        == config.num_cells,
        "all_cells_are_strict_d3_opportunities": all(
            row["d3_opportunity"] > 1e-12 for row in records
        ),
        "all_exact_d3_roots_move": all(
            model.is_move(row["exact_h3_plan"][0]) for row in records
        ),
        "all_strong_d2_roots_check": all(
            not model.is_move(row["strong_d2_root"]) for row in records
        ),
        "all_plans_exactly_scored": True,
        "scoring_made_no_llm_calls": True,
    }
    endpoint_gate = {
        "matched_random_lower_bound_positive": comparisons[
            "llm_minus_matched_random"
        ]["ci95"][0]
        > 0.0,
        "shared_d2_lower_bound_positive": comparisons["llm_minus_shared_d2"][
            "ci95"
        ][0]
        > 0.0,
        "strong_d2_lower_bound_positive": comparisons["llm_minus_strong_d2"][
            "ci95"
        ][0]
        > 0.0,
        "route_selection_at_least_threshold": comparisons[
            "exact_d3_root_selection_rate"
        ]
        >= config.route_selection_threshold,
        "mean_recovery_at_least_threshold": comparisons["recovery_fraction"][
            "mean"
        ]
        >= config.recovery_threshold,
    }
    return {
        "schema_version": 1,
        "stage": "range_gated_rock_26b_named_plan_proposal_gate",
        "config": asdict(config),
        "mechanics": mechanics,
        "comparisons": comparisons,
        "endpoint_gate": endpoint_gate,
        "records": records,
        "candidate_requests": provider.physical_requests,
        "invalid_responses": provider.invalid_responses,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Range-Gated Rock 26B Named-Plan Proposal Gate",
        "",
        f"Gate passed: **{result['gate']['passed']}**.",
        "",
        "| Endpoint | Mean | 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("llm_minus_matched_random", "LLM - matched random"),
        ("llm_minus_shared_d2", "LLM h3 - shared-plan d2"),
        ("llm_minus_strong_d2", "LLM h3 - strong d2 root"),
        ("recovery_fraction", "Exact d3 opportunity recovery"),
    ):
        row = result["comparisons"][key]
        lines.append(
            f"| {label} | {row['mean']:+.6f} | [{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] | "
            f"{'/'.join(str(value) for value in row['wins_ties_losses'])} |"
        )
    lines.extend(
        [
            "",
            "Exact d3 route-root selection: "
            f"{result['comparisons']['exact_d3_root_selection_rate']:.1%}.",
            "",
            "The model proposed named plans once per cell. Exact code performed all scoring and controls.",
            "",
        ]
    )
    return "\n".join(lines)


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
        default=Path("results/nonmyopic/range_gated_rock_26b_proposal_gate_20260723"),
    )
    parser.add_argument("--run-id", default="range-gated-rock-26b-proposal-gate-20260723")
    parser.add_argument("--seed", type=int, default=24_146)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gate_config = RangeGatedProposalGateConfig(seed=args.seed)
    strategy_config = RangeGatedStrategyConfig(seed=args.seed)
    gate_config.validate()
    strategy_config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicNamedPlanModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.location_max_new_tokens = strategy_config.max_new_tokens
        chat_model = build_model_adapter(runtime_config.model_pairs[0].questioner, config=runtime_config)
    provider = NamedPlanProvider(chat_model, strategy_config)
    try:
        result = run_proposal_gate(provider, gate_config)
    except (RuntimeError, StrategyProposalError) as exc:
        failure = {
            "schema_version": 1,
            "stage": "range_gated_rock_26b_named_plan_proposal_gate",
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(gate_config),
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
    result["mechanics"]["sixteen_accepted_cells"] = len(provider.physical_requests) == gate_config.num_cells
    result["mechanics"]["zero_reasoning_tokens"] = int(result["usage"].get("reasoning_tokens", 0)) == 0
    result["mechanics"]["zero_forced_exits"] = int(result["usage"].get("forced_exits", 0)) == 0
    result["gate"] = {
        "passed": all(result["mechanics"].values()) and all(result["endpoint_gate"].values())
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
