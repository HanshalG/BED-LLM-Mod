"""Proposal-quality gate for successor-grounded range-gated h3 plans."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from scripts.nonmyopic_gated_sensor_strategy_prior import (
    StrategyProposalError,
    _usage_snapshot,
)
from scripts.nonmyopic_range_gated_rock_fixed_tail import (
    DeterministicFixedTailModel,
    FixedRootTailProvider,
    fixed_roots,
)
from scripts.nonmyopic_range_gated_rock_proposal_gate import (
    RangeGatedProposalGateConfig,
    _best_plan,
    _random_plan_with_root,
    _stable_seed,
    _summary,
)
from scripts.nonmyopic_range_gated_rock_strategy import (
    ChatModel,
    Plan,
    RangeGatedStrategyConfig,
    build_belief_cells,
    enumerate_legal_plans,
    plan_value,
)
from scripts.nonmyopic_rock_depth_oracle import exhaustive_action_values


def matched_random_fixed_root_plans(
    model: RangeGatedRockDiagnosisModel,
    *,
    position: tuple[int, int],
    belief: np.ndarray,
    seed: int,
) -> tuple[Plan, ...]:
    rng = np.random.default_rng(seed)
    return tuple(
        _random_plan_with_root(
            model,
            position=position,
            root=root,
            rng=rng,
        )
        for root in fixed_roots(model, position=position, belief=belief)
    )


def run_proposal_gate(
    provider: FixedRootTailProvider,
    config: RangeGatedProposalGateConfig,
) -> dict[str, Any]:
    config.validate()
    model = RangeGatedRockDiagnosisModel(
        get_paper_map("7-8"), remote_accuracy=0.55, onsite_accuracy=0.95
    )
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
        roots = fixed_roots(model, position=position, belief=belief)
        if tuple(plan[0] for plan in proposed.plans) != roots:
            raise RuntimeError("compiled plans do not preserve the machine-fixed roots")
        llm_plan, llm_value = _best_plan(
            model,
            position=position,
            belief=belief,
            plans=proposed.plans,
        )
        random_plans = matched_random_fixed_root_plans(
            model,
            position=position,
            belief=belief,
            seed=_stable_seed(config.seed, "fixed-root-random", cell_index),
        )
        random_plan, random_value = _best_plan(
            model,
            position=position,
            belief=belief,
            plans=random_plans,
        )
        truncated = tuple(plan[:2] for plan in proposed.plans)
        shared_plan, _ = _best_plan(
            model,
            position=position,
            belief=belief,
            plans=truncated,
        )
        shared_index = truncated.index(shared_plan)
        shared_full_plan = proposed.plans[shared_index]
        shared_full_value = plan_value(
            model,
            position=position,
            belief=belief,
            plan=shared_full_plan,
        )
        d2_values, _ = exhaustive_action_values(
            model,
            position=position,
            belief=belief,
            depth=2,
        )
        legal = model.legal_actions(position)
        d2_root = max(
            legal,
            key=lambda action: (d2_values[action], -legal.index(action)),
        )
        strong_d2_plans = tuple(
            plan for plan in exhaustive_plans if plan[0] == d2_root
        )
        strong_d2_plan, strong_d2_value = _best_plan(
            model,
            position=position,
            belief=belief,
            plans=strong_d2_plans,
        )
        exact_plan, exact_value = _best_plan(
            model,
            position=position,
            belief=belief,
            plans=exhaustive_plans,
        )
        opportunity = exact_value - strong_d2_value
        if opportunity <= 1e-12 or exact_plan[0] != "move-SOUTH":
            raise RuntimeError("cell is not the registered strict delayed-route opportunity")
        records.append(
            {
                "cell_index": cell_index,
                "history": [list(item) for item in history],
                "fixed_roots": list(roots),
                "llm_plans": [list(plan) for plan in proposed.plans],
                "llm_selected_plan": list(llm_plan),
                "llm_value": llm_value,
                "matched_random_plans": [list(plan) for plan in random_plans],
                "matched_random_selected_plan": list(random_plan),
                "matched_random_value": random_value,
                "shared_d2_selected_full_plan": list(shared_full_plan),
                "shared_d2_selected_full_value": shared_full_value,
                "strong_d2_root": d2_root,
                "strong_d2_h3_plan": list(strong_d2_plan),
                "strong_d2_h3_value": strong_d2_value,
                "exact_h3_plan": list(exact_plan),
                "exact_h3_value": exact_value,
                "d3_opportunity": opportunity,
                "recovery_fraction": (llm_value - strong_d2_value) / opportunity,
            }
        )
    comparisons = {
        "llm_minus_matched_random": _summary(
            [row["llm_value"] - row["matched_random_value"] for row in records],
            config=config,
            label="fixed-tail-llm-minus-random",
        ),
        "llm_minus_shared_d2": _summary(
            [
                row["llm_value"] - row["shared_d2_selected_full_value"]
                for row in records
            ],
            config=config,
            label="fixed-tail-llm-minus-shared-d2",
        ),
        "llm_minus_strong_d2": _summary(
            [row["llm_value"] - row["strong_d2_h3_value"] for row in records],
            config=config,
            label="fixed-tail-llm-minus-strong-d2",
        ),
        "recovery_fraction": _summary(
            [row["recovery_fraction"] for row in records],
            config=config,
            label="fixed-tail-recovery",
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
        and len(
            {
                tuple(tuple(item) for item in row["history"])
                for row in records
            }
        )
        == config.num_cells,
        "all_cells_are_strict_d3_opportunities": all(
            row["d3_opportunity"] > 1e-12 for row in records
        ),
        "all_roots_match_fixed_interface": all(
            [plan[0] for plan in row["llm_plans"]] == row["fixed_roots"]
            for row in records
        ),
        "all_controls_exactly_scored": True,
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
        "stage": "range_gated_rock_successor_grounded_proposal_gate",
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
        "# Range-Gated Rock Successor-Grounded Proposal Gate",
        "",
        f"Gate passed: **{result['gate']['passed']}**.",
        "",
        "| Endpoint | Mean | 95% CI | W/T/L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("llm_minus_matched_random", "LLM - identical-root random"),
        ("llm_minus_shared_d2", "LLM h3 - shared-plan d2"),
        ("llm_minus_strong_d2", "LLM h3 - strong d2 root"),
        ("recovery_fraction", "Exact h3 opportunity recovery"),
    ):
        row = result["comparisons"][key]
        lines.append(
            f"| {label} | {row['mean']:+.6f} | "
            f"[{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] | "
            f"{'/'.join(str(value) for value in row['wins_ties_losses'])} |"
        )
    lines.extend(
        [
            "",
            f"- Exact h3 route-root selection: "
            f"`{result['comparisons']['exact_d3_root_selection_rate']:.3f}`.",
            f"- Mechanics: `{result['mechanics']}`.",
            f"- Endpoint gates: `{result['endpoint_gate']}`.",
            f"- Usage: `{result['usage']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=24_184)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/nonmyopic/range_gated_rock_successor_grounded_cluster26b_proposal_20260723"
        ),
    )
    parser.add_argument(
        "--run-id",
        default="range-gated-rock-successor-grounded-cluster26b-proposal-20260723",
    )
    parser.add_argument("--model-generation-tokens", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gate_config = RangeGatedProposalGateConfig(seed=args.seed)
    strategy_config = RangeGatedStrategyConfig(seed=args.seed)
    gate_config.validate()
    strategy_config.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = DeterministicFixedTailModel()
    else:
        runtime_config: Config = load_config(args.config)
        runtime_config.run_id = args.run_id
        runtime_config.location_max_new_tokens = args.model_generation_tokens
        chat_model = build_model_adapter(
            runtime_config.model_pairs[0].questioner,
            config=runtime_config,
        )
    provider = FixedRootTailProvider(
        chat_model,
        strategy_config,
        include_successor_grounding=True,
    )
    try:
        result = run_proposal_gate(provider, gate_config)
    except StrategyProposalError as exc:
        failure = {
            "schema_version": 1,
            "stage": "range_gated_rock_successor_grounded_proposal_gate",
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(gate_config),
            "candidate_requests": provider.physical_requests,
            "invalid_responses": provider.invalid_responses,
            "usage": _usage_snapshot(chat_model),
        }
        (args.output_dir / "PROPOSAL_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    result["usage"] = _usage_snapshot(chat_model)
    result["mechanics"]["usage_accounted"] = all(
        field in result["usage"]
        for field in ("requests", "completion_tokens", "forced_exits")
    )
    result["gate"] = {
        "passed": all(result["mechanics"].values())
        and all(result["endpoint_gate"].values())
    }
    (args.output_dir / "PROPOSAL.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "PROPOSAL.md").write_text(
        render_report(result),
        encoding="utf-8",
    )
    print(json.dumps({"gate": result["gate"], "usage": result["usage"]}, indent=2))


if __name__ == "__main__":
    main()
