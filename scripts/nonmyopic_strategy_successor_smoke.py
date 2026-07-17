"""Serving gate for the distinct StrategyEIG model-capability successor probe."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Protocol

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.rock_diagnosis import RockDiagnosisModel, get_paper_map
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.nonmyopic_copex_strategy_prior import (
    ContinuousProposalError,
    ContinuousStrategyProvider,
    DeterministicContinuousModel,
    L3Config,
)
from scripts.nonmyopic_rock_strategy_prior import (
    L1Config,
    LLMRockStrategyProvider,
    StrategyProposalError,
    DeterministicStrategyModel,
)


class ChatModel(Protocol):
    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]: ...


@dataclass(frozen=True)
class SuccessorSmokeConfig:
    seed: int = 31_003
    l1_cells: int = 5
    l3_cells: int = 5
    num_particles: int = 64
    num_strategies: int = 4
    l1_horizon: int = 2
    l3_horizon: int = 4

    def validate(self) -> None:
        if self.l1_cells <= 0 or self.l3_cells <= 0:
            raise ValueError("smoke cell counts must be positive")
        if self.l1_cells + self.l3_cells != 10:
            raise ValueError("the successor serving gate requires exactly ten generation cells")
        if self.num_particles < 4 or self.num_strategies < 2:
            raise ValueError("smoke requires addressable particles and at least two strategies")


class _RoutingDeterministicModel:
    """No-spend model for exercising both prompt/parser routes in tests and dry runs."""

    def __init__(self) -> None:
        self._rock = DeterministicStrategyModel()
        self._continuous = DeterministicContinuousModel()

    def chat_complete(
        self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1
    ) -> list[str]:
        if "Rock coordinates by ID:" in messages[-1]["content"]:
            return self._rock.chat_complete(messages, temperature, num_responses)
        return self._continuous.chat_complete(messages, temperature, num_responses)

    def usage_snapshot(self) -> dict[str, Any]:
        return {"backend": "dry_run", "requests": 0, "cost_usd": 0.0, "forced_exits": 0}


def _run_l1_cells(provider: LLMRockStrategyProvider, config: SuccessorSmokeConfig) -> None:
    for cell_index in range(config.l1_cells):
        map_name = ("3-6", "5-7")[cell_index % 2]
        model = RockDiagnosisModel(get_paper_map(map_name))
        rng = np.random.default_rng(config.seed + cell_index)
        belief = rng.dirichlet(np.ones(len(model.hidden_states)))
        provider.propose_strategies(
            model,
            map_name=map_name,
            trial_index=cell_index,
            position=model.map_spec.start_position,
            belief=belief,
            history=(),
            horizon=config.l1_horizon,
        )


def _run_l3_cells(provider: ContinuousStrategyProvider, config: SuccessorSmokeConfig) -> None:
    for cell_index in range(config.l3_cells):
        rng = np.random.default_rng(config.seed + 100 + cell_index)
        particles = rng.random((config.num_particles, 2))
        probabilities = rng.dirichlet(np.ones(config.num_particles))
        provider.strategies(
            cell_index,
            np.asarray((0.1 + 0.15 * cell_index, 0.8 - 0.1 * cell_index), dtype=float),
            particles,
            probabilities,
            config.l3_horizon,
        )


def _usage(chat_model: Any) -> dict[str, Any]:
    snapshot = getattr(chat_model, "usage_snapshot", None)
    return snapshot() if callable(snapshot) else {"backend": "unknown"}


def run_successor_smoke(chat_model: ChatModel, config: SuccessorSmokeConfig) -> dict[str, Any]:
    """Run ten strict, no-repair strategy cells across the original L1 and L3 prompts."""

    config.validate()
    l1_provider = LLMRockStrategyProvider(
        chat_model,
        L1Config(
            num_trials_per_map=1,
            num_rounds=1,
            num_strategies=config.num_strategies,
            planning_horizon=config.l1_horizon,
            bootstrap_replicates=1,
            validation_retries=0,
            trial_concurrency=1,
        ),
    )
    l3_provider = ContinuousStrategyProvider(
        chat_model,
        L3Config(
            num_trials=1,
            num_rounds=1,
            num_particles=config.num_particles,
            num_strategies=config.num_strategies,
            planning_horizon=config.l3_horizon,
            rollout_samples=1,
            grid_resolution=2,
            bootstrap_replicates=1,
            validation_retries=0,
            trial_concurrency=1,
        ),
    )
    try:
        _run_l1_cells(l1_provider, config)
        _run_l3_cells(l3_provider, config)
        usage = _usage(chat_model)
        requests = len(l1_provider.physical_requests) + len(l3_provider.accepted_requests)
        forced_exits = int(usage.get("forced_exits", 0) or 0)
        if requests != 10:
            raise RuntimeError(f"serving gate expected ten successful generation cells, got {requests}")
        if l1_provider.invalid_responses or l3_provider.invalid_responses:
            raise RuntimeError("serving gate accepted a repaired or invalid generation cell")
        if forced_exits:
            raise RuntimeError(f"serving gate observed {forced_exits} forced reasoning/output exits")
    except (ContinuousProposalError, StrategyProposalError, RuntimeError) as exc:
        setattr(exc, "l1_requests", l1_provider.physical_requests)
        setattr(exc, "l1_invalid_responses", l1_provider.invalid_responses)
        setattr(exc, "l3_requests", l3_provider.accepted_requests)
        setattr(exc, "l3_invalid_responses", l3_provider.invalid_responses)
        raise
    return {
        "schema_version": 1,
        "stage": "strategy_successor_interface_smoke",
        "status": "passed",
        "config": asdict(config),
        "checks": {
            "successful_generation_cells": requests,
            "l1_successful_cells": len(l1_provider.physical_requests),
            "l3_successful_cells": len(l3_provider.accepted_requests),
            "zero_invalid_or_repaired_cells": True,
            "zero_forced_exits": True,
        },
        "l1_requests": l1_provider.physical_requests,
        "l3_requests": l3_provider.accepted_requests,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config_nonmyopic_strategy_successor_smoke_qwen397_thinking_openrouter.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/strategy_successor_qwen397_interface_smoke/20260718"),
    )
    parser.add_argument("--run-id", default="nonmyopic-strategy-successor-qwen397-interface-smoke-20260718")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = SuccessorSmokeConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        chat_model: ChatModel = _RoutingDeterministicModel()
    else:
        runtime: Config = load_config(args.config)
        runtime.run_id = args.run_id
        chat_model = build_model_adapter(runtime.model_pairs[0].questioner, config=runtime)
    try:
        result = run_successor_smoke(chat_model, config)
    except (ContinuousProposalError, StrategyProposalError, RuntimeError) as exc:
        failure = {
            "schema_version": 1,
            "stage": "strategy_successor_interface_smoke",
            "status": "failed_closed",
            "error": str(exc),
            "config": asdict(config),
            "l1_requests": getattr(exc, "l1_requests", []),
            "l1_invalid_responses": getattr(exc, "l1_invalid_responses", []),
            "l3_requests": getattr(exc, "l3_requests", []),
            "l3_invalid_responses": getattr(exc, "l3_invalid_responses", []),
            "usage": _usage(chat_model),
        }
        (args.output_dir / "SMOKE_FAILURE.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise
    result["run_id"] = args.run_id
    result["dry_run"] = args.dry_run
    (args.output_dir / "SMOKE.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["checks"], "usage": result["usage"]}, indent=2))


if __name__ == "__main__":
    main()
