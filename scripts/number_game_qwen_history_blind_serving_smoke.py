#!/usr/bin/env python3
"""Run the exact-10 Qwen history-blind support serving gate."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.number_game_generator_aware_bed import (
    MAX_TOKENS,
    initial_messages,
    proposal_response_format,
)
from scripts.number_game_item_isolated_codec import (
    parse_proposals_item_isolated,
)
from scripts.number_game_predictive_risk_replication import (
    SeededStructuredAdapter,
    TEMPERATURE,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-history-blind-serving-smoke-1"
MODEL_ID = "qwen/qwen3.7-plus"
MODEL_SEEDS = tuple(range(8_900_000, 8_900_010))
EXPECTED_REQUESTS = 10
CONCURRENCY = 10
RUN_BUDGET_USD = 0.12
PROJECTED_COST_USD = 0.02
MIN_DRAW_VALID = 16
MIN_POOL_VALID = 24
MIN_SECOND_DRAW_NOVEL = 2


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PerRequestSeedStructuredAdapter(SeededStructuredAdapter):
    """Bind a distinct seed to each concurrent structured request."""

    def __init__(
        self,
        spec: ModelSpec,
        config: Config,
    ) -> None:
        super().__init__(spec, config, request_seed=0)
        self._per_request_seed = threading.local()

    def _payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        *,
        disable_reasoning: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages,
            temperature,
            n,
            max_tokens,
            disable_reasoning=disable_reasoning,
            response_format=response_format,
        )
        seed = getattr(self._per_request_seed, "value", None)
        if seed is None:
            raise RuntimeError("per-request seed was not bound")
        payload["seed"] = int(seed)
        return payload

    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages: Sequence[list[dict[str, str]]],
        seeds: Sequence[int],
        *,
        temperature: float,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        if len(batch_messages) != len(seeds):
            raise ValueError("message and seed counts differ")

        def request(
            item: tuple[list[dict[str, str]], int],
        ) -> str:
            messages, seed = item
            self._per_request_seed.value = int(seed)
            try:
                return self._complete_request(
                    messages,
                    temperature,
                    1,
                    max_new_tokens,
                    response_format=response_format,
                )[0]
            finally:
                del self._per_request_seed.value

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            return list(executor.map(request, zip(batch_messages, seeds)))


def build_adapter(
    *,
    run_id: str,
    output_dir: Path,
    concurrency: int,
    run_budget_usd: float,
    projected_cost_usd: float,
) -> PerRequestSeedStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=215.0,
        openrouter_run_budget_usd=run_budget_usd,
        openrouter_projected_cost_usd=projected_cost_usd,
        openrouter_concurrency=concurrency,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return PerRequestSeedStructuredAdapter(
        ModelSpec(
            model=MODEL_ID,
            backend="openrouter",
            max_model_len=65536,
        ),
        config,
    )


def pool_diagnostics(
    responses: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(responses) % 2:
        raise ValueError("pooled responses require an even draw count")
    draw_diagnostics = []
    pools = []
    for pool_index in range(len(responses) // 2):
        merged_extensions = set()
        valid_counts = []
        novel_contributions = []
        local_diagnostics = []
        for response in responses[2 * pool_index : 2 * pool_index + 2]:
            support, diagnostic = parse_proposals_item_isolated(response)
            valid_counts.append(len(support))
            local_diagnostics.append(diagnostic)
            before = len(merged_extensions)
            merged_extensions.update(
                hypothesis.extension for hypothesis in support
            )
            novel_contributions.append(len(merged_extensions) - before)
            draw_diagnostics.append(diagnostic)
        pools.append(
            {
                "pool_index": pool_index,
                "valid_unique_count": len(merged_extensions),
                "draw_valid_counts": valid_counts,
                "draw_novel_contributions": novel_contributions,
                "draw_diagnostics": local_diagnostics,
            }
        )
    return draw_diagnostics, pools


def serving_gates(
    *,
    draw_diagnostics: Sequence[dict[str, Any]],
    pools: Sequence[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, bool]:
    gates = {
        "exact_ten_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "exact_ten_http_attempts": (
            usage["http_attempts"] == EXPECTED_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": (
            usage["provider_error_retries"] == 0
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_smoke_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "all_ten_draws_are_strict_json": (
            len(draw_diagnostics) == EXPECTED_REQUESTS
            and all(
                diagnostic["codec_mode"] == "strict_json"
                for diagnostic in draw_diagnostics
            )
        ),
        "every_draw_has_at_least_sixteen_valid": all(
            diagnostic["valid_unique_count"] >= MIN_DRAW_VALID
            for diagnostic in draw_diagnostics
        ),
        "every_pool_has_at_least_twenty_four_valid": (
            len(pools) == EXPECTED_REQUESTS // 2
            and all(
                pool["valid_unique_count"] >= MIN_POOL_VALID
                for pool in pools
            )
        ),
        "every_second_draw_adds_at_least_two_extensions": all(
            pool["draw_novel_contributions"][1]
            >= MIN_SECOND_DRAW_NOVEL
            for pool in pools
        ),
    }
    gates["all_pass"] = all(gates.values())
    return gates


def run_smoke(
    *,
    output_dir: Path,
    run_id: str,
    adapter: PerRequestSeedStructuredAdapter | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    adapter = adapter or build_adapter(
        run_id=run_id,
        output_dir=output_dir,
        concurrency=CONCURRENCY,
        run_budget_usd=RUN_BUDGET_USD,
        projected_cost_usd=PROJECTED_COST_USD,
    )
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        [initial_messages() for _ in MODEL_SEEDS],
        MODEL_SEEDS,
        temperature=TEMPERATURE,
        response_format=proposal_response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    checkpoint(
        raw_path,
        {
            "seeds": list(MODEL_SEEDS),
            "responses": list(responses),
        },
    )
    draw_diagnostics, pools = pool_diagnostics(responses)
    usage = summarize_usage(adapter.usage_snapshot())
    gates = serving_gates(
        draw_diagnostics=draw_diagnostics,
        pools=pools,
        usage=usage,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gated_null",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "model_seeds": list(MODEL_SEEDS),
            "temperature": TEMPERATURE,
            "reasoning": False,
            "prompt": "initial Number Game prompt with no observations",
            "pool_size": 2,
            "expected_requests": EXPECTED_REQUESTS,
            "semantic_repair": False,
            "efficacy_used_for_authorization": False,
            "run_budget_usd": RUN_BUDGET_USD,
        },
        "usage": usage,
        "gates": gates,
        "draw_valid_counts": [
            diagnostic["valid_unique_count"]
            for diagnostic in draw_diagnostics
        ],
        "pool_valid_counts": [
            pool["valid_unique_count"] for pool in pools
        ],
        "second_draw_novel_contributions": [
            pool["draw_novel_contributions"][1] for pool in pools
        ],
        "draw_diagnostics": list(draw_diagnostics),
        "raw_responses_sha256": sha256_file(raw_path),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    try:
        result = run_smoke(output_dir=args.output_dir, run_id=args.run_id)
    except Exception as exc:
        checkpoint(
            args.output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
