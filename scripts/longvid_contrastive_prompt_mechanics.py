#!/usr/bin/env python3
"""Run prompt-only contrastive LongVid belief mechanics on new tasks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
import scripts.longvid_contrastive_path_belief_mechanics as base
from scripts.longvid_contrastive_prompt_serving_smoke import (
    prompt_only_rank_schema_messages,
    prompt_only_support_messages,
)


INTERFACE_VERSION = "longvid-contrastive-prompt-mechanics-1"
LAYOUT_SEED = 270744
TASK_LAYOUT = (
    (1404, (2, 0)),
    (2703, (0, 19)),
    (1867, (5, 4)),
    (1295, (4, 0)),
)
TASK_LAYOUT_HASH = (
    "90090d9912857fb7f4dea4992933e845e4a0f9367504f52215ce9160d3e10ba0"
)
EXCLUDED_PRIOR_ROWS = (
    955,
    1802,
    540,
    479,
    1332,
    1068,
    2156,
    2062,
    1689,
    1648,
)
PROMPT_SMOKE_SHA256 = (
    "fe332c80f79bc4f8eaff6dbbe2d9cc7831e1dc0b6648a07d8037784ac7890068"
)
STRUCTURAL_CONFIRMATION_SHA256 = (
    "895e448c047ca7afa924393da0a4f637a21735c8a770336fa17db0a72fdd4081"
)


class OrdinaryChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class PromptOnlyBridge:
    """Expose prompt-only chat through the base runner's codec protocol."""

    def __init__(self, delegate: OrdinaryChatModel) -> None:
        self.delegate = delegate

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        name = response_format["json_schema"]["name"]
        if name == "longvid_semantic_belief_support":
            messages = [
                prompt_only_support_messages(value)
                for value in batch_messages
            ]
        elif name == "longvid_contrastive_path_rank":
            messages = [
                prompt_only_rank_schema_messages(value)
                for value in batch_messages
            ]
        else:
            raise ValueError(f"unsupported prompt codec {name}")
        return self.delegate.chat_complete_messages_batched(
            messages,
            temperature=temperature,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
        )

    def usage_snapshot(self) -> dict[str, Any]:
        return self.delegate.usage_snapshot()


class _FixtureStructuredModel(base.DeterministicFixtureModel):
    GREEDY_ROOT = {1404: 0, 2703: 0, 1867: 5, 1295: 0}
    ORACLE_ROOT = {1404: 2, 2703: 19, 1867: 4, 1295: 4}


class PromptFixtureModel:
    def __init__(self) -> None:
        self.structured = _FixtureStructuredModel()

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        is_rank = "TRAJECTORY_A=" in batch_messages[0][-1]["content"]
        response_format = (
            base.rank_response_format()
            if is_rank
            else base.support_response_format()
        )
        return self.structured.chat_complete_messages_batched_structured(
            batch_messages,
            temperature=temperature,
            block_size=block_size,
            response_format=response_format,
            max_new_tokens=max_new_tokens,
        )

    def usage_snapshot(self) -> dict[str, Any]:
        return self.structured.usage_snapshot()


def verify_bindings() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "prompt_smoke": (
            root
            / "results/nonmyopic/longvid_contrastive_prompt_serving_smoke/"
            "longvid-contrastive-prompt-smoke-20260727T183000Z/SMOKE.json"
        ),
        "structural_confirmation": (
            root
            / "results/nonmyopic/longvid_four_hop_tradeoff_confirmation_v2/"
            "CONFIRMATION.json"
        ),
    }
    expected = {
        "prompt_smoke": PROMPT_SMOKE_SHA256,
        "structural_confirmation": STRUCTURAL_CONFIRMATION_SHA256,
    }
    for name, path in paths.items():
        if base.sha256_file(path) != expected[name]:
            raise ValueError(f"{name} artifact hash does not match")


def run_prompt_mechanics(
    config: Config,
    *,
    qa_path: Path,
    caption_path: Path,
    raw_path: Path,
    model: OrdinaryChatModel,
) -> dict[str, Any]:
    verify_bindings()
    payload = base.run_mechanics(
        config,
        qa_path=qa_path,
        caption_path=caption_path,
        raw_path=raw_path,
        model=PromptOnlyBridge(model),
        task_layout=TASK_LAYOUT,
        task_layout_hash=TASK_LAYOUT_HASH,
        excluded_prior_rows=EXCLUDED_PRIOR_ROWS,
        response_format_name="prompt_only_flat_json",
        interface_version=INTERFACE_VERSION,
        layout_seed=LAYOUT_SEED,
    )
    payload["protocol"]["prompt_smoke_sha256"] = PROMPT_SMOKE_SHA256
    payload["protocol"][
        "structural_confirmation_sha256"
    ] = STRUCTURAL_CONFIRMATION_SHA256
    return payload


def _nonthinking_spec(spec: Any) -> Any:
    return replace(
        spec,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )


def _build_model(config: Config) -> OrdinaryChatModel:
    spec = _nonthinking_spec(config.model_pairs[0].questioner)
    if spec.model != base.MODEL_ID:
        raise ValueError("LongVid prompt mechanics config selects the wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qa-path", type=Path, required=True)
    parser.add_argument("--caption-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = base.PROJECTED_COST_USD
    config.openrouter_run_budget_usd = base.MAX_COST_USD
    config.openrouter_concurrency = 8
    config.openrouter_max_retries = 0
    config.openrouter_max_output_tokens = base.MAX_NEW_TOKENS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: OrdinaryChatModel | None = None
    try:
        model = PromptFixtureModel() if args.dry_run else _build_model(config)
        payload = run_prompt_mechanics(
            config,
            qa_path=args.qa_path,
            caption_path=args.caption_path,
            raw_path=raw_path,
            model=model,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, base.MechanicsExecutionError):
            failure["usage"] = exc.usage
        elif model is not None:
            failure["usage"] = base._usage(PromptOnlyBridge(model))
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        base._checkpoint(args.output_dir / "MECHANICS_FAILURE.json", failure)
        raise
    base._checkpoint(args.output_dir / "MECHANICS.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "summary": payload["summary"],
                "diagnostics": payload["diagnostics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
