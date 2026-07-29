#!/usr/bin/env python3
"""Robust keyed-line LongVid serving smoke and two-pair mechanics gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.strict_rows import parse_keyed_pipe_rows
from helpers import Config, load_config
from model_factory import build_model_adapter
import scripts.longvid_contrastive_path_belief_mechanics as base
import scripts.longvid_contrastive_prompt_serving_smoke as smoke


INTERFACE_VERSION = "longvid-robust-keyed-belief-1"
MECHANICS_LAYOUT_SEED = 40410
MECHANICS_TASK_LAYOUT = (
    (319, (0, 9)),
    (120, (2, 1)),
)
MECHANICS_TASK_LAYOUT_HASH = (
    "a2783ca4a1a1140e427e031820eddc0a2e055582ab4c4fb03fc68dd149c46a9f"
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
    1404,
    2703,
    1867,
    1295,
    2989,
    549,
    2642,
    2345,
)
STRUCTURAL_CONFIRMATION_SHA256 = (
    "895e448c047ca7afa924393da0a4f637a21735c8a770336fa17db0a72fdd4081"
)
EXPECTED_SMOKE_REQUESTS = 10
EXPECTED_MECHANICS_REQUESTS = 22
MAX_TRANSPORT_RETRIES = 2
SMOKE_MAX_COST_USD = 0.20
MECHANICS_MAX_COST_USD = 0.50
MAX_NEW_TOKENS = 900


class OrdinaryChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


def keyed_support_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    result = [dict(message) for message in messages]
    result[0]["content"] = re.sub(
        r"Return only the schema-conforming object\.?",
        "",
        result[0]["content"],
    ).strip()
    result[0]["content"] += (
        "\nReturn exactly six pipe-delimited rows and no other text. The rows "
        "may be in any order, but keys H1 through H6 must each occur exactly "
        "once. Format: Hn|weight|anchor|query|hypothesis. weight is an integer "
        "1..100 and at least two weights differ. Every field is nonempty and "
        "contains no pipe or newline."
    )
    return result


def keyed_rank_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    result = [dict(message) for message in messages]
    result[0]["content"] = re.sub(
        r"Return only the schema-conforming object\.?",
        "",
        result[0]["content"],
    ).strip()
    result[0]["content"] += (
        "\nReturn exactly one pipe-delimited row and no other text: "
        "R|choice|confidence|unresolved_need. choice is A or B; confidence is "
        "an integer 51..100; unresolved_need is nonempty and contains no pipe "
        "or newline."
    )
    return result


def parse_keyed_support(text: str) -> list[dict[str, Any]]:
    keys = [f"H{index}" for index in range(1, base.SUPPORT_SIZE + 1)]
    rows = parse_keyed_pipe_rows(
        text,
        expected_keys=keys,
        value_fields=4,
    )
    support: list[dict[str, Any]] = []
    for key in keys:
        weight_text, anchor, query, hypothesis = rows[key]
        if not re.fullmatch(r"\d{1,3}", weight_text):
            raise ValueError(f"{key} weight is not an integer")
        weight = int(weight_text)
        if not 1 <= weight <= 100:
            raise ValueError(f"{key} weight is outside 1..100")
        support.append(
            {
                "hypothesis": hypothesis,
                "weight": weight,
                "anchor": anchor,
                "query": query,
            }
        )
    if len({base._normalize(row["hypothesis"]) for row in support}) != 6:
        raise ValueError("support hypotheses must be distinct")
    if len({base._normalize(row["query"]) for row in support}) != 6:
        raise ValueError("support queries must be distinct")
    if len({row["weight"] for row in support}) < 2:
        raise ValueError("support must express nonuniform confidence")
    return support


def parse_keyed_rank(text: str) -> dict[str, Any]:
    row = parse_keyed_pipe_rows(
        text,
        expected_keys=["R"],
        value_fields=3,
    )["R"]
    choice, confidence_text, unresolved_need = row
    if choice not in {"A", "B"}:
        raise ValueError("rank choice is invalid")
    if not re.fullmatch(r"\d{2,3}", confidence_text):
        raise ValueError("rank confidence is not an integer")
    confidence = int(confidence_text)
    if not 51 <= confidence <= 100:
        raise ValueError("rank confidence is outside 51..100")
    return {
        "choice": choice,
        "confidence": confidence,
        "unresolved_need": unresolved_need,
    }


def smoke_rank_messages(
    supports: list[list[dict[str, Any]]],
) -> list[dict[str, str]]:
    trajectories = []
    for candidate, branch_supports in zip(("A", "B"), supports, strict=True):
        trajectories.append(
            {
                "candidate": candidate,
                "queries": [
                    f"synthetic branch {candidate} query {index}"
                    for index in range(1, 5)
                ],
                "belief_supports": [
                    base._support_for_prompt(support_value)
                    for support_value in branch_supports
                ],
            }
        )
    return keyed_rank_messages(
        [
            {
                "role": "system",
                "content": (
                    "Compare two blinded retrieval trajectories from their "
                    "evolving semantic belief supports and queries. Choose the "
                    "one more likely to expose a complete four-link evidence "
                    "chain. Do not answer the question or reason aloud."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION={smoke.QUESTION}\nTRAJECTORIES="
                    + json.dumps(trajectories, separators=(",", ":"))
                ),
            },
        ]
    )


class KeyedCodecBridge:
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
            converted = [keyed_support_messages(value) for value in batch_messages]
        elif name == "longvid_contrastive_path_rank":
            converted = [keyed_rank_messages(value) for value in batch_messages]
        else:
            raise ValueError(f"unsupported keyed codec {name}")
        return self.delegate.chat_complete_messages_batched(
            converted,
            temperature=temperature,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
        )

    def usage_snapshot(self) -> dict[str, Any]:
        return self.delegate.usage_snapshot()


class KeyedFixtureModel:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses = []
        for offset, messages in enumerate(batch_messages):
            user = messages[-1]["content"]
            if "TRAJECTORY_A=" in user or "TRAJECTORIES=" in user:
                final = "STAGE=FINAL_FOUR_STEP" in user
                responses.append(
                    f"R|{'B' if final else 'A'}|80|fixture evidence bridge"
                )
                continue
            observation_match = re.search(
                r"OBSERVATION:\n(.+?)\nPREVIOUS_SUPPORT:",
                user,
                flags=re.DOTALL,
            )
            anchor = "QUESTION"
            if observation_match:
                question_match = re.search(
                    r"QUESTION:\n(.+?)\nPREVIOUS_QUERY:",
                    user,
                    flags=re.DOTALL,
                )
                query_match = re.search(
                    r"PREVIOUS_QUERY:\n(.+?)\nOBSERVATION:",
                    user,
                    flags=re.DOTALL,
                )
                excluded = set(
                    base.tokenize(
                        question_match.group(1) if question_match else ""
                    )
                ).union(
                    base.tokenize(query_match.group(1) if query_match else "")
                )
                tokens = [
                    token
                    for token in base.tokenize(observation_match.group(1))
                    if token not in excluded
                ]
                anchor = tokens[0] if tokens else "caption"
            suffix = self.requests + offset
            rows = [
                (
                    f"H{index}|{10 + index + suffix % 3}|{anchor}|"
                    f"{anchor} fixture query {suffix} {index}|"
                    f"Fixture chain {suffix} hypothesis {index}"
                )
                for index in range(1, 7)
            ]
            responses.append("\n".join(reversed(rows)))
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


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
        raise ValueError("LongVid robust keyed config selects the wrong model")
    return build_model_adapter(spec, config)


def verify_mechanics_bindings() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "results/nonmyopic/longvid_four_hop_tradeoff_confirmation_v2/"
        "CONFIRMATION.json"
    )
    if base.sha256_file(path) != STRUCTURAL_CONFIRMATION_SHA256:
        raise ValueError("structural confirmation artifact hash does not match")
    if base.layout_hash_for(MECHANICS_TASK_LAYOUT) != MECHANICS_TASK_LAYOUT_HASH:
        raise ValueError("mechanics task layout hash does not match")


def run_mechanics(
    config: Config,
    *,
    qa_path: Path,
    caption_path: Path,
    raw_path: Path,
    model: OrdinaryChatModel,
) -> dict[str, Any]:
    verify_mechanics_bindings()
    payload = base.run_mechanics(
        config,
        qa_path=qa_path,
        caption_path=caption_path,
        raw_path=raw_path,
        model=KeyedCodecBridge(model),
        task_layout=MECHANICS_TASK_LAYOUT,
        task_layout_hash=MECHANICS_TASK_LAYOUT_HASH,
        excluded_prior_rows=EXCLUDED_PRIOR_ROWS,
        response_format_name="order_insensitive_keyed_pipe_rows",
        interface_version=INTERFACE_VERSION,
        layout_seed=MECHANICS_LAYOUT_SEED,
        support_parser=parse_keyed_support,
        rank_parser=parse_keyed_rank,
    )
    usage = payload["usage"]
    summary = payload["summary"]
    diagnostics = payload["diagnostics"]
    gates = {
        "exact_22_accepted_requests": (
            usage["physical_requests"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "http_attempts_within_24": (
            EXPECTED_MECHANICS_REQUESTS
            <= usage["http_attempts"]
            <= EXPECTED_MECHANICS_REQUESTS + MAX_TRANSPORT_RETRIES
        ),
        "transport_retries_at_most_2": (
            usage["retry_count"] <= MAX_TRANSPORT_RETRIES
        ),
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_outputs_parse": True,
        "endpoint_loaded_after_policy_freeze": bool(
            payload["protocol"]["endpoint_loaded_after_policy_freeze"]
        ),
        "all_four_paths_complete": (
            diagnostics["complete_unique_path_count"] == 4
        ),
        "at_least_14_of_16_refreshes_change": (
            diagnostics["changed_refresh_count"] >= 14
        ),
        "all_16_refreshes_have_four_valid_anchors": (
            diagnostics["valid_anchor_refresh_count"] == 16
        ),
        "both_pairs_rankable": summary["rankable_task_count"] == 2,
        "final_rank_correct_2_of_2": (
            summary["final_pairwise_correct"] == 2
        ),
        "final_rank_beats_immediate": (
            summary["final_pairwise_accuracy"]
            > summary["immediate_pairwise_accuracy"]
        ),
        "both_policy_changes_correct": (
            summary["policy_change_count"] == 2
            and summary["correct_policy_change_count"] == 2
        ),
        "final_coverage_gain_exactly_2": (
            summary["final_coverage_gain_over_immediate"] == 2
        ),
        "cost_at_most_0_50": (
            usage["adapter_cost_usd"] <= MECHANICS_MAX_COST_USD
        ),
    }
    gates["all_pass"] = all(gates.values())
    payload["status"] = "passed" if gates["all_pass"] else "gate_failed"
    payload["gates"] = gates
    payload["protocol"].update(
        {
            "expected_requests": EXPECTED_MECHANICS_REQUESTS,
            "max_transport_retries": MAX_TRANSPORT_RETRIES,
            "semantic_repairs_or_reissues": 0,
            "max_cost_usd": MECHANICS_MAX_COST_USD,
            "structural_confirmation_sha256": (
                STRUCTURAL_CONFIRMATION_SHA256
            ),
        }
    )
    return payload


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "mechanics"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qa-path", type=Path)
    parser.add_argument("--caption-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_concurrency = 8
    config.openrouter_max_retries = MAX_TRANSPORT_RETRIES
    config.openrouter_max_output_tokens = MAX_NEW_TOKENS
    is_smoke = args.mode == "smoke"
    config.openrouter_projected_cost_usd = 0.08 if is_smoke else 0.20
    config.openrouter_run_budget_usd = (
        SMOKE_MAX_COST_USD if is_smoke else MECHANICS_MAX_COST_USD
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model = KeyedFixtureModel() if args.dry_run else _build_model(config)

    if is_smoke:
        payload = smoke.run_smoke(
            config,
            raw_path=raw_path,
            model=model,
            support_message_builder=keyed_support_messages,
            rank_message_builder=smoke_rank_messages,
            support_parser=parse_keyed_support,
            rank_parser=parse_keyed_rank,
            interface_version=INTERFACE_VERSION,
            response_format_name="order_insensitive_keyed_pipe_rows",
            max_transport_retries=MAX_TRANSPORT_RETRIES,
        )
        filename = "SMOKE.json"
    else:
        if args.qa_path is None or args.caption_path is None:
            raise ValueError("mechanics mode requires QA and caption paths")
        payload = run_mechanics(
            config,
            qa_path=args.qa_path,
            caption_path=args.caption_path,
            raw_path=raw_path,
            model=model,
        )
        filename = "MECHANICS.json"
    payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    _write(args.output_dir / filename, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "diagnostics": payload["diagnostics"],
                "summary": payload.get("summary"),
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
