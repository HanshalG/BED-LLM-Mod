#!/usr/bin/env python3
"""Run fixed-line contrastive LongVid belief mechanics on new tasks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Protocol, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
import scripts.longvid_contrastive_path_belief_mechanics as base


INTERFACE_VERSION = "longvid-contrastive-line-mechanics-1"
LAYOUT_SEED = 270746
TASK_LAYOUT = (
    (2989, (0, 17)),
    (549, (2, 6)),
    (2642, (4, 0)),
    (2345, (11, 1)),
)
TASK_LAYOUT_HASH = (
    "77bd6d1c93c84e2968502541f6a087dfd4257b8ee0ee710845504ca36c1c5909"
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
)
STRUCTURAL_CONFIRMATION_SHA256 = (
    "895e448c047ca7afa924393da0a4f637a21735c8a770336fa17db0a72fdd4081"
)
REAL_TASK_LINE_EVIDENCE_SHA256 = (
    "2b5f009008b48be2f1413933bd7a5244dc4a422463f72e8bbe0f2895dfbc6161"
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


def line_support_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    result = [dict(message) for message in messages]
    result[0]["content"] = re.sub(
        r"Return only the schema-conforming object\.?",
        "",
        result[0]["content"],
    ).strip()
    result[0]["content"] += (
        "\nReturn exactly six lines and no other text. Each line is "
        "Hn|weight|anchor|query|hypothesis. Labels must be H1 through H6 in "
        "order. weight is an integer 1..100 and at least two weights differ. "
        "Fields must be nonempty and cannot contain the pipe character."
    )
    return result


def line_rank_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    result = [dict(message) for message in messages]
    result[0]["content"] = re.sub(
        r"Return only the schema-conforming object\.?",
        "",
        result[0]["content"],
    ).strip()
    result[0]["content"] += (
        "\nReturn exactly one line and no other text: "
        "choice|confidence|unresolved_need. choice is A or B; confidence is an "
        "integer 51..100; unresolved_need is nonempty and contains no pipe."
    )
    return result


def parse_line_support(text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != base.SUPPORT_SIZE:
        raise ValueError("line support must contain exactly six nonempty lines")
    support = []
    for index, line in enumerate(lines, start=1):
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 5:
            raise ValueError(f"H{index} line must contain exactly five fields")
        label, weight_text, anchor, query, hypothesis = fields
        if label != f"H{index}":
            raise ValueError(f"support label {label!r} is out of order")
        if not re.fullmatch(r"\d{1,3}", weight_text):
            raise ValueError(f"H{index} weight is not an integer")
        weight = int(weight_text)
        if not 1 <= weight <= 100:
            raise ValueError(f"H{index} weight is outside 1..100")
        if not anchor or not query or not hypothesis:
            raise ValueError(f"H{index} contains an empty field")
        support.append(
            {
                "hypothesis": hypothesis,
                "weight": weight,
                "anchor": anchor,
                "query": query,
            }
        )
    if len({base._normalize(row["hypothesis"]) for row in support}) != 6:
        raise ValueError("line support hypotheses must be distinct")
    if len({base._normalize(row["query"]) for row in support}) != 6:
        raise ValueError("line support queries must be distinct")
    if len({row["weight"] for row in support}) < 2:
        raise ValueError("line support must express nonuniform confidence")
    return support


def parse_line_rank(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("line rank must contain exactly one nonempty line")
    fields = [field.strip() for field in lines[0].split("|")]
    if len(fields) != 3:
        raise ValueError("line rank must contain exactly three fields")
    choice, confidence_text, unresolved_need = fields
    if choice not in {"A", "B"}:
        raise ValueError("line rank choice must be A or B")
    if not re.fullmatch(r"\d{2,3}", confidence_text):
        raise ValueError("line rank confidence is not an integer")
    confidence = int(confidence_text)
    if not 51 <= confidence <= 100:
        raise ValueError("line rank confidence is outside 51..100")
    if not unresolved_need:
        raise ValueError("line rank unresolved need is empty")
    return {
        "choice": choice,
        "confidence": confidence,
        "unresolved_need": unresolved_need,
    }


class LineCodecBridge:
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
            messages = [line_support_messages(value) for value in batch_messages]
        elif name == "longvid_contrastive_path_rank":
            messages = [line_rank_messages(value) for value in batch_messages]
        else:
            raise ValueError(f"unsupported line codec {name}")
        return self.delegate.chat_complete_messages_batched(
            messages,
            temperature=temperature,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
        )

    def usage_snapshot(self) -> dict[str, Any]:
        return self.delegate.usage_snapshot()


class LineFixtureModel:
    GREEDY_ROOT = {2989: 0, 549: 2, 2642: 0, 2345: 1}
    ORACLE_ROOT = {2989: 17, 549: 6, 2642: 4, 2345: 11}

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
            if "TRAJECTORY_A=" in user:
                row = int(re.search(r"ROW_ID=(\d+)", user).group(1))
                final = "STAGE=FINAL_FOUR_STEP" in user
                roots = [
                    int(value)
                    for value in re.findall(r'"root_index":(\d+)', user)
                ]
                target = (
                    self.ORACLE_ROOT[row] if final else self.GREEDY_ROOT[row]
                )
                choice = "A" if roots[0] == target else "B"
                responses.append(
                    f"{choice}|80|fixture missing evidence bridge"
                )
                continue
            question_match = re.search(
                r"QUESTION:\n(.+?)(?:\nPREVIOUS_QUERY:|\n\nGenerate)",
                user,
                flags=re.DOTALL,
            )
            query_match = re.search(
                r"PREVIOUS_QUERY:\n(.+?)\nOBSERVATION:",
                user,
                flags=re.DOTALL,
            )
            observation_match = re.search(
                r"OBSERVATION:\n(.+?)\nPREVIOUS_SUPPORT:",
                user,
                flags=re.DOTALL,
            )
            if question_match and query_match and observation_match:
                excluded = set(base.tokenize(question_match.group(1))).union(
                    base.tokenize(query_match.group(1))
                )
                anchors = [
                    token
                    for token in base.tokenize(observation_match.group(1))
                    if token not in excluded
                ]
                anchor = anchors[0] if anchors else "caption"
            else:
                anchor = "QUESTION"
            suffix = self.requests + offset
            responses.append(
                "\n".join(
                    (
                        f"H{index}|{10 + index + (suffix % 3)}|{anchor}|"
                        f"{anchor} fixture query {suffix} {index}|"
                        f"Fixture chain {suffix} hypothesis {index}"
                    )
                    for index in range(1, base.SUPPORT_SIZE + 1)
                )
            )
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


def verify_bindings() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "structural_confirmation": (
            root
            / "results/nonmyopic/longvid_four_hop_tradeoff_confirmation_v2/"
            "CONFIRMATION.json"
        ),
        "real_task_line_evidence": (
            root
            / "results/nonmyopic/longvid_four_hop_ranking_mechanics/"
            "longvid-four-hop-ranking-mechanics-20260727T150132Z/"
            "MECHANICS_FAILURE.json"
        ),
    }
    expected = {
        "structural_confirmation": STRUCTURAL_CONFIRMATION_SHA256,
        "real_task_line_evidence": REAL_TASK_LINE_EVIDENCE_SHA256,
    }
    for name, path in paths.items():
        if base.sha256_file(path) != expected[name]:
            raise ValueError(f"{name} artifact hash does not match")


def run_line_mechanics(
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
        model=LineCodecBridge(model),
        task_layout=TASK_LAYOUT,
        task_layout_hash=TASK_LAYOUT_HASH,
        excluded_prior_rows=EXCLUDED_PRIOR_ROWS,
        response_format_name="fixed_six_line_support_and_rank",
        interface_version=INTERFACE_VERSION,
        layout_seed=LAYOUT_SEED,
        support_parser=parse_line_support,
        rank_parser=parse_line_rank,
    )
    payload["protocol"][
        "structural_confirmation_sha256"
    ] = STRUCTURAL_CONFIRMATION_SHA256
    payload["protocol"][
        "real_task_line_evidence_sha256"
    ] = REAL_TASK_LINE_EVIDENCE_SHA256
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
        raise ValueError("LongVid line mechanics config selects the wrong model")
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
    config.openrouter_max_output_tokens = 900
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: OrdinaryChatModel | None = None
    try:
        model = LineFixtureModel() if args.dry_run else _build_model(config)
        payload = run_line_mechanics(
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
            failure["usage"] = base._usage(LineCodecBridge(model))
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
