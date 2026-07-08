from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _empty_bucket() -> dict[str, int]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "unknown_completion_token_records": 0,
    }


def _add_record(bucket: dict[str, int], record: dict[str, Any]) -> None:
    bucket["calls"] += 1
    prompt_tokens = record.get("prompt_tokens")
    completion_tokens = record.get("completion_tokens")
    total_tokens = record.get("total_tokens")
    if isinstance(prompt_tokens, int):
        bucket["prompt_tokens"] += prompt_tokens
    if isinstance(completion_tokens, int):
        bucket["completion_tokens"] += completion_tokens
    else:
        bucket["unknown_completion_token_records"] += 1
    if isinstance(total_tokens, int):
        bucket["total_tokens"] += total_tokens
    elif isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        bucket["total_tokens"] += prompt_tokens + completion_tokens


def summarize_llm_token_usage(log_path: Path | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "log_path": None if log_path is None else str(log_path),
        "total": _empty_bucket(),
        "by_call_type": {},
        "by_model": {},
    }
    if log_path is None or not log_path.exists():
        return summary

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if record.get("event") != "llm_token_usage":
                continue

            _add_record(summary["total"], record)
            call_type = str(record.get("call_type", "unknown"))
            call_bucket = summary["by_call_type"].setdefault(call_type, _empty_bucket())
            _add_record(call_bucket, record)
            model = str(record.get("model", "unknown"))
            model_bucket = summary["by_model"].setdefault(model, _empty_bucket())
            _add_record(model_bucket, record)

    return summary


def merge_token_usage_summaries(summaries: list[dict[str, Any] | None]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "log_path": None,
        "total": _empty_bucket(),
        "by_call_type": {},
        "by_model": {},
    }
    for summary in summaries:
        if not summary:
            continue
        _merge_bucket(merged["total"], summary.get("total", {}))
        for call_type, bucket in summary.get("by_call_type", {}).items():
            _merge_bucket(
                merged["by_call_type"].setdefault(str(call_type), _empty_bucket()),
                bucket,
            )
        for model, bucket in summary.get("by_model", {}).items():
            _merge_bucket(
                merged["by_model"].setdefault(str(model), _empty_bucket()),
                bucket,
            )
    return merged


def _merge_bucket(target: dict[str, int], source: dict[str, Any]) -> None:
    for key in target:
        value = source.get(key, 0)
        if isinstance(value, int):
            target[key] += value


def token_usage_report_lines(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["- LLM token usage: no token usage records found."]
    total = summary.get("total", {})
    calls = int(total.get("calls", 0))
    if calls <= 0:
        return ["- LLM token usage: no token usage records found."]

    lines = [
        "## LLM Token Usage",
        "",
        (
            f"- Calls: {calls}; prompt tokens: {int(total.get('prompt_tokens', 0))}; "
            f"completion tokens: {int(total.get('completion_tokens', 0))}; "
            f"total tokens: {int(total.get('total_tokens', 0))}"
        ),
    ]
    unknown = int(total.get("unknown_completion_token_records", 0))
    if unknown:
        lines.append(f"- Records with unknown completion tokens: {unknown}")
    lines.extend(
        [
            "",
            "| call type | calls | prompt tokens | completion tokens | total tokens |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for call_type, bucket in sorted(summary.get("by_call_type", {}).items()):
        lines.append(
            f"| `{call_type}` | {int(bucket.get('calls', 0))} | "
            f"{int(bucket.get('prompt_tokens', 0))} | "
            f"{int(bucket.get('completion_tokens', 0))} | "
            f"{int(bucket.get('total_tokens', 0))} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize BED-LLM token usage records from a run log.")
    parser.add_argument("log_path", type=Path)
    args = parser.parse_args()
    summary = summarize_llm_token_usage(args.log_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
