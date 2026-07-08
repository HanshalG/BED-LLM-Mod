import json

from scripts.llm_token_usage import (
    merge_token_usage_summaries,
    summarize_llm_token_usage,
    token_usage_report_lines,
)


def test_summarize_llm_token_usage_ignores_non_json_log_lines(tmp_path):
    log_path = tmp_path / "run.log"
    records = [
        "plain progress line",
        json.dumps(
            {
                "event": "llm_token_usage",
                "model": "test/model",
                "call_type": "chat",
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            }
        ),
        json.dumps({"event": "other", "prompt_tokens": 100}),
        json.dumps(
            {
                "event": "llm_token_usage",
                "model": "test/model",
                "call_type": "forced_final",
                "prompt_tokens": 5,
                "completion_tokens": None,
                "total_tokens": None,
            }
        ),
    ]
    log_path.write_text("\n".join(records) + "\n", encoding="utf-8")

    summary = summarize_llm_token_usage(log_path)

    assert summary["total"] == {
        "calls": 2,
        "prompt_tokens": 8,
        "completion_tokens": 4,
        "total_tokens": 7,
        "unknown_completion_token_records": 1,
    }
    assert summary["by_call_type"]["chat"]["total_tokens"] == 7
    assert summary["by_call_type"]["forced_final"]["unknown_completion_token_records"] == 1
    assert summary["by_model"]["test/model"]["calls"] == 2


def test_merge_token_usage_summaries_and_report_lines():
    first = {
        "total": {
            "calls": 1,
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
            "unknown_completion_token_records": 0,
        },
        "by_call_type": {
            "chat": {
                "calls": 1,
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
                "unknown_completion_token_records": 0,
            }
        },
        "by_model": {},
    }
    second = {
        "total": {
            "calls": 2,
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "unknown_completion_token_records": 0,
        },
        "by_call_type": {
            "batched_chat": {
                "calls": 2,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "unknown_completion_token_records": 0,
            }
        },
        "by_model": {},
    }

    merged = merge_token_usage_summaries([first, second])
    lines = token_usage_report_lines(merged)

    assert merged["total"]["calls"] == 3
    assert merged["total"]["total_tokens"] == 37
    assert "- Calls: 3; prompt tokens: 13; completion tokens: 24; total tokens: 37" in lines
    assert "| `chat` | 1 | 3 | 4 | 7 |" in lines
    assert "| `batched_chat` | 2 | 10 | 20 | 30 |" in lines
