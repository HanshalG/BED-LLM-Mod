from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "hover_support_regeneration_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "hover_support_regeneration_smoke",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _hypothesis_lines() -> list[str]:
    return [
        f"H{index:02d}|{index}|chain hypothesis {index}"
        for index in range(1, 9)
    ]


def test_parse_initial_accepts_exact_flat_grammar():
    text = "\n".join(
        _hypothesis_lines()
        + [f"R{index:02d}|{index * 10}" for index in range(1, 4)]
    )

    parsed = MODULE.parse_initial(text, root_count=3)

    assert len(parsed["hypotheses"]) == 8
    assert parsed["immediate_scores"] == [10, 20, 30]


def test_parse_initial_rejects_extra_text():
    text = "\n".join(
        ["Here you go:"]
        + _hypothesis_lines()
        + ["R01|10"]
    )

    with pytest.raises(ValueError, match="wrong line count"):
        MODULE.parse_initial(text, root_count=1)


def test_parse_refresh_requires_distinct_known_nonroot_proposals():
    valid = "\n".join(
        _hypothesis_lines()
        + ["N01|C002", "N02|C003", "N03|C004"]
    )
    parsed = MODULE.parse_refresh(
        valid,
        candidate_count=5,
        root_candidate_index=0,
    )
    assert parsed["proposal_indexes"] == [1, 2, 3]

    duplicate = valid.replace("N03|C004", "N03|C003")
    with pytest.raises(ValueError, match="distinct"):
        MODULE.parse_refresh(
            duplicate,
            candidate_count=5,
            root_candidate_index=0,
        )


def test_parse_future_scores_accepts_only_canonical_lines():
    assert MODULE.parse_future_scores(
        "R01|0\nR02|100",
        root_count=2,
    ) == [0, 100]
    with pytest.raises(ValueError, match="canonical"):
        MODULE.parse_future_scores(
            "R01|00\nR02|100",
            root_count=2,
        )


def test_shuffled_bundles_is_a_derangement():
    rows = [{"id": index} for index in range(4)]
    shuffled = MODULE._shuffled_bundles(rows)

    assert [row["id"] for row in shuffled] == [1, 2, 3, 0]
    assert all(row is not rows[index] for index, row in enumerate(shuffled))


def test_config_is_nonreasoning_fail_closed_and_budgeted():
    from helpers import load_config

    config = load_config(
        "configs/config_hover_support_regeneration_smoke_openrouter.yaml"
    )

    assert config.model_pairs[0].questioner.model == "openai/gpt-5.4"
    assert config.openrouter_max_retries == 0
    assert config.openrouter_concurrency == 24
    assert config.openrouter_run_budget_usd == pytest.approx(0.50)
    assert config.openrouter_budget_usd == pytest.approx(105.0)


class _FakeAdapter:
    def __init__(self):
        self.requests = 0

    def chat_complete_messages_batched(self, messages, **kwargs):
        del kwargs
        outputs = []
        for message_list in messages:
            self.requests += 1
            payload = json.loads(message_list[-1]["content"])
            grammar = payload["exact_output_lines"]
            lines = []
            for line in grammar:
                label = line.split("|", 1)[0]
                if label.startswith("H"):
                    lines.append(
                        f"{label}|10|distinct hypothesis {label} "
                        f"request {self.requests}"
                    )
                elif label.startswith("N"):
                    proposal = int(label[1:])
                    lines.append(f"{label}|C{proposal + 20:03d}")
                else:
                    lines.append(f"{label}|{int(label[1:]) * 7}")
            outputs.append("\n".join(lines))
        return outputs

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.01,
            "http_attempts": self.requests,
            "retry_count": 0,
            "forced_exits": 0,
        }


def test_serving_stage_has_exact_ten_call_shape(monkeypatch, tmp_path):
    task = {
        "task_id": MODULE.TASK_IDS[0],
        "claim": "A claim.",
        "candidate_titles": [
            f"Candidate {index}" for index in range(100)
        ],
    }
    monkeypatch.setattr(MODULE, "load_fixture", lambda path: [task])
    monkeypatch.setattr(
        MODULE,
        "load_root_texts",
        lambda database_path, tasks, root_count: [
            [f"Root text {index}" for index in range(root_count)]
        ],
    )
    config = type(
        "Config",
        (),
        {
            "openrouter_max_output_tokens": 2048,
            "openrouter_concurrency": 24,
        },
    )()
    adapter = _FakeAdapter()

    parsed, usage = MODULE.run_model_stage(
        config,
        stage="serving",
        fixture_path=tmp_path / "fixture.json",
        database_path=tmp_path / "wiki.db",
        raw_path=tmp_path / "raw.json",
        model_adapter=adapter,
    )

    assert usage["physical_requests"] == 10
    assert len(parsed["initials"]) == 1
    assert len(parsed["refreshes"][0]) == 8
    assert len(parsed["future_scores"][0]["aligned"]) == 8
    assert (tmp_path / "raw.json").exists()
