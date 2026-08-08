from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image
import pytest

from scripts import bongard_openworld_luna_naive_first_link as naive
from scripts import bongard_openworld_luna_naive_first_link_daily_execute as daily
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_vlm_bed as bed


_DEVELOPMENT_TASK_IDS = tuple(
    task_id
    for block_id in development.BLOCK_ORDER
    for task_id in development.verify_protocol_manifest(
        naive.MAIN_MANIFEST
    )["task_ids_by_block"][block_id]
)


def _image_bytes(index: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (30, 24), (40 + index, 90, 130)).save(
        output, format="JPEG"
    )
    return output.getvalue()


def _task(index: int, *, mechanics: bool = False) -> bed.VisualTask:
    image_ids = tuple(f"image-{image_index:02d}" for image_index in range(14))
    labels = {
        image_id: image_index % 2 == 0
        for image_index, image_id in enumerate(image_ids)
    }
    return bed.VisualTask(
        task_id=(
            f"mechanics-{index:012d}"
            if mechanics
            else _DEVELOPMENT_TASK_IDS[index]
        ),
        image_ids=image_ids,
        initial_history=tuple(
            sorted((image_id, labels[image_id]) for image_id in image_ids[:4])
        ),
        candidate_ids=image_ids[4:12],
        endpoint_ids=image_ids[12:14],
        image_bytes={
            image_id: _image_bytes(image_index)
            for image_index, image_id in enumerate(image_ids)
        },
        actual_labels=labels,
        hidden_values=(
            f"secret concept {index}",
            f"images/{index:04d}/pos__0__source.jpg",
        ),
    )


class FixtureAdapter:
    def __init__(self):
        self.requests = 0

    def chat_complete_seeded_messages_batched_structured(
        self, batch_messages, seeds, **kwargs
    ):
        del kwargs
        assert len(batch_messages) == len(seeds)
        self.requests += len(batch_messages)
        responses = []
        for messages in batch_messages:
            request = json.loads(messages[0]["content"][0]["text"])
            responses.append(
                json.dumps(
                    {"first_image_id": request["display_order"][4]}
                )
            )
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": self.requests * 100,
            "forced_exits": 0,
            "forced_final_requests": 0,
            "adapter_prompt_tokens": self.requests * 500,
            "adapter_completion_tokens": self.requests * 120,
            "adapter_cost_usd": self.requests * 0.001,
        }


def _mechanics_tasks() -> list[bed.VisualTask]:
    return [_task(index, mechanics=True) for index in range(4)]


def _development_tasks() -> list[bed.VisualTask]:
    return [_task(index) for index in range(development.TASKS)]


def _smoke(tmp_path: Path, monkeypatch) -> Path:
    tasks = _mechanics_tasks()
    monkeypatch.setattr(naive.bed, "load_mechanics_tasks", lambda: tasks)
    result = naive.run_smoke(
        output_dir=tmp_path / "smoke",
        run_id="naive-smoke-fixture",
        tasks=tasks,
        adapter=FixtureAdapter(),
    )
    assert result["status"] == "passed"
    return tmp_path / "smoke/RESULT.json"


def test_prompt_exposes_candidates_but_omits_endpoints_and_labels() -> None:
    task = naive.seal_unqueried_labels(_task(0))
    case = naive.make_case(task, seed=1234, suffix="privacy")
    messages = naive.build_messages(case)
    request = json.loads(messages[0]["content"][0]["text"])
    assert set(request["selectable_image_ids"]) == set(task.candidate_ids)
    assert not (set(task.endpoint_ids) & set(request["display_order"]))
    assert naive.prompt_errors(case, messages) == []
    assert sum(
        item.get("type") == "image_url"
        for item in messages[0]["content"]
        if isinstance(item, dict)
    ) == 12
    for candidate in task.candidate_ids:
        assert candidate not in {
            row["image_id"] for row in request["observed_labels"]
        }
    with pytest.raises(ValueError, match="not selectable"):
        naive.parse_choice(
            json.dumps({"first_image_id": task.endpoint_ids[0]}), task
        )


def test_prompt_audit_rejects_materialized_unqueried_labels() -> None:
    task = _task(0)
    case = naive.make_case(task, seed=1234, suffix="unsealed")
    assert "unqueried_labels_materialized" in naive.prompt_errors(
        case, naive.build_messages(case)
    )


def test_reasoning_adapter_payload_enables_medium_and_excludes_trace(
    monkeypatch,
) -> None:
    def base_payload(*args, **kwargs):
        del args, kwargs
        return {
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": 50,
            "n": 1,
            "reasoning": {"enabled": False, "exclude": True},
        }

    monkeypatch.setattr(serving.LunaVisionAdapter, "_payload", base_payload)
    adapter = object.__new__(naive.LunaReasoningAdapter)
    payload = adapter._payload([], 0.0, 1, response_format=naive.response_format())
    assert payload["reasoning"] == {"effort": "medium", "exclude": True}


def test_exact_ten_reasoning_smoke_passes_and_replays(
    tmp_path: Path, monkeypatch
) -> None:
    path = _smoke(tmp_path, monkeypatch)
    result = json.loads(path.read_text())
    assert result["gates"]["positive_reasoning_tokens"]
    assert result["gates"]["candidate_and_endpoint_labels_remain_unaccessed"]
    assert result["usage"]["adapter_requests"] == 10
    assert naive.verify_smoke_result(path)["verified"]


def test_development_choices_are_endpoint_blind_and_replay(
    tmp_path: Path, monkeypatch
) -> None:
    smoke = _smoke(tmp_path, monkeypatch)
    tasks = _development_tasks()
    result = naive.run_block(
        output_dir=tmp_path / "block-a",
        run_id="naive-block-a-fixture",
        block_id="a",
        smoke_result=smoke,
        tasks=tasks,
        adapter=FixtureAdapter(),
    )
    assert result["status"] == "passed"
    assert result["protocol"]["candidate_labels_accessed"] is False
    assert result["protocol"]["endpoint_labels_accessed"] is False
    assert len(result["choices"]) == development.BLOCK_SIZES["a"]
    replay = naive.verify_block_result(
        tmp_path / "block-a/RESULT.json",
        smoke_result=smoke,
        tasks=tasks,
    )
    assert replay["verified"]
    assert len(replay["choices"]) == development.BLOCK_SIZES["a"]


def test_prior_naive_block_requires_replayable_privacy_and_ledger_chain(
    tmp_path: Path, monkeypatch
) -> None:
    smoke = _smoke(tmp_path, monkeypatch)
    tasks = _development_tasks()
    monkeypatch.setattr(
        naive.bed,
        "load_validation_partition_tasks",
        lambda *args, **kwargs: tasks,
    )
    output = tmp_path / "block-a"
    naive.run_block(
        output_dir=output,
        run_id="naive-block-a-fixture",
        block_id="a",
        smoke_result=smoke,
        tasks=tasks,
        adapter=FixtureAdapter(),
    )
    ledger = tmp_path / "supplement-a.json"
    ledger.write_text(
        json.dumps(
            {
                "interface_version": daily.INTERFACE_VERSION,
                "date": development.BLOCK_EARLIEST_DATES["a"],
                "daily_cap_usd": 5.0,
                "account_wide_usage_counts_against_cap": True,
                "recorded_actual_spend_usd": 4.80,
                "naive_first_link": {
                    "status": "passed",
                    "model": naive.MODEL_ID,
                    "reasoning_effort": naive.REASONING_EFFORT,
                    "actual_cost_usd": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )
    execution = {
        "interface_version": daily.INTERFACE_VERSION,
        "status": "complete_reconciled",
        "result_sha256": development.sha256_file(output / "RESULT.json"),
        "ledger_sha256": development.sha256_file(ledger),
    }
    (output / "EXECUTION.json").write_text(
        json.dumps(execution), encoding="utf-8"
    )
    monkeypatch.setattr(daily, "SMOKE_RESULT", smoke)
    monkeypatch.setitem(daily.BLOCK_DIRS, "a", output)
    monkeypatch.setitem(daily.SUPPLEMENTAL_LEDGERS, "a", ledger)
    assert daily._naive_predecessor("a")["block_id"] == "a"

    raw_path = output / "private/RAW_RESPONSES.json"
    raw = json.loads(raw_path.read_text())
    raw["candidate_labels_accessed"] = True
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    result_path = output / "RESULT.json"
    result = json.loads(result_path.read_text())
    result["raw_responses_sha256"] = development.sha256_file(raw_path)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="raw request manifest changed"):
        daily._naive_predecessor("a")


def test_analysis_joins_choices_to_main_all_action_cache(
    tmp_path: Path, monkeypatch
) -> None:
    smoke = _smoke(tmp_path, monkeypatch)
    tasks = _development_tasks()
    block_paths = []
    choices = {}
    for block_id in development.BLOCK_ORDER:
        output = tmp_path / f"block-{block_id}"
        result = naive.run_block(
            output_dir=output,
            run_id=f"naive-block-{block_id}-fixture",
            block_id=block_id,
            smoke_result=smoke,
            tasks=tasks,
            adapter=FixtureAdapter(),
        )
        block_paths.append(output / "RESULT.json")
        choices.update(
            {row["task_id"]: row["first_image_id"] for row in result["choices"]}
        )

    trees = []
    for task in tasks:
        naive_first = choices[task.task_id]
        dynamic_first = next(
            candidate
            for candidate in task.candidate_ids
            if candidate != naive_first
        )
        action_rows = {}
        for candidate in task.candidate_ids:
            brier = 0.20 if candidate == naive_first else 0.10
            action_rows[candidate] = {
                "first_image_id": candidate,
                "second_image_id": next(
                    other for other in task.candidate_ids if other != candidate
                ),
                "final_history_key": f"history-{task.task_id}-{candidate}",
                "endpoint": {
                    "mean_brier": brier,
                    "mean_log_loss": brier + 0.1,
                    "accuracy": 1.0 - brier,
                    "mean_truth_probability": 1.0 - brier,
                    "rows": [],
                },
            }
        trees.append(
            {
                "task_id": task.task_id,
                "all_first_action_paths": action_rows,
                "policies": {
                    "dynamic_depth2": {
                        "first_image_id": dynamic_first,
                        "final_history_key": action_rows[dynamic_first][
                            "final_history_key"
                        ],
                        "endpoint": action_rows[dynamic_first]["endpoint"],
                    },
                    "myopic_width": {
                        "first_image_id": naive_first,
                        "final_history_key": action_rows[naive_first][
                            "final_history_key"
                        ],
                        "endpoint": action_rows[naive_first]["endpoint"],
                    },
                },
            }
        )
    main = {
        "status": "development_signal",
        "protocol": {
            "interface_version": development.INTERFACE_VERSION,
            "task_count": development.TASKS,
            "endpoint_labels_accessed_only_after_all_blocks_replayed": True,
        },
        "trees": trees,
    }
    main_path = tmp_path / "COMBINED_RESULT.json"
    main_path.write_text(json.dumps(main), encoding="utf-8")
    result = naive.analyze(
        block_results=block_paths,
        smoke_result=smoke,
        main_combined_result=main_path,
        output_path=tmp_path / "NAIVE_RESULT.json",
        tasks=tasks,
        main_verification={
            "verified": True,
            "status": "development_signal",
            "result_sha256": development.sha256_file(main_path),
        },
    )
    assert result["status"] == "dynamic_beats_naive_thinking"
    assert result["gates"]["all_pass"]
    assert result["dynamic_minus_naive"]["mean_brier"]["mean_difference"] < 0
    assert result["dynamic_vs_naive_changed_final_histories"] == development.TASKS
    assert result["authorizes_main_confirmation"] is False


def _catalog() -> dict:
    return {
        "data": [
            {
                "id": naive.MODEL_ID,
                "architecture": {"input_modalities": ["text", "image"]},
                "supported_parameters": [
                    "reasoning",
                    "response_format",
                    "structured_outputs",
                ],
                "top_provider": {"max_completion_tokens": 128_000},
                "pricing": {"prompt": "0.0000001", "completion": "0.0000006"},
            }
        ]
    }


def test_daily_preflight_reserves_full_reasoning_prompt_and_output(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 8, 12, tzinfo=ZoneInfo("Europe/London"))
    live = {
        "total_credits_usd": 275.0,
        "total_usage_usd": 220.0,
        "balance_usd": 55.0,
    }
    result = daily.preflight_smoke(
        now=now,
        output_dir=tmp_path / "smoke",
        supplemental_ledger=tmp_path / "ledger.json",
        live_reader=lambda: live,
        catalog_reader=_catalog,
    )
    assert result["status"] == "ready_without_paid_calls"
    assert result["model"]["reasoning_supported"]
    assert (
        result["model"]["covered_prompt_tokens_at_live_price"]
        >= daily.MIN_RESERVED_PROMPT_TOKENS
    )
    assert result["budget"]["naive_run_cap_usd"] == 0.20
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0


def test_supplemental_execution_preserves_main_ledger_hash(
    tmp_path: Path,
) -> None:
    main_ledger = tmp_path / "main-ledger.json"
    main_ledger.write_text(
        json.dumps(
            {
                "opening_total_credits_usd": 275.0,
                "opening_total_usage_usd": 220.0,
                "opening_balance_usd": 55.0,
                "recorded_actual_spend_usd": 4.70,
            }
        ),
        encoding="utf-8",
    )
    before = development.sha256_file(main_ledger)
    output = tmp_path / "baseline"
    supplement = tmp_path / "supplement.json"
    preflight = {
        "date": "2026-08-11",
        "live_credits": {
            "total_credits_usd": 275.0,
            "total_usage_usd": 224.70,
            "balance_usd": 50.30,
        },
        "budget": {"spent_before_naive_usd": 4.70},
        "main_predecessor": {"ledger_path": str(main_ledger)},
    }

    def runner(*, output_dir, run_id):
        del run_id
        output_dir.mkdir(parents=True)
        result = {
            "status": "passed",
            "usage": {"run_cost_usd": 0.05},
        }
        (output_dir / "RESULT.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

    live_after = {
        "total_credits_usd": 275.0,
        "total_usage_usd": 224.75,
        "balance_usd": 50.25,
    }
    daily._execute(
        preflight=preflight,
        output_dir=output,
        ledger_path=supplement,
        run_id="fixture",
        runner=runner,
        runner_kwargs={},
        live_reader=lambda: live_after,
    )
    assert development.sha256_file(main_ledger) == before
    ledger = json.loads(supplement.read_text())
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(4.75)
    assert ledger["reconciliation"][
        "remaining_daily_allowance_usd"
    ] == pytest.approx(0.25)
    assert (output / "EXECUTION.json").is_file()
