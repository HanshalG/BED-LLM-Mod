from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image
import pytest

from scripts import bongard_openworld_luna_claim_report as claim_report
from scripts import bongard_openworld_luna_development32_daily_execute as daily
from scripts import bongard_openworld_luna_vlm_development as development
from scripts import bongard_openworld_luna_vlm_mechanics_tree as mechanics
from scripts import bongard_openworld_vlm_bed as bed


_DEVELOPMENT_TASK_IDS = tuple(
    sorted(
        development.source_audit._task_layout(row)["task_id"]
        for row in development.source_audit.split_validation_rows(
            development.source_audit.load_rows("val")
        )[1]
    )
)


def _image_bytes(index: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (36, 28), (50 + index, 100, 150)).save(
        output, format="JPEG"
    )
    return output.getvalue()


def _task(index: int) -> bed.VisualTask:
    image_ids = tuple(f"image-{image_index:02d}" for image_index in range(14))
    labels = {
        image_id: image_index % 2 == 0
        for image_index, image_id in enumerate(image_ids)
    }
    return bed.VisualTask(
        task_id=_DEVELOPMENT_TASK_IDS[index],
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


def _request(messages):
    return json.loads(
        next(
            item["text"]
            for item in messages[0]["content"]
            if item.get("type") == "text"
        )
    )


def _response(messages, *, seed: int | None = None) -> str:
    request = _request(messages)
    image_ids = request["image_order"]
    observed = {
        item["image_id"]: item["label"] == "positive"
        for item in request["observed_labels"]
    }
    initial_ids = set(image_ids[:4])
    extras = sorted(set(observed) - initial_ids)
    first_extra = extras[0] if extras else None
    rows = []
    for hypothesis_index, hypothesis_id in enumerate(bed.HYPOTHESIS_IDS):
        probabilities = []
        centered = (hypothesis_index - 4.5) / 4.5
        for image_index, image_id in enumerate(image_ids):
            if image_id in observed:
                value = (
                    90 - hypothesis_index
                    if observed[image_id]
                    else 10 + hypothesis_index
                )
            else:
                seed_scale = 0 if seed is None else seed % 7
                amplitude = 8 + seed_scale * 3 + (image_index % 4) * 2
                if first_extra == "image-05":
                    amplitude = 35 if image_id != first_extra else 8
                elif first_extra is not None:
                    amplitude = 8
                base = 65 if image_index % 2 == 0 else 35
                value = min(95, max(5, round(base + amplitude * centered)))
            probabilities.append(value)
        rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "rule": (
                    f"{request['task_id']} {first_extra or 'root'} "
                    f"semantic visual rule {hypothesis_index + 1}"
                ),
                "history_weight": 20 - hypothesis_index,
                "positive_probabilities": probabilities,
            }
        )
    return json.dumps({"hypotheses": rows})


class FixtureAdapter:
    def __init__(self):
        self.requests = 0

    def chat_complete_messages_batched_structured(self, batch_messages, **kwargs):
        del kwargs
        self.requests += len(batch_messages)
        return [_response(messages) for messages in batch_messages]

    def chat_complete_seeded_messages_batched_structured(
        self, batch_messages, seeds, **kwargs
    ):
        del kwargs
        assert len(batch_messages) == len(seeds)
        self.requests += len(batch_messages)
        return [
            _response(messages, seed=seed)
            for messages, seed in zip(batch_messages, seeds, strict=True)
        ]

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": self.requests * 100,
            "adapter_completion_tokens": self.requests * 200,
            "adapter_cost_usd": self.requests * 0.001,
        }


def _mechanics_result(tmp_path: Path) -> Path:
    output = tmp_path / "mechanics"
    raw_path = output / "private/RAW_RESPONSES.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("{}", encoding="utf-8")
    result = {
        "status": "mechanics_pass",
        "protocol": {
            "interface_version": mechanics.INTERFACE_VERSION,
            "model": development.MODEL_ID,
            "development_accessed": False,
            "confirmation_accessed": False,
            "sealed_test_accessed": False,
            "expected_total_requests": 82,
        },
        "usage": {"run_cost_usd": 0.20},
        "gates": {"all_pass": True},
        "raw_responses_sha256": development.sha256_file(raw_path),
    }
    path = output / "RESULT.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def _protocol_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "protocol/MANIFEST.json"
    development.build_protocol_manifest(output_path=path)
    return path


def test_development_blocks_are_exact_disjoint_and_endpoint_sealed() -> None:
    tasks = [_task(index) for index in range(32)]
    observed = []
    for block_id, expected_size in development.BLOCK_SIZES.items():
        block = development.development_tasks_for_block(tasks, block_id)
        assert len(block) == expected_size
        observed.extend(task.task_id for task in block)
        sealed = [development.seal_endpoint_labels(task) for task in block]
        assert all(
            not (set(task.endpoint_ids) & set(task.actual_labels))
            for task in sealed
        )
    assert len(observed) == len(set(observed)) == 32


def test_block_run_is_endpoint_blind_and_replays(tmp_path: Path) -> None:
    tasks = [_task(index) for index in range(32)]
    mechanics_result = _mechanics_result(tmp_path)
    protocol_manifest = _protocol_manifest(tmp_path)
    result = development.run_block(
        output_dir=tmp_path / "block-a",
        run_id="fixture-a",
        block_id="a",
        mechanics_result=mechanics_result,
        protocol_manifest=protocol_manifest,
        all_development_tasks=tasks,
        adapter=FixtureAdapter(),
    )
    assert result["status"] == "block_mechanics_pass"
    assert result["gates"]["endpoint_labels_remain_sealed"]
    assert result["protocol"]["endpoint_labels_accessed"] is False
    assert result["protocol"]["first_stage_requests"] == 264
    assert result["protocol"]["conditioned_branch_requests"] == 128
    assert result["protocol"]["history_blind_branch_requests"] == 128
    assert result["paired_request_diagnostics"]["gates"][
        "each_pair_is_adjacent_dynamic_then_blind"
    ]
    assert result["paired_request_diagnostics"]["gates"][
        "each_pair_shares_one_dispatch_batch"
    ]
    assert "pooled_policy_metrics" not in result
    assert result["protocol"]["distinct_final_history_requests"] >= 8 * 4
    assert all(len(tree["all_first_action_paths"]) == 8 for tree in result["trees"])
    replay = development.replay_block(
        result_path=tmp_path / "block-a/RESULT.json",
        all_development_tasks=tasks,
    )
    assert replay["verified"]
    assert replay["block_id"] == "a"


def test_combined_analysis_opens_endpoints_only_after_all_blocks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks = [_task(index) for index in range(32)]
    mechanics_result = _mechanics_result(tmp_path)
    protocol_manifest = _protocol_manifest(tmp_path)
    paths = []
    for block_id in development.BLOCK_ORDER:
        output = tmp_path / f"block-{block_id}"
        development.run_block(
            output_dir=output,
            run_id=f"fixture-{block_id}",
            block_id=block_id,
            mechanics_result=mechanics_result,
            protocol_manifest=protocol_manifest,
            all_development_tasks=tasks,
            adapter=FixtureAdapter(),
        )
        paths.append(output / "RESULT.json")
    label_access = []

    def load_tasks(partition, *, include_endpoint_labels=True):
        assert partition == "development"
        label_access.append(include_endpoint_labels)
        if include_endpoint_labels:
            return tasks
        return [development.seal_endpoint_labels(task) for task in tasks]

    monkeypatch.setattr(bed, "load_validation_partition_tasks", load_tasks)
    combined_path = tmp_path / "combined.json"
    result = development.analyze_combined(
        block_results=paths,
        output_path=combined_path,
    )
    assert label_access == [False, True]
    assert result["protocol"][
        "endpoint_labels_accessed_only_after_all_blocks_replayed"
    ]
    assert result["protocol"]["confirmation_accessed"] is False
    assert set(result["pooled_policy_metrics"]) == set(mechanics.POLICIES)
    assert len(result["trees"]) == 32
    assert set(result["dynamic_vs_history_blind"]) == {
        "mean_brier",
        "mean_log_loss",
    }
    assert "dynamic_vs_history_blind_relative_brier_improvement" in result
    assert set(result["ranking_fidelity"]) == set(mechanics.SCORE_POLICIES)
    assert all(
        "dynamic_history_blind_changed_final_histories" in row
        for row in result["blockwise_dynamic_vs_myopic"].values()
    )
    assert all(
        math.isfinite(metrics["mean_brier"])
        for metrics in result["pooled_policy_metrics"].values()
    )
    verification = daily.verify_combined_result(
        result_path=combined_path,
        block_results=paths,
        all_development_tasks=tasks,
    )
    assert verification["verified"]
    assert verification["status"] == result["status"]
    assert verification["authorizes_confirmation_preregistration"] is result[
        "authorizes_confirmation_preregistration"
    ]
    claim = claim_report.build_claim_report(
        result,
        result_sha256=development.sha256_file(combined_path),
        independent_verification=verification,
    )
    assert claim["status"] == "claim_scope_frozen"
    assert claim["authorizes_confirmation_preregistration"] is result[
        "authorizes_confirmation_preregistration"
    ]


def test_combined_analysis_refuses_missing_block(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        development.analyze_combined(
            block_results=[],
            output_path=tmp_path / "combined.json",
            all_development_tasks=[_task(index) for index in range(32)],
        )


def test_spearman_ranking_fidelity_matches_order_and_reversal() -> None:
    assert development.spearman_correlation(
        [1, 2, 3], [10, 20, 30]
    ) == pytest.approx(1.0)
    assert development.spearman_correlation(
        [1, 2, 3], [30, 20, 10]
    ) == pytest.approx(-1.0)


def test_block_date_gate_is_frozen() -> None:
    ledger = {
        "date": "2026-08-11",
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
    }
    development._validate_daily_ledger(
        ledger,
        block_id="a",
        now=datetime(2026, 8, 11, 9, tzinfo=ZoneInfo("Europe/London")),
    )
    with pytest.raises(RuntimeError, match="forbidden before"):
        development._validate_daily_ledger(
            {**ledger, "date": "2026-08-10"},
            block_id="a",
            now=datetime(2026, 8, 10, 9, tzinfo=ZoneInfo("Europe/London")),
        )


def test_protocol_manifest_is_opaque_and_exact(tmp_path: Path) -> None:
    result = development.build_protocol_manifest(
        output_path=tmp_path / "MANIFEST.json"
    )
    assert result["status"] == "frozen"
    assert result["gates"]["all_pass"]
    assert len(result["tasks"]) == 32
    assert {
        block_id: sum(row["block_id"] == block_id for row in result["tasks"])
        for block_id in development.BLOCK_ORDER
    } == development.BLOCK_SIZES
    public_text = json.dumps(result["tasks"])
    assert "concept" not in public_text
    assert "caption" not in public_text
    assert "images/" not in public_text
