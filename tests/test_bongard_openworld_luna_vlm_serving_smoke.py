from __future__ import annotations

from datetime import datetime
from io import BytesIO
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

from scripts import bongard_openworld_luna_vlm_serving_smoke as smoke
from scripts import bongard_openworld_vlm_bed as bed


def _image_bytes(color: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), (color, 100, 150)).save(output, format="JPEG")
    return output.getvalue()


def _task(index: int) -> bed.VisualTask:
    image_ids = tuple(f"image-{image_index:02d}" for image_index in range(14))
    labels = {image_id: image_index % 2 == 0 for image_index, image_id in enumerate(image_ids)}
    return bed.VisualTask(
        task_id=f"task-{index:012d}",
        image_ids=image_ids,
        initial_history=tuple(
            sorted((image_id, labels[image_id]) for image_id in image_ids[:4])
        ),
        candidate_ids=image_ids[4:12],
        endpoint_ids=image_ids[12:14],
        image_bytes={image_id: _image_bytes(40 + image_index) for image_index, image_id in enumerate(image_ids)},
        actual_labels=labels,
        hidden_values=(f"secret concept {index}", f"images/{index:04d}/pos__0__source.jpg"),
    )


def _response(case: smoke.SmokeCase) -> str:
    labels = dict(case.history)
    rows = []
    for hypothesis_index, hypothesis_id in enumerate(bed.HYPOTHESIS_IDS):
        probabilities = []
        for image_index, image_id in enumerate(case.task.image_ids):
            value = 20 + ((image_index * 11 + hypothesis_index * 7) % 60)
            if image_id in labels:
                value = 90 - hypothesis_index if labels[image_id] else 10 + hypothesis_index
            probabilities.append(value)
        rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "rule": (
                    f"history {case.case_id} semantic rule {hypothesis_index + 1}"
                ),
                "history_weight": 20 - hypothesis_index,
                "positive_probabilities": probabilities,
            }
        )
    return json.dumps({"hypotheses": rows})


class FixtureAdapter:
    def __init__(self, cases):
        self.cases = cases

    def chat_complete_messages_batched_structured(self, batch_messages, **kwargs):
        del kwargs
        assert len(batch_messages) == smoke.EXPECTED_REQUESTS
        return [_response(case) for case in self.cases]

    def usage_snapshot(self):
        return {
            "adapter_requests": 10,
            "http_attempts": 10,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": 1000,
            "adapter_completion_tokens": 2000,
            "adapter_cost_usd": 0.02,
        }


class RetriedFixtureAdapter(FixtureAdapter):
    def usage_snapshot(self):
        usage = super().usage_snapshot()
        usage.update(
            {
                "http_attempts": 11,
                "retry_count": 1,
                "provider_error_retries": 0,
            }
        )
        return usage


def test_serving_adapter_reserves_luna_attempt_cost(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    adapter = smoke._adapter(output_dir=tmp_path, run_id="precharge-test")
    assert adapter.max_request_cost_usd == smoke.MAX_REQUEST_COST_USD == 0.004


def test_exact_ten_fixture_passes_without_endpoint_access(tmp_path: Path) -> None:
    tasks = [_task(1), _task(2)]
    cases = smoke.build_smoke_cases(tasks)
    result = smoke.run_smoke(
        output_dir=tmp_path / "run",
        run_id="fixture",
        tasks=tasks,
        adapter=FixtureAdapter(cases),
    )
    assert result["status"] == "passed"
    assert result["gates"]["all_pass"]
    assert result["usage"]["adapter_requests"] == 10
    assert result["protocol"]["actual_candidate_labels_accessed"] is False
    assert result["protocol"]["endpoint_labels_accessed"] is False
    raw = json.loads((tmp_path / "run/private/RAW_RESPONSES.json").read_text())
    assert raw["endpoint_labels_accessed"] is False


def test_exact_ten_fixture_allows_one_bounded_transport_retry(
    tmp_path: Path,
) -> None:
    tasks = [_task(1), _task(2)]
    cases = smoke.build_smoke_cases(tasks)
    result = smoke.run_smoke(
        output_dir=tmp_path / "run",
        run_id="retried-fixture",
        tasks=tasks,
        adapter=RetriedFixtureAdapter(cases),
    )
    assert result["status"] == "passed"
    assert result["gates"]["all_pass"]
    assert result["gates"]["total_retries_within_preregistered_bound"]
    assert result["usage"]["retry_count"] == 1


def test_transport_retry_gates_enforce_frozen_bound_and_accounting() -> None:
    base = {
        "adapter_requests": 10,
        "http_attempts": 14,
        "retry_count": 4,
        "provider_error_retries": 2,
    }
    assert all(
        smoke.transport_retry_gates(base, expected_requests=10).values()
    )

    too_many = base | {"http_attempts": 15, "retry_count": 5}
    assert not smoke.transport_retry_gates(
        too_many, expected_requests=10
    )["total_retries_within_preregistered_bound"]

    mismatched = base | {"http_attempts": 13}
    assert not smoke.transport_retry_gates(
        mismatched, expected_requests=10
    )["http_attempts_equal_accepted_plus_retries"]

    excess_provider = base | {"provider_error_retries": 5}
    assert not smoke.transport_retry_gates(
        excess_provider, expected_requests=10
    )["provider_error_retries_are_bounded_subset"]

    noninteger = base | {"retry_count": 1.5, "http_attempts": 11.5}
    assert not smoke.transport_retry_gates(
        noninteger, expected_requests=10
    )["transport_usage_counts_are_nonnegative_integers"]


def test_transport_retry_allowance_scales_at_two_percent() -> None:
    usage = {
        "adapter_requests": 344,
        "http_attempts": 351,
        "retry_count": 7,
        "provider_error_retries": 0,
    }
    assert all(
        smoke.transport_retry_gates(usage, expected_requests=344).values()
    )
    usage["http_attempts"] = 352
    usage["retry_count"] = 8
    assert not smoke.transport_retry_gates(
        usage, expected_requests=344
    )["total_retries_within_preregistered_bound"]


def test_branch_sensitivity_ignores_the_just_labelled_image() -> None:
    task = _task(1)
    candidate = task.candidate_ids[0]
    beliefs = []
    for label in (False, True):
        case = smoke.SmokeCase(
            case_id="shared-case",
            task=task,
            history=tuple(sorted((*task.initial_history, (candidate, label)))),
            kind="branch",
            branch_candidate_id=candidate,
            branch_label=label,
        )
        value = json.loads(_response(case))
        for index, row in enumerate(value["hypotheses"]):
            row["rule"] = f"shared semantic rule {index + 1}"
        beliefs.append(
            bed.parse_belief_response(
                json.dumps(value),
                image_ids=task.image_ids,
                history=case.history,
            )
        )
    sensitivity = smoke.branch_sensitivity(
        beliefs[0], beliefs[1], candidate_id=candidate
    )
    assert sensitivity["unobserved_prediction_mae"] == 0.0
    assert sensitivity["rule_jaccard"] == 1.0
    assert sensitivity["material"] is False


def test_branch_obedience_rejects_support_change_that_ignores_negative_label() -> None:
    task = _task(1)
    candidate = task.candidate_ids[0]
    beliefs = []
    rows = []
    for label in (False, True):
        case = smoke.SmokeCase(
            case_id="obedience-adversary",
            task=task,
            history=tuple(sorted((*task.initial_history, (candidate, label)))),
            kind="branch",
            branch_candidate_id=candidate,
            branch_label=label,
        )
        value = json.loads(_response(case))
        candidate_index = task.image_ids.index(candidate)
        for index, row in enumerate(value["hypotheses"]):
            row["rule"] = (
                f"{'positive' if label else 'negative'} branch rule {index + 1}"
            )
            row["positive_probabilities"][candidate_index] = 90
        belief = bed.parse_belief_response(
            json.dumps(value), image_ids=task.image_ids, history=case.history
        )
        beliefs.append(belief)
        rows.append((candidate, label, belief))

    assert smoke.branch_sensitivity(
        beliefs[0], beliefs[1], candidate_id=candidate
    )["material"]
    obedience = smoke.branch_label_obedience(rows)
    assert obedience["positive_mean_brier"] < 0.25
    assert obedience["negative_mean_brier"] > 0.25
    assert smoke.branch_label_obedience_passes(obedience) is False


def test_terminal_obedience_requires_both_new_query_labels() -> None:
    task = _task(1)
    histories = [
        tuple(
            sorted(
                (
                    *task.initial_history,
                    (task.candidate_ids[0], True),
                    (task.candidate_ids[1], False),
                )
            )
        ),
        tuple(
            sorted(
                (
                    *task.initial_history,
                    (task.candidate_ids[2], False),
                    (task.candidate_ids[3], True),
                )
            )
        ),
    ]
    beliefs = []
    for index, history in enumerate(histories):
        case = smoke.SmokeCase(
            case_id=f"terminal-{index}",
            task=task,
            history=history,
            kind="final",
        )
        beliefs.append(
            bed.parse_belief_response(
                _response(case), image_ids=task.image_ids, history=history
            )
        )
    obedience = smoke.terminal_label_obedience(
        [(task.initial_history, belief) for belief in beliefs]
    )
    assert obedience["queried_label_count"] == 4
    assert obedience["negative_label_count"] == 2
    assert obedience["positive_label_count"] == 2
    assert smoke.terminal_label_obedience_passes(obedience)

    value = json.loads(_response(
        smoke.SmokeCase(
            case_id="terminal-adversary",
            task=task,
            history=histories[0],
            kind="final",
        )
    ))
    negative_id = task.candidate_ids[1]
    negative_index = task.image_ids.index(negative_id)
    for row in value["hypotheses"]:
        row["positive_probabilities"][negative_index] = 90
    adversary = bed.parse_belief_response(
        json.dumps(value), image_ids=task.image_ids, history=histories[0]
    )
    failed = smoke.terminal_label_obedience(
        [(task.initial_history, adversary), (task.initial_history, beliefs[1])]
    )
    assert failed["negative_mean_brier"] > 0.25
    assert smoke.terminal_label_obedience_passes(failed) is False


def test_luna_payload_removes_unsupported_sampling_parameters(monkeypatch) -> None:
    def base_payload(*args, **kwargs):
        del args, kwargs
        return {
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": 50,
            "n": 1,
            "response_format": bed.belief_response_format(),
        }

    monkeypatch.setattr(smoke.SeededStructuredAdapter, "_payload", base_payload)
    adapter = object.__new__(smoke.LunaVisionAdapter)
    payload = adapter._payload(
        [],
        0.0,
        1,
        response_format=bed.belief_response_format(),
    )
    assert "temperature" not in payload
    assert payload["reasoning"]["enabled"] is False
    assert payload["provider"]["require_parameters"] is False


def test_luna_payload_overrides_static_seed_with_bound_pair_seed(monkeypatch) -> None:
    def base_payload(*args, **kwargs):
        del args, kwargs
        return {"seed": 7}

    monkeypatch.setattr(smoke.SeededStructuredAdapter, "_payload", base_payload)
    adapter = object.__new__(smoke.LunaVisionAdapter)
    import threading

    adapter._per_request_seed = threading.local()
    adapter._per_request_seed.value = 12345
    payload = adapter._payload([], 0.0, 1)
    assert payload["seed"] == 12345
    assert payload["reasoning"] == {"enabled": False, "exclude": True}


def test_seeded_dispatch_uses_fixed_ordered_concurrency_batches(monkeypatch) -> None:
    batches = []

    class RecordingExecutor:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def map(self, function, items):
            batch = list(items)
            batches.append([seed for _, seed in batch])
            return [function(item) for item in batch]

    monkeypatch.setattr(smoke, "ThreadPoolExecutor", RecordingExecutor)
    adapter = object.__new__(smoke.LunaVisionAdapter)
    import threading

    adapter._per_request_seed = threading.local()
    adapter.concurrency = 2
    adapter._complete_request = lambda *args, **kwargs: [
        str(adapter._per_request_seed.value)
    ]
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        [[{"role": "user", "content": []}]] * 5,
        [101, 102, 103, 104, 105],
        temperature=0.0,
        response_format={"type": "json_schema"},
    )
    assert batches == [[101, 102], [103, 104], [105]]
    assert responses == ["101", "102", "103", "104", "105"]


def test_ledger_refuses_before_august_tenth() -> None:
    ledger = {
        "date": "2026-08-09",
        "timezone": smoke.TIMEZONE,
        "daily_cap_usd": 5.0,
    }
    try:
        smoke._validate_ledger(
            ledger,
            datetime(2026, 8, 9, 12, tzinfo=ZoneInfo(smoke.TIMEZONE)),
        )
    except RuntimeError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("pre-August-10 spend should fail")


def test_reconcile_ledger_uses_max_of_posted_and_local() -> None:
    ledger = {
        "date": "2026-08-10",
        "timezone": smoke.TIMEZONE,
        "daily_cap_usd": 5.0,
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": 0.4,
    }
    result = smoke.reconcile_ledger(
        ledger=ledger,
        measured_cost_usd=0.2,
        live_after={
            "total_credits_usd": 130.0,
            "total_usage_usd": 100.5,
            "balance_usd": 29.5,
        },
        status="passed",
    )
    assert math.isclose(result["recorded_actual_spend_usd"], 0.6)
    assert math.isclose(
        result["reconciliation"]["remaining_daily_allowance_usd"], 4.4
    )


def test_initialize_daily_ledger_binds_live_account_usage(tmp_path: Path) -> None:
    now = datetime(2026, 8, 10, 9, tzinfo=ZoneInfo(smoke.TIMEZONE))
    path = tmp_path / "2026-08-10.json"
    ledger = smoke.initialize_daily_ledger(
        path=path,
        live={
            "total_credits_usd": 140.0,
            "total_usage_usd": 110.0,
            "balance_usd": 30.0,
        },
        now=now,
    )
    assert ledger["daily_cap_usd"] == 5.0
    assert ledger["opening_total_usage_usd"] == 110.0
    assert ledger["account_wide_usage_counts_against_cap"] is True
    assert ledger["first_authorized_block"]["expected_requests"] == 10
    assert json.loads(path.read_text()) == ledger


def test_execute_smoke_initializes_and_reconciles_without_double_counting(
    tmp_path: Path,
) -> None:
    live_values = iter(
        [
            {
                "total_credits_usd": 140.0,
                "total_usage_usd": 110.0,
                "balance_usd": 30.0,
            },
            {
                "total_credits_usd": 140.0,
                "total_usage_usd": 110.015,
                "balance_usd": 29.985,
            },
        ]
    )

    def runner(**kwargs):
        del kwargs
        return {"status": "passed", "usage": {"run_cost_usd": 0.02}}

    ledger_path = tmp_path / "ledger.json"
    result = smoke.execute_smoke(
        output_dir=tmp_path / "run",
        run_id="fixture",
        ledger_path=ledger_path,
        now=datetime(2026, 8, 10, 9, tzinfo=ZoneInfo(smoke.TIMEZONE)),
        live_reader=lambda: next(live_values),
        smoke_runner=runner,
    )
    ledger = json.loads(ledger_path.read_text())
    assert result["status"] == "passed"
    assert math.isclose(ledger["recorded_actual_spend_usd"], 0.02)
    assert math.isclose(
        ledger["bongard_luna_vlm_serving_smoke"]["actual_cost_usd"],
        0.02,
    )
    assert math.isclose(
        ledger["first_authorized_block"]["actual_cost_usd"], 0.02
    )
