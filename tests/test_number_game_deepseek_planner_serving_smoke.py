import json

from scripts import number_game_deepseek_planner_serving_smoke as smoke


def _usage() -> dict:
    return {
        "adapter_requests": 10,
        "http_attempts": 10,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.05,
    }


def test_serving_cases_are_exact_ten_and_cover_each_history_depth() -> None:
    cases = smoke.serving_cases()

    assert len(cases) == 10
    assert [len(case["observations"]) for case in cases] == [
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    ]


def test_serving_gates_accept_clean_exact_ten_fixture() -> None:
    cases = smoke.serving_cases()
    diagnostics = [
        {"valid_unique_count": 20 if not case["observations"] else 12}
        for case in cases
    ]

    gates = smoke.serving_gates(
        diagnostics=diagnostics,
        cases=cases,
        usage=_usage(),
    )

    assert gates["all_pass"]


def test_serving_gates_fail_on_retry_or_thin_conditioned_support() -> None:
    cases = smoke.serving_cases()
    diagnostics = [{"valid_unique_count": 20} for _ in cases]
    diagnostics[-1]["valid_unique_count"] = 7
    usage = _usage()
    usage["retry_count"] = 1
    usage["http_attempts"] = 11

    gates = smoke.serving_gates(
        diagnostics=diagnostics,
        cases=cases,
        usage=usage,
    )

    assert not gates["zero_retries"]
    assert not gates["all_conditioned_supports_have_at_least_eight_valid"]
    assert not gates["all_pass"]


class _FakeAdapter:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def chat_complete_messages_batched_structured(self, *args, **kwargs):
        return self.responses

    def usage_snapshot(self):
        return {
            **_usage(),
            "adapter_cost_usd": 0.05,
            "adapter_prompt_tokens": 100,
            "adapter_completion_tokens": 1000,
        }


def test_run_smoke_writes_public_result_without_raw_text(tmp_path) -> None:
    expressions = [
        f"n % 23 == {index}" for index in range(23)
    ] + ["n < 50"]
    response = json.dumps(
        {
            "hypotheses": [
                {"name": f"rule_{index}", "expression": expression}
                for index, expression in enumerate(expressions)
            ]
        }
    )
    cases = smoke.serving_cases()
    responses = []
    for case in cases:
        observations = case["observations"]
        items = []
        for index in range(24):
            positives = [
                f"n == {number}"
                for number, label in observations
                if label
            ]
            positives.append(f"n % 29 == {index}")
            negatives = [
                f"n != {number}"
                for number, label in observations
                if not label
            ]
            expression = f"({' or '.join(positives)})"
            if negatives:
                expression += " and " + " and ".join(negatives)
            items.append({"name": f"rule_{index}", "expression": expression})
        responses.append(json.dumps({"hypotheses": items}))
    responses[0] = response
    responses[1] = response

    result = smoke.run_smoke(
        output_dir=tmp_path,
        run_id="fake",
        adapter=_FakeAdapter(responses),
    )

    assert result["status"] == "passed"
    assert responses[0] not in json.dumps(result)
    assert (tmp_path / "private" / "RAW_RESPONSES.json").exists()
