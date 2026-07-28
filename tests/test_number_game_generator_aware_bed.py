from __future__ import annotations

import json
import re

import pytest

from scripts.number_game_generator_aware_bed import (
    NUM_PROPOSALS,
    RuleHypothesis,
    candidate_roots,
    compile_expression,
    evaluate_policy_root,
    generator_aware_score,
    merge_controlled_support,
    parse_proposals,
    proposal_response_format,
    query_eig,
    run_experiment,
)


def _rule(name: str, expression: str) -> RuleHypothesis:
    return RuleHypothesis(name, expression, compile_expression(expression))


def _payload(expressions: list[str]) -> str:
    assert len(expressions) == NUM_PROPOSALS
    return json.dumps(
        {
            "hypotheses": [
                {"name": f"rule-{index}", "expression": expression}
                for index, expression in enumerate(expressions)
            ]
        }
    )


def test_expression_compiler_accepts_dsl_and_rejects_unsafe_code():
    extension = compile_expression(
        "divisible(n, 3) and (n < 20 or ends_with(n, 6))"
    )

    assert extension[6]
    assert extension[18]
    assert not extension[21]
    with pytest.raises(ValueError, match="forbidden syntax|only documented"):
        compile_expression("__import__('os').system('id')")
    with pytest.raises(ValueError, match="constant rules"):
        compile_expression("n == n")


def test_response_schema_is_strict_and_fixed_width():
    response_format = proposal_response_format()
    schema = response_format["json_schema"]["schema"]
    hypotheses = schema["properties"]["hypotheses"]

    assert response_format["json_schema"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert hypotheses["minItems"] == hypotheses["maxItems"] == NUM_PROPOSALS
    assert hypotheses["items"]["additionalProperties"] is False


def test_parser_filters_invalid_inconsistent_and_duplicate_extensions():
    base = [
        "divisible(n, 2)",
        "n % 2 == 0",
        "divisible(n, 3)",
        "n % 5 == 1",
        "n < 50",
        "n > 20",
        "is_square(n)",
        "is_prime(n)",
        "digit_sum(n) < 8",
        "ends_with(n, 6)",
        "is_power_of_two(n)",
        "divisible(n, 7)",
        "n % 9 == 5",
        "n < 30 and divisible(n, 2)",
        "divisible(n, 3) or divisible(n, 7)",
        "divisible(n, 3) and n % 2 == 1",
        "is_prime(n + 1)",
        "is_square((n + 2) // 2) and (n + 2) % 2 == 0",
        "n >= 40 and n <= 60",
        "digit_sum(n) % 2 == 0",
        "n % 4 == 1",
        "n % 6 == 2",
        "n < 10 or n > 90",
        "ends_with(n, 2) or ends_with(n, 7)",
    ]
    parsed, diagnostics = parse_proposals(
        _payload(base), observations=((2, True),)
    )

    assert all(rule.extension[2] for rule in parsed)
    assert diagnostics["rejected"]["duplicate_extension"] == 1
    assert diagnostics["rejected"]["inconsistent"] > 0


def test_query_eig_and_controlled_support_include_simulated_truth():
    support = [_rule("even", "divisible(n, 2)"), _rule("three", "divisible(n, 3)")]
    generated = [_rule("five", "divisible(n, 5)"), support[0]]

    assert query_eig(support, 2) == pytest.approx(0.6931471805599453)
    controlled = merge_controlled_support(support[0], generated)
    assert controlled[0] == support[0]
    assert len(controlled) == 2


def test_generator_score_uses_branch_conditioned_support():
    support = [
        _rule("even", "divisible(n, 2)"),
        _rule("three", "divisible(n, 3)"),
    ]
    branches = {
        (2, False): [_rule("odd", "n % 2 == 1"), _rule("five", "divisible(n, 5)")],
        (2, True): [_rule("even", "divisible(n, 2)"), _rule("square", "is_square(n)")],
    }

    score = generator_aware_score(support, 2, branches)

    assert score >= query_eig(support, 2)


def test_candidate_roots_are_distinct_and_include_matched_controls():
    support = [
        _rule("even", "divisible(n, 2)"),
        _rule("three", "divisible(n, 3)"),
        _rule("five", "divisible(n, 5)"),
        _rule("square", "is_square(n)"),
        _rule("prime", "is_prime(n)"),
        _rule("ending6", "ends_with(n, 6)"),
        _rule("small", "n < 30"),
        _rule("large", "n > 70"),
    ]

    roots, metadata = candidate_roots(support, seed=7)

    assert len(roots) == len(set(roots)) == 8
    assert metadata["myopic_root"] in roots
    assert metadata["fixed_depth_two_root"] in roots


def test_endpoint_uses_same_generated_branch_for_each_policy_root():
    target = _rule("even", "divisible(n, 2)")
    branches = {
        (2, False): [_rule("odd", "n % 2 == 1")],
        (2, True): [
            target,
            _rule("multiples4", "divisible(n, 4)"),
        ],
    }

    result = evaluate_policy_root(
        policy="test",
        root=2,
        targets={"even": target},
        branches=branches,
    )

    assert result["root"] == 2
    assert result["truth_extension_coverage_rate"] == 1.0
    assert result["targets"][0]["branch_support_size"] == 2


class _FakeAdapter:
    def __init__(self) -> None:
        self.requests = 0

    @staticmethod
    def _expressions(root: int | None, label: bool | None) -> list[str]:
        candidates = []
        for modulus in range(2, 24):
            for remainder in range(modulus):
                candidates.append(f"n % {modulus} == {remainder}")
        candidates.extend(f"n < {threshold}" for threshold in range(1, 101))
        candidates.extend(f"n > {threshold}" for threshold in range(100))
        selected = []
        extensions = set()
        for expression in candidates:
            extension = compile_expression(expression)
            if root is not None and extension[root] != label:
                continue
            if extension in extensions:
                continue
            selected.append(expression)
            extensions.add(extension)
            if len(selected) == NUM_PROPOSALS:
                return selected
        raise AssertionError("test proposal pool was too small")

    def chat_complete_messages_batched_structured(
        self,
        messages,
        *,
        temperature,
        block_size,
        response_format,
        max_new_tokens,
    ):
        del temperature, block_size, response_format, max_new_tokens
        self.requests += len(messages)
        responses = []
        for request in messages:
            prompt = request[-1]["content"]
            match = re.search(
                r"Is (\\d+) in the concept\\? (YES|NO)", prompt
            )
            root = int(match.group(1)) if match else None
            label = match.group(2) == "YES" if match else None
            responses.append(
                _payload(self._expressions(root, label))
            )
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "run_cost_usd": 0.0,
        }


def test_fake_end_to_end_rehearsal_makes_exact_17_calls(tmp_path):
    adapter = _FakeAdapter()

    result = run_experiment(
        output_dir=tmp_path,
        run_id="fake-number-game",
        adapter=adapter,
    )

    assert result["status"] in {"passed", "mechanics_failed"}
    assert adapter.requests == 17
    assert result["gates"]["exact_17_accepted_requests"]
    assert (tmp_path / "RESULT.json").exists()
    assert (tmp_path / "MODEL.json").exists()
    assert (tmp_path / "private" / "RAW_RESPONSES.json").exists()
