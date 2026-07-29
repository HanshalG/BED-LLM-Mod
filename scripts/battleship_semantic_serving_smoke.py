#!/usr/bin/env python3
"""Qualify fresh Battleship questions and independent semantic translation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Protocol, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.battleship_llm_native_opportunity_audit import (
    EXPECTED_COMMIT,
    EXPECTED_TRAJECTORY_SHA256,
    git_commit,
    sample_official_prior,
    sha256_file,
)
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_predictive_risk_replication import (
    SeededStructuredAdapter,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "battleship-semantic-serving-smoke-1"
SOURCE_AUDIT_SHA256 = (
    "16aff4012fd5992341f161c01a4dfab48d562e04fe2591eb6fd87d85feba0872"
)
DEFAULT_EXTERNAL_ROOT = REPO_ROOT / "external/battleship"
PLANNER_MODEL_ID = "openai/gpt-5.4-mini"
TRANSLATOR_MODEL_IDS = (
    "openai/gpt-5.4-mini",
    "google/gemini-2.5-flash",
)
PLANNER_SEED = 39_700
TRANSLATOR_SEEDS = (39_800, 39_900)
BOARD_SEEDS = (39_710, 39_711)
NUM_PLANNER_CALLS = 2
QUESTIONS_PER_PLANNER = 4
SELECTED_PER_PLANNER = 2
NUM_SELECTED_QUESTIONS = NUM_PLANNER_CALLS * SELECTED_PER_PLANNER
EXPECTED_REQUESTS = NUM_PLANNER_CALLS + (
    NUM_SELECTED_QUESTIONS * len(TRANSLATOR_MODEL_IDS)
)
NUM_SAMPLES_PER_BLOCK = 4096
PLANNER_TEMPERATURE = 0.7
TRANSLATOR_TEMPERATURE = 0.0
PLANNER_MAX_TOKENS = 900
TRANSLATOR_MAX_TOKENS = 700
CONCURRENCY = 8
PROJECTED_COST_USD = 0.10
RUN_BUDGET_USD = 0.25
MIN_YES_PREVALENCE = 0.05
MAX_YES_PREVALENCE = 0.95
MIN_TRANSLATOR_AGREEMENT = 0.97
MIN_AGREEING_QUESTIONS = 3
MIN_UNIQUE_BEHAVIORS = 3

_YES_NO_PREFIX = re.compile(
    r"^(is|are|does|do|has|have|can|could|would|will|was|were)\b",
    re.IGNORECASE,
)
_SAFE_FUNCTIONS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "range": range,
    "sum": sum,
}
_SAFE_NUMPY_CALLS = {
    "all",
    "any",
    "count_nonzero",
    "isin",
    "max",
    "min",
    "sum",
    "unique",
    "where",
}
_SAFE_ARRAY_ATTRIBUTES = {"all", "any", "max", "min", "shape", "size", "sum"}
_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Set,
    ast.GeneratorExp,
    ast.ListComp,
    ast.SetComp,
    ast.comprehension,
    ast.Attribute,
    ast.IfExp,
)


class StructuredModel(Protocol):
    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class ServingExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def strict_json_object(response: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not exact JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys are {sorted(value)}, expected {sorted(expected)}"
        )


def planner_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "battleship_questions",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["questions"],
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": QUESTIONS_PER_PLANNER,
                        "maxItems": QUESTIONS_PER_PLANNER,
                        "items": {
                            "type": "string",
                            "minLength": 10,
                            "maxLength": 240,
                        },
                    }
                },
            },
        },
    }


def translator_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "battleship_boolean_expression",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["expression"],
                "properties": {
                    "expression": {
                        "type": "string",
                        "minLength": 4,
                        "maxLength": 1000,
                    }
                },
            },
        },
    }


def parse_planner_response(response: str) -> list[str]:
    value = strict_json_object(response, label="planner response")
    _exact_keys(value, {"questions"}, "planner response")
    questions = value["questions"]
    if not isinstance(questions, list) or len(questions) != QUESTIONS_PER_PLANNER:
        raise ValueError("planner must return exactly four questions")
    parsed = []
    for index, question in enumerate(questions):
        if not isinstance(question, str):
            raise ValueError(f"question {index} is not a string")
        text = " ".join(question.split())
        if not 10 <= len(text) <= 240:
            raise ValueError(f"question {index} has invalid length")
        if not text.endswith("?") or not _YES_NO_PREFIX.match(text):
            raise ValueError(f"question {index} is not a direct yes/no question")
        parsed.append(text)
    if len({text.casefold() for text in parsed}) != QUESTIONS_PER_PLANNER:
        raise ValueError("planner questions are not unique")
    return parsed


def parse_translation_response(response: str) -> str:
    value = strict_json_object(response, label="translation response")
    _exact_keys(value, {"expression"}, "translation response")
    expression = value["expression"]
    if not isinstance(expression, str):
        raise ValueError("translation expression is not a string")
    expression = expression.strip()
    if not 4 <= len(expression) <= 1000:
        raise ValueError("translation expression has invalid length")
    return expression


def planner_messages(call_index: int) -> list[dict[str, str]]:
    request = {
        "task": (
            "Propose four distinct semantic yes/no questions about a hidden "
            "standard Battleship board. These questions will be translated "
            "into executable predicates and used for Bayesian experimental "
            "design before any shots have been taken."
        ),
        "board": {
            "size": "8x8",
            "rows": "A through H",
            "columns": "1 through 8",
            "ships": "one each of lengths 2, 3, 4, and 5",
            "touching": "ships may touch but may not overlap",
        },
        "requirements": [
            "Every question must have an objective Boolean answer from the full board.",
            "Use spatial or relational structure, not subjective language.",
            "Questions may concern cells, regions, counts, lengths, or orientations.",
            "Make the four questions behaviorally diverse and likely to split plausible boards.",
            "Do not mention probabilities, information gain, code, or this request.",
            "Return direct grammatical yes/no questions ending in a question mark.",
        ],
        "independent_batch": call_index + 1,
    }
    return [
        {
            "role": "system",
            "content": (
                "You design informative natural-language experiments for "
                "Battleship. Return only the strict JSON schema."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def translator_messages(question: str) -> list[dict[str, str]]:
    request = {
        "task": (
            "Translate the Battleship yes/no question into one pure Python "
            "boolean expression. The expression is evaluated separately on "
            "each complete board."
        ),
        "question": question,
        "encoding": {
            "true_board": (
                "8x8 NumPy integer array; 0 is water; ship IDs 1,2,3,4 "
                "have lengths 2,3,4,5 respectively"
            ),
            "partial_board": (
                "8x8 NumPy integer array; -1 means unknown, 0 miss, positive hit; "
                "for this smoke it is entirely unknown"
            ),
            "coordinates": "row A is index 0; column 1 is index 0",
        },
        "allowed": [
            "one expression only, without return",
            "true_board, partial_board, np",
            "boolean/comparison/arithmetic/indexing operations",
            "np.any, np.all, np.sum, np.count_nonzero, np.unique, np.isin, np.where",
            "safe any/all/sum/len/range/min/max/abs/int/bool and comprehensions",
        ],
        "forbidden": [
            "imports, assignments, lambdas, function definitions, statements",
            "file/network/system access or any name beginning with underscore",
            "explanations, markdown, or code fences",
        ],
        "example_expression": "bool(np.any(true_board[0:4, 0:4] > 0))",
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a precise semantic compiler. Return only the strict "
                "JSON schema and preserve the question's literal meaning."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


class _SafeExpressionValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.bound_names: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"forbidden syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        if node.is_async:
            raise ValueError("async comprehensions are forbidden")
        if not isinstance(node.target, ast.Name):
            raise ValueError("only simple comprehension variables are allowed")
        if node.target.id.startswith("_"):
            raise ValueError("private comprehension name is forbidden")
        self.bound_names.add(node.target.id)
        self.visit(node.iter)
        for condition in node.ifs:
            self.visit(condition)

    def _visit_comprehension_expression(
        self,
        generators: Sequence[ast.comprehension],
        elements: Sequence[ast.AST],
    ) -> None:
        if len(generators) > 2:
            raise ValueError("at most two comprehension generators are allowed")
        for generator in generators:
            self.visit(generator)
        for element in elements:
            self.visit(element)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension_expression(node.generators, [node.elt])

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension_expression(node.generators, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension_expression(node.generators, [node.elt])

    def visit_Name(self, node: ast.Name) -> None:
        allowed = {
            "true_board",
            "partial_board",
            "np",
            *list(_SAFE_FUNCTIONS),
            *list(self.bound_names),
        }
        if node.id not in allowed or node.id.startswith("_"):
            raise ValueError(f"forbidden name: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise ValueError("private attribute access is forbidden")
        allowed = _SAFE_NUMPY_CALLS | _SAFE_ARRAY_ATTRIBUTES
        if node.attr not in allowed:
            raise ValueError(f"forbidden attribute: {node.attr}")
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id not in _SAFE_FUNCTIONS:
                raise ValueError(f"forbidden call: {node.func.id}")
            if node.func.id == "range":
                if not 1 <= len(node.args) <= 3:
                    raise ValueError("range requires one to three arguments")
                if not all(
                    isinstance(argument, ast.Constant)
                    and isinstance(argument.value, int)
                    and abs(argument.value) <= 64
                    for argument in node.args
                ):
                    raise ValueError("range bounds must be integer constants <=64")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr not in (
                _SAFE_NUMPY_CALLS | (_SAFE_ARRAY_ATTRIBUTES - {"shape", "size"})
            ):
                raise ValueError(f"forbidden method call: {node.func.attr}")
            self.visit(node.func)
        else:
            raise ValueError("indirect calls are forbidden")
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg.startswith("_"):
                raise ValueError("expanded or private call keywords are forbidden")
            self.visit(keyword.value)


def compile_safe_expression(expression: str) -> Any:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"expression syntax error: {exc}") from exc
    validator = _SafeExpressionValidator()
    validator.visit(tree)
    return compile(tree, "<battleship-expression>", "eval")


def evaluate_expression(
    compiled: Any,
    boards: np.ndarray,
    partial_board: np.ndarray,
) -> np.ndarray:
    values = []
    for true_board in boards:
        globals_value = {
            "np": np,
            "__builtins__": _SAFE_FUNCTIONS,
            "true_board": true_board,
            "partial_board": partial_board,
        }
        value = eval(
            compiled,
            globals_value,
            {},
        )
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(
                f"expression returned {type(value).__name__}, not bool"
            )
        values.append(bool(value))
    return np.asarray(values, dtype=np.uint8)


def _usage(models: Sequence[StructuredModel]) -> dict[str, Any]:
    snapshots = [model.usage_snapshot() for model in models]
    totals: dict[str, Any] = {}
    integer_keys = (
        "adapter_requests",
        "http_attempts",
        "retry_count",
        "provider_error_retries",
        "adapter_reasoning_tokens",
        "forced_exits",
        "adapter_prompt_tokens",
        "adapter_completion_tokens",
    )
    for key in (*integer_keys, "adapter_cost_usd"):
        totals[key] = sum(float(item.get(key, 0) or 0) for item in snapshots)
    for key in integer_keys:
        totals[key] = int(totals[key])
    totals["run_cost_usd"] = totals.pop("adapter_cost_usd")
    totals["models"] = snapshots
    return totals


def _checkpoint_private(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def aggregate_gates(
    *,
    question_metrics: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    selected_question_count: int,
) -> dict[str, bool]:
    agreement_passes = sum(
        item["translator_agreement"] >= MIN_TRANSLATOR_AGREEMENT
        for item in question_metrics
    )
    behavior_hashes = {
        item["reference_behavior_sha256"] for item in question_metrics
    }
    gates = {
        "exact_10_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "exact_10_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_ten_schemas_parse": (
            selected_question_count == NUM_SELECTED_QUESTIONS
            and len(question_metrics) == NUM_SELECTED_QUESTIONS
        ),
        "four_selected_questions_globally_unique": (
            selected_question_count == NUM_SELECTED_QUESTIONS
        ),
        "all_eight_translations_safe_boolean": all(
            item["safe_boolean_translations"] == len(TRANSLATOR_MODEL_IDS)
            for item in question_metrics
        ),
        "all_eight_translations_nontrivial_on_both_blocks": all(
            item["all_translations_nontrivial_on_both_blocks"]
            for item in question_metrics
        ),
        "at_least_three_of_four_translator_agreements_at_least_0_97": (
            agreement_passes >= MIN_AGREEING_QUESTIONS
        ),
        "at_least_three_distinct_reference_behaviors": (
            len(behavior_hashes) >= MIN_UNIQUE_BEHAVIORS
        ),
        "cost_at_most_0_25": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def run_smoke(
    *,
    external_root: Path,
    planner_model: StructuredModel,
    translator_models: Sequence[StructuredModel],
    raw_path: Path,
) -> dict[str, Any]:
    if len(translator_models) != len(TRANSLATOR_MODEL_IDS):
        raise ValueError("exactly two translator models are required")
    models = (planner_model, *translator_models)
    raw: dict[str, Any] = {
        "planner_responses": [],
        "selected_questions": [],
        "translator_responses": {},
        "policy_endpoint_accessed": False,
        "released_trajectory_questions_prompted": False,
    }
    try:
        trajectory_path = (
            external_root / "docs/static/data/trajectory_samples.json"
        )
        if git_commit(external_root) != EXPECTED_COMMIT:
            raise ValueError("official Battleship commit changed")
        if sha256_file(trajectory_path) != EXPECTED_TRAJECTORY_SHA256:
            raise ValueError("official trajectory artifact changed")

        planner_responses = (
            planner_model.chat_complete_messages_batched_structured(
                [
                    planner_messages(index)
                    for index in range(NUM_PLANNER_CALLS)
                ],
                temperature=PLANNER_TEMPERATURE,
                block_size=NUM_PLANNER_CALLS,
                response_format=planner_response_format(),
                max_new_tokens=PLANNER_MAX_TOKENS,
            )
        )
        raw["planner_responses"] = list(planner_responses)
        _checkpoint_private(raw_path, raw)
        if len(planner_responses) != NUM_PLANNER_CALLS:
            raise ValueError("planner response count changed")
        question_batches = [
            parse_planner_response(response)
            for response in planner_responses
        ]
        selected_questions = [
            question
            for batch in question_batches
            for question in batch[:SELECTED_PER_PLANNER]
        ]
        if (
            len({question.casefold() for question in selected_questions})
            != NUM_SELECTED_QUESTIONS
        ):
            raise ValueError("selected questions are not globally unique")
        raw["selected_questions"] = selected_questions

        translated_expressions: list[list[str]] = []
        for model_id, model in zip(
            TRANSLATOR_MODEL_IDS, translator_models, strict=True
        ):
            responses = model.chat_complete_messages_batched_structured(
                [
                    translator_messages(question)
                    for question in selected_questions
                ],
                temperature=TRANSLATOR_TEMPERATURE,
                block_size=NUM_SELECTED_QUESTIONS,
                response_format=translator_response_format(),
                max_new_tokens=TRANSLATOR_MAX_TOKENS,
            )
            raw["translator_responses"][model_id] = list(responses)
            _checkpoint_private(raw_path, raw)
            if len(responses) != NUM_SELECTED_QUESTIONS:
                raise ValueError(
                    f"translator response count changed for {model_id}"
                )
            translated_expressions.append(
                [
                    parse_translation_response(response)
                    for response in responses
                ]
            )

        blocks = sample_official_prior(
            external_root,
            seeds=BOARD_SEEDS,
            num_samples=NUM_SAMPLES_PER_BLOCK,
        )
        partial_board = np.full((8, 8), -1, dtype=int)
        question_metrics = []
        for question_index, question in enumerate(selected_questions):
            outcomes_by_translator = []
            translator_metrics = []
            for translator_index, model_id in enumerate(
                TRANSLATOR_MODEL_IDS
            ):
                expression = translated_expressions[translator_index][
                    question_index
                ]
                compiled = compile_safe_expression(expression)
                block_outcomes = [
                    evaluate_expression(compiled, boards, partial_board)
                    for boards in blocks
                ]
                prevalence = [
                    float(outcomes.mean()) for outcomes in block_outcomes
                ]
                outcomes_by_translator.append(block_outcomes)
                translator_metrics.append(
                    {
                        "model": model_id,
                        "expression_sha256": hashlib.sha256(
                            expression.encode("utf-8")
                        ).hexdigest(),
                        "yes_prevalence": prevalence,
                        "nontrivial_on_both_blocks": all(
                            MIN_YES_PREVALENCE
                            <= value
                            <= MAX_YES_PREVALENCE
                            for value in prevalence
                        ),
                    }
                )
            agreements = [
                float(
                    np.mean(
                        outcomes_by_translator[0][block_index]
                        == outcomes_by_translator[1][block_index]
                    )
                )
                for block_index in range(len(blocks))
            ]
            reference = b"".join(
                outcomes.tobytes()
                for outcomes in outcomes_by_translator[0]
            )
            question_metrics.append(
                {
                    "question_index": question_index,
                    "question": question,
                    "safe_boolean_translations": len(translator_metrics),
                    "all_translations_nontrivial_on_both_blocks": all(
                        item["nontrivial_on_both_blocks"]
                        for item in translator_metrics
                    ),
                    "translator_agreement_by_block": agreements,
                    "translator_agreement": min(agreements),
                    "reference_behavior_sha256": hashlib.sha256(
                        reference
                    ).hexdigest(),
                    "translations": translator_metrics,
                }
            )
        usage = _usage(models)
    except Exception as exc:
        _checkpoint_private(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}", _usage(models)
        ) from exc

    gates = aggregate_gates(
        question_metrics=question_metrics,
        usage=usage,
        selected_question_count=len(selected_questions),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_audit_sha256": SOURCE_AUDIT_SHA256,
            "official_commit": EXPECTED_COMMIT,
            "official_trajectory_sha256": EXPECTED_TRAJECTORY_SHA256,
            "planner_model": PLANNER_MODEL_ID,
            "translator_models": list(TRANSLATOR_MODEL_IDS),
            "planner_seed": PLANNER_SEED,
            "translator_seeds": list(TRANSLATOR_SEEDS),
            "board_seeds": list(BOARD_SEEDS),
            "num_samples_per_block": NUM_SAMPLES_PER_BLOCK,
            "expected_requests": EXPECTED_REQUESTS,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "policy_endpoint_accessed": False,
            "released_trajectory_questions_prompted": False,
        },
        "metrics": {
            "selected_question_count": len(selected_questions),
            "safe_boolean_translation_count": sum(
                item["safe_boolean_translations"]
                for item in question_metrics
            ),
            "nontrivial_translation_count": sum(
                sum(
                    translation["nontrivial_on_both_blocks"]
                    for translation in item["translations"]
                )
                for item in question_metrics
            ),
            "agreement_passing_question_count": sum(
                item["translator_agreement"] >= MIN_TRANSLATOR_AGREEMENT
                for item in question_metrics
            ),
            "unique_reference_behavior_count": len(
                {
                    item["reference_behavior_sha256"]
                    for item in question_metrics
                }
            ),
        },
        "questions": question_metrics,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self, role: str, variant: int = 0) -> None:
        self.role = role
        self.variant = variant
        self.requests = 0

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, response_format, max_new_tokens
        responses = []
        questions = (
            [
                "Is there any ship tile on cell D4?",
                "Is the length-5 ship oriented horizontally?",
                "Is any ship tile in columns 1 through 4?",
                "Are at least seven ship tiles on dark squares?",
            ],
            [
                "Is there any ship tile on cell E5?",
                "Does the length-3 ship touch row A?",
                "Is any ship tile on the main diagonal?",
                "Are more ship tiles above row E than below it?",
            ],
        )
        expressions = {
            questions[0][0]: "bool(true_board[3, 3] > 0)",
            questions[0][1]: (
                "bool(len(np.unique(np.where(true_board == 4)[0])) == 1)"
            ),
            questions[1][0]: "bool(true_board[4, 4] > 0)",
            questions[1][1]: "bool(np.any(true_board[0, :] == 2))",
        }
        for messages in batch_messages:
            request = json.loads(messages[-1]["content"])
            if self.role == "planner":
                batch_index = int(request["independent_batch"]) - 1
                responses.append(
                    json.dumps({"questions": questions[batch_index]})
                )
            else:
                expression = expressions[request["question"]]
                responses.append(json.dumps({"expression": expression}))
            self.requests += 1
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
            "adapter_cost_usd": 0.0,
            "model": f"fixture-{self.role}-{self.variant}",
        }


def _adapter(
    *,
    model: str,
    seed: int,
    run_id: str,
    output_dir: Path,
    max_tokens: int,
) -> SeededStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=500.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=max_tokens,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return SeededStructuredAdapter(
        ModelSpec(model=model, backend="openrouter", max_model_len=262_144),
        config,
        request_seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=DEFAULT_EXTERNAL_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"

    if args.dry_run:
        planner: StructuredModel = DeterministicFixtureModel("planner")
        translators: Sequence[StructuredModel] = (
            DeterministicFixtureModel("translator", 0),
            DeterministicFixtureModel("translator", 1),
        )
    else:
        planner = _adapter(
            model=PLANNER_MODEL_ID,
            seed=PLANNER_SEED,
            run_id=f"{args.run_id}-planner",
            output_dir=args.output_dir,
            max_tokens=PLANNER_MAX_TOKENS,
        )
        translators = tuple(
            _adapter(
                model=model,
                seed=seed,
                run_id=f"{args.run_id}-translator-{index}",
                output_dir=args.output_dir,
                max_tokens=TRANSLATOR_MAX_TOKENS,
            )
            for index, (model, seed) in enumerate(
                zip(TRANSLATOR_MODEL_IDS, TRANSLATOR_SEEDS, strict=True)
            )
        )

    try:
        payload = run_smoke(
            external_root=args.external_root.resolve(),
            planner_model=planner,
            translator_models=translators,
            raw_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
        checkpoint(args.output_dir / "SERVING.json", payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "metrics": payload["metrics"],
                    "gates": payload["gates"],
                    "usage": payload["usage"],
                },
                indent=2,
            )
        )
    except ServingExecutionError as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "source_audit_sha256": SOURCE_AUDIT_SHA256,
                "expected_requests": EXPECTED_REQUESTS,
                "reasoning_requested": False,
                "repairs_or_reissues": 0,
            },
            "error": str(exc),
            "usage": exc.usage,
            "private_raw_sha256": sha256_file(raw_path),
        }
        checkpoint(args.output_dir / "FAILURE.json", failure)
        raise


if __name__ == "__main__":
    main()
