#!/usr/bin/env python3
"""Gate a target-blind LLM bank of executable Battleship experiments."""

from __future__ import annotations

import argparse
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
    FiniteHorizonQuestionPlanner,
    compile_program,
    dedupe_by_joint_behavior,
    evaluate_program,
    extract_stage_zero_programs,
    git_commit,
    sample_official_prior,
    sha256_file,
)
from scripts.battleship_semantic_serving_smoke import (
    ServingExecutionError,
    compile_safe_expression,
    evaluate_expression,
    strict_json_object,
)
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_predictive_risk_replication import (
    SeededStructuredAdapter,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "battleship-executable-candidate-bank-1"
SOURCE_AUDIT_SHA256 = (
    "16aff4012fd5992341f161c01a4dfab48d562e04fe2591eb6fd87d85feba0872"
)
FAILED_TRANSLATION_SHA256 = (
    "27ae82a7dc531a8457a777c68c211be8e92d576b7ff89798ca58c592a2b9a625"
)
DEFAULT_EXTERNAL_ROOT = REPO_ROOT / "external/battleship"
MODEL_ID = "openai/gpt-5.4"
MODEL_SEED = 40_000
BOARD_SEEDS = (40_010, 40_011)
NUM_CALLS = 10
CANDIDATES_PER_CALL = 6
EXPECTED_REQUESTS = NUM_CALLS
NUM_SAMPLES_PER_BLOCK = 4096
TEMPERATURE = 0.7
MAX_TOKENS = 2500
CONCURRENCY = 10
PROJECTED_COST_USD = 0.25
RUN_BUDGET_USD = 0.50
MIN_YES_PREVALENCE = 0.05
MAX_YES_PREVALENCE = 0.95
MIN_VALID_PER_CALL = 3
MIN_UNIQUE_VALID = 20
MIN_NOVEL_BEHAVIORS = 8
MAX_PLANNING_BANK = 25
MIN_D2_OVER_D1 = 0.01
MIN_NONMYOPIC_OVER_GREEDY_EIG = 0.02
_YES_NO_PREFIX = re.compile(
    r"^(is|are|does|do|has|have|can|could|would|will|was|were)\b",
    re.IGNORECASE,
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


def response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "battleship_executable_experiments",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidates"],
                "properties": {
                    "candidates": {
                        "type": "array",
                        "minItems": CANDIDATES_PER_CALL,
                        "maxItems": CANDIDATES_PER_CALL,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["question", "expression"],
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "minLength": 10,
                                    "maxLength": 240,
                                },
                                "expression": {
                                    "type": "string",
                                    "minLength": 4,
                                    "maxLength": 1000,
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def parse_response(response: str) -> list[dict[str, str]]:
    value = strict_json_object(response, label="candidate response")
    if set(value) != {"candidates"}:
        raise ValueError("candidate response has unexpected keys")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) != CANDIDATES_PER_CALL:
        raise ValueError("response must contain exactly six candidates")
    parsed = []
    seen = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != {
            "question",
            "expression",
        }:
            raise ValueError(f"candidate {index} has unexpected shape")
        question = " ".join(str(candidate["question"]).split())
        expression = str(candidate["expression"]).strip()
        if not 10 <= len(question) <= 240:
            raise ValueError(f"candidate {index} question length is invalid")
        if not question.endswith("?") or not _YES_NO_PREFIX.match(question):
            raise ValueError(f"candidate {index} is not a direct yes/no question")
        if not 4 <= len(expression) <= 1000:
            raise ValueError(f"candidate {index} expression length is invalid")
        key = question.casefold()
        if key in seen:
            raise ValueError("questions within one response are not unique")
        seen.add(key)
        parsed.append({"question": question, "expression": expression})
    return parsed


def candidate_messages(batch_index: int) -> list[dict[str, str]]:
    slot_requirements = [
        "a single cell or compact region of at most four cells",
        "a count threshold on a compact region or one named ship",
        "a named ship's orientation or location",
        "a parity, checkerboard, symmetry, or count-comparison property",
        "a relation between two regions or between ships",
        "an original property unlike the first five",
    ]
    request = {
        "task": (
            "Directly define six executable yes/no measurements for Bayesian "
            "experimental design on an unseen Battleship board. The Python "
            "expression is the operational definition of the experiment; "
            "there is no later translator."
        ),
        "board": {
            "size": "8x8",
            "rows": "A-H map to indices 0-7",
            "columns": "1-8 map to indices 0-7",
            "true_board": (
                "NumPy integer array: water 0; ship IDs 1,2,3,4 have "
                "lengths 2,3,4,5"
            ),
            "partial_board": (
                "NumPy integer array: hidden -1, water 0, ship IDs 1-4; "
                "it is fully hidden at this gate"
            ),
        },
        "allowed_expression_language": [
            "one pure expression returning bool or np.bool_",
            "true_board, partial_board, np",
            "boolean, comparison, arithmetic, indexing, slices",
            "np.any, np.all, np.sum, np.count_nonzero, np.unique, np.isin, np.where",
            "bool, int, len, range, min, max, abs, sum, any, all",
            "at most two comprehensions, each over constant range bounds no larger than 64",
        ],
        "forbidden": [
            "imports, statements, assignment, lambda, function definitions",
            "np.array, reshape, argwhere, arbitrary methods, private names",
            "hardcoded True/False or expressions independent of true_board",
            "answers that use the hidden target board outside the expression",
        ],
        "quality": [
            "Aim for Yes probability between 20% and 80% under random legal boards.",
            "Avoid broad existential questions such as any ship in half the board.",
            "Make all six induced board partitions behaviorally different.",
            "Keep question text literally consistent with its expression.",
            "Use the six slot requirements in order.",
        ],
        "slot_requirements": slot_requirements,
        "independent_batch": batch_index + 1,
        "examples": [
            {
                "question": "Is cell B2 occupied by a ship?",
                "expression": "bool(true_board[1, 1] > 0)",
            },
            {
                "question": "Is the length-4 ship vertical?",
                "expression": (
                    "bool(len(np.unique(np.where(true_board == 3)[1])) == 1)"
                ),
            },
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You invent executable scientific measurements. Return only "
                "the strict JSON schema. Do not reveal reasoning."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def _usage(model: StructuredModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "adapter_requests": int(snapshot.get("adapter_requests", 0) or 0),
        "http_attempts": int(snapshot.get("http_attempts", 0) or 0),
        "retry_count": int(snapshot.get("retry_count", 0) or 0),
        "provider_error_retries": int(
            snapshot.get("provider_error_retries", 0) or 0
        ),
        "adapter_reasoning_tokens": int(
            snapshot.get("adapter_reasoning_tokens", 0) or 0
        ),
        "forced_exits": int(snapshot.get("forced_exits", 0) or 0),
        "adapter_prompt_tokens": int(
            snapshot.get("adapter_prompt_tokens", 0) or 0
        ),
        "adapter_completion_tokens": int(
            snapshot.get("adapter_completion_tokens", 0) or 0
        ),
        "run_cost_usd": float(snapshot.get("adapter_cost_usd", 0.0) or 0.0),
        "model": snapshot,
    }


def _checkpoint_private(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def released_behavior_keys(
    external_root: Path,
    blocks: Sequence[np.ndarray],
) -> set[bytes]:
    payload = json.loads(
        (
            external_root / "docs/static/data/trajectory_samples.json"
        ).read_text()
    )
    keys = set()
    partial_board = np.full((8, 8), -1, dtype=int)
    for program in extract_stage_zero_programs(payload):
        answer = compile_program(program["fn_str"])
        outcomes = [
            evaluate_program(answer, boards, partial_board)
            for boards in blocks
        ]
        keys.add(b"".join(outcome.tobytes() for outcome in outcomes))
    return keys


def evaluate_candidates(
    candidate_batches: Sequence[Sequence[dict[str, str]]],
    blocks: Sequence[np.ndarray],
    *,
    released_keys: set[bytes],
) -> tuple[list[dict[str, Any]], list[list[np.ndarray]], list[int]]:
    partial_board = np.full((8, 8), -1, dtype=int)
    valid_candidates = []
    valid_outcomes = []
    valid_per_call = [0 for _ in candidate_batches]
    for call_index, candidates in enumerate(candidate_batches):
        for candidate_index, candidate in enumerate(candidates):
            expression = candidate["expression"]
            try:
                compiled = compile_safe_expression(expression)
                outcomes = [
                    evaluate_expression(compiled, boards, partial_board)
                    for boards in blocks
                ]
                prevalence = [float(outcome.mean()) for outcome in outcomes]
                if not all(
                    MIN_YES_PREVALENCE <= value <= MAX_YES_PREVALENCE
                    for value in prevalence
                ):
                    continue
            except Exception:
                continue
            behavior_key = b"".join(
                outcome.tobytes() for outcome in outcomes
            )
            valid_per_call[call_index] += 1
            valid_candidates.append(
                {
                    "call_index": call_index,
                    "candidate_index": candidate_index,
                    "question": candidate["question"],
                    "expression_sha256": hashlib.sha256(
                        expression.encode("utf-8")
                    ).hexdigest(),
                    "behavior_sha256": hashlib.sha256(
                        behavior_key
                    ).hexdigest(),
                    "yes_prevalence": prevalence,
                    "novel_vs_released_bank": behavior_key not in released_keys,
                }
            )
            valid_outcomes.append(outcomes)
    return valid_candidates, valid_outcomes, valid_per_call


def planning_results(
    candidates: Sequence[dict[str, Any]],
    outcomes_by_candidate: Sequence[Sequence[np.ndarray]],
    blocks: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    planning_candidates = list(candidates[:MAX_PLANNING_BANK])
    planning_outcomes = list(outcomes_by_candidate[:MAX_PLANNING_BANK])
    results = []
    for block_index, boards in enumerate(blocks):
        outcomes = np.stack(
            [candidate[block_index] for candidate in planning_outcomes]
        )
        occupancy = (boards > 0).reshape(len(boards), -1)
        planner = FiniteHorizonQuestionPlanner(
            outcomes,
            occupancy,
            epsilon=0.1,
        )
        depth_results = {}
        for depth in (1, 2, 3):
            root = planner.root_evaluation(depth)
            depth_results[str(depth)] = {
                "best_value": root.best_value,
                "best_indices": list(root.best_indices),
                "best_questions": [
                    planning_candidates[index]["question"]
                    for index in root.best_indices
                ],
                "best_behavior_sha256": [
                    planning_candidates[index]["behavior_sha256"]
                    for index in root.best_indices
                ],
            }
        receding = {
            str(depth): planner.receding_horizon_value(
                total_questions=3,
                planning_horizon=depth,
            )
            for depth in (1, 2, 3)
        }
        greedy_index, greedy_eig = planner.greedy_eig_root()
        greedy_terminal = planner.greedy_eig_terminal_value(
            total_questions=3
        )
        results.append(
            {
                "block_index": block_index,
                "seed": BOARD_SEEDS[block_index],
                "depth": depth_results,
                "three_question_receding_hit_probability": receding,
                "greedy_eig": {
                    "root_index": greedy_index,
                    "root_question": planning_candidates[greedy_index][
                        "question"
                    ],
                    "root_eig_bits": greedy_eig,
                    "three_question_hit_probability": greedy_terminal,
                },
            }
        )
    return results


def aggregate_gates(
    *,
    usage: dict[str, Any],
    candidate_batches: Sequence[Sequence[dict[str, str]]],
    valid_per_call: Sequence[int],
    unique_candidates: Sequence[dict[str, Any]],
    block_results: Sequence[dict[str, Any]],
) -> dict[str, bool]:
    d1_d2_distinct = []
    d2_gain = []
    d3_no_regression = []
    nonmyopic_beats_greedy = []
    for block in block_results:
        roots = [
            set(
                block["depth"][str(depth)]["best_behavior_sha256"]
            )
            for depth in (1, 2, 3)
        ]
        d1_d2_distinct.append(roots[0].isdisjoint(roots[1]))
        values = block["three_question_receding_hit_probability"]
        d2_gain.append(values["2"] - values["1"] >= MIN_D2_OVER_D1)
        d3_no_regression.append(values["3"] >= values["2"] - 1e-12)
        nonmyopic_beats_greedy.append(
            max(values["2"], values["3"])
            - block["greedy_eig"]["three_question_hit_probability"]
            >= MIN_NONMYOPIC_OVER_GREEDY_EIG
        )
    gates = {
        "exact_10_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "exact_10_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_ten_schemas_parse": len(candidate_batches) == NUM_CALLS,
        "each_call_has_at_least_three_valid_nontrivial_candidates": all(
            count >= MIN_VALID_PER_CALL for count in valid_per_call
        ),
        "at_least_20_unique_valid_behaviors": (
            len(unique_candidates) >= MIN_UNIQUE_VALID
        ),
        "at_least_8_behaviors_novel_vs_released_bank": (
            sum(
                candidate["novel_vs_released_bank"]
                for candidate in unique_candidates
            )
            >= MIN_NOVEL_BEHAVIORS
        ),
        "depth1_depth2_roots_disjoint_on_both_blocks": all(
            d1_d2_distinct
        ),
        "depth2_three_question_gain_at_least_0_01_on_both_blocks": all(
            d2_gain
        ),
        "depth3_does_not_regress_vs_depth2_on_both_blocks": all(
            d3_no_regression
        ),
        "best_nonmyopic_beats_greedy_eig_by_0_02_on_both_blocks": all(
            nonmyopic_beats_greedy
        ),
        "cost_at_most_0_50": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def run_gate(
    *,
    external_root: Path,
    model: StructuredModel,
    raw_path: Path,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "responses": [],
        "policy_endpoint_accessed": False,
        "hidden_target_accessed": False,
        "failed_translation_responses_used": False,
    }
    try:
        trajectory_path = (
            external_root / "docs/static/data/trajectory_samples.json"
        )
        if git_commit(external_root) != EXPECTED_COMMIT:
            raise ValueError("official Battleship commit changed")
        if sha256_file(trajectory_path) != EXPECTED_TRAJECTORY_SHA256:
            raise ValueError("official trajectory artifact changed")
        responses = model.chat_complete_messages_batched_structured(
            [candidate_messages(index) for index in range(NUM_CALLS)],
            temperature=TEMPERATURE,
            block_size=NUM_CALLS,
            response_format=response_format(),
            max_new_tokens=MAX_TOKENS,
        )
        raw["responses"] = list(responses)
        _checkpoint_private(raw_path, raw)
        if len(responses) != NUM_CALLS:
            raise ValueError("response count changed")
        candidate_batches = [parse_response(response) for response in responses]

        blocks = sample_official_prior(
            external_root,
            seeds=BOARD_SEEDS,
            num_samples=NUM_SAMPLES_PER_BLOCK,
        )
        source_keys = released_behavior_keys(external_root, blocks)
        valid_candidates, valid_outcomes, valid_per_call = evaluate_candidates(
            candidate_batches,
            blocks,
            released_keys=source_keys,
        )
        unique_candidates, unique_outcomes = dedupe_by_joint_behavior(
            valid_candidates, valid_outcomes
        )
        if len(unique_candidates) < 3:
            raise ValueError("fewer than three unique valid candidates")
        block_results = planning_results(
            unique_candidates,
            unique_outcomes,
            blocks,
        )
        usage = _usage(model)
    except Exception as exc:
        _checkpoint_private(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}", _usage(model)
        ) from exc

    gates = aggregate_gates(
        usage=usage,
        candidate_batches=candidate_batches,
        valid_per_call=valid_per_call,
        unique_candidates=unique_candidates,
        block_results=block_results,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_audit_sha256": SOURCE_AUDIT_SHA256,
            "failed_translation_sha256": FAILED_TRANSLATION_SHA256,
            "model": MODEL_ID,
            "model_seed": MODEL_SEED,
            "board_seeds": list(BOARD_SEEDS),
            "num_samples_per_block": NUM_SAMPLES_PER_BLOCK,
            "num_calls": NUM_CALLS,
            "candidates_per_call": CANDIDATES_PER_CALL,
            "max_planning_bank": MAX_PLANNING_BANK,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "policy_endpoint_accessed": False,
            "hidden_target_accessed": False,
            "failed_translation_responses_used": False,
        },
        "counts": {
            "raw_candidates": NUM_CALLS * CANDIDATES_PER_CALL,
            "valid_nontrivial_candidates": len(valid_candidates),
            "valid_per_call": valid_per_call,
            "unique_valid_behaviors": len(unique_candidates),
            "novel_vs_released_bank": sum(
                candidate["novel_vs_released_bank"]
                for candidate in unique_candidates
            ),
            "planning_bank_size": min(
                len(unique_candidates), MAX_PLANNING_BANK
            ),
        },
        "candidate_bank": list(unique_candidates[:MAX_PLANNING_BANK]),
        "blocks": block_results,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self) -> None:
        self.requests = 0

    @staticmethod
    def _candidate(question: str, expression: str) -> dict[str, str]:
        return {"question": question, "expression": expression}

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
        cells = [
            (row, column)
            for row in range(8)
            for column in range(8)
        ]
        for messages in batch_messages:
            request = json.loads(messages[-1]["content"])
            batch = int(request["independent_batch"]) - 1
            candidates = []
            if batch == 0:
                candidates.extend(
                    [
                        self._candidate(
                            "Is cell D4 occupied by a ship?",
                            "bool(true_board[3, 3] > 0)",
                        ),
                        self._candidate(
                            "Is either cell D4 or D5 occupied by a ship?",
                            "bool(np.any(true_board[3, 3:5] > 0))",
                        ),
                        self._candidate(
                            "Is any cell in the D4 to E5 square occupied by a ship?",
                            "bool(np.any(true_board[3:5, 3:5] > 0))",
                        ),
                    ]
                )
            start = max(0, batch * CANDIDATES_PER_CALL - len(candidates))
            for row, column in cells[start:]:
                if len(candidates) == CANDIDATES_PER_CALL:
                    break
                coordinate = f"{chr(65 + row)}{column + 1}"
                candidates.append(
                    self._candidate(
                        f"Is cell {coordinate} occupied by a ship?",
                        f"bool(true_board[{row}, {column}] > 0)",
                    )
                )
            responses.append(json.dumps({"candidates": candidates}))
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
            "model": "fixture",
        }


def _adapter(
    *,
    run_id: str,
    output_dir: Path,
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
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return SeededStructuredAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=262_144),
        config,
        request_seed=MODEL_SEED,
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
    model: StructuredModel = (
        DeterministicFixtureModel()
        if args.dry_run
        else _adapter(run_id=args.run_id, output_dir=args.output_dir)
    )
    try:
        payload = run_gate(
            external_root=args.external_root.resolve(),
            model=model,
            raw_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
        checkpoint(args.output_dir / "GATE.json", payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "counts": payload["counts"],
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
                "failed_translation_sha256": FAILED_TRANSLATION_SHA256,
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
