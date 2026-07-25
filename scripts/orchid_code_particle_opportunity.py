#!/usr/bin/env python3
"""Screen Orchid for executable-particle clarification opportunity."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any, Iterable, Protocol, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.atd_code_first_link_audit import evaluate_outputs, spearman


INTERFACE_VERSION = "orchid-code-particle-opportunity-1"
ORCHID_REPOSITORY = "https://huggingface.co/datasets/SII-YDD/Orchid"
ORCHID_COMMIT = "55ddeb0d3670d22d420e072816cf4f034cf62caf"
ORCHID_HEVAL_SHA256 = (
    "412b5caf99df978482af62d4e3ea7dec089cb457b99d47fcd5d9eb56329e0082"
)
ATD_REPOSITORY = "https://github.com/kasia-kobalczyk/active-task-disambiguation"
ATD_COMMIT = "4c8ecb4d4ffdbffcc611366743fc1e2461037772"
ELIGIBLE_MANIFEST_SHA256 = (
    "cbc79146077f4d7b8d4d68365b4d816acf45777a15ae3e14f2fe5c62b1e96920"
)
MODEL_ID = "openai/gpt-5.4-mini"
SEED = 24_374
SPLIT_SEED = 24_375
TASK_IDS = (97, 133)
EXPECTED_TEST_COUNTS = {97: 8, 133: 12}
EXPECTED_SPLITS = {
    97: {"query": (2, 5, 6, 7), "holdout": (0, 1, 3, 4)},
    133: {"query": (0, 3, 5, 6, 7, 9), "holdout": (1, 2, 4, 8, 10, 11)},
}
PARTICLE_COUNT = 8
MIN_VALID_PARTICLES = 6
EXPECTED_REQUESTS = len(TASK_IDS) * PARTICLE_COUNT
MAX_COST_USD = 0.50
EXECUTION_TIMEOUT_SECONDS = 1.0


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class OpportunityExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def entropy_from_outcomes(outcomes: Iterable[str]) -> float:
    values = list(outcomes)
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def extract_python_program(response: str, entry_point: str) -> str:
    text = response.strip()
    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0].strip()
    elif text.startswith("```") and "```" in text[3:]:
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if not text or len(text) > 20_000:
        raise ValueError("program is empty or too long")
    tree = ast.parse(text)
    definitions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if entry_point not in definitions:
        raise ValueError(f"program does not define {entry_point}")
    compile(tree, "<orchid-particle>", "exec")
    return text


def _canonical_target_output(text: str) -> str:
    return str(ast.literal_eval(text))


def _particle_messages(
    *,
    task_id: int,
    ambiguous_prompt: str,
    entry_point: str,
    sample_index: int,
) -> list[dict[str, str]]:
    request = {
        "benchmark_task_id": task_id,
        "ambiguous_function_request": ambiguous_prompt,
        "required_entry_point": entry_point,
        "interpretation_sample": sample_index,
    }
    return [
        {
            "role": "system",
            "content": (
                "Write one complete executable Python implementation for the "
                "ambiguous function request. Preserve the required function name "
                "and signature. Choose one concrete plausible interpretation; "
                "different interpretation_sample values should explore materially "
                "different plausible behavior when the request permits it. Return "
                "only Python source code, optionally in one python code fence. Do "
                "not explain the code and do not include tests."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _eligible_manifest(rows: Sequence[dict[str, Any]]) -> list[int]:
    eligible = []
    for index, row in enumerate(rows):
        tests = row["test_case"]
        if index in {0, 1, 2} or len(tests) < 8:
            continue
        valid = True
        for test in tests:
            try:
                ast.parse("f(" + test["input"] + ")", mode="eval")
                ast.literal_eval(test["output"])
            except Exception:
                valid = False
                break
            if test.get("relation") != "==":
                valid = False
                break
        if valid:
            eligible.append(index)
    random.Random(SEED).shuffle(eligible)
    return eligible


def verify_sources(orchid_root: Path, atd_root: Path) -> Path:
    orchid_commit = subprocess.check_output(
        ["git", "-C", str(orchid_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if orchid_commit != ORCHID_COMMIT:
        raise ValueError(
            f"Orchid commit is {orchid_commit}, expected {ORCHID_COMMIT}"
        )
    atd_commit = subprocess.check_output(
        ["git", "-C", str(atd_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if atd_commit != ATD_COMMIT:
        raise ValueError(f"ATD commit is {atd_commit}, expected {ATD_COMMIT}")
    dataset_path = orchid_root / "Orchid-HEval" / "data.jsonl"
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if digest != ORCHID_HEVAL_SHA256:
        raise ValueError(f"Orchid HEval hash is {digest}, expected {ORCHID_HEVAL_SHA256}")
    rows = _load_rows(dataset_path)
    manifest = _eligible_manifest(rows)
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest_digest != ELIGIBLE_MANIFEST_SHA256:
        raise ValueError("Orchid eligible manifest hash does not match")
    if tuple(manifest[: len(TASK_IDS)]) != TASK_IDS:
        raise ValueError("Orchid opportunity task order does not match")
    for task_id in TASK_IDS:
        if len(rows[task_id]["test_case"]) != EXPECTED_TEST_COUNTS[task_id]:
            raise ValueError(f"task {task_id} test count changed")
        indices = list(range(EXPECTED_TEST_COUNTS[task_id]))
        random.Random(SPLIT_SEED + task_id).shuffle(indices)
        split = len(indices) // 2
        actual = {
            "query": tuple(sorted(indices[:split])),
            "holdout": tuple(sorted(indices[split:])),
        }
        if actual != EXPECTED_SPLITS[task_id]:
            raise ValueError(f"task {task_id} split changed")
    return dataset_path


def _target_blind_task_metadata(
    dataset_path: Path,
) -> dict[int, dict[str, Any]]:
    rows = _load_rows(dataset_path)
    metadata = {}
    for task_id in TASK_IDS:
        row = rows[task_id]
        metadata[task_id] = {
            "task_id": task_id,
            "ambiguous_prompt": row["Vagueness_prompt"],
            "entry_point": row["entry_point"],
            "inputs": [test["input"] for test in row["test_case"]],
            "query_indices": EXPECTED_SPLITS[task_id]["query"],
            "holdout_indices": EXPECTED_SPLITS[task_id]["holdout"],
        }
    return metadata


def _load_target_outputs(dataset_path: Path) -> dict[int, list[str]]:
    rows = _load_rows(dataset_path)
    return {
        task_id: [
            _canonical_target_output(test["output"])
            for test in rows[task_id]["test_case"]
        ]
        for task_id in TASK_IDS
    }


def score_task(
    *,
    task_id: int,
    particle_outputs: Sequence[Sequence[str]],
    query_indices: Sequence[int],
    holdout_indices: Sequence[int],
    target_outputs: Sequence[str],
) -> dict[str, Any]:
    holdout_scores = [
        sum(outputs[index] == target_outputs[index] for index in holdout_indices)
        / len(holdout_indices)
        for outputs in particle_outputs
    ]
    initial_endpoint = statistics.fmean(holdout_scores)
    query_rows = []
    for query_index in query_indices:
        outcomes = [outputs[query_index] for outputs in particle_outputs]
        survivors = [
            index
            for index, output in enumerate(outcomes)
            if output == target_outputs[query_index]
        ]
        endpoint = (
            statistics.fmean(holdout_scores[index] for index in survivors)
            if survivors
            else 0.0
        )
        query_rows.append(
            {
                "test_index": query_index,
                "immediate_eig": entropy_from_outcomes(outcomes),
                "unique_outcomes": len(set(outcomes)),
                "survivor_count": len(survivors),
                "posterior_holdout_pass_fraction": endpoint,
            }
        )
    selected_position = max(
        range(len(query_rows)),
        key=lambda index: (query_rows[index]["immediate_eig"], -index),
    )
    oracle_position = max(
        range(len(query_rows)),
        key=lambda index: (
            query_rows[index]["posterior_holdout_pass_fraction"],
            -index,
        ),
    )
    endpoints = [
        row["posterior_holdout_pass_fraction"] for row in query_rows
    ]
    eigs = [row["immediate_eig"] for row in query_rows]
    selected_endpoint = endpoints[selected_position]
    oracle_endpoint = endpoints[oracle_position]
    return {
        "task_id": task_id,
        "valid_particle_count": len(particle_outputs),
        "query_count": len(query_indices),
        "holdout_count": len(holdout_indices),
        "initial_holdout_pass_fraction": initial_endpoint,
        "informative_query_count": sum(value >= 0.30 for value in eigs),
        "endpoint_range": max(endpoints) - min(endpoints),
        "myopic_test_index": query_rows[selected_position]["test_index"],
        "oracle_test_index": query_rows[oracle_position]["test_index"],
        "myopic_endpoint": selected_endpoint,
        "oracle_endpoint": oracle_endpoint,
        "oracle_gain_over_initial": oracle_endpoint - initial_endpoint,
        "oracle_gap_over_myopic": oracle_endpoint - selected_endpoint,
        "eig_endpoint_spearman": spearman(eigs, endpoints),
        "queries": query_rows,
    }


def _usage_snapshot(model: ChatModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "adapter_cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
        "generator": snapshot,
    }


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_opportunity(
    config: Config,
    *,
    orchid_root: Path,
    atd_root: Path,
    raw_path: Path,
    model_adapter: ChatModel,
) -> dict[str, Any]:
    dataset_path = verify_sources(orchid_root, atd_root)
    metadata = _target_blind_task_metadata(dataset_path)
    messages = [
        _particle_messages(
            task_id=task_id,
            ambiguous_prompt=metadata[task_id]["ambiguous_prompt"],
            entry_point=metadata[task_id]["entry_point"],
            sample_index=sample_index,
        )
        for task_id in TASK_IDS
        for sample_index in range(PARTICLE_COUNT)
    ]
    raw: dict[str, Any] = {"task_ids": list(TASK_IDS)}
    try:
        responses = model_adapter.chat_complete_messages_batched(
            messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["responses"] = responses
        valid_outputs: dict[int, list[list[str]]] = {}
        parse_errors: dict[int, list[dict[str, str]]] = {}
        for task_offset, task_id in enumerate(TASK_IDS):
            task_outputs = []
            errors = []
            task = metadata[task_id]
            calls = [
                f"{task['entry_point']}({input_text})"
                for input_text in task["inputs"]
            ]
            start = task_offset * PARTICLE_COUNT
            for sample_index, response in enumerate(
                responses[start : start + PARTICLE_COUNT]
            ):
                try:
                    program = extract_python_program(
                        response, task["entry_point"]
                    )
                    outputs = evaluate_outputs(
                        atd_root,
                        program,
                        calls,
                        timeout=EXECUTION_TIMEOUT_SECONDS,
                    )
                    if len(outputs) != len(calls) or any(
                        output in {"error", "timed out"} for output in outputs
                    ):
                        raise ValueError("program failed at least one frozen input")
                    task_outputs.append(outputs)
                except Exception as exc:
                    errors.append(
                        {
                            "sample_index": str(sample_index),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            if len(task_outputs) < MIN_VALID_PARTICLES:
                raise ValueError(
                    f"task {task_id} has {len(task_outputs)} valid particles; "
                    f"{MIN_VALID_PARTICLES} required"
                )
            valid_outputs[task_id] = task_outputs
            parse_errors[task_id] = errors
        raw["target_blind_execution_complete"] = True
        _checkpoint(raw_path, raw)
        usage = _usage_snapshot(model_adapter)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise OpportunityExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model_adapter),
        ) from exc

    # Output partitions are complete before target values are loaded.
    target_blind_eigs = {
        task_id: [
            entropy_from_outcomes(
                outputs[query_index] for outputs in valid_outputs[task_id]
            )
            for query_index in metadata[task_id]["query_indices"]
        ]
        for task_id in TASK_IDS
    }
    target_outputs = _load_target_outputs(dataset_path)
    task_records = [
        score_task(
            task_id=task_id,
            particle_outputs=valid_outputs[task_id],
            query_indices=metadata[task_id]["query_indices"],
            holdout_indices=metadata[task_id]["holdout_indices"],
            target_outputs=target_outputs[task_id],
        )
        for task_id in TASK_IDS
    ]
    for task in task_records:
        task["invalid_particles"] = parse_errors[task["task_id"]]
        expected_eigs = target_blind_eigs[task["task_id"]]
        observed_eigs = [row["immediate_eig"] for row in task["queries"]]
        if observed_eigs != expected_eigs:
            raise AssertionError("target loading changed a frozen EIG score")

    generator = usage["generator"]
    mean_oracle_gap = statistics.fmean(
        task["oracle_gap_over_myopic"] for task in task_records
    )
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": int(generator.get("http_attempts", -1))
        == EXPECTED_REQUESTS,
        "zero_transport_retries": int(generator.get("retry_count", -1)) == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": int(generator.get("forced_exits", -1)) == 0,
        "at_least_six_valid_particles_each": all(
            task["valid_particle_count"] >= MIN_VALID_PARTICLES
            for task in task_records
        ),
        "initial_support_unsaturated_each": all(
            0.10 <= task["initial_holdout_pass_fraction"] <= 0.90
            for task in task_records
        ),
        "at_least_two_informative_queries_each": all(
            task["informative_query_count"] >= 2 for task in task_records
        ),
        "endpoint_range_at_least_0_10_each": all(
            task["endpoint_range"] >= 0.10 for task in task_records
        ),
        "oracle_gain_at_least_0_10_each": all(
            task["oracle_gain_over_initial"] >= 0.10 for task in task_records
        ),
        "oracle_root_differs_from_myopic_at_least_one": any(
            task["oracle_test_index"] != task["myopic_test_index"]
            for task in task_records
        ),
        "mean_oracle_gap_over_myopic_at_least_0_05": mean_oracle_gap >= 0.05,
        "cost_at_most_0_50": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "orchid_repository": ORCHID_REPOSITORY,
            "orchid_commit": ORCHID_COMMIT,
            "orchid_heval_sha256": ORCHID_HEVAL_SHA256,
            "atd_repository": ATD_REPOSITORY,
            "atd_commit": ATD_COMMIT,
            "eligible_manifest_sha256": ELIGIBLE_MANIFEST_SHA256,
            "model": MODEL_ID,
            "seed": SEED,
            "split_seed": SPLIT_SEED,
            "task_ids": list(TASK_IDS),
            "particle_count": PARTICLE_COUNT,
            "expected_requests": EXPECTED_REQUESTS,
            "ambiguity_variant": "Vagueness_prompt",
            "reasoning_requested": False,
            "test_outputs_loaded_after_all_particle_partitions_and_eig_scores": True,
            "branch_regeneration_calls": 0,
            "repairs_or_reissues": 0,
        },
        "metrics": {
            "task_count": len(task_records),
            "mean_initial_holdout_pass_fraction": statistics.fmean(
                task["initial_holdout_pass_fraction"] for task in task_records
            ),
            "mean_myopic_endpoint": statistics.fmean(
                task["myopic_endpoint"] for task in task_records
            ),
            "mean_oracle_endpoint": statistics.fmean(
                task["oracle_endpoint"] for task in task_records
            ),
            "mean_oracle_gap_over_myopic": mean_oracle_gap,
        },
        "tasks": task_records,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses = []
        for messages in batch_messages:
            request = json.loads(messages[-1]["content"])
            entry_point = request["required_entry_point"]
            responses.append(
                f"def {entry_point}(*args, **kwargs):\n"
                "    return args[0] if args else None\n"
            )
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
            "http_attempts": self.requests,
            "retry_count": 0,
            "forced_exits": 0,
        }


def _build_model(config: Config) -> ChatModel:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("Orchid opportunity config selects the wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--orchid-root", type=Path, required=True)
    parser.add_argument("--atd-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.15
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = EXPECTED_REQUESTS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
        payload = run_opportunity(
            config,
            orchid_root=args.orchid_root,
            atd_root=args.atd_root,
            raw_path=raw_path,
            model_adapter=model,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, OpportunityExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "OPPORTUNITY_FAILURE.json", failure)
        raise
    output = args.output_dir / "OPPORTUNITY.json"
    _checkpoint(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "metrics": payload["metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
