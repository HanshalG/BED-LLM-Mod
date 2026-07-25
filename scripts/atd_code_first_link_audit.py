#!/usr/bin/env python3
"""Audit whether code-hypothesis EIG ranks external program correctness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


INTERFACE_VERSION = "atd-code-first-link-audit-1"
ATD_COMMIT = "4c8ecb4d4ffdbffcc611366743fc1e2461037772"
HUMANEVAL_SHA256 = "882c3d56432b2b5b9e568398d7ebdf54f2c84fdb05fef3b833a3d935ad71861c"
EXPECTED_TASK_IDS = (
    5,
    6,
    17,
    26,
    33,
    36,
    38,
    39,
    41,
    50,
    54,
    55,
    64,
    70,
    73,
    74,
    76,
    77,
    81,
    82,
    90,
    91,
    93,
    95,
    96,
    98,
    101,
    103,
    106,
    107,
    109,
    110,
    111,
    114,
    115,
    118,
    121,
    122,
    123,
    134,
    138,
    139,
    141,
    143,
    147,
    154,
    159,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _program_string(code_sample: str, entry_point: str, prompt: str) -> str:
    if "```python" in code_sample:
        code_sample = code_sample.split("```python", 1)[1].split("```", 1)[0]
    marker = f"def {entry_point}"
    marker_index = code_sample.find(marker)
    if marker_index >= 0:
        signature_end = code_sample.find("\n", marker_index)
        if signature_end >= 0:
            code_sample = code_sample[signature_end + 1 :]
    return prompt + code_sample


def entropy_from_outcomes(outcomes: Iterable[str]) -> float:
    values = list(outcomes)
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in ordered[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    ranks_a = _average_ranks(values_a)
    ranks_b = _average_ranks(values_b)
    mean_a = statistics.fmean(ranks_a)
    mean_b = statistics.fmean(ranks_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    denominator_a = sum((a - mean_a) ** 2 for a in ranks_a)
    denominator_b = sum((b - mean_b) ** 2 for b in ranks_b)
    if denominator_a <= 0.0 or denominator_b <= 0.0:
        return None
    return numerator / math.sqrt(denominator_a * denominator_b)


def _prepare_execution_module(atd_root: str) -> Any:
    code_root = str(Path(atd_root) / "src" / "code-generation")
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    import _execution  # type: ignore[import-not-found]

    return _execution


def _output_worker(
    queue: Any,
    atd_root: str,
    program: str,
    questions: list[str],
    timeout: float,
) -> None:
    execution = _prepare_execution_module(atd_root)
    execution.reliability_guard()
    outputs = execution.get_outputs(program, questions, timeout)
    queue.put([str(output) for output in outputs])


def _test_worker(
    queue: Any,
    atd_root: str,
    program: str,
    test: str,
    entry_point: str,
    timeout: float,
) -> None:
    execution = _prepare_execution_module(atd_root)
    execution.reliability_guard()
    queue.put(execution.run_test_case(program, test, entry_point, timeout))


def _run_isolated(
    target: Any,
    args: tuple[Any, ...],
    *,
    wall_timeout: float,
    fallback: Any,
) -> Any:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=target, args=(queue, *args))
    process.start()
    process.join(wall_timeout)
    if process.is_alive():
        process.kill()
        process.join()
        return fallback
    if queue.empty():
        return fallback
    return queue.get()


def evaluate_outputs(
    atd_root: Path,
    program: str,
    questions: list[str],
    *,
    timeout: float,
) -> list[str]:
    fallback = ["timed out"] * len(questions)
    return _run_isolated(
        _output_worker,
        (str(atd_root), program, questions, timeout),
        wall_timeout=max(2.0, timeout * len(questions) + 1.0),
        fallback=fallback,
    )


def evaluate_hidden_test(
    atd_root: Path,
    program: str,
    test: str,
    entry_point: str,
    *,
    timeout: float,
) -> bool:
    result = _run_isolated(
        _test_worker,
        (str(atd_root), program, test, entry_point, timeout),
        wall_timeout=max(2.0, timeout + 1.0),
        fallback=False,
    )
    return result is True


def _load_tasks(dataset_path: Path) -> dict[int, dict[str, Any]]:
    tasks: dict[int, dict[str, Any]] = {}
    with dataset_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            task_id = int(str(row["task_id"]).split("/")[-1])
            tasks[task_id] = row
    return tasks


def _trace_paths(atd_root: Path) -> dict[int, Path]:
    base = atd_root / "results" / "code-generation" / "HumanEval"
    paths: dict[int, Path] = {}
    for path in base.glob(
        "*/active-reasoning/gpt-4o-mini/iter_0/listed_hypothesis.json"
    ):
        task_id = int(path.parts[-5])
        run_dir = path.parent
        if (run_dir / "questions.json").exists():
            paths[task_id] = run_dir
    return paths


def _verify_source(atd_root: Path) -> tuple[Path, dict[int, Path]]:
    commit = subprocess.check_output(
        ["git", "-C", str(atd_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != ATD_COMMIT:
        raise ValueError(f"Active Task Disambiguation commit is {commit}, expected {ATD_COMMIT}")
    dataset_path = (
        atd_root / "data" / "code-generation" / "HumanEval_for_code_generation.jsonl"
    )
    digest = _sha256(dataset_path)
    if digest != HUMANEVAL_SHA256:
        raise ValueError(f"HumanEval data hash is {digest}, expected {HUMANEVAL_SHA256}")
    trace_paths = _trace_paths(atd_root)
    if tuple(sorted(trace_paths)) != EXPECTED_TASK_IDS:
        raise ValueError("complete released GPT-4o-mini trace IDs do not match the frozen manifest")
    return dataset_path, trace_paths


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else float("nan")


def score_task(
    task_id: int,
    questions: list[str],
    particle_outputs: list[list[str]],
    canonical_outputs: list[str],
    particle_correct: list[bool],
) -> dict[str, Any]:
    if not particle_outputs or not questions:
        return {"task_id": task_id, "usable": False, "failure": "empty support or query set"}
    if any(len(outputs) != len(questions) for outputs in particle_outputs):
        return {"task_id": task_id, "usable": False, "failure": "output width mismatch"}

    initial_pass = sum(particle_correct) / len(particle_correct)
    rows = []
    for query_index, query in enumerate(questions):
        outcomes = [outputs[query_index] for outputs in particle_outputs]
        true_output = canonical_outputs[query_index]
        survivors = [
            particle_index
            for particle_index, output in enumerate(outcomes)
            if output == true_output
        ]
        pass_fraction = (
            sum(particle_correct[index] for index in survivors) / len(survivors)
            if survivors
            else 0.0
        )
        rows.append(
            {
                "query": query,
                "true_output": true_output,
                "immediate_eig": entropy_from_outcomes(outcomes),
                "unique_outcomes": len(set(outcomes)),
                "survivor_count": len(survivors),
                "posterior_pass_fraction": pass_fraction,
            }
        )

    selected_index = max(
        range(len(rows)),
        key=lambda index: (rows[index]["immediate_eig"], rows[index]["query"]),
    )
    oracle_index = max(
        range(len(rows)),
        key=lambda index: (rows[index]["posterior_pass_fraction"], rows[index]["query"]),
    )
    selected_pass = rows[selected_index]["posterior_pass_fraction"]
    oracle_pass = rows[oracle_index]["posterior_pass_fraction"]
    candidate_mean = _mean(row["posterior_pass_fraction"] for row in rows)
    rho = spearman(
        [row["immediate_eig"] for row in rows],
        [row["posterior_pass_fraction"] for row in rows],
    )
    return {
        "task_id": task_id,
        "usable": len(particle_correct) >= 8 and len(rows) >= 3,
        "particle_count": len(particle_correct),
        "query_count": len(rows),
        "initial_pass_fraction": initial_pass,
        "selected_query_index": selected_index,
        "oracle_query_index": oracle_index,
        "selected_pass_fraction": selected_pass,
        "oracle_pass_fraction": oracle_pass,
        "candidate_mean_pass_fraction": candidate_mean,
        "selected_gain_over_initial": selected_pass - initial_pass,
        "selected_gain_over_candidate_mean": selected_pass - candidate_mean,
        "top1_regret": oracle_pass - selected_pass,
        "pass_fraction_range": max(
            row["posterior_pass_fraction"] for row in rows
        )
        - min(row["posterior_pass_fraction"] for row in rows),
        "spearman_eig_vs_pass": rho,
        "queries": rows,
    }


def summarize(task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in task_rows if row.get("usable")]
    finite_rhos = [
        row["spearman_eig_vs_pass"]
        for row in usable
        if row["spearman_eig_vs_pass"] is not None
    ]
    metrics = {
        "task_count": len(task_rows),
        "usable_task_count": len(usable),
        "mean_initial_pass_fraction": _mean(
            row["initial_pass_fraction"] for row in usable
        ),
        "unsaturated_task_count": sum(
            0.02 <= row["initial_pass_fraction"] <= 0.95 for row in usable
        ),
        "dynamic_range_task_count": sum(
            row["pass_fraction_range"] >= 0.10 for row in usable
        ),
        "mean_selected_gain_over_initial": _mean(
            row["selected_gain_over_initial"] for row in usable
        ),
        "mean_selected_gain_over_candidate_mean": _mean(
            row["selected_gain_over_candidate_mean"] for row in usable
        ),
        "mean_top1_regret": _mean(row["top1_regret"] for row in usable),
        "rho_task_count": len(finite_rhos),
        "mean_within_task_spearman": _mean(finite_rhos),
        "positive_rho_task_count": sum(rho > 0 for rho in finite_rhos),
    }
    gates = {
        "usable_tasks": metrics["usable_task_count"] >= 35,
        "unsaturated_tasks": metrics["unsaturated_task_count"] >= 25,
        "dynamic_range_tasks": metrics["dynamic_range_task_count"] >= 25,
        "selected_gain_over_initial": metrics["mean_selected_gain_over_initial"] >= 0.05,
        "selected_gain_over_candidate_mean": (
            metrics["mean_selected_gain_over_candidate_mean"] >= 0.02
        ),
        "mean_within_task_spearman": metrics["mean_within_task_spearman"] >= 0.10,
        "mean_top1_regret": metrics["mean_top1_regret"] <= 0.15,
    }
    return {
        "metrics": metrics,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }


def run_audit(
    *,
    atd_root: Path,
    output_path: Path,
    output_timeout: float,
    test_timeout: float,
) -> dict[str, Any]:
    dataset_path, trace_paths = _verify_source(atd_root)
    tasks = _load_tasks(dataset_path)

    prepared: dict[int, dict[str, Any]] = {}
    for task_id in EXPECTED_TASK_IDS:
        task = tasks[task_id]
        run_dir = trace_paths[task_id]
        hypotheses = json.loads((run_dir / "listed_hypothesis.json").read_text())["0"]
        questions = json.loads((run_dir / "questions.json").read_text())["0"]
        particle_programs = [
            _program_string(row["content"], task["entry_point"], task["prompt"])
            for row in hypotheses
        ]
        canonical_program = _program_string(
            task["canonical_solution"], task["entry_point"], task["prompt"]
        )
        canonical_outputs = evaluate_outputs(
            atd_root,
            canonical_program,
            questions,
            timeout=output_timeout,
        )
        particle_outputs = [
            evaluate_outputs(atd_root, program, questions, timeout=output_timeout)
            for program in particle_programs
        ]
        prepared[task_id] = {
            "task": task,
            "questions": questions,
            "particle_programs": particle_programs,
            "canonical_outputs": canonical_outputs,
            "particle_outputs": particle_outputs,
        }
        print(f"scored target-blind query partitions for task {task_id}", flush=True)

    # Hidden HumanEval tests are evaluated only after every EIG partition is frozen.
    task_rows = []
    for task_id in EXPECTED_TASK_IDS:
        item = prepared[task_id]
        task = item["task"]
        particle_correct = [
            evaluate_hidden_test(
                atd_root,
                program,
                task["test"],
                task["entry_point"],
                timeout=test_timeout,
            )
            for program in item["particle_programs"]
        ]
        task_rows.append(
            score_task(
                task_id,
                item["questions"],
                item["particle_outputs"],
                item["canonical_outputs"],
                particle_correct,
            )
        )
        print(f"evaluated hidden tests for task {task_id}", flush=True)

    summary = summarize(task_rows)
    artifact = {
        "interface_version": INTERFACE_VERSION,
        "source": {
            "repository": "https://github.com/kasia-kobalczyk/active-task-disambiguation",
            "commit": ATD_COMMIT,
            "dataset_sha256": HUMANEVAL_SHA256,
            "released_model": "gpt-4o-mini",
            "released_strategy": "active-reasoning",
            "released_iteration": 0,
            "task_ids": list(EXPECTED_TASK_IDS),
        },
        "execution": {
            "openrouter_calls": 0,
            "reasoning_tokens": 0,
            "output_timeout_seconds": output_timeout,
            "hidden_test_timeout_seconds": test_timeout,
            "particle_semantics": "released completion multiplicity retained",
            "error_semantics": "error and timeout retained as outcomes",
        },
        **summary,
        "tasks": task_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atd-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nonmyopic/atd_code_first_link_audit/AUDIT.json"),
    )
    parser.add_argument("--output-timeout", type=float, default=0.2)
    parser.add_argument("--test-timeout", type=float, default=0.5)
    args = parser.parse_args()
    artifact = run_audit(
        atd_root=args.atd_root.resolve(),
        output_path=args.output.resolve(),
        output_timeout=args.output_timeout,
        test_timeout=args.test_timeout,
    )
    print(json.dumps({"metrics": artifact["metrics"], "gates": artifact["gates"]}, indent=2))


if __name__ == "__main__":
    main()
