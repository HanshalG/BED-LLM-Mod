#!/usr/bin/env python3
"""Run the frozen untouched BrowseComp terminal semantic development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts import browsecomp_plus_cached_semantic_information as cached
from scripts import browsecomp_plus_semantic_bed_manifest as parent_manifest
from scripts import browsecomp_plus_semantic_mechanics as mechanics
from scripts import browsecomp_plus_terminal_semantic_information_smoke as smoke
from scripts import extract_browsecomp_plus_open_mechanics as decryptor


INTERFACE_VERSION = "browsecomp-plus-terminal-semantic-development-1"
DEVELOPMENT_IDS = (
    "1161",
    "193",
    "1155",
    "418",
    "6",
    "98",
    "237",
    "314",
    "261",
    "562",
)
DEVELOPMENT_ORDERED_SHA256 = (
    "63ac5756a4cb90c64b29bd45b41510ed1733913b8cd890951f67eed9b4451216"
)
DECRYPTED_SOURCE_SHA256 = (
    "44a4ef3520e7398a249df073b1ff1e8ff837ee2e97afa88f61b44ead01c4461e"
)
PARENT_MANIFEST_SHA256 = decryptor.MANIFEST_SHA256
EXPECTED_REQUESTS = 130
MAX_COST_USD = 1.00
PROJECTED_COST_USD = 0.78
RANDOM_SEED = 24_409
TEMPERATURE = 0.0


class DevelopmentExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _source_hash(tasks: Sequence[Mapping[str, Any]]) -> str:
    encoded = (
        "\n".join(
            json.dumps(task, ensure_ascii=False, sort_keys=True)
            for task in tasks
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_development_tasks(
    manifest_path: Path,
    parquet_dir: Path,
) -> list[dict[str, Any]]:
    if mechanics.sha256_file(manifest_path) != PARENT_MANIFEST_SHA256:
        raise ValueError("BrowseComp-Plus parent manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = tuple(
        str(value)
        for value in manifest["splits"]["development"]["task_ids"][:10]
    )
    if selected != DEVELOPMENT_IDS:
        raise ValueError("development selection changed")
    if _ordered_hash(selected) != DEVELOPMENT_ORDERED_SHA256:
        raise ValueError("development selection hash changed")

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read official shards") from exc

    encrypted_rows: list[dict[str, Any]] = []
    for name, expected_hash in parent_manifest.PARQUET_SHA256.items():
        path = parquet_dir / name
        if mechanics.sha256_file(path) != expected_hash:
            raise ValueError(f"{name} hash changed")
        table = parquet.read_table(
            path,
            filters=[("query_id", "in", list(DEVELOPMENT_IDS))],
        )
        encrypted_rows.extend(table.to_pylist())
    row_by_id = {
        str(row["query_id"]): row for row in encrypted_rows
    }
    if set(row_by_id) != set(DEVELOPMENT_IDS):
        raise ValueError("development IDs did not reproduce")
    return [
        decryptor.decrypt_value(row_by_id[query_id])
        for query_id in DEVELOPMENT_IDS
    ]


def load_decrypted_source(path: Path) -> list[dict[str, Any]]:
    if mechanics.sha256_file(path) != DECRYPTED_SOURCE_SHA256:
        raise ValueError("decrypted development source hash changed")
    tasks = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if tuple(str(task["query_id"]) for task in tasks) != DEVELOPMENT_IDS:
        raise ValueError("decrypted development task order changed")
    if _source_hash(tasks) != DECRYPTED_SOURCE_SHA256:
        raise ValueError("decrypted development canonical hash changed")
    return tasks


def initial_messages(task: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Act as open-world semantic belief and experimental-design "
                "machinery for a hard factual research question. Generate "
                "eight distinct candidate final answers with positive integer "
                "weights summing exactly to 100. Generate six diverse first "
                "search queries that could reveal different evidence paths. "
                "Do not score the queries and do not reveal reasoning. Return "
                "only eight ordered H lines and six ordered Q lines."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": task["query"],
                    "exact_output_lines": (
                        mechanics._belief_grammar()
                        + [
                            f"Q{index:02d}|distinct first search query"
                            for index in range(1, mechanics.ROOT_COUNT + 1)
                        ]
                    ),
                },
                separators=(",", ":"),
            ),
        },
    ]


def parse_initial(text: str) -> tuple[mechanics.Belief, list[str]]:
    lines = mechanics._response_lines(text)
    expected = mechanics.HYPOTHESIS_COUNT + mechanics.ROOT_COUNT
    if len(lines) != expected:
        raise ValueError("initial response has wrong line count")
    belief = mechanics._parse_belief(
        lines[: mechanics.HYPOTHESIS_COUNT]
    )
    queries = []
    for index, line in enumerate(
        lines[mechanics.HYPOTHESIS_COUNT :],
        start=1,
    ):
        parts = line.split("|")
        if len(parts) != 2 or parts[0] != f"Q{index:02d}":
            raise ValueError("invalid root query line")
        queries.append(
            mechanics._clean_text_field(
                parts[1],
                maximum=mechanics.QUERY_MAX_CHARS,
            )
        )
    if len({mechanics.normalize_query(query) for query in queries}) != len(
        queries
    ):
        raise ValueError("root queries must be normalized-distinct")
    return belief, queries


def refresh_messages(
    task: Mapping[str, Any],
    *,
    initial_belief: mechanics.Belief,
    root_query: str,
    root_documents: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Regenerate the open-world answer belief after the first "
                "search observation. Preserve, revise, add, or drop candidate "
                "answers based on the returned text. Then choose one adaptive "
                "second search query targeting the most useful unresolved "
                "evidence made available by this observation. Use positive "
                "integer weights summing exactly to 100. Do not score either "
                "query and do not reveal reasoning. Return only eight ordered "
                "H lines and one A01 line."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": task["query"],
                    "prior_belief": mechanics._belief_payload(
                        initial_belief
                    ),
                    "first_query": root_query,
                    "first_search_results": mechanics._document_payload(
                        root_documents
                    ),
                    "exact_output_lines": (
                        mechanics._belief_grammar()
                        + ["A01|adaptive second query"]
                    ),
                },
                separators=(",", ":"),
            ),
        },
    ]


def run_model_stage(
    config: Config,
    *,
    manifest_path: Path,
    parquet_dir: Path,
    raw_path: Path,
    source_path: Path | None = None,
    model_adapter: Any | None = None,
    tasks_override: Sequence[dict[str, Any]] | None = None,
) -> tuple[
    dict[str, Any],
    list[list[smoke.MassBelief]],
    dict[str, Any],
]:
    tasks = (
        list(tasks_override)
        if tasks_override is not None
        else (
            load_decrypted_source(source_path)
            if source_path is not None
            else load_development_tasks(manifest_path, parquet_dir)
        )
    )
    if tuple(str(task["query_id"]) for task in tasks) != DEVELOPMENT_IDS:
        raise ValueError("development task order changed")
    source_hash = _source_hash(tasks)
    retrievers = [
        mechanics.TaskBM25(mechanics._task_documents(task))
        for task in tasks
    ]
    model = (
        model_adapter
        if model_adapter is not None
        else mechanics._build_model(config)
    )
    raw: dict[str, Any] = {
        "task_ids": list(DEVELOPMENT_IDS),
        "decrypted_source_sha256": source_hash,
    }
    try:
        initial_responses = mechanics._complete(
            model,
            [initial_messages(task) for task in tasks],
            config,
        )
        raw["initial_responses"] = initial_responses
        mechanics._checkpoint(raw_path, raw)
        initials = [
            parse_initial(response) for response in initial_responses
        ]

        branches_by_task: list[list[mechanics.Branch]] = []
        for retriever, (_, queries) in zip(
            retrievers,
            initials,
            strict=True,
        ):
            branches_by_task.append(
                [
                    mechanics.Branch(
                        root_index=root_index,
                        strategy=mechanics.Strategy(
                            root_query=query,
                            direct_score=0,
                            future_intent="",
                        ),
                        root_documents=retriever.search(query),
                    )
                    for root_index, query in enumerate(queries)
                ]
            )

        specs = [
            (task_index, root_index)
            for task_index in range(len(tasks))
            for root_index in range(mechanics.ROOT_COUNT)
        ]
        refresh_responses = mechanics._complete(
            model,
            [
                refresh_messages(
                    tasks[task_index],
                    initial_belief=initials[task_index][0],
                    root_query=branches_by_task[task_index][
                        root_index
                    ].strategy.root_query,
                    root_documents=branches_by_task[task_index][
                        root_index
                    ].root_documents,
                )
                for task_index, root_index in specs
            ],
            config,
        )
        raw["refresh_responses"] = refresh_responses
        mechanics._checkpoint(raw_path, raw)
        for response, (task_index, root_index) in zip(
            refresh_responses,
            specs,
            strict=True,
        ):
            branch = branches_by_task[task_index][root_index]
            branch.root_belief, branch.adaptive_query = (
                mechanics.parse_refresh(response)
            )
            branch.adaptive_documents = retrievers[task_index].search(
                branch.adaptive_query
            )

        terminal_responses = mechanics._complete(
            model,
            [
                smoke.terminal_mass_messages(
                    tasks[task_index],
                    initial_belief=initials[task_index][0],
                    branch=branches_by_task[task_index][root_index],
                )
                for task_index, root_index in specs
            ],
            config,
        )
        raw["terminal_responses"] = terminal_responses
        mechanics._checkpoint(raw_path, raw)
        parsed_terminals = [
            smoke.parse_terminal_masses(response)
            for response in terminal_responses
        ]
        terminals: list[list[smoke.MassBelief]] = [
            [] for _ in tasks
        ]
        for terminal, (task_index, _) in zip(
            parsed_terminals,
            specs,
            strict=True,
        ):
            terminals[task_index].append(terminal)
        usage = mechanics._usage_snapshot(model)
    except Exception as exc:
        mechanics._checkpoint(raw_path, raw)
        raise DevelopmentExecutionError(
            f"{type(exc).__name__}: {exc}",
            mechanics._usage_snapshot(model),
        ) from exc

    reconstructed = {
        "tasks": tasks,
        "initials": [(belief, []) for belief, _ in initials],
        "branches": branches_by_task,
        "decrypted_source_sha256": source_hash,
    }
    return reconstructed, terminals, usage


def _mean_difference(
    selections: Sequence[dict[str, Any]],
    left_key: str,
    right_key: str,
) -> float:
    return sum(
        selection[left_key] - selection[right_key]
        for selection in selections
    ) / len(selections)


def _counts(
    selections: Sequence[dict[str, Any]],
    left_key: str,
    right_key: str,
) -> dict[str, int]:
    wins = sum(
        selection[left_key] > selection[right_key]
        for selection in selections
    )
    losses = sum(
        selection[left_key] < selection[right_key]
        for selection in selections
    )
    return {
        "wins": wins,
        "losses": losses,
        "ties": len(selections) - wins - losses,
    }


def analyze(
    reconstructed: dict[str, Any],
    terminals: Sequence[Sequence[smoke.MassBelief]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    base = smoke.analyze(reconstructed, terminals, usage)
    records = base["records"]
    selections = base["selections"]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_task.setdefault(record["task_id"], []).append(record)

    changed_beliefs = 0
    adaptive_differences = 0
    for task_index, branches in enumerate(reconstructed["branches"]):
        initial = reconstructed["initials"][task_index][0]
        for branch in branches:
            assert branch.root_belief is not None
            assert branch.adaptive_query is not None
            changed_beliefs += (
                branch.root_belief.signature != initial.signature
            )
            adaptive_differences += (
                mechanics.normalize_query(branch.adaptive_query)
                != mechanics.normalize_query(branch.strategy.root_query)
            )

    for selection in selections:
        rows = by_task[selection["task_id"]]
        random_root = (
            RANDOM_SEED + int(selection["task_id"])
        ) % mechanics.ROOT_COUNT
        myopic = rows[selection["myopic_root"]]
        d2 = rows[selection["d2_root"]]
        random_record = rows[random_root]
        selection.update(
            {
                "random_root": random_root,
                "myopic_total_gold": myopic["total_gold"],
                "d2_total_gold": d2["total_gold"],
                "random_terminal_truth_mass": random_record[
                    "terminal_truth_mass"
                ],
                "random_total_evidence": random_record[
                    "total_evidence"
                ],
                "random_total_gold": random_record["total_gold"],
            }
        )

    pairwise = base["summary"]["pairwise"]
    truth_counts = _counts(
        selections,
        "d2_terminal_truth_mass",
        "myopic_terminal_truth_mass",
    )
    evidence_counts = _counts(
        selections,
        "d2_total_evidence",
        "myopic_total_evidence",
    )
    gold_counts = _counts(
        selections,
        "d2_total_gold",
        "myopic_total_gold",
    )
    d2_myopic_truth = _mean_difference(
        selections,
        "d2_terminal_truth_mass",
        "myopic_terminal_truth_mass",
    )
    d2_myopic_evidence = _mean_difference(
        selections,
        "d2_total_evidence",
        "myopic_total_evidence",
    )
    d2_myopic_gold = _mean_difference(
        selections,
        "d2_total_gold",
        "myopic_total_gold",
    )
    d2_random_truth = _mean_difference(
        selections,
        "d2_terminal_truth_mass",
        "random_terminal_truth_mass",
    )
    d2_random_evidence = _mean_difference(
        selections,
        "d2_total_evidence",
        "random_total_evidence",
    )
    root_differences = sum(
        selection["d2_root"] != selection["myopic_root"]
        for selection in selections
    )
    gates = {
        "exact_130_requests": (
            usage["physical_requests"] == EXPECTED_REQUESTS
            and usage["http_attempts"] == EXPECTED_REQUESTS
            and len(records) == 60
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_1": usage["adapter_cost_usd"] <= MAX_COST_USD,
        "changed_root_beliefs_at_least_50": changed_beliefs >= 50,
        "adaptive_queries_differ_at_least_50": (
            adaptive_differences >= 50
        ),
        "total_entropy_dynamic_on_at_least_8_tasks": (
            base["summary"]["dynamic_task_count"] >= 8
        ),
        "total_truth_pairs_at_least_20": (
            pairwise["total_entropy_vs_total_truth"]["pairs"] >= 20
        ),
        "total_evidence_pairs_at_least_60": (
            pairwise["total_entropy_vs_total_evidence"]["pairs"] >= 60
        ),
        "total_entropy_truth_accuracy_at_least_0_60": (
            pairwise["total_entropy_vs_total_truth"]["accuracy"] >= 0.60
        ),
        "total_entropy_evidence_accuracy_at_least_0_58": (
            pairwise["total_entropy_vs_total_evidence"]["accuracy"] >= 0.58
        ),
        "total_entropy_gold_accuracy_at_least_0_55": (
            pairwise["total_entropy_vs_total_gold"]["accuracy"] >= 0.55
        ),
        "future_entropy_evidence_accuracy_at_least_0_55": (
            pairwise["future_entropy_vs_future_evidence"]["accuracy"] >= 0.55
        ),
        "d2_root_differs_on_at_least_3_tasks": root_differences >= 3,
        "d2_myopic_truth_mean_positive": d2_myopic_truth > 0.0,
        "d2_myopic_truth_wins_at_least_2": truth_counts["wins"] >= 2,
        "d2_myopic_truth_losses_at_most_2": truth_counts["losses"] <= 2,
        "d2_myopic_evidence_mean_positive": d2_myopic_evidence > 0.0,
        "d2_myopic_evidence_wins_at_least_3": (
            evidence_counts["wins"] >= 3
        ),
        "d2_myopic_evidence_losses_at_most_2": (
            evidence_counts["losses"] <= 2
        ),
        "d2_myopic_gold_mean_nonnegative": d2_myopic_gold >= 0.0,
        "d2_myopic_gold_losses_at_most_2": gold_counts["losses"] <= 2,
        "d2_random_truth_mean_positive": d2_random_truth > 0.0,
        "d2_random_evidence_mean_positive": d2_random_evidence > 0.0,
    }
    return {
        "schema_version": 1,
        "status": (
            "untouched_development_pass"
            if all(gates.values())
            else "untouched_development_failure"
        ),
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "task_ids": list(DEVELOPMENT_IDS),
            "ordered_sha256": DEVELOPMENT_ORDERED_SHA256,
            "decrypted_source_sha256": reconstructed[
                "decrypted_source_sha256"
            ],
            "model": mechanics.MODEL_ID,
            "temperature": TEMPERATURE,
            "expected_requests": EXPECTED_REQUESTS,
            "projected_cost_usd": PROJECTED_COST_USD,
            "max_cost_usd": MAX_COST_USD,
            "development_tasks_were_untouched_before_run": True,
            "holdout_authorized": False,
            "oatml_used": False,
        },
        "summary": {
            "task_count": len(selections),
            "root_count": len(records),
            "changed_root_belief_count": changed_beliefs,
            "adaptive_query_difference_count": adaptive_differences,
            "dynamic_task_count": base["summary"]["dynamic_task_count"],
            "pairwise": pairwise,
            "d2_root_difference_count": root_differences,
            "d2_vs_myopic": {
                "truth_mass_mean_difference": d2_myopic_truth,
                "truth_mass_counts": truth_counts,
                "total_evidence_mean_difference": d2_myopic_evidence,
                "total_evidence_counts": evidence_counts,
                "total_gold_mean_difference": d2_myopic_gold,
                "total_gold_counts": gold_counts,
            },
            "d2_vs_random": {
                "truth_mass_mean_difference": d2_random_truth,
                "total_evidence_mean_difference": d2_random_evidence,
            },
            "gates": gates,
            "all_gates_pass": all(gates.values()),
        },
        "selections": selections,
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = min(
        config.openrouter_concurrency,
        60,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        reconstructed, terminals, usage = run_model_stage(
            config,
            manifest_path=args.manifest,
            parquet_dir=args.parquet_dir,
            raw_path=raw_path,
            source_path=args.source_path,
        )
        result = analyze(reconstructed, terminals, usage)
        result["protocol"]["private_raw_sha256"] = mechanics.sha256_file(
            raw_path
        )
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, DevelopmentExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = mechanics.sha256_file(raw_path)
        output = args.output_dir / "DEVELOPMENT_FAILURE.json"
        output.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output = args.output_dir / "RESULT.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
