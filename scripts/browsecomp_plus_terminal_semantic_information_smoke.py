#!/usr/bin/env python3
"""Run the frozen all-root BrowseComp semantic-information terminal smoke."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts import browsecomp_plus_cached_semantic_information as cached
from scripts import browsecomp_plus_semantic_mechanics as mechanics
from scripts import browsecomp_plus_semantic_mechanics_posthoc as posthoc


INTERFACE_VERSION = "browsecomp-plus-terminal-semantic-information-smoke-1"
EXPECTED_REQUESTS = 30
MAX_COST_USD = 0.75
PROJECTED_COST_USD = 0.30
TEMPERATURE = 0.0


@dataclass(frozen=True)
class MassBelief:
    hypotheses: tuple[str, ...]
    probabilities: tuple[float, ...]


class SmokeExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def parse_terminal_masses(text: str) -> MassBelief:
    lines = mechanics._response_lines(text)
    if len(lines) != mechanics.HYPOTHESIS_COUNT:
        raise ValueError("wrong terminal hypothesis line count")
    hypotheses: list[str] = []
    masses: list[int] = []
    for index, line in enumerate(lines, start=1):
        parts = line.split("|")
        if len(parts) != 3 or parts[0] != f"H{index:02d}":
            raise ValueError("invalid terminal hypothesis line")
        masses.append(
            mechanics._canonical_integer(
                parts[1],
                minimum=0,
                maximum=100,
            )
        )
        hypothesis = " ".join(parts[2].split())
        if not hypothesis:
            raise ValueError("terminal hypothesis is empty")
        hypotheses.append(hypothesis)
    normalized = {
        mechanics.normalize_answer(hypothesis)
        for hypothesis in hypotheses
    }
    if "" in normalized or len(normalized) != mechanics.HYPOTHESIS_COUNT:
        raise ValueError("terminal hypotheses must be normalized-distinct")
    total_mass = sum(masses)
    if total_mass <= 0:
        raise ValueError("terminal masses must have positive total")
    return MassBelief(
        hypotheses=tuple(hypotheses),
        probabilities=tuple(mass / total_mass for mass in masses),
    )


def terminal_mass_messages(
    task: Mapping[str, Any],
    *,
    initial_belief: mechanics.Belief,
    branch: mechanics.Branch,
) -> list[dict[str, str]]:
    assert branch.root_belief is not None
    assert branch.adaptive_query is not None
    assert branch.adaptive_documents is not None
    return [
        {
            "role": "system",
            "content": (
                "Regenerate the open-world answer belief after both search "
                "observations. Return exactly eight distinct short candidate "
                "answers with nonnegative integer belief masses from 0 to 100. "
                "The masses are unnormalized scores and need not sum to 100, "
                "but at least one must be positive. Preserve, revise, add, or "
                "drop answers based on the returned text. Do not score the "
                "queries and do not reveal reasoning. Return only the eight "
                "ordered H lines."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": task["query"],
                    "initial_belief": mechanics._belief_payload(
                        initial_belief
                    ),
                    "belief_after_first_search": mechanics._belief_payload(
                        branch.root_belief
                    ),
                    "first_query": branch.strategy.root_query,
                    "first_results": mechanics._document_payload(
                        branch.root_documents
                    ),
                    "second_query": branch.adaptive_query,
                    "second_results": mechanics._document_payload(
                        branch.adaptive_documents
                    ),
                    "exact_output_lines": [
                        (
                            f"H{index:02d}|0..100 unnormalized belief mass|"
                            "short candidate final answer"
                        )
                        for index in range(
                            1,
                            mechanics.HYPOTHESIS_COUNT + 1,
                        )
                    ],
                },
                separators=(",", ":"),
            ),
        },
    ]


def semantic_entropy(
    hypotheses: Sequence[str],
    probabilities: Sequence[float],
) -> float:
    cluster_ids = cached.semantic_cluster_ids(hypotheses)
    cluster_probabilities = [0.0] * (max(cluster_ids) + 1)
    for cluster_id, probability in zip(
        cluster_ids,
        probabilities,
        strict=True,
    ):
        cluster_probabilities[cluster_id] += probability
    return cached.entropy(cluster_probabilities)


def mechanics_entropy(belief: mechanics.Belief) -> float:
    return semantic_entropy(
        belief.hypotheses,
        [weight / 100.0 for weight in belief.weights],
    )


def mass_truth_probability(belief: MassBelief, answer: str) -> float:
    return sum(
        probability
        for hypothesis, probability in zip(
            belief.hypotheses,
            belief.probabilities,
            strict=True,
        )
        if cached.alias_equivalent(hypothesis, answer)
    )


def run_model_stage(
    config: Config,
    *,
    source_path: Path,
    cached_raw_path: Path,
    raw_path: Path,
    model_adapter: Any | None = None,
) -> tuple[dict[str, Any], list[list[MassBelief]], dict[str, Any]]:
    reconstructed = posthoc.reconstruct(
        source_path=source_path,
        raw_path=cached_raw_path,
    )
    specs = [
        (task_index, root_index)
        for task_index in range(len(reconstructed["tasks"]))
        for root_index in range(mechanics.ROOT_COUNT)
    ]
    model = (
        model_adapter
        if model_adapter is not None
        else mechanics._build_model(config)
    )
    raw = {
        "task_ids": list(mechanics.TASK_IDS),
        "root_order": [
            {"task_index": task_index, "root_index": root_index}
            for task_index, root_index in specs
        ],
    }
    try:
        responses = mechanics._complete(
            model,
            [
                terminal_mass_messages(
                    reconstructed["tasks"][task_index],
                    initial_belief=reconstructed["initials"][
                        task_index
                    ][0],
                    branch=reconstructed["branches"][task_index][
                        root_index
                    ],
                )
                for task_index, root_index in specs
            ],
            config,
        )
        raw["terminal_responses"] = responses
        mechanics._checkpoint(raw_path, raw)
        parsed = [
            parse_terminal_masses(response) for response in responses
        ]
        terminal_beliefs: list[list[MassBelief]] = [
            [] for _ in reconstructed["tasks"]
        ]
        for belief, (task_index, _) in zip(parsed, specs, strict=True):
            terminal_beliefs[task_index].append(belief)
        usage = mechanics._usage_snapshot(model)
    except Exception as exc:
        if raw:
            mechanics._checkpoint(raw_path, raw)
        raise SmokeExecutionError(
            f"{type(exc).__name__}: {exc}",
            mechanics._usage_snapshot(model),
        ) from exc
    return reconstructed, terminal_beliefs, usage


def _pairwise_summary(
    records: Sequence[dict[str, Any]],
    *,
    score_key: str,
    endpoint_key: str,
) -> dict[str, float | int]:
    accuracy, pairs = cached._weighted_pairwise(
        records,
        score_key=score_key,
        endpoint_key=endpoint_key,
    )
    return {"accuracy": accuracy, "pairs": pairs}


def analyze(
    reconstructed: dict[str, Any],
    terminal_beliefs: Sequence[Sequence[MassBelief]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    dynamic_task_count = 0

    for task_index, task in enumerate(reconstructed["tasks"]):
        initial = reconstructed["initials"][task_index][0]
        branches = reconstructed["branches"][task_index]
        terminals = terminal_beliefs[task_index]
        initial_entropy = mechanics_entropy(initial)
        initial_truth = cached.truth_mass(initial, task["answer"])
        evidence_ids = {
            str(document["docid"]) for document in task["evidence_docs"]
        }
        gold_ids = {
            str(document["docid"]) for document in task["gold_docs"]
        }
        task_records = []
        for branch, terminal in zip(branches, terminals, strict=True):
            assert branch.root_belief is not None
            assert branch.adaptive_documents is not None
            root_entropy = mechanics_entropy(branch.root_belief)
            terminal_entropy = semantic_entropy(
                terminal.hypotheses,
                terminal.probabilities,
            )
            root_truth = cached.truth_mass(
                branch.root_belief,
                task["answer"],
            )
            terminal_truth = mass_truth_probability(
                terminal,
                task["answer"],
            )
            root_ids = {
                str(document["docid"])
                for document in branch.root_documents
            }
            adaptive_ids = {
                str(document["docid"])
                for document in branch.adaptive_documents
            }
            record = {
                "task_id": str(task["query_id"]),
                "root_index": branch.root_index,
                "root_entropy_drop": initial_entropy - root_entropy,
                "future_entropy_drop": root_entropy - terminal_entropy,
                "total_entropy_drop": initial_entropy - terminal_entropy,
                "root_truth_mass": root_truth,
                "terminal_truth_mass": terminal_truth,
                "future_truth_mass_gain": terminal_truth - root_truth,
                "total_truth_mass_gain": terminal_truth - initial_truth,
                "immediate_evidence": len(root_ids & evidence_ids),
                "future_evidence_gain": (
                    len((root_ids | adaptive_ids) & evidence_ids)
                    - len(root_ids & evidence_ids)
                ),
                "total_evidence": len(
                    (root_ids | adaptive_ids) & evidence_ids
                ),
                "total_gold": len((root_ids | adaptive_ids) & gold_ids),
            }
            records.append(record)
            task_records.append(record)

        total_scores = [
            record["total_entropy_drop"] for record in task_records
        ]
        dynamic_task_count += (
            max(total_scores) - min(total_scores) > 1e-12
        )
        myopic_root = mechanics._argmax(
            [record["root_entropy_drop"] for record in task_records]
        )
        d2_root = mechanics._argmax(total_scores)
        selections.append(
            {
                "task_id": str(task["query_id"]),
                "myopic_root": myopic_root,
                "d2_root": d2_root,
                "myopic_terminal_truth_mass": task_records[myopic_root][
                    "terminal_truth_mass"
                ],
                "d2_terminal_truth_mass": task_records[d2_root][
                    "terminal_truth_mass"
                ],
                "myopic_total_evidence": task_records[myopic_root][
                    "total_evidence"
                ],
                "d2_total_evidence": task_records[d2_root][
                    "total_evidence"
                ],
            }
        )

    pairwise = {
        "total_entropy_vs_total_truth": _pairwise_summary(
            records,
            score_key="total_entropy_drop",
            endpoint_key="total_truth_mass_gain",
        ),
        "total_entropy_vs_total_evidence": _pairwise_summary(
            records,
            score_key="total_entropy_drop",
            endpoint_key="total_evidence",
        ),
        "total_entropy_vs_total_gold": _pairwise_summary(
            records,
            score_key="total_entropy_drop",
            endpoint_key="total_gold",
        ),
        "future_entropy_vs_future_truth": _pairwise_summary(
            records,
            score_key="future_entropy_drop",
            endpoint_key="future_truth_mass_gain",
        ),
        "future_entropy_vs_future_evidence": _pairwise_summary(
            records,
            score_key="future_entropy_drop",
            endpoint_key="future_evidence_gain",
        ),
    }

    def comparison_counts(
        left_key: str,
        right_key: str,
    ) -> tuple[int, int, int]:
        wins = sum(
            selection[left_key] > selection[right_key]
            for selection in selections
        )
        losses = sum(
            selection[left_key] < selection[right_key]
            for selection in selections
        )
        return wins, losses, len(selections) - losses

    truth_wins, truth_losses, truth_nonworse = comparison_counts(
        "d2_terminal_truth_mass",
        "myopic_terminal_truth_mass",
    )
    evidence_wins, evidence_losses, evidence_nonworse = comparison_counts(
        "d2_total_evidence",
        "myopic_total_evidence",
    )
    root_differences = sum(
        selection["d2_root"] != selection["myopic_root"]
        for selection in selections
    )
    gates = {
        "exact_30_requests": (
            usage["physical_requests"] == EXPECTED_REQUESTS
            and usage["http_attempts"] == EXPECTED_REQUESTS
            and len(records) == EXPECTED_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_75": usage["adapter_cost_usd"] <= MAX_COST_USD,
        "total_entropy_dynamic_on_5_tasks": dynamic_task_count == 5,
        "total_truth_pairs_at_least_20": (
            pairwise["total_entropy_vs_total_truth"]["pairs"] >= 20
        ),
        "total_evidence_pairs_at_least_30": (
            pairwise["total_entropy_vs_total_evidence"]["pairs"] >= 30
        ),
        "total_entropy_truth_accuracy_at_least_0_60": (
            pairwise["total_entropy_vs_total_truth"]["accuracy"] >= 0.60
        ),
        "total_entropy_evidence_accuracy_at_least_0_60": (
            pairwise["total_entropy_vs_total_evidence"]["accuracy"] >= 0.60
        ),
        "future_entropy_truth_accuracy_at_least_0_55": (
            pairwise["future_entropy_vs_future_truth"]["accuracy"] >= 0.55
        ),
        "future_entropy_evidence_accuracy_at_least_0_55": (
            pairwise["future_entropy_vs_future_evidence"]["accuracy"] >= 0.55
        ),
        "d2_root_differs_on_at_least_1_task": root_differences >= 1,
        "d2_truth_wins_at_least_1_task": truth_wins >= 1,
        "d2_truth_losses_at_most_1_task": truth_losses <= 1,
        "d2_truth_nonworse_on_at_least_4_tasks": truth_nonworse >= 4,
        "d2_evidence_wins_at_least_1_task": evidence_wins >= 1,
        "d2_evidence_losses_at_most_1_task": evidence_losses <= 1,
        "d2_evidence_nonworse_on_at_least_4_tasks": (
            evidence_nonworse >= 4
        ),
    }
    return {
        "schema_version": 1,
        "status": (
            "terminal_semantic_information_pass"
            if all(gates.values())
            else "terminal_semantic_information_failure"
        ),
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_sha256": mechanics.SOURCE_SHA256,
            "cached_raw_sha256": posthoc.RAW_SHA256,
            "model": mechanics.MODEL_ID,
            "temperature": TEMPERATURE,
            "expected_requests": EXPECTED_REQUESTS,
            "projected_cost_usd": PROJECTED_COST_USD,
            "max_cost_usd": MAX_COST_USD,
            "posthoc_open_mechanics": True,
            "development_authorized": False,
            "oatml_used": False,
        },
        "summary": {
            "task_count": len(selections),
            "root_count": len(records),
            "dynamic_task_count": dynamic_task_count,
            "pairwise": pairwise,
            "d2_root_difference_count": root_differences,
            "d2_truth_wins": truth_wins,
            "d2_truth_losses": truth_losses,
            "d2_truth_nonworse": truth_nonworse,
            "d2_evidence_wins": evidence_wins,
            "d2_evidence_losses": evidence_losses,
            "d2_evidence_nonworse": evidence_nonworse,
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
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--cached-raw-path", type=Path, required=True)
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
        EXPECTED_REQUESTS,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        reconstructed, terminal_beliefs, usage = run_model_stage(
            config,
            source_path=args.source_path,
            cached_raw_path=args.cached_raw_path,
            raw_path=raw_path,
        )
        result = analyze(reconstructed, terminal_beliefs, usage)
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
        if isinstance(exc, SmokeExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = mechanics.sha256_file(raw_path)
        output = args.output_dir / "SMOKE_FAILURE.json"
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
