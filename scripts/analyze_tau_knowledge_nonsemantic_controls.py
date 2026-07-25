#!/usr/bin/env python3
"""Compare tau V3.1 with frozen nonsemantic tree-scoring controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tau_knowledge_first_link_scorer import pairwise_ranking_points
from scripts.tau_knowledge_retrieval_opportunity import analyze_record


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """
    a about after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down
    during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just me more most
    my myself no nor not now of off on once only or other our ours ourselves
    out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why will with you
    your yours yourself yourselves
    """.split()
)


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _unique_documents(
    first: Sequence[dict[str, Any]],
    followup: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for document in [*first, *followup]:
        document_id = str(document["id"])
        if document_id not in documents:
            documents[document_id] = document
        elif float(document["bm25_score"]) > float(
            documents[document_id]["bm25_score"]
        ):
            documents[document_id] = document
    return list(documents.values())


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def _lexical_idf(record: dict[str, Any]) -> dict[str, float]:
    documents: dict[str, set[str]] = {}
    for branch in record["first_branches"]:
        for document in branch["first_results"]:
            documents.setdefault(
                str(document["id"]),
                _tokens(f"{document['title']} {document['content']}"),
            )
        for followup in branch["followups"]:
            for document in followup["results"]:
                documents.setdefault(
                    str(document["id"]),
                    _tokens(f"{document['title']} {document['content']}"),
                )
    document_count = len(documents)
    frequencies: dict[str, int] = {}
    for tokens in documents.values():
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
    return {
        token: math.log((1 + document_count) / (1 + frequency)) + 1
        for token, frequency in frequencies.items()
    }


def _pair_score_tables(
    record: dict[str, Any],
) -> dict[str, list[list[float]]]:
    idf = _lexical_idf(record)
    initial_text = " ".join(
        [
            str(record["opening"]),
            *map(str, record["initial_information_need_hypotheses"]),
        ]
    )
    tables = {"bm25_sum": [], "novel_count": [], "lexical_idf": []}
    for branch in record["first_branches"]:
        first = branch["first_results"]
        first_ids = {str(document["id"]) for document in first}
        objective = _tokens(
            " ".join(
                [
                    initial_text,
                    *map(
                        str,
                        branch["refreshed_information_need_hypotheses"],
                    ),
                ]
            )
        )
        denominator = sum(idf.get(token, 1.0) for token in objective) or 1.0
        bm25_scores = []
        novel_counts = []
        lexical_scores = []
        for followup in branch["followups"]:
            documents = _unique_documents(first, followup["results"])
            bm25_scores.append(
                sum(float(document["bm25_score"]) for document in documents)
            )
            novel_counts.append(
                float(
                    len(
                        {
                            str(document["id"])
                            for document in followup["results"]
                        }
                        - first_ids
                    )
                )
            )
            lexical_scores.append(
                sum(
                    sum(
                        idf.get(token, 1.0)
                        for token in (
                            objective
                            & _tokens(
                                f"{document['title']} {document['content']}"
                            )
                        )
                    )
                    for document in documents
                )
                / denominator
            )
        tables["bm25_sum"].append(bm25_scores)
        tables["novel_count"].append(novel_counts)
        tables["lexical_idf"].append(lexical_scores)
    return tables


def _policy_summary(
    records: Sequence[dict[str, Any]],
    semantic_values: Sequence[int],
    table_builder: Callable[
        [dict[str, Any]], dict[str, list[list[float]]]
    ] = _pair_score_tables,
) -> dict[str, Any]:
    policies: dict[str, list[dict[str, Any]]] = {
        "bm25_sum": [],
        "novel_count": [],
        "lexical_idf": [],
    }
    for case_index, record in enumerate(records):
        endpoint = analyze_record(record)
        pair_values = endpoint["pair_counts"]
        root_values = [max(values) for values in pair_values]
        tables = table_builder(record)
        for name, table in tables.items():
            root_scores = [max(scores) for scores in table]
            root_index = _argmax(root_scores)
            followup_indices = [_argmax(scores) for scores in table]
            followup_index = followup_indices[root_index]
            root_points, root_comparable = pairwise_ranking_points(
                root_scores, root_values
            )
            continuation_points = 0.0
            continuation_comparable = 0
            optimal_count = 0
            regret = 0
            for scores, values, selected in zip(
                table, pair_values, followup_indices, strict=True
            ):
                points, comparable = pairwise_ranking_points(scores, values)
                continuation_points += points
                continuation_comparable += comparable
                optimal = max(values)
                optimal_count += values[selected] == optimal
                regret += optimal - values[selected]
            selected_value = pair_values[root_index][followup_index]
            policies[name].append(
                {
                    "task_id": record["task_id"],
                    "root_index": root_index,
                    "followup_index": followup_index,
                    "selected_value": selected_value,
                    "semantic_advantage": (
                        semantic_values[case_index] - selected_value
                    ),
                    "root_points": root_points,
                    "root_comparable": root_comparable,
                    "continuation_points": continuation_points,
                    "continuation_comparable": continuation_comparable,
                    "optimal_count": optimal_count,
                    "regret": regret,
                }
            )

    summaries = {}
    for name, rows in policies.items():
        root_comparable = sum(row["root_comparable"] for row in rows)
        continuation_comparable = sum(
            row["continuation_comparable"] for row in rows
        )
        advantages = [row["semantic_advantage"] for row in rows]
        summaries[name] = {
            "endpoint_total": sum(row["selected_value"] for row in rows),
            "semantic_advantage_total": sum(advantages),
            "semantic_win_count": sum(value > 0 for value in advantages),
            "semantic_loss_count": sum(value < 0 for value in advantages),
            "semantic_tie_count": sum(value == 0 for value in advantages),
            "root_pairwise_accuracy": (
                sum(row["root_points"] for row in rows) / root_comparable
                if root_comparable
                else 0.0
            ),
            "root_pairwise_comparable_count": root_comparable,
            "continuation_pairwise_accuracy": (
                sum(row["continuation_points"] for row in rows)
                / continuation_comparable
                if continuation_comparable
                else 0.0
            ),
            "continuation_pairwise_comparable_count": (
                continuation_comparable
            ),
            "optimal_followup_count": sum(
                row["optimal_count"] for row in rows
            ),
            "total_regret": sum(row["regret"] for row in rows),
            "case_diagnostics": rows,
        }
    return summaries


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    semantic_rows = payload["summary"]["policy_diagnostics"]
    semantic_values = [
        row["nonmyopic_receding_value"] for row in semantic_rows
    ]
    controls = _policy_summary(payload["records"], semantic_values)
    semantic = {
        "endpoint_total": sum(semantic_values),
        "root_pairwise_accuracy": payload["summary"][
            "nonmyopic_root_pairwise_accuracy"
        ],
        "continuation_pairwise_accuracy": payload["summary"][
            "focused_pairwise_accuracy"
        ],
    }
    load_bearing_checks = {
        name: {
            "endpoint_advantage_at_least_3": (
                semantic["endpoint_total"] - control["endpoint_total"] >= 3
            ),
            "more_wins_than_losses": (
                control["semantic_win_count"]
                > control["semantic_loss_count"]
            ),
            "higher_root_accuracy": (
                semantic["root_pairwise_accuracy"]
                > control["root_pairwise_accuracy"]
            ),
            "higher_continuation_accuracy": (
                semantic["continuation_pairwise_accuracy"]
                > control["continuation_pairwise_accuracy"]
            ),
        }
        for name, control in controls.items()
    }
    for checks in load_bearing_checks.values():
        checks["all_pass"] = all(checks.values())
    return {
        "status": "posthoc_nonsemantic_control",
        "semantic_policy": semantic,
        "controls": controls,
        "load_bearing_checks": load_bearing_checks,
        "all_load_bearing_checks_pass": all(
            checks["all_pass"] for checks in load_bearing_checks.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.confirmation.read_bytes()
    payload = json.loads(raw)
    result = {
        "confirmation_sha256": hashlib.sha256(raw).hexdigest(),
        **analyze(payload),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
