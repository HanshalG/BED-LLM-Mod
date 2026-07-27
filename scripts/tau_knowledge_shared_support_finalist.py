#!/usr/bin/env python3
"""Score tau-Knowledge finalists independently on one shared need support."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.tau_knowledge_balanced_finalist_duel import (
    SERVING_TASK_COUNT,
    _clean_excerpt,
    _selected_vs_all,
    load_finalist_records,
)
from scripts.tau_knowledge_retrieval_opportunity import (
    GateExecutionError,
    SCHEMA_VERSION,
    _build_model,
    _checkpoint,
    _usage_snapshot,
)
from scripts.tau_knowledge_shared_comparative_pooling import (
    DEVELOPMENT_TASK_COUNT,
    MAX_INPUT_CHARS,
    _root_values,
)


INTERFACE_VERSION = 1
MAP_REPLICATES = 2
CALLS_PER_CHANGED_TASK = 4
PRESENTATION_SEED = 24423
SUPPORT_WEIGHTS = {"C": 2, "P": 1, "U": 0}


def compact_support_input(
    record: dict[str, Any],
    *,
    root_index: int,
    replicate_index: int,
) -> dict[str, Any]:
    branch = record["first_branches"][root_index]
    hypotheses = list(record["initial_information_need_hypotheses"])
    need_order = list(range(len(hypotheses)))
    followup_order = list(range(len(branch["followups"])))
    rng = random.Random(
        PRESENTATION_SEED
        + 10_007 * int(record["task_id"].split("_")[-1])
        + 101 * root_index
        + replicate_index
    )
    rng.shuffle(need_order)
    rng.shuffle(followup_order)
    document_refs: dict[str, str] = {}
    catalog: list[dict[str, str]] = []

    def add_results(results: Sequence[dict[str, Any]]) -> list[str]:
        refs = []
        for result in results:
            document_id = str(result["id"])
            if document_id not in document_refs:
                ref = f"D{len(document_refs) + 1}"
                document_refs[document_id] = ref
                catalog.append(
                    {
                        "ref": ref,
                        "title": str(result["title"]),
                        "excerpt": _clean_excerpt(str(result["content"])),
                    }
                )
            refs.append(document_refs[document_id])
        return refs

    payload = {
        "customer_opening": record["opening"],
        "shared_information_needs": [
            {
                "need_id": f"H{index + 1}",
                "need": hypotheses[index],
            }
            for index in need_order
        ],
        "candidate": {
            "root_query": branch["query"],
            "first_result_refs": add_results(branch["first_results"]),
            "refreshed_information_needs": branch[
                "refreshed_information_need_hypotheses"
            ],
            "followups": [
                {
                    "followup_id": f"F{index + 1}",
                    "query": branch["followups"][index]["query"],
                    "result_refs": add_results(
                        branch["followups"][index]["results"]
                    ),
                }
                for index in followup_order
            ],
        },
        "document_catalog": catalog,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    if len(encoded) > MAX_INPUT_CHARS:
        raise ValueError("shared-support input exceeds character cap")
    return payload


def support_messages(
    record: dict[str, Any],
    *,
    root_index: int,
    replicate_index: int,
) -> list[dict[str, str]]:
    payload = compact_support_input(
        record,
        root_index=root_index,
        replicate_index=replicate_index,
    )
    need_count = len(record["initial_information_need_hypotheses"])
    return [
        {
            "role": "system",
            "content": (
                "Map one retrieval plan onto a fixed shared information-need "
                "support. Required-document labels and evaluation answers are "
                "hidden. Ground every label only in supplied document evidence. "
                "Return strict lines and no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Choose the one displayed followup whose returned evidence combines "
                "best with the first results. Then label every shared need using "
                "C only when that evidence clearly covers it, P when evidence is "
                "useful but incomplete, and U when it is unsupported. Refreshed "
                "needs are fallible clues, not evidence. Do not reward query wording, "
                "document count, verbosity, or duplicated evidence. Output exactly "
                f"{need_count + 1} lines: first `BEST|F1` through `BEST|F4`, then "
                "one `Hn|C`, `Hn|P`, or `Hn|U` line for every supplied H ID exactly "
                "once. Data: "
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_support_map(text: str, *, need_count: int) -> dict[str, Any]:
    if text != text.strip():
        raise ValueError("support map must be canonical text")
    lines = text.splitlines()
    if len(lines) != need_count + 1:
        raise ValueError("support map has wrong line count")
    best_fields = lines[0].split("|")
    if (
        len(best_fields) != 2
        or best_fields[0] != "BEST"
        or best_fields[1] not in {"F1", "F2", "F3", "F4"}
    ):
        raise ValueError("support map has invalid BEST line")
    labels: dict[str, str] = {}
    for line in lines[1:]:
        fields = line.split("|")
        if (
            len(fields) != 2
            or fields[0] not in {f"H{index + 1}" for index in range(need_count)}
            or fields[1] not in SUPPORT_WEIGHTS
            or fields[0] in labels
        ):
            raise ValueError("support map has invalid hypothesis line")
        labels[fields[0]] = fields[1]
    if set(labels) != {f"H{index + 1}" for index in range(need_count)}:
        raise ValueError("support map is not a complete hypothesis mapping")
    return {
        "best_followup_index": int(best_fields[1][1:]) - 1,
        "labels": labels,
        "score": sum(SUPPORT_WEIGHTS[label] for label in labels.values()),
    }


def categorical_agreement(
    left: dict[str, Any],
    right: dict[str, Any],
) -> float:
    left_labels = left["labels"]
    right_labels = right["labels"]
    if set(left_labels) != set(right_labels):
        raise ValueError("support maps have different need IDs")
    return sum(
        left_labels[key] == right_labels[key] for key in left_labels
    ) / len(left_labels)


def resolve_support_scores(
    *,
    myopic_root: int,
    nonmyopic_root: int,
    myopic_maps: Sequence[dict[str, Any]],
    nonmyopic_maps: Sequence[dict[str, Any]],
) -> tuple[int, str]:
    if myopic_root == nonmyopic_root:
        return myopic_root, "same_finalist"
    if len(myopic_maps) != MAP_REPLICATES or len(nonmyopic_maps) != MAP_REPLICATES:
        raise ValueError("unexpected support-map replicate count")
    myopic_scores = [int(item["score"]) for item in myopic_maps]
    nonmyopic_scores = [int(item["score"]) for item in nonmyopic_maps]
    if min(nonmyopic_scores) > max(myopic_scores):
        return nonmyopic_root, "robust_nonmyopic_override"
    return myopic_root, "myopic_fallback"


def summarize(
    records: Sequence[dict[str, Any]],
    maps: dict[str, dict[str, list[dict[str, Any]]]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    expected_tasks = (
        SERVING_TASK_COUNT if stage == "serving_smoke" else DEVELOPMENT_TASK_COUNT
    )
    rows = []
    myopic_top_points = 0
    myopic_top_comparable = 0
    selected_top_points = 0
    selected_top_comparable = 0
    agreements = []
    for record in records:
        task_id = str(record["task_id"])
        myopic_root = int(record["myopic_finalist_index"])
        nonmyopic_root = int(record["nonmyopic_finalist_index"])
        if myopic_root == nonmyopic_root:
            selected_root, decision = myopic_root, "same_finalist"
            task_maps = None
        else:
            task_maps = maps[task_id]
            selected_root, decision = resolve_support_scores(
                myopic_root=myopic_root,
                nonmyopic_root=nonmyopic_root,
                myopic_maps=task_maps["myopic"],
                nonmyopic_maps=task_maps["nonmyopic"],
            )
            agreements.extend(
                [
                    categorical_agreement(*task_maps["myopic"]),
                    categorical_agreement(*task_maps["nonmyopic"]),
                ]
            )
        _immediate, values = _root_values(record)
        my_points, my_count = _selected_vs_all(values, myopic_root)
        selected_points, selected_count = _selected_vs_all(
            values, selected_root
        )
        myopic_top_points += my_points
        myopic_top_comparable += my_count
        selected_top_points += selected_points
        selected_top_comparable += selected_count
        rows.append(
            {
                "task_id": task_id,
                "myopic_finalist_index": myopic_root,
                "nonmyopic_finalist_index": nonmyopic_root,
                "selected_root_index": selected_root,
                "decision": decision,
                "support_maps": task_maps,
                "oracle_tail_values": values,
                "myopic_oracle_tail": values[myopic_root],
                "selected_oracle_tail": values[selected_root],
                "advantage_over_myopic": (
                    values[selected_root] - values[myopic_root]
                ),
            }
        )
    changed_count = sum(
        row["myopic_finalist_index"] != row["nonmyopic_finalist_index"]
        for row in rows
    )
    expected_requests = changed_count * CALLS_PER_CHANGED_TASK
    generator = usage["generator"]
    gates = {
        "all_tasks_complete": len(records) == expected_tasks,
        "exact_logical_request_count": int(usage["physical_requests"])
        == expected_requests,
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "zero_forced_exits": int(generator.get("forced_exits", 0)) == 0,
        "all_changed_tasks_have_complete_maps": len(maps) == changed_count,
    }
    myopic_total = sum(row["myopic_oracle_tail"] for row in rows)
    selected_total = sum(row["selected_oracle_tail"] for row in rows)
    advantages = [row["advantage_over_myopic"] for row in rows]
    override_count = sum(
        row["decision"] == "robust_nonmyopic_override" for row in rows
    )
    mean_agreement = sum(agreements) / len(agreements) if agreements else 1.0
    myopic_top_accuracy = (
        myopic_top_points / myopic_top_comparable
        if myopic_top_comparable
        else 0.0
    )
    selected_top_accuracy = (
        selected_top_points / selected_top_comparable
        if selected_top_comparable
        else 0.0
    )
    summary = {
        "num_tasks": len(records),
        "changed_finalist_count": changed_count,
        "robust_nonmyopic_override_count": override_count,
        "mean_categorical_replicate_agreement": mean_agreement,
        "myopic_selected_oracle_tail_total": myopic_total,
        "support_selected_oracle_tail_total": selected_total,
        "support_advantage_over_myopic": selected_total - myopic_total,
        "support_vs_myopic_wins": sum(value > 0 for value in advantages),
        "support_vs_myopic_ties": sum(value == 0 for value in advantages),
        "support_vs_myopic_losses": sum(value < 0 for value in advantages),
        "myopic_selected_vs_all_accuracy": myopic_top_accuracy,
        "support_selected_vs_all_accuracy": selected_top_accuracy,
        "selected_vs_all_accuracy_gain": (
            selected_top_accuracy - myopic_top_accuracy
        ),
        "task_diagnostics": rows,
    }
    if stage == "development":
        gates.update(
            {
                "categorical_replicate_agreement_at_least_0_70": (
                    mean_agreement >= 0.70
                ),
                "robust_nonmyopic_overrides_at_least_3": override_count >= 3,
                "oracle_tail_gain_over_myopic_at_least_4": (
                    selected_total - myopic_total >= 4
                ),
                "wins_exceed_losses_by_at_least_3": (
                    sum(value > 0 for value in advantages)
                    - sum(value < 0 for value in advantages)
                    >= 3
                ),
                "losses_at_most_1": sum(value < 0 for value in advantages) <= 1,
                "selected_vs_all_accuracy_at_least_0_70": (
                    selected_top_accuracy >= 0.70
                ),
                "selected_vs_all_accuracy_gain_at_least_0_08": (
                    selected_top_accuracy - myopic_top_accuracy >= 0.08
                ),
            }
        )
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def run_gate(
    config: Config,
    *,
    stage: str,
    primary_artifact: str | Path,
    secondary_artifact: str | Path,
    rank_artifact: str | Path,
    raw_checkpoint_path: Path | None,
) -> dict[str, Any]:
    records, rank_payload = load_finalist_records(
        primary_artifact,
        secondary_artifact,
        rank_artifact,
    )
    if stage == "serving_smoke":
        records = records[:SERVING_TASK_COUNT]
    elif stage != "development":
        raise ValueError("stage must be serving_smoke or development")
    changed = [
        record
        for record in records
        if record["myopic_finalist_index"]
        != record["nonmyopic_finalist_index"]
    ]
    model = _build_model(config)
    raw: dict[str, Any] = {}
    try:
        messages = []
        keys = []
        for record in changed:
            for candidate_name, root_key in (
                ("myopic", "myopic_finalist_index"),
                ("nonmyopic", "nonmyopic_finalist_index"),
            ):
                for replicate_index in range(MAP_REPLICATES):
                    messages.append(
                        support_messages(
                            record,
                            root_index=int(record[root_key]),
                            replicate_index=replicate_index,
                        )
                    )
                    keys.append(
                        (
                            str(record["task_id"]),
                            candidate_name,
                            replicate_index,
                        )
                    )
        responses = model.chat_complete_messages_batched(
            messages,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["support_maps"] = [
            {
                "task_id": task_id,
                "candidate": candidate,
                "replicate": replicate,
                "response": response,
            }
            for (task_id, candidate, replicate), response in zip(
                keys, responses, strict=True
            )
        ]
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        parsed = []
        records_by_id = {str(record["task_id"]): record for record in records}
        for (task_id, _candidate, _replicate), response in zip(
            keys, responses, strict=True
        ):
            parsed.append(
                parse_support_map(
                    response,
                    need_count=len(
                        records_by_id[task_id][
                            "initial_information_need_hypotheses"
                        ]
                    ),
                )
            )
        maps: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for (task_id, candidate, _replicate), support_map in zip(
            keys, parsed, strict=True
        ):
            maps.setdefault(
                task_id, {"myopic": [], "nonmyopic": []}
            )[candidate].append(support_map)
        usage = _usage_snapshot(model)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc

    summary = summarize(records, maps, usage, stage=stage)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "interface_version": INTERFACE_VERSION,
            "model": config.model_pairs[0].questioner.model,
            "primary_artifact_sha256": hashlib.sha256(
                Path(primary_artifact).read_bytes()
            ).hexdigest(),
            "secondary_artifact_sha256": hashlib.sha256(
                Path(secondary_artifact).read_bytes()
            ).hexdigest(),
            "rank_artifact_sha256": hashlib.sha256(
                Path(rank_artifact).read_bytes()
            ).hexdigest(),
            "rank_artifact_status": rank_payload["status"],
            "task_ids": [record["task_id"] for record in records],
            "support_weights": SUPPORT_WEIGHTS,
            "map_replicates_per_candidate": MAP_REPLICATES,
            "candidate_maps_independent": True,
            "robust_override_rule": "min(nonmyopic)>max(myopic)",
            "presentation_seed": PRESENTATION_SEED,
            "oracle_continuation_used_for_first_link_endpoint_only": True,
            "required_documents_hidden_from_model": True,
            "reasoning_disabled": True,
            "raw_responses_private_and_untracked": True,
            "development_tasks_previously_open": stage == "development",
        },
        "summary": summary,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "development"),
        required=True,
    )
    parser.add_argument("--primary-artifact", type=Path, required=True)
    parser.add_argument("--secondary-artifact", type=Path, required=True)
    parser.add_argument("--rank-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_max_output_tokens = min(
        config.openrouter_max_output_tokens, 512
    )
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.12
        config.openrouter_run_budget_usd = 0.50
    else:
        config.openrouter_projected_cost_usd = 0.80
        config.openrouter_run_budget_usd = 1.75
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json"
        if args.stage == "serving_smoke"
        else "DEVELOPMENT.json"
    )
    try:
        payload = run_gate(
            config,
            stage=args.stage,
            primary_artifact=args.primary_artifact,
            secondary_artifact=args.secondary_artifact,
            rank_artifact=args.rank_artifact,
            raw_checkpoint_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        (args.output_dir / f"{args.stage.upper()}_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path = args.output_dir / output_name
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output_path),
                "summary": payload["summary"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
