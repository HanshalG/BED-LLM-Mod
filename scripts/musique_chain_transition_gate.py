#!/usr/bin/env python3
"""Measure path-dependent LLM belief recovery on two-hop MuSiQue questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence
import unicodedata

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.movielens_profile_dynamics_gate import _parse_json_object


DATA_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
SELECTION_SEED = 24314
CHAIN_COUNT = 8
ACTION_COUNT = 6
PREVIOUS_IDS = (
    "2hop__554167_451128",
    "2hop__256336_714772",
    "2hop__171254_383727",
    "2hop__6827_55848",
    "2hop__131818_161450",
    "2hop__635187_861533",
    "2hop__819974_129669",
    "2hop__809785_606637",
    "2hop__286268_97805",
    "2hop__70584_198548",
    "2hop__131275_72870",
    "2hop__228_90265",
    "2hop__559273_152023",
    "2hop__486392_35739",
)
SMOKE_IDS = (
    "2hop__78756_198548",
    "2hop__329676_119915",
)
FORMAL_IDS = (
    "2hop__267938_92763",
    "2hop__58168_1783",
    "2hop__85931_108632",
    "2hop__747306_72813",
    "2hop__133102_417697",
    "2hop__462179_643013",
    "2hop__785711_63853",
    "2hop__96414_47902",
    "2hop__6736_6733",
    "2hop__739909_807845",
    "2hop__132472_684936",
    "2hop__499003_853511",
)


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def _normalized(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
    )
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value))


def _text_hash(value: str) -> str:
    return hashlib.sha256(_clean_text(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_ids(stage: str) -> tuple[str, ...]:
    if stage == "serving_smoke":
        return SMOKE_IDS
    if stage == "formal":
        return FORMAL_IDS
    raise ValueError("stage must be serving_smoke or formal")


def is_eligible_row(row: dict[str, Any]) -> bool:
    decomposition = row.get("question_decomposition", [])
    paragraphs = row.get("paragraphs", [])
    if not row.get("answerable") or len(decomposition) != 2 or len(paragraphs) != 20:
        return False
    answer = _normalized(str(row.get("answer", "")))
    if not answer or answer in {"yes", "no"}:
        return False
    if answer in _normalized(str(row.get("question", ""))):
        return False
    if any(
        _normalized(str(paragraph.get("title", ""))) == answer
        or answer in _normalized(str(paragraph.get("title", "")))
        for paragraph in paragraphs
    ):
        return False
    return (
        int(decomposition[0]["paragraph_support_idx"])
        != int(decomposition[1]["paragraph_support_idx"])
    )


def load_selected_rows(data_path: str | Path, stage: str) -> list[dict[str, Any]]:
    path = Path(data_path)
    if _sha256(path) != DATA_SHA256:
        raise ValueError("MuSiQue data hash does not match the frozen dev artifact")
    frozen_all = SMOKE_IDS + FORMAL_IDS
    wanted = selected_ids(stage)
    rows_by_id: dict[str, dict[str, Any]] = {}
    eligible_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row_id = str(row.get("id", ""))
            if is_eligible_row(row) and row_id not in PREVIOUS_IDS:
                eligible_ids.append(row_id)
            if row_id in frozen_all:
                rows_by_id[row_id] = row
    reproduced = tuple(
        str(value)
        for value in np.random.default_rng(SELECTION_SEED).choice(
            sorted(eligible_ids),
            size=len(frozen_all),
            replace=False,
        )
    )
    if reproduced != frozen_all:
        raise ValueError("frozen target-blind MuSiQue selection does not reproduce")
    missing = [row_id for row_id in wanted if row_id not in rows_by_id]
    if missing:
        raise ValueError(f"frozen MuSiQue rows are missing: {missing}")
    rows = [rows_by_id[row_id] for row_id in wanted]
    for row in rows:
        if not row.get("answerable"):
            raise ValueError("selected MuSiQue row is not answerable")
        if len(row.get("question_decomposition", [])) != 2:
            raise ValueError("selected MuSiQue row is not two-hop")
        if len(row.get("paragraphs", [])) != 20:
            raise ValueError("selected MuSiQue row does not have 20 paragraphs")
    return rows


def _schema() -> dict[str, Any]:
    return {
        "roots": [
            {
                "id": f"r{index + 1}",
                "first_doc_id": "exact available document ID",
                "continuations": [
                    {
                        "id": f"r{index + 1}{letter}",
                        "second_doc_id": "different exact available document ID",
                        "bridge_entity_hypothesis": "short entity or value",
                    }
                    for letter in (
                        ("a", "b") if index < 2 else ("a",)
                    )
                ],
            }
            for index in range(ACTION_COUNT)
        ]
    }


def chain_messages(
    row: dict[str, Any],
    *,
    previous_chains: Sequence[dict[str, str]] | None = None,
    opened_document: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    available_documents = [
        {
            "doc_id": f"d{index + 1:02d}",
            "title": str(paragraph["title"]),
        }
        for index, paragraph in enumerate(row["paragraphs"])
    ]
    payload: dict[str, Any] = {
        "question": row["question"],
        "available_documents": available_documents,
    }
    if previous_chains is not None:
        payload["previous_candidate_chains_for_context_only"] = list(previous_chains)
    if opened_document is not None:
        payload["newly_opened_document"] = opened_document
    refresh_instruction = (
        " Rebuild the support from all visible evidence. Retain compatible chains, "
        "remove contradicted chains, and introduce new chains enabled by the opened "
        "document."
        if opened_document is not None
        else ""
    )
    return [
        {
            "role": "system",
            "content": (
                "Maintain a target-blind belief support over possible two-document "
                "reasoning chains. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate exactly {ACTION_COUNT} root rows with pairwise-distinct "
                "first_doc_id values. Roots r1 and r2 must each contain exactly two "
                "distinct continuations; roots r3 through r6 must each contain exactly "
                "one continuation. This yields eight ordered chains without duplicate "
                "pairs. Every document ID must be copied exactly from "
                "available_documents, and each second_doc_id must differ from its "
                "root first_doc_id. Use titles only to reason about which IDs belong in "
                "the chain. The first document should reveal a bridge entity "
                "needed to use the second document. Do not claim that any title is gold "
                "or supporting. Do not answer from outside the visible evidence."
                f"{refresh_instruction} Preserve root and continuation IDs exactly. Return "
                + json.dumps(_schema(), ensure_ascii=True, separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_chains(
    text: str,
    *,
    available_titles: Sequence[str],
) -> list[dict[str, str]]:
    rows = _parse_json_object(text).get("roots")
    if not isinstance(rows, list) or len(rows) != ACTION_COUNT:
        raise ValueError(f"roots must contain exactly {ACTION_COUNT} rows")
    doc_ids = [f"d{index + 1:02d}" for index in range(len(available_titles))]
    title_by_doc_id = dict(zip(doc_ids, available_titles, strict=True))
    pairs: set[tuple[str, str]] = set()
    first_doc_ids: set[str] = set()
    parsed: list[dict[str, str]] = []
    chain_index = 0
    for index, row in enumerate(rows):
        expected_id = f"r{index + 1}"
        if not isinstance(row, dict) or row.get("id") != expected_id:
            raise ValueError("root IDs or order changed")
        first_doc_id = row.get("first_doc_id")
        if first_doc_id not in title_by_doc_id:
            raise ValueError("root document ID is not available")
        first_title = title_by_doc_id[first_doc_id]
        continuations = row.get("continuations")
        expected_count = 2 if index < 2 else 1
        if not isinstance(continuations, list) or len(continuations) != expected_count:
            raise ValueError("root continuation count changed")
        if first_doc_id in first_doc_ids:
            raise ValueError("root document IDs must be unique")
        first_doc_ids.add(first_doc_id)
        for continuation_index, continuation in enumerate(continuations):
            continuation_id = (
                f"{expected_id}{chr(ord('a') + continuation_index)}"
            )
            if (
                not isinstance(continuation, dict)
                or continuation.get("id") != continuation_id
            ):
                raise ValueError("continuation IDs or order changed")
            second_doc_id = continuation.get("second_doc_id")
            bridge = continuation.get("bridge_entity_hypothesis")
            if second_doc_id not in title_by_doc_id:
                raise ValueError("continuation document ID is not available")
            second_title = title_by_doc_id[second_doc_id]
            if first_doc_id == second_doc_id:
                raise ValueError("chain document IDs must differ")
            if not isinstance(bridge, str) or not _clean_text(bridge):
                raise ValueError("bridge_entity_hypothesis must be nonempty")
            pair = (first_doc_id, second_doc_id)
            if pair in pairs:
                raise ValueError("ordered title pairs must be unique")
            pairs.add(pair)
            parsed.append(
                {
                    "id": f"c{chain_index + 1}",
                    "first_doc_id": first_doc_id,
                    "second_doc_id": second_doc_id,
                    "first_title": first_title,
                    "second_title": second_title,
                    "bridge_entity_hypothesis": _clean_text(bridge),
                }
            )
            chain_index += 1
    if len(parsed) != CHAIN_COUNT:
        raise ValueError(f"root structure must flatten to {CHAIN_COUNT} chains")
    if len(first_doc_ids) != ACTION_COUNT:
        raise ValueError(
            f"chains must use exactly {ACTION_COUNT} distinct first titles"
        )
    return parsed


def candidate_actions(chains: Sequence[dict[str, str]]) -> list[str]:
    actions: list[str] = []
    for chain in chains:
        doc_id = chain["first_doc_id"]
        if doc_id not in actions:
            actions.append(doc_id)
    if len(actions) != ACTION_COUNT:
        raise ValueError("candidate chain support has the wrong action count")
    return actions


def binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log(probability) - (
        1.0 - probability
    ) * math.log(1.0 - probability)


def immediate_eig_values(
    chains: Sequence[dict[str, str]],
    actions: Sequence[str],
) -> list[float]:
    count = len(chains)
    if count == 0:
        raise ValueError("cannot score an empty chain support")
    return [
        binary_entropy(
            sum(chain["first_doc_id"] == action for chain in chains) / count
        )
        for action in actions
    ]


def gold_doc_ids(row: dict[str, Any]) -> tuple[str, str]:
    doc_id_by_paragraph_idx = {
        int(paragraph["idx"]): f"d{index + 1:02d}"
        for index, paragraph in enumerate(row["paragraphs"])
    }
    decomposition = row["question_decomposition"]
    return tuple(
        doc_id_by_paragraph_idx[int(step["paragraph_support_idx"])]
        for step in decomposition
    )  # type: ignore[return-value]


def truth_coverage(
    chains: Sequence[dict[str, str]],
    truth_doc_ids: tuple[str, str],
) -> int:
    return int(
        any(
            chain["first_doc_id"] == truth_doc_ids[0]
            and chain["second_doc_id"] == truth_doc_ids[1]
            for chain in chains
        )
    )


def _usage(model: Any, *, cumulative_run: bool = False) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    request_key = "requests" if cumulative_run else "adapter_requests"
    reasoning_key = (
        "reasoning_tokens" if cumulative_run else "adapter_reasoning_tokens"
    )
    cost_key = "run_cost_usd" if cumulative_run else "adapter_cost_usd"
    return {
        "physical_requests": int(snapshot[request_key]),
        "reasoning_tokens": int(snapshot[reasoning_key]),
        "adapter_cost_usd": float(snapshot[cost_key]),
        "model": snapshot,
    }


def _write_raw_checkpoint(
    path: Path | None,
    *,
    stage: str,
    row_ids: Sequence[str],
    raw: dict[str, Any],
) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_stage": stage,
                "row_ids": list(row_ids),
                "responses": raw,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    expected_rows = len(selected_ids(stage))
    expected_branches = expected_rows * ACTION_COUNT
    expected_requests = expected_rows * (1 + ACTION_COUNT)
    branch_count = sum(len(record["branches"]) for record in records)
    base_gates = {
        "all_rows_completed": len(records) == expected_rows,
        "all_branches_completed": branch_count == expected_branches,
        "exact_physical_request_count": (
            int(usage["physical_requests"]) == expected_requests
        ),
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
    }
    if stage == "serving_smoke":
        gates = dict(base_gates)
        gates["all_pass"] = all(gates.values())
        return {
            "num_rows": len(records),
            "num_branches": branch_count,
            "gates": gates,
        }

    initial_omissions = sum(
        int(record["initial_truth_coverage"]) == 0 for record in records
    )
    recoveries = sum(bool(record["omission_recovered"]) for record in records)
    oracle_gains = [
        float(record["oracle_truth_coverage_gain"]) for record in records
    ]
    spreads = [float(record["branch_truth_coverage_spread"]) for record in records]
    regrets = [
        float(record["immediate_eig_truth_coverage_regret"])
        for record in records
    ]
    gold_root_candidates = sum(
        bool(record["gold_root_is_candidate"]) for record in records
    )
    summary = {
        "num_rows": len(records),
        "num_branches": branch_count,
        "gold_root_candidate_count": gold_root_candidates,
        "initial_truth_omission_count": initial_omissions,
        "omission_recovery_count": recoveries,
        "mean_oracle_truth_coverage_gain": float(np.mean(oracle_gains)),
        "rows_with_branch_truth_coverage_spread": sum(value > 0.0 for value in spreads),
        "mean_immediate_eig_truth_coverage_regret": float(np.mean(regrets)),
        "rows_with_immediate_eig_truth_coverage_regret": sum(
            value > 0.0 for value in regrets
        ),
    }
    gates = {
        **base_gates,
        "gold_root_candidate_count_at_least_8": gold_root_candidates >= 8,
        "initial_truth_omission_count_at_least_4": initial_omissions >= 4,
        "omission_recovery_count_at_least_3": recoveries >= 3,
        "mean_oracle_truth_coverage_gain_at_least_0_15": (
            summary["mean_oracle_truth_coverage_gain"] >= 0.15
        ),
        "branch_spread_count_at_least_4": (
            summary["rows_with_branch_truth_coverage_spread"] >= 4
        ),
        "mean_immediate_eig_regret_at_least_0_10": (
            summary["mean_immediate_eig_truth_coverage_regret"] >= 0.10
        ),
        "immediate_eig_regret_count_at_least_3": (
            summary["rows_with_immediate_eig_truth_coverage_regret"] >= 3
        ),
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def run_gate(
    config: Config,
    *,
    data_path: str | Path,
    stage: str,
    raw_checkpoint_path: Path | None = None,
    initial_raw_override: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows = load_selected_rows(data_path, stage)
    if len(config.model_pairs) != 1:
        raise ValueError("MuSiQue gate requires exactly one model pair")
    model = build_model_adapter(config.model_pairs[0].questioner, config)
    row_ids = [str(row["id"]) for row in rows]
    raw: dict[str, Any] = {}

    if initial_raw_override is None:
        initial_raw = model.chat_complete_messages_batched(
            [chain_messages(row) for row in rows],
            temperature=float(config.generation_temperature_diverse),
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
    else:
        initial_raw = list(initial_raw_override)
        if len(initial_raw) != len(rows):
            raise ValueError("resumed initial response count does not match stage rows")
    raw["initial_chains"] = initial_raw
    _write_raw_checkpoint(
        raw_checkpoint_path, stage=stage, row_ids=row_ids, raw=raw
    )
    initial_chains = [
        parse_chains(
            response,
            available_titles=[
                str(paragraph["title"]) for paragraph in row["paragraphs"]
            ],
        )
        for response, row in zip(initial_raw, rows, strict=True)
    ]

    flat_row_indices: list[int] = []
    flat_actions: list[str] = []
    for row_index, chains in enumerate(initial_chains):
        for action in candidate_actions(chains):
            flat_row_indices.append(row_index)
            flat_actions.append(action)
    paragraph_maps = [
        {
            f"d{index + 1:02d}": paragraph
            for index, paragraph in enumerate(row["paragraphs"])
        }
        for row in rows
    ]
    refresh_prompts = []
    for row_index, action in zip(flat_row_indices, flat_actions, strict=True):
        paragraph = paragraph_maps[row_index][action]
        refresh_prompts.append(
            chain_messages(
                rows[row_index],
                previous_chains=initial_chains[row_index],
                opened_document={
                    "doc_id": action,
                    "title": str(paragraph["title"]),
                    "text": str(paragraph["paragraph_text"]),
                },
            )
        )
    refresh_raw = model.chat_complete_messages_batched(
        refresh_prompts,
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["refreshed_chains"] = refresh_raw
    _write_raw_checkpoint(
        raw_checkpoint_path, stage=stage, row_ids=row_ids, raw=raw
    )
    refreshed_chains = [
        parse_chains(
            response,
            available_titles=[
                str(paragraph["title"])
                for paragraph in rows[row_index]["paragraphs"]
            ],
        )
        for response, row_index in zip(
            refresh_raw, flat_row_indices, strict=True
        )
    ]

    records: list[dict[str, Any]] = []
    offset = 0
    for row_index, row in enumerate(rows):
        chains = initial_chains[row_index]
        actions = candidate_actions(chains)
        eig_values = immediate_eig_values(chains, actions)
        truth_doc_ids = gold_doc_ids(row)
        initial_coverage = truth_coverage(chains, truth_doc_ids)
        branches = []
        branch_coverages: list[int] = []
        for action_index, action in enumerate(actions):
            branch_chains = refreshed_chains[offset + action_index]
            coverage = truth_coverage(branch_chains, truth_doc_ids)
            branch_coverages.append(coverage)
            branches.append(
                {
                    "action_index": action_index,
                    "opened_doc_id": action,
                    "opened_title_hash": _text_hash(
                        str(paragraph_maps[row_index][action]["title"])
                    ),
                    "truth_coverage": coverage,
                    "generated_pair_hashes": [
                        _text_hash(
                            chain["first_doc_id"] + "\n" + chain["second_doc_id"]
                        )
                        for chain in branch_chains
                    ],
                }
            )
        offset += ACTION_COUNT
        oracle_coverage = max(branch_coverages)
        d1_index = int(np.argmax(np.asarray(eig_values, dtype=float)))
        random_index = int(
            np.random.default_rng(
                SELECTION_SEED * 1000 + row_index
            ).integers(0, ACTION_COUNT)
        )
        records.append(
            {
                "row_id": row["id"],
                "question_hash": _text_hash(str(row["question"])),
                "truth_doc_ids": list(truth_doc_ids),
                "initial_pair_hashes": [
                    _text_hash(
                        chain["first_doc_id"] + "\n" + chain["second_doc_id"]
                    )
                    for chain in chains
                ],
                "candidate_action_doc_ids": actions,
                "gold_root_is_candidate": truth_doc_ids[0] in actions,
                "initial_truth_coverage": initial_coverage,
                "immediate_eig_values": eig_values,
                "immediate_eig_selected_index": d1_index,
                "random_selected_index": random_index,
                "oracle_truth_coverage": oracle_coverage,
                "oracle_truth_coverage_gain": (
                    oracle_coverage - initial_coverage
                ),
                "omission_recovered": (
                    initial_coverage == 0 and oracle_coverage == 1
                ),
                "branch_truth_coverage_spread": (
                    max(branch_coverages) - min(branch_coverages)
                ),
                "immediate_eig_truth_coverage_regret": (
                    oracle_coverage - branch_coverages[d1_index]
                ),
                "random_truth_coverage_regret": (
                    oracle_coverage - branch_coverages[random_index]
                ),
                "branches": branches,
            }
        )

    usage = _usage(model, cumulative_run=initial_raw_override is not None)
    summary = summarize(records, usage, stage=stage)
    return {
        "schema_version": 3,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "interface_version": 3,
            "selection_seed": SELECTION_SEED,
            "data_sha256": DATA_SHA256,
            "row_ids": row_ids,
            "chain_count": CHAIN_COUNT,
            "action_count": ACTION_COUNT,
            "target_is_ordered_support_title_pair": True,
            "gold_decomposition_hidden_from_all_prompts": True,
            "gold_answer_hidden_from_all_prompts": True,
            "paragraph_support_flags_hidden_from_all_prompts": True,
            "opened_document_is_the_only_branch_observation": True,
            "no_llm_likelihood_or_answerer": True,
            "initial_responses_reused": initial_raw_override is not None,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--resume-initial-raw",
        type=Path,
        help="Reuse hash-locked initial responses from a pre-branch failure.",
    )
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "formal"),
        default="formal",
    )
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.05
        config.openrouter_run_budget_usd = 0.50
    else:
        config.openrouter_projected_cost_usd = 0.15
        config.openrouter_run_budget_usd = 1.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_run_dir = args.private_raw_dir / args.run_id
    private_run_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_run_dir / "RAW_RESPONSES.json"
    initial_raw_override = None
    initial_raw_sha256 = None
    if args.resume_initial_raw is not None:
        initial_raw_sha256 = _sha256(args.resume_initial_raw)
        resumed_payload = json.loads(args.resume_initial_raw.read_text())
        initial_raw_override = resumed_payload["responses"]["initial_chains"]
    output_name = (
        "SERVING_SMOKE.json" if args.stage == "serving_smoke" else "GATE.json"
    )
    failure_name = (
        "SERVING_SMOKE_FAILURE.json"
        if args.stage == "serving_smoke"
        else "GATE_FAILURE.json"
    )
    try:
        payload = run_gate(
            config,
            data_path=args.data_path,
            stage=args.stage,
            raw_checkpoint_path=raw_path,
            initial_raw_override=initial_raw_override,
        )
        payload["protocol"]["private_raw_sha256"] = _sha256(raw_path)
        payload["protocol"]["reused_initial_raw_sha256"] = initial_raw_sha256
    except Exception as exc:
        failure = {
            "schema_version": 3,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_responses_path": str(raw_path),
        }
        (args.output_dir / failure_name).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
