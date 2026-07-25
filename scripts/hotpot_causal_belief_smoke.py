#!/usr/bin/env python3
"""Run a causal belief-state bottleneck smoke on one HotpotQA bridge task."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.hotpot_directional_unlock_audit import (
    SOURCE_SHA256,
    _support_sentence_texts,
    contains_phrase,
    normalize_text,
    sha256_file,
    split_ids,
    title_bm25_root,
)


MODEL_ID = "openai/gpt-5.4"
INTERFACE_VERSION = "hotpot-causal-belief-smoke-1"
TASK_ID = "5ae0036a55429942ec259bdf"
ROOT_CONTEXT_INDICES = (2, 3, 5, 0)
HYPOTHESIS_COUNT = 8
EXPECTED_REQUESTS = 10
BLINDING_SEED = 24351
RANDOM_CONTROL_SEED = 24352
ALIGNED_IS_A = (False, True, False, True)
MAX_COST_USD = 0.50


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _strict_object(text: str) -> dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("response must be one JSON object")
    return payload


def _score(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("score must not be Boolean")
    if isinstance(value, int):
        score = value
    elif (
        isinstance(value, str)
        and value.isdigit()
        and (len(value) == 1 or not value.startswith("0"))
    ):
        score = int(value)
    else:
        raise ValueError("score must be an integer or canonical digit string")
    if not 0 <= score <= 100:
        raise ValueError("score is outside 0..100")
    return score


def _hypothesis_keys() -> list[str]:
    return [
        f"hypothesis_{index}" for index in range(1, HYPOTHESIS_COUNT + 1)
    ]


def parse_hypotheses(payload: dict[str, Any]) -> list[str]:
    keys = _hypothesis_keys()
    if set(payload) != set(keys):
        raise ValueError("hypothesis response has unexpected keys")
    hypotheses = []
    for key in keys:
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError("hypothesis must be a nonempty string")
        hypotheses.append(" ".join(value.split()))
    normalized = {normalize_text(value) for value in hypotheses}
    if len(normalized) != HYPOTHESIS_COUNT or "" in normalized:
        raise ValueError("hypotheses must be unique after normalization")
    return hypotheses


def parse_initial(text: str) -> dict[str, Any]:
    payload = _strict_object(text)
    expected = set(_hypothesis_keys()) | {
        f"root_{index}_score" for index in range(1, 5)
    }
    if set(payload) != expected:
        raise ValueError("initial response has unexpected keys")
    hypotheses = parse_hypotheses(
        {key: payload[key] for key in _hypothesis_keys()}
    )
    return {
        "hypotheses": hypotheses,
        "root_scores": [
            _score(payload[f"root_{index}_score"]) for index in range(1, 5)
        ],
    }


def parse_refresh(text: str) -> list[str]:
    return parse_hypotheses(_strict_object(text))


def scorer_keys(candidate_count: int) -> list[str]:
    return [
        f"state_{state}_title_{index}_score"
        for state in ("a", "b", "c")
        for index in range(1, candidate_count + 1)
    ]


def parse_scorer(text: str, candidate_count: int) -> dict[str, list[int]]:
    payload = _strict_object(text)
    expected = set(scorer_keys(candidate_count))
    if set(payload) != expected:
        raise ValueError("continuation scorer has unexpected keys")
    return {
        state: [
            _score(payload[f"state_{state}_title_{index}_score"])
            for index in range(1, candidate_count + 1)
        ]
        for state in ("a", "b", "c")
    }


def parse_final(text: str) -> dict[str, Any]:
    payload = _strict_object(text)
    if set(payload) != {"answer", "confidence"}:
        raise ValueError("final answer response has unexpected keys")
    answer = payload["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("final answer must be nonempty")
    return {"answer": " ".join(answer.split()), "confidence": _score(payload["confidence"])}


def initial_messages(
    question: str,
    titles: Sequence[str],
    root_titles: Sequence[str],
) -> list[dict[str, str]]:
    schema = {
        **{
            key: "one distinct plausible answer/support-chain hypothesis"
            for key in _hypothesis_keys()
        },
        **{
            f"root_{index}_score": "integer 0..100"
            for index in range(1, 5)
        },
    }
    payload = {
        "question": question,
        "candidate_titles": [
            {"id": f"C{index}", "title": title}
            for index, title in enumerate(titles, start=1)
        ],
        "root_candidates": [
            {"id": f"R{index}", "title": title}
            for index, title in enumerate(root_titles, start=1)
        ],
        "required_output": schema,
    }
    return [
        {
            "role": "system",
            "content": (
                "You maintain an open-world belief state for a two-document "
                "question. Generate eight diverse plausible answer/support-chain "
                "hypotheses from the question and candidate article titles. Score "
                "each root only by how much reading that one article alone is likely "
                "to advance the answer. Use the full 0..100 range to rank roots. "
                "Return exactly the requested flat JSON object, with no markdown or "
                "explanation."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def refresh_messages(
    question: str,
    initial_hypotheses: Sequence[str],
    root_title: str,
    root_paragraph: str,
) -> list[dict[str, str]]:
    payload = {
        "question": question,
        "prior_hypotheses": list(initial_hypotheses),
        "revealed_article": {
            "title": root_title,
            "paragraph": root_paragraph,
        },
        "required_output": {
            key: "one distinct unresolved answer/support-chain hypothesis"
            for key in _hypothesis_keys()
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "Refresh the open-world belief state after one article is revealed. "
                "Return eight distinct unresolved hypotheses about the answer and "
                "which evidence or article is still needed. Preserve supported ideas, "
                "revise contradicted ones, and make the state sufficient for another "
                "agent to choose the next article without seeing the question or "
                "revealed paragraph. Do not score actions. Return exactly the requested "
                "flat JSON object and no explanation."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def scorer_messages(
    *,
    aligned: Sequence[str],
    shuffled: Sequence[str],
    initial: Sequence[str],
    candidates: Sequence[str],
    aligned_is_a: bool,
) -> list[dict[str, str]]:
    state_a, state_b = (
        (aligned, shuffled) if aligned_is_a else (shuffled, aligned)
    )
    payload = {
        "state_a_unresolved_hypotheses": list(state_a),
        "state_b_unresolved_hypotheses": list(state_b),
        "state_c_unresolved_hypotheses": list(initial),
        "candidate_titles": [
            {"id": f"T{index}", "title": title}
            for index, title in enumerate(candidates, start=1)
        ],
        "required_output": {
            key: "integer 0..100" for key in scorer_keys(len(candidates))
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "Score candidate article titles using only each supplied unresolved "
                "belief state. Treat states A, B, and C independently. A high score "
                "means reading that title is likely to resolve the state's missing "
                "evidence and answer uncertainty. Rank candidates sharply; do not "
                "give every title the same score. You do not know the original "
                "question or prior article. Return exactly the requested flat JSON "
                "object with no markdown or explanation."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def final_messages(
    question: str,
    evidence: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Answer the question using only the supplied articles. Return exactly "
                'one JSON object {"answer":"...","confidence":0} with confidence '
                "as an integer from 0 to 100 and no explanation."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"question": question, "revealed_articles": list(evidence)},
                separators=(",", ":"),
            ),
        },
    ]


def _argmax(values: Sequence[int]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _normalized_state(values: Sequence[str]) -> str:
    return normalize_text(" ".join(values))


def _candidate_titles(titles: Sequence[str], root_context_index: int) -> list[str]:
    return [
        title for index, title in enumerate(titles) if index != root_context_index
    ]


def policy_selections(
    *,
    immediate_scores: Sequence[int],
    continuation_rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    aligned_max = [max(row["aligned_scores"]) for row in continuation_rows]
    shuffled_max = [max(row["shuffled_scores"]) for row in continuation_rows]
    initial_max = [max(row["initial_scores"]) for row in continuation_rows]
    root_scores = {
        "myopic": list(immediate_scores),
        "fixed": [
            immediate + continuation
            for immediate, continuation in zip(immediate_scores, initial_max)
        ],
        "model_aware": [
            immediate + continuation
            for immediate, continuation in zip(immediate_scores, aligned_max)
        ],
        "shuffled": [
            immediate + continuation
            for immediate, continuation in zip(immediate_scores, shuffled_max)
        ],
    }
    roots = {name: _argmax(scores) for name, scores in root_scores.items()}
    roots["random"] = random.Random(RANDOM_CONTROL_SEED).randrange(4)
    selections: dict[str, dict[str, int]] = {}
    for name, root_index in roots.items():
        row = continuation_rows[root_index]
        score_key = {
            "fixed": "initial_scores",
            "shuffled": "shuffled_scores",
        }.get(name, "aligned_scores")
        selections[name] = {
            "root_index": root_index,
            "followup_candidate_index": _argmax(row[score_key]),
        }
    return selections


def support_coverage(
    *,
    selection: dict[str, int],
    titles: Sequence[str],
    support_titles: set[str],
) -> tuple[int, list[str]]:
    root_context_index = ROOT_CONTEXT_INDICES[selection["root_index"]]
    candidates = _candidate_titles(titles, root_context_index)
    chosen = [
        titles[root_context_index],
        candidates[selection["followup_candidate_index"]],
    ]
    return len(set(chosen) & support_titles), chosen


def token_f1(prediction: str, answer: str) -> float:
    predicted = normalize_text(prediction).split()
    target = normalize_text(answer).split()
    if not predicted or not target:
        return 0.0
    remaining: dict[str, int] = {}
    for token in target:
        remaining[token] = remaining.get(token, 0) + 1
    overlap = 0
    for token in predicted:
        if remaining.get(token, 0) > 0:
            overlap += 1
            remaining[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(target)
    return 2 * precision * recall / (precision + recall)


def _answer_and_enabling_titles(row: dict[str, Any]) -> tuple[str, str]:
    support = row["supporting_facts"]
    support_titles = list(dict.fromkeys(str(value) for value in support["title"]))
    texts = _support_sentence_texts(
        support_titles=[str(value) for value in support["title"]],
        support_sentence_ids=[int(value) for value in support["sent_id"]],
        context=row["context"],
    )
    answer = normalize_text(str(row["answer"]))
    answer_title = next(
        title
        for title in support_titles
        if contains_phrase(normalize_text(texts[title]), answer)
    )
    enabling_title = next(title for title in support_titles if title != answer_title)
    return answer_title, enabling_title


def load_task(path: Path) -> dict[str, Any]:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("HotpotQA source hash mismatch")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow==21.0.0 is required") from exc
    table = pq.read_table(path)
    matches = [row for row in table.to_pylist() if str(row["id"]) == TASK_ID]
    if len(matches) != 1:
        raise ValueError("frozen HotpotQA smoke task does not reproduce")
    row = matches[0]
    metadata = pq.read_table(path, columns=["id", "type", "level"]).to_pylist()
    splits = split_ids(metadata)
    if TASK_ID not in set(splits["opportunity"]):
        raise ValueError("smoke task is not in the public opportunity split")
    titles = [str(value) for value in row["context"]["title"]]
    if len(titles) != 10:
        raise ValueError("smoke task does not have ten context titles")
    root_titles = [titles[index] for index in ROOT_CONTEXT_INDICES]
    if title_bm25_root(str(row["question"]), titles) != root_titles[0]:
        raise ValueError("frozen title-BM25 root does not reproduce")
    answer_title, enabling_title = _answer_and_enabling_titles(row)
    if answer_title != root_titles[0] or enabling_title != root_titles[1]:
        raise ValueError("frozen answer/enabling roles do not reproduce")
    return row


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot["adapter_requests"]),
        "reasoning_tokens": int(snapshot["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(snapshot["adapter_cost_usd"]),
        "generator": snapshot,
    }


def _build_model(config: Config) -> Any:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("Hotpot smoke config selects the wrong model")
    return build_model_adapter(spec, config)


def _checkpoint(path: Path, raw: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_smoke(
    config: Config,
    *,
    data_path: Path,
    raw_path: Path,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    row = load_task(data_path)
    model = model_adapter if model_adapter is not None else _build_model(config)
    titles = [str(value) for value in row["context"]["title"]]
    paragraphs = [
        " ".join(str(sentence) for sentence in sentences)
        for sentences in row["context"]["sentences"]
    ]
    root_titles = [titles[index] for index in ROOT_CONTEXT_INDICES]
    raw: dict[str, Any] = {"task_id": TASK_ID}
    try:
        initial_response = model.chat_complete_messages_batched(
            [initial_messages(str(row["question"]), titles, root_titles)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=config.openrouter_max_output_tokens,
        )[0]
        raw["initial"] = initial_response
        initial = parse_initial(initial_response)
        refresh_responses = model.chat_complete_messages_batched(
            [
                refresh_messages(
                    str(row["question"]),
                    initial["hypotheses"],
                    titles[context_index],
                    paragraphs[context_index],
                )
                for context_index in ROOT_CONTEXT_INDICES
            ],
            temperature=0.0,
            block_size=4,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["refreshes"] = refresh_responses
        refreshes = [parse_refresh(text) for text in refresh_responses]
        scorer_responses = model.chat_complete_messages_batched(
            [
                scorer_messages(
                    aligned=refreshes[root_index],
                    shuffled=refreshes[(root_index + 1) % 4],
                    initial=initial["hypotheses"],
                    candidates=_candidate_titles(titles, context_index),
                    aligned_is_a=ALIGNED_IS_A[root_index],
                )
                for root_index, context_index in enumerate(ROOT_CONTEXT_INDICES)
            ],
            temperature=0.0,
            block_size=4,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["scorers"] = scorer_responses
        parsed_scorers = [parse_scorer(text, 9) for text in scorer_responses]
        continuation_rows = []
        for root_index, scores in enumerate(parsed_scorers):
            aligned_label = "a" if ALIGNED_IS_A[root_index] else "b"
            shuffled_label = "b" if ALIGNED_IS_A[root_index] else "a"
            continuation_rows.append(
                {
                    "root_index": root_index,
                    "aligned_label": aligned_label.upper(),
                    "aligned_scores": scores[aligned_label],
                    "shuffled_scores": scores[shuffled_label],
                    "initial_scores": scores["c"],
                }
            )
        selections = policy_selections(
            immediate_scores=initial["root_scores"],
            continuation_rows=continuation_rows,
        )
        support_titles = set(str(value) for value in row["supporting_facts"]["title"])
        policies = {}
        for name, selection in selections.items():
            coverage, chosen_titles = support_coverage(
                selection=selection,
                titles=titles,
                support_titles=support_titles,
            )
            policies[name] = {
                **selection,
                "selected_titles": chosen_titles,
                "support_coverage": coverage,
            }
        model_evidence = [
            {
                "title": title,
                "paragraph": paragraphs[titles.index(title)],
            }
            for title in policies["model_aware"]["selected_titles"]
        ]
        final_response = model.chat_complete_messages_batched(
            [final_messages(str(row["question"]), model_evidence)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=config.openrouter_max_output_tokens,
        )[0]
        raw["final"] = final_response
        _checkpoint(raw_path, raw)
        final = parse_final(final_response)
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc

    answer_title, enabling_title = _answer_and_enabling_titles(row)
    state_names = [_normalized_state(values) for values in refreshes]
    initial_name = _normalized_state(initial["hypotheses"])
    aligned_variation = sum(
        len(set(item["aligned_scores"])) > 1 for item in continuation_rows
    )
    aligned_shuffled_changes = sum(
        item["aligned_scores"] != item["shuffled_scores"]
        for item in continuation_rows
    )
    aligned_initial_changes = sum(
        item["aligned_scores"] != item["initial_scores"]
        for item in continuation_rows
    )
    enabling_root_index = root_titles.index(enabling_title)
    enabling_candidates = _candidate_titles(
        titles, ROOT_CONTEXT_INDICES[enabling_root_index]
    )
    enabling_followup = enabling_candidates[
        _argmax(continuation_rows[enabling_root_index]["aligned_scores"])
    ]
    final_f1 = token_f1(final["answer"], str(row["answer"]))
    generator = usage["generator"]
    gates = {
        "exact_10_physical_requests": usage["physical_requests"] == 10,
        "exact_10_http_attempts": int(generator.get("http_attempts", -1)) == 10,
        "zero_transport_retries": int(generator.get("retry_count", -1)) == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": int(generator.get("forced_exits", -1)) == 0,
        "all_responses_parsed_without_repair": True,
        "all_refreshed_states_differ_from_initial": all(
            name != initial_name for name in state_names
        ),
        "all_refreshed_states_pairwise_distinct": len(set(state_names)) == 4,
        "balanced_blinding_labels": sum(ALIGNED_IS_A) == 2,
        "aligned_score_variation_at_least_3_roots": aligned_variation >= 3,
        "aligned_shuffled_vectors_differ_at_least_3_roots": (
            aligned_shuffled_changes >= 3
        ),
        "aligned_initial_vectors_differ_at_least_3_roots": (
            aligned_initial_changes >= 3
        ),
        "enabling_aligned_followup_selects_answer_support": (
            enabling_followup == answer_title
        ),
        "model_aware_selects_enabling_root": (
            policies["model_aware"]["selected_titles"][0] == enabling_title
        ),
        "model_aware_support_coverage_equals_2": (
            policies["model_aware"]["support_coverage"] == 2
        ),
        "model_aware_coverage_exceeds_myopic": (
            policies["model_aware"]["support_coverage"]
            > policies["myopic"]["support_coverage"]
        ),
        "model_aware_coverage_at_least_fixed": (
            policies["model_aware"]["support_coverage"]
            >= policies["fixed"]["support_coverage"]
        ),
        "model_aware_coverage_at_least_shuffled": (
            policies["model_aware"]["support_coverage"]
            >= policies["shuffled"]["support_coverage"]
        ),
        "final_answer_token_f1_positive": final_f1 > 0.0,
        "cost_at_most_0_50": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "task_id": TASK_ID,
            "root_context_indices_one_based": [
                value + 1 for value in ROOT_CONTEXT_INDICES
            ],
            "blinding_seed": BLINDING_SEED,
            "aligned_is_a": list(ALIGNED_IS_A),
            "random_control_seed": RANDOM_CONTROL_SEED,
            "question_hidden_from_continuation_scorer": True,
            "root_evidence_hidden_from_continuation_scorer": True,
            "reasoning_requested": False,
            "repairs_or_scientific_retries": 0,
        },
        "summary": {
            "gates": gates,
            "aligned_score_variation_count": aligned_variation,
            "aligned_shuffled_vector_change_count": aligned_shuffled_changes,
            "aligned_initial_vector_change_count": aligned_initial_changes,
            "enabling_root_index": enabling_root_index,
            "enabling_aligned_followup_is_answer": enabling_followup == answer_title,
            "final_answer_token_f1": final_f1,
        },
        "initial": initial,
        "refreshes": refreshes,
        "continuation_rows": continuation_rows,
        "policies": policies,
        "final_answer": final,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.15
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 4
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        payload = run_smoke(config, data_path=args.data, raw_path=raw_path)
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        (args.output_dir / "SMOKE_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output = args.output_dir / "SMOKE.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "summary": payload["summary"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
