#!/usr/bin/env python3
"""Build a non-gating diagnostic for tau receding V3 zero-padded scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.tau_knowledge_receding_continuation import (
    load_public_records,
    summarize,
)
from scripts.tau_knowledge_receding_continuation_v3 import _score_schema


def normalize_diagnostic_scores(text: str) -> tuple[dict[str, Any], bool]:
    payload = _parse_json_object(text)
    if set(payload) != set(_score_schema()):
        raise ValueError("diagnostic response has unexpected keys")
    scores = []
    noncanonical = False
    for index in range(1, 5):
        value = payload[f"followup_{index}_score"]
        if not isinstance(value, str) or not value.isdigit():
            raise ValueError("diagnostic score is not a digit string")
        score = int(value)
        if not (
            0 <= score <= 9
            or 30 <= score <= 39
            or 60 <= score <= 69
            or 90 <= score <= 99
        ):
            raise ValueError("diagnostic score is outside a valid band")
        noncanonical |= value != str(score)
        scores.append(score)
    return {
        "scores": scores,
        "rationales": ["posthoc zero-padding diagnostic"] * 4,
    }, noncanonical


def analyze(
    *,
    input_artifact: str | Path,
    failure_artifact: str | Path,
    private_raw_path: str | Path,
) -> dict[str, Any]:
    records, myopic_scores, nonmyopic_scores = load_public_records(
        input_artifact,
        stage="development",
    )
    failure = json.loads(Path(failure_artifact).read_text(encoding="utf-8"))
    raw_path = Path(private_raw_path)
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if raw_sha256 != failure.get("private_raw_sha256"):
        raise ValueError("private raw response hash does not match failure")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    responses = raw.get("focused_continuations")
    if not isinstance(responses, list) or len(responses) != 100:
        raise ValueError("private raw response count does not match")
    parsed = []
    noncanonical_indices = []
    for index, text in enumerate(responses):
        score, noncanonical = normalize_diagnostic_scores(text)
        parsed.append(score)
        if noncanonical:
            noncanonical_indices.append(index)
    continuation_scores = [
        parsed[index * 5 : (index + 1) * 5]
        for index in range(len(records))
    ]
    summary = summarize(
        records,
        continuation_scores,
        failure["usage"],
        stage="development",
        myopic_scores=myopic_scores,
        nonmyopic_scores=nonmyopic_scores,
    )
    return {
        "status": "posthoc_diagnostic_not_a_gate_result",
        "warning": (
            "Leading-zero normalization was not preregistered and cannot "
            "release confirmation."
        ),
        "private_raw_sha256": raw_sha256,
        "num_responses": len(responses),
        "num_noncanonical_responses": len(noncanonical_indices),
        "noncanonical_response_indices": noncanonical_indices,
        "normalized_continuation_scores": continuation_scores,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--failure-artifact", type=Path, required=True)
    parser.add_argument("--private-raw-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        input_artifact=args.input_artifact,
        failure_artifact=args.failure_artifact,
        private_raw_path=args.private_raw_path,
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"]["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
