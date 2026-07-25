#!/usr/bin/env python3
"""Analyze selection stability of the disclosed ClariQ V2 likelihood run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clariq_multisample_likelihood_development import (
    build_likelihood,
    policy_scores,
)


INPUT_SHA256 = (
    "6daf922d9bfca5c2c28a843fddde58970b13219aa1981ea1836f033cc069c4fc"
)
BOOTSTRAP_SAMPLES = 1_000
BOOTSTRAP_SEED_BASE = 24_402


def _selected(
    maps: dict[str, list[str]],
    indices: Sequence[int],
) -> str:
    likelihoods = {
        question_id: build_likelihood(
            [samples[index] for index in indices],
            len(samples[0]),
        )
        for question_id, samples in maps.items()
    }
    return policy_scores(likelihoods)["depth_two_question_id"]


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    rows = {row["topic_id"]: row for row in payload["rows"]}
    records = []
    for topic_id, maps in payload["maps"].items():
        full = payload["policies"][topic_id]["depth_two_question_id"]
        myopic = payload["policies"][topic_id]["myopic_question_id"]
        sample_count = len(next(iter(maps.values())))
        leave_one_out = [
            _selected(
                maps,
                [index for index in range(sample_count) if index != omitted],
            )
            for omitted in range(sample_count)
        ]
        subset_three = [
            _selected(maps, indices)
            for indices in itertools.combinations(range(sample_count), 3)
        ]
        rng = random.Random(BOOTSTRAP_SEED_BASE + int(topic_id))
        bootstrap = [
            _selected(
                maps,
                [rng.randrange(sample_count) for _ in range(sample_count)],
            )
            for _ in range(BOOTSTRAP_SAMPLES)
        ]
        scores = payload["policies"][topic_id]["depth_two_scores"]
        top = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        records.append(
            {
                "topic_id": topic_id,
                "full_depth_two_question_id": full,
                "myopic_question_id": myopic,
                "root_changed": full != myopic,
                "full_score_margin": top[0][1] - top[1][1],
                "single_sample_modal_count": rows[topic_id][
                    "single_sample_depth_two_modal_count"
                ],
                "leave_one_out_agreement_count": leave_one_out.count(full),
                "leave_one_out_selections": leave_one_out,
                "three_sample_subset_agreement_count": subset_three.count(full),
                "three_sample_subset_selections": subset_three,
                "bootstrap_full_selection_rate": (
                    bootstrap.count(full) / BOOTSTRAP_SAMPLES
                ),
                "bootstrap_top_selections": Counter(bootstrap).most_common(5),
                "oracle_tail_gain_over_myopic": (
                    rows[topic_id]["depth_two_oracle_tail"]
                    - rows[topic_id]["myopic_oracle_tail"]
                ),
            }
        )
    changed = [row for row in records if row["root_changed"]]
    return {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            "analysis_type": "disclosed_development_posthoc",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed_base": BOOTSTRAP_SEED_BASE,
            "api_calls": 0,
        },
        "summary": {
            "topic_count": len(records),
            "changed_root_topic_count": len(changed),
            "changed_roots_with_single_sample_modal_count_at_least_3": sum(
                row["single_sample_modal_count"] >= 3 for row in changed
            ),
            "changed_roots_with_leave_one_out_agreement_at_least_3": sum(
                row["leave_one_out_agreement_count"] >= 3 for row in changed
            ),
            "unchanged_root_topic_count": len(records) - len(changed),
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != INPUT_SHA256:
        raise ValueError("ClariQ V2 input hash does not match")
    result = analyze(json.loads(args.input.read_text(encoding="utf-8")))
    result["protocol"]["input_sha256"] = INPUT_SHA256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
