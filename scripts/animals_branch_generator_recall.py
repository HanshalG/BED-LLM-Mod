#!/usr/bin/env python3
"""Screen target-blind belief generators on fixed Animals branch histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.animals.beliefs import _update_beliefs_many
from helpers import load_config
from model_factory import build_model_adapter


def branch_histories(record: dict[str, Any]) -> list[list[dict[str, str]]]:
    base = [
        message
        for item in record["history"]
        for message in (
            {"role": "assistant", "content": str(item["question"])},
            {"role": "user", "content": str(item["answer"])},
        )
    ]
    return [
        base
        + [
            {"role": "assistant", "content": str(candidate["question"])},
            {"role": "user", "content": answer},
        ]
        for candidate in record["candidate_dynamics"]
        for answer in ("Yes", "No")
    ]


def run_screen(
    records: list[dict[str, Any]],
    config: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    model = build_model_adapter(config.model_pairs[0].questioner, config=config)
    screened = []
    for record in records:
        histories = branch_histories(record)
        beliefs = _update_beliefs_many(
            histories,
            record["belief_support"],
            model,
            False,
            config,
        )
        supports = [list(state.hypotheses) for state in beliefs]
        target_key = str(record["target_measurement_only"]).strip().casefold()
        branch_hits = [
            any(str(item).strip().casefold() == target_key for item in support)
            for support in supports
        ]
        candidates = []
        for index, source in enumerate(record["candidate_dynamics"]):
            yes_hit = branch_hits[2 * index]
            no_hit = branch_hits[2 * index + 1]
            candidates.append(
                {
                    "question": source["question"],
                    "p_yes": float(source["p_yes"]),
                    "p_no": float(source["p_no"]),
                    "support_if_yes": supports[2 * index],
                    "support_if_no": supports[2 * index + 1],
                    "truth_covered_if_yes": yes_hit,
                    "truth_covered_if_no": no_hit,
                    "expected_truth_coverage": (
                        float(source["p_yes"]) * float(yes_hit)
                        + float(source["p_no"]) * float(no_hit)
                    ),
                }
            )
        screened.append(
            {
                "state_index": record["state_index"],
                "target_measurement_only": record["target_measurement_only"],
                "history": record["history"],
                "belief_support": record["belief_support"],
                "candidate_dynamics": candidates,
                "truth_covered_in_branch_union": any(branch_hits),
            }
        )
    union_hits = sum(
        record["truth_covered_in_branch_union"] for record in screened
    )
    summary = {
        "num_states": len(screened),
        "num_branches": sum(
            2 * len(record["candidate_dynamics"]) for record in screened
        ),
        "states_with_truth_in_branch_union": union_hits,
        "branch_union_truth_recall": (
            union_hits / len(screened) if screened else None
        ),
        "states_with_nonzero_candidate_coverage": sum(
            max(
                candidate["expected_truth_coverage"]
                for candidate in record["candidate_dynamics"]
            )
            > 0.0
            for record in screened
        ),
    }
    return screened, summary, model.usage_snapshot()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    source = json.loads(args.input.read_text())
    records = source.get("records")
    if not isinstance(records, list):
        raise ValueError("input does not contain records")
    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    screened, summary, usage = run_screen(records, config)
    payload = {
        "schema_version": 1,
        "status": "development_only",
        "target_measurement_only": True,
        "source_path": str(args.input),
        "config_path": str(args.config),
        "run_id": args.run_id,
        "summary": summary,
        "records": screened,
        "usage": usage,
    }
    (args.output_dir / "GENERATOR_RECALL.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
