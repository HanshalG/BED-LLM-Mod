#!/usr/bin/env python3
"""Sample a fixed Animals benchmark pool from an LLM's implicit prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.animals.beliefs import filter_valid_animal_names_batched
from environments.animals.prompts import generate_original_animals_system_prompt
from helpers import clean_generated_belief_labels, convert_string_to_array, load_config
from model_factory import build_model_adapter


NUM_CALLS = 16
NAMES_PER_CALL = 16
DEVELOPMENT_SIZE = 20
HOLDOUT_SIZE = 60


def dedupe_names(names: list[str]) -> list[str]:
    seen = set()
    result = []
    for name in names:
        key = name.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            result.append(name.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split-seed", type=int, default=24284)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    model = build_model_adapter(config.model_pairs[0].questioner, config=config)
    messages = [
        [
            generate_original_animals_system_prompt(NAMES_PER_CALL),
            {"role": "user", "content": "Generate the animal list now."},
        ]
        for _ in range(NUM_CALLS)
    ]
    completions = model.chat_complete_messages_batched(
        messages,
        temperature=config.generation_temperature_diverse,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw_names = [
        name
        for completion in completions
        for name in convert_string_to_array(completion)
    ]
    cleaned = dedupe_names(clean_generated_belief_labels(raw_names))
    validated = dedupe_names(
        filter_valid_animal_names_batched(
            cleaned,
            model,
            config.batched_block_size,
        )
    )
    required = DEVELOPMENT_SIZE + HOLDOUT_SIZE
    if len(validated) < required:
        raise RuntimeError(
            f"implicit-prior sampler produced {len(validated)}/{required} "
            "required unique validated animals"
        )
    shuffled = list(validated)
    random.Random(args.split_seed).shuffle(shuffled)
    payload = {
        "schema_version": 1,
        "status": "fixed_before_policy_responses",
        "run_id": args.run_id,
        "split_seed": args.split_seed,
        "num_generation_calls": NUM_CALLS,
        "names_per_call": NAMES_PER_CALL,
        "raw_name_count": len(raw_names),
        "unique_cleaned_name_count": len(cleaned),
        "unique_validated_name_count": len(validated),
        "development_targets": shuffled[:DEVELOPMENT_SIZE],
        "holdout_targets": shuffled[
            DEVELOPMENT_SIZE:DEVELOPMENT_SIZE + HOLDOUT_SIZE
        ],
        "unused_validated_targets": shuffled[required:],
        "raw_completions": completions,
        "usage": model.usage_snapshot(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "raw_name_count",
                    "unique_cleaned_name_count",
                    "unique_validated_name_count",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
