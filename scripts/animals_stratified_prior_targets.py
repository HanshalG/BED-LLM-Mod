#!/usr/bin/env python3
"""Sample an Animals benchmark prior from explicit semantic strata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.animals.beliefs import filter_valid_animal_names_batched
from helpers import clean_generated_belief_labels, convert_string_to_array, load_config
from model_factory import build_model_adapter
from scripts.animals_implicit_prior_targets import dedupe_names


STRATA = (
    "mammals",
    "birds",
    "reptiles",
    "amphibians",
    "fish",
    "insects",
    "arachnids and myriapods",
    "mollusks, crustaceans, echinoderms, and cnidarians",
)
NAMES_PER_STRATUM = 16
DEVELOPMENT_SIZE = 20
HOLDOUT_SIZE = 60


def generation_messages(stratum: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Generate exactly 16 diverse existing animal names from the "
                f"taxonomic stratum: {stratum}. Include both familiar and less "
                "common examples. Each name must denote one animal or commonly "
                "used animal label. List one name per line with no numbering, "
                "punctuation, explanation, or extra text. Do not repeat names."
            ),
        },
        {"role": "user", "content": "Generate the list now."},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split-seed", type=int, default=24285)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    model = build_model_adapter(config.model_pairs[0].questioner, config=config)
    completions = model.chat_complete_messages_batched(
        [generation_messages(stratum) for stratum in STRATA],
        temperature=config.generation_temperature_diverse,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    per_stratum = []
    for stratum, completion in zip(STRATA, completions):
        cleaned = dedupe_names(
            clean_generated_belief_labels(
                convert_string_to_array(completion)
            )
        )
        per_stratum.append({"stratum": stratum, "cleaned_names": cleaned})
    all_cleaned = dedupe_names(
        [
            name
            for entry in per_stratum
            for name in entry["cleaned_names"]
        ]
    )
    validated = dedupe_names(
        filter_valid_animal_names_batched(
            all_cleaned,
            model,
            config.batched_block_size,
        )
    )
    required = DEVELOPMENT_SIZE + HOLDOUT_SIZE
    if len(validated) < required:
        raise RuntimeError(
            f"stratified sampler produced {len(validated)}/{required} "
            "required unique validated animals"
        )
    shuffled = list(validated)
    random.Random(args.split_seed).shuffle(shuffled)
    payload = {
        "schema_version": 1,
        "status": "fixed_before_policy_responses",
        "strata": list(STRATA),
        "names_per_stratum": NAMES_PER_STRATUM,
        "split_seed": args.split_seed,
        "per_stratum": per_stratum,
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
                "status": payload["status"],
                "unique_validated_name_count": len(validated),
                "per_stratum_cleaned_counts": {
                    entry["stratum"]: len(entry["cleaned_names"])
                    for entry in per_stratum
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
