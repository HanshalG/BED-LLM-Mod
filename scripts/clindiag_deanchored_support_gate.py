#!/usr/bin/env python3
"""Run the fresh ClinDiag de-anchored support-refresh serving gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts.clindiag_fixed_slot_support_gate import (
    deanchored_refresh_differential_messages,
    run_smoke,
)


SELECTION_SEED = 24297
SMOKE_IDS = ("21991897", "rare70")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generator-model", default="openai/gpt-5.4")
    parser.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "SERVING_SMOKE.json"
    try:
        payload = run_smoke(
            config,
            args.data_zip,
            generator_model=args.generator_model,
            judge_model=args.judge_model,
            source_ids=SMOKE_IDS,
            selection_seed=SELECTION_SEED,
            refresh_message_builder=deanchored_refresh_differential_messages,
            refresh_mode="deanchored_full_rebuild",
        )
    except Exception as exc:
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "runtime_failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
