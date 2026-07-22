"""Audit the three direct-vLLM Gemma 4 12B RockSample[15,15] seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.analyze_nonmyopic_rocksample_15_15_vllm_multiseed import (
    analyze_run_set,
    plot_entropy,
    render_summary,
)


RUN_KEYS = (
    "12b_vllm",
    "12b_vllm_seed_24107",
    "12b_vllm_seed_24108",
)
FRESH_RUN_KEYS = RUN_KEYS[1:]
BOOTSTRAP_SEED = 24109


def analyze(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    return analyze_run_set(
        payloads,
        run_keys=RUN_KEYS,
        fresh_run_keys=FRESH_RUN_KEYS,
        bootstrap_seed=BOOTSTRAP_SEED,
        claim="gemma_12b_15_rock_gain_replicates_across_two_fresh_seeds",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs=3, type=Path)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()
    audit = analyze(
        [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    )
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.summary_output.write_text(
        render_summary(
            audit,
            title="RockSample[15,15] Gemma 4 12B Multi-Seed Robustness",
        ),
        encoding="utf-8",
    )
    plot_entropy(
        audit,
        args.plot_output,
        title="RockSample[15,15]: Gemma 4 12B seed robustness",
    )
    print(
        json.dumps(
            {
                "all_12_fresh_seed_intervals_passed": audit[
                    "all_12_fresh_seed_intervals_passed"
                ],
                "all_three_direct_vllm_seeds_passed": audit[
                    "all_three_direct_vllm_seeds_passed"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
