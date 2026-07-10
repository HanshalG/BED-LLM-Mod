#!/usr/bin/env python3
"""Deterministic five-task smoke for Paprika adapter mechanics (no LLM claim)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.experiment import run_from_config
from environments.paprika_customer_service.data import PAPRIKA_COMMIT, PAPRIKA_REPOSITORY
from helpers import Config


class _RoutingQuestioner:
    def chat_complete(self, messages, temperature, num_responses=1):
        del temperature, num_responses
        text = messages[-1]["content"]
        if '"refined_hypotheses"' in text:
            count = int(re.search(r"exactly (\d+)", text).group(1))
            return [json.dumps({"refined_hypotheses": [f"refined candidate cause {i} with remedy" for i in range(count)]})]
        if '"keep_indices"' in text:
            indices = [int(value) for value in re.findall(r"^(\d+):", text, re.MULTILINE)]
            return [json.dumps({"keep_indices": indices})]
        if '"hypotheses"' in text:
            count = int(re.search(r"exactly (\d+)", text).group(1))
            return [json.dumps({"hypotheses": [f"candidate cause {i} with remedy {i}" for i in range(count)]})]
        if '"candidates"' in text:
            count = int(re.search(r"exactly (\d+)", text).group(1))
            return [json.dumps({"candidates": [{"query": f"Run diagnostic check {i}?", "outcomes": ["positive", "negative", "unknown"]} for i in range(count)]})]
        if '"probabilities"' in text:
            return [json.dumps({"probabilities": {"positive": 0.6, "negative": 0.3, "unknown": 0.1}})]
        if '"outcome"' in text and '"clean"' in text:
            return [json.dumps({"outcome": "positive", "clean": True})]
        if "Reply with <VALID>" in text:
            return ["<NOTVALID>"]
        raise RuntimeError("Unrecognized smoke prompt")

    def chat_complete_messages_batched(
        self, batch_messages, temperature, block_size, max_new_tokens=None
    ):
        del block_size, max_new_tokens
        return [self.chat_complete(messages, temperature)[0] for messages in batch_messages]


class _RoutingCustomer:
    def chat_complete(self, messages, temperature, num_responses=1):
        del messages, temperature, num_responses
        return ["The diagnostic result is positive."]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/path_e/step0a_adapter_smoke"))
    args = parser.parse_args()
    config = Config(
        task="paprika_customer_service",
        method_names=["EIG"],
        paprika_data_path=str(args.data_path),
        paprika_verify_official_hash=True,
        paprika_num_trials=5,
        paprika_num_rounds=2,
        paprika_num_hypotheses=3,
        paprika_num_candidates=2,
    )
    run_result, summary = run_from_config(
        config, _RoutingQuestioner(), _RoutingCustomer(), output_dir=args.output_dir
    )
    coverage = [value for trial in run_result.trials for value in [trial.final_metrics["answer_set_coverage"]]]
    report = {
        "status": "adapter_mechanics_only",
        "not_llm_evidence": True,
        "source_repository": PAPRIKA_REPOSITORY,
        "source_commit": PAPRIKA_COMMIT,
        "num_tasks": len(run_result.trials),
        "rounds_per_task": [len(trial.rounds) for trial in run_result.trials],
        "mean_final_answer_set_coverage": sum(coverage) / len(coverage),
        "metric_traces": summary.metrics,
    }
    report_path = args.output_dir / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
