#!/usr/bin/env python3
"""Confirm expected support retention on the sealed Animals holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.analyze_animals_transition_selectors import (
    select_index,
    within_state_pairwise_accuracy,
)
from scripts.animals_cabed_aligned_v13 import (
    OversampledAlignedAnimalsEnvironment,
    RecordingBatchedSemanticModel,
)
from scripts.animals_support_expansion_policy import (
    BOOTSTRAP_SEED,
    CANDIDATE_WIDTH,
    TARGET_POOL_SHA256,
    _bootstrap_interval,
    _private_payload,
    _sign_test_pvalue,
    load_target_pool,
    run_state,
    sha256_file,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "animals-support-retention-aligned-holdout-1"
CONFIG_SHA256 = (
    "0f291564e034103be1021dc2fb1a46d47a5be3f4b0719f5781e171d1f6ace052"
)
PROJECTED_COST_USD = 1.0
RUN_COST_CAP_USD = 2.0
HOLDOUT_SIZE = 60
MIN_SELECTOR_CHANGES = 20


class SupportRetentionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        usage: Mapping[str, Any],
        records: Sequence[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.usage = dict(usage)
        self.records = list(records)


def holdout_targets(pool: Mapping[str, Any]) -> list[str]:
    targets = list(pool["holdout_targets"])
    if len(targets) != HOLDOUT_SIZE or len(set(targets)) != HOLDOUT_SIZE:
        raise ValueError("support-retention holdout must contain 60 names")
    return targets


def _selected_values(
    records: Sequence[Mapping[str, Any]],
    score_key: str,
    value,
) -> tuple[list[int], list[float]]:
    indices = [
        select_index(record["candidate_dynamics"], score_key)
        for record in records
    ]
    values = [
        float(value(record["candidate_dynamics"][index]))
        for record, index in zip(records, indices, strict=True)
    ]
    return indices, values


def _comparison(
    selected: Sequence[float],
    baseline: Sequence[float],
) -> dict[str, Any]:
    differences = [
        left - right
        for left, right in zip(selected, baseline, strict=True)
    ]
    wins = sum(value > 0.0 for value in differences)
    ties = sum(value == 0.0 for value in differences)
    losses = sum(value < 0.0 for value in differences)
    return {
        "mean_paired_gain": float(np.mean(differences)),
        "paired_bootstrap_95": _bootstrap_interval(
            differences,
            seed=BOOTSTRAP_SEED,
        ),
        "wins_ties_losses": [wins, ties, losses],
        "exact_one_sided_sign_p": _sign_test_pvalue(wins, losses),
    }


def _augmented_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        output.append(
            {
                **record,
                "candidate_dynamics": [
                    {
                        **candidate,
                        "realized_truth_coverage": float(
                            candidate["realized_endpoint"][
                                "truth_covered"
                            ]
                        ),
                    }
                    for candidate in record["candidate_dynamics"]
                ],
            }
        )
    return output


def summarize_holdout(
    records: Sequence[dict[str, Any]],
    usage: Mapping[str, Any],
) -> dict[str, Any]:
    score_keys = {
        "support_retention": "expected_current_support_retention",
        "immediate_eig": "immediate_eig",
        "support_expansion": "expected_support_size",
    }
    selected_indices: dict[str, list[int]] = {}
    selector_metrics: dict[str, dict[str, float | int]] = {}
    for name, score_key in score_keys.items():
        indices, coverage = _selected_values(
            records,
            score_key,
            lambda candidate: candidate["realized_endpoint"][
                "truth_covered"
            ],
        )
        _, expected_coverage = _selected_values(
            records,
            score_key,
            lambda candidate: candidate["expected_truth_coverage"],
        )
        _, truth_probability = _selected_values(
            records,
            score_key,
            lambda candidate: candidate["realized_endpoint"][
                "uniform_truth_probability"
            ],
        )
        selected_indices[name] = indices
        selector_metrics[name] = {
            "mean_realized_truth_coverage": float(np.mean(coverage)),
            "mean_expected_truth_coverage": float(
                np.mean(expected_coverage)
            ),
            "mean_uniform_truth_probability": float(
                np.mean(truth_probability)
            ),
            "recoveries_after_initial_omission": sum(
                not bool(
                    record["truth_covered_before_counterfactuals"]
                )
                and bool(value)
                for record, value in zip(records, coverage, strict=True)
            ),
        }

    random_indices = [
        int(record["selector_indices"]["random"])
        for record in records
    ]
    random_coverage = [
        float(
            record["candidate_dynamics"][index]["realized_endpoint"][
                "truth_covered"
            ]
        )
        for record, index in zip(records, random_indices, strict=True)
    ]
    random_expected_coverage = [
        float(
            record["candidate_dynamics"][index][
                "expected_truth_coverage"
            ]
        )
        for record, index in zip(records, random_indices, strict=True)
    ]
    random_truth_probability = [
        float(
            record["candidate_dynamics"][index]["realized_endpoint"][
                "uniform_truth_probability"
            ]
        )
        for record, index in zip(records, random_indices, strict=True)
    ]
    selector_metrics["random"] = {
        "mean_realized_truth_coverage": float(
            np.mean(random_coverage)
        ),
        "mean_expected_truth_coverage": float(
            np.mean(random_expected_coverage)
        ),
        "mean_uniform_truth_probability": float(
            np.mean(random_truth_probability)
        ),
        "recoveries_after_initial_omission": sum(
            not bool(record["truth_covered_before_counterfactuals"])
            and bool(value)
            for record, value in zip(
                records,
                random_coverage,
                strict=True,
            )
        ),
    }

    retention_indices = selected_indices["support_retention"]
    eig_indices = selected_indices["immediate_eig"]
    retention_coverage = [
        float(
            record["candidate_dynamics"][index]["realized_endpoint"][
                "truth_covered"
            ]
        )
        for record, index in zip(
            records,
            retention_indices,
            strict=True,
        )
    ]
    eig_coverage = [
        float(
            record["candidate_dynamics"][index]["realized_endpoint"][
                "truth_covered"
            ]
        )
        for record, index in zip(records, eig_indices, strict=True)
    ]
    retention_expected = [
        float(
            record["candidate_dynamics"][index][
                "expected_truth_coverage"
            ]
        )
        for record, index in zip(
            records,
            retention_indices,
            strict=True,
        )
    ]
    eig_expected = [
        float(
            record["candidate_dynamics"][index][
                "expected_truth_coverage"
            ]
        )
        for record, index in zip(records, eig_indices, strict=True)
    ]
    retention_vs_eig = _comparison(
        retention_coverage,
        eig_coverage,
    )
    retention_vs_eig["mean_expected_coverage_gain"] = float(
        np.mean(
            [
                left - right
                for left, right in zip(
                    retention_expected,
                    eig_expected,
                    strict=True,
                )
            ]
        )
    )
    retention_vs_random = _comparison(
        retention_coverage,
        random_coverage,
    )
    pairwise_records = _augmented_records(records)
    pairwise = {
        name: within_state_pairwise_accuracy(
            pairwise_records,
            score_key,
            "realized_truth_coverage",
        )
        for name, score_key in score_keys.items()
    }

    wins, _ties, losses = retention_vs_eig["wins_ties_losses"]
    random_wins, _random_ties, random_losses = retention_vs_random[
        "wins_ties_losses"
    ]
    bootstrap_low = retention_vs_eig["paired_bootstrap_95"][0]
    selector_changes = sum(
        left != right
        for left, right in zip(
            retention_indices,
            eig_indices,
            strict=True,
        )
    )
    retries = int(usage.get("retry_count", 0))
    gates = {
        "all_60_states_completed": len(records) == HOLDOUT_SIZE,
        "all_states_have_three_candidates": all(
            len(record["candidate_dynamics"]) == CANDIDATE_WIDTH
            for record in records
        ),
        "retention_changes_at_least_20_selections": (
            selector_changes >= MIN_SELECTOR_CHANGES
        ),
        "eig_endpoint_not_saturated": (
            0.05
            <= selector_metrics["immediate_eig"][
                "mean_realized_truth_coverage"
            ]
            <= 0.95
        ),
        "retention_realized_gain_positive": (
            retention_vs_eig["mean_paired_gain"] > 0.0
        ),
        "retention_bootstrap_lower_bound_positive": bootstrap_low > 0.0,
        "retention_sign_test_at_most_0_05": (
            retention_vs_eig["exact_one_sided_sign_p"] <= 0.05
        ),
        "retention_wins_exceed_losses": wins > losses,
        "retention_expected_coverage_gain_positive": (
            retention_vs_eig["mean_expected_coverage_gain"] > 0.0
        ),
        "retention_recovers_at_least_as_many_as_eig": (
            selector_metrics["support_retention"][
                "recoveries_after_initial_omission"
            ]
            >= selector_metrics["immediate_eig"][
                "recoveries_after_initial_omission"
            ]
        ),
        "retention_beats_seeded_random_mean": (
            retention_vs_random["mean_paired_gain"] > 0.0
        ),
        "retention_wins_exceed_losses_vs_random": (
            random_wins > random_losses
        ),
        "zero_reasoning_tokens": int(
            usage.get("adapter_reasoning_tokens", 0)
        )
        == 0,
        "zero_forced_exits": int(usage.get("forced_exits", 0)) == 0,
        "zero_retries": retries == 0,
        "within_cost_cap": float(
            usage.get("adapter_cost_usd", 0.0)
        )
        <= RUN_COST_CAP_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "selector_metrics": selector_metrics,
        "support_retention_vs_eig": retention_vs_eig,
        "support_retention_vs_random": retention_vs_random,
        "within_state_realized_coverage_pairwise": pairwise,
        "selector_changes_vs_eig": selector_changes,
        "gates": gates,
    }


def run_holdout(
    config: Config,
    *,
    pool: Mapping[str, Any],
    model: RecordingBatchedSemanticModel,
) -> dict[str, Any]:
    targets = holdout_targets(pool)
    prior_targets = [
        *pool["development_targets"],
        *pool["holdout_targets"],
        *pool["unused_validated_targets"],
    ]
    env = OversampledAlignedAnimalsEnvironment(
        config=config,
        answerer=model,
        target_animals=list(prior_targets),
    )
    records: list[dict[str, Any]] = []
    try:
        for state_index, target in enumerate(targets):
            records.append(
                run_state(
                    env,
                    model,
                    config,
                    prior_targets=prior_targets,
                    target=target,
                    state_index=state_index,
                )
            )
    except Exception as exc:
        raise SupportRetentionError(
            f"{type(exc).__name__}: {exc}",
            usage=model.usage_snapshot(),
            records=records,
        ) from exc
    usage = model.usage_snapshot()
    summary = summarize_holdout(records, usage)
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "passed"
            if summary["gates"]["all_pass"]
            else "holdout_gate_failed"
        ),
        "protocol": {
            "target_pool_sha256": TARGET_POOL_SHA256,
            "config_sha256": CONFIG_SHA256,
            "target_count": len(targets),
            "prior_target_count": len(prior_targets),
            "candidate_width": CANDIDATE_WIDTH,
            "primary_selector": "expected_current_support_retention",
            "primary_endpoint": "realized_branch_truth_inclusion",
            "primary_control": "immediate_eig",
            "secondary_control": "seeded_random",
            "shared_candidates_and_counterfactual_branches": True,
            "semantic_table_is_target_blind": True,
            "semantic_table_defines_likelihood_and_realized_answer": True,
            "policy_generation_prompts_receive_no_target_field": True,
            "support_retention_selected_posthoc_on_development": True,
            "development_artifact_sha256": (
                "c6797b987b69c1abcc819f6b935c32436919da1863fd98388e9efc6f9f80e091"
            ),
            "output_cap_increased_for_serving_only": True,
            "output_cap_tokens": 16_384,
            "reasoning_requested": False,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    if sha256_file(args.config) != CONFIG_SHA256:
        raise ValueError("Animals support-retention config hash changed")
    pool = load_target_pool(args.target_pool)
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = RUN_COST_CAP_USD
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"

    delegate = build_model_adapter(
        config.model_pairs[0].questioner,
        config,
    )
    model = RecordingBatchedSemanticModel(delegate, config)
    private_path = private_dir / "RAW_RESPONSES.json"
    try:
        payload = run_holdout(config, pool=pool, model=model)
    except Exception as exc:
        private_path.write_text(
            json.dumps(
                _private_payload(model),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        failure = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "error": f"{type(exc).__name__}: {exc}",
            "usage": getattr(exc, "usage", model.usage_snapshot()),
            "completed_records": getattr(exc, "records", []),
            "private_raw_sha256": sha256_file(private_path),
        }
        (args.output_dir / "HOLDOUT_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    private_path.write_text(
        json.dumps(
            _private_payload(model),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    payload["protocol"]["private_raw_sha256"] = sha256_file(
        private_path
    )
    (args.output_dir / "HOLDOUT.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "summary": payload["summary"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
