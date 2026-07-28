#!/usr/bin/env python3
"""Evaluate a frozen predictive-risk Number Game root on fresh target rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.discoverphysics_dark_matter_structured_replication_v3 import (
    DefaultRoutingStructuredAdapter,
)
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_generator_aware_bed import (
    INTERFACE_VERSION as SOURCE_INTERFACE_VERSION,
    MAX_TOKENS,
    RuleHypothesis,
    choose_predictive_bayes_risk_root,
    compile_expression,
    evaluate_policy_root,
    initial_messages,
    parse_proposals,
    predictive_bayes_risk_scores,
    proposal_response_format,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-predictive-risk-holdout-1"
MODEL_ID = "openai/gpt-5.4"
SOURCE_MODEL_SHA256 = (
    "bf45eb9f5dd8da0289d834671952b53f9dff8fc7144190fe1ecfab50045bcecb"
)
SOURCE_RESULT_SHA256 = (
    "6c1de697892c387df02ab0a2f657f05b1a2164872192653b407281e8e22a453c"
)
EXPECTED_PREDICTIVE_RISK_ROOT = 34
EXPECTED_MYOPIC_ROOT = 48
EXPECTED_REQUESTS = 1
MIN_VALID_TARGETS = 16
MIN_NOVEL_TARGETS = 8
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 26069
RUN_BUDGET_USD = 0.20
PROJECTED_COST_USD = 0.05


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rule(item: dict[str, Any]) -> RuleHypothesis:
    return RuleHypothesis(
        name=item["name"],
        expression=item["expression"],
        extension=compile_expression(item["expression"]),
    )


def load_source(
    model_path: Path,
    result_path: Path,
) -> tuple[
    list[RuleHypothesis],
    list[int],
    dict[tuple[int, bool], list[RuleHypothesis]],
    dict[str, Any],
]:
    if sha256_file(model_path) != SOURCE_MODEL_SHA256:
        raise ValueError("source MODEL.json hash changed")
    if sha256_file(result_path) != SOURCE_RESULT_SHA256:
        raise ValueError("source RESULT.json hash changed")
    model = json.loads(model_path.read_text())
    result = json.loads(result_path.read_text())
    if model["protocol"]["interface_version"] != SOURCE_INTERFACE_VERSION:
        raise ValueError("source interface changed")
    initial = [_rule(item) for item in model["initial"]]
    roots = [int(root) for root in model["roots"]]
    branches = {}
    for key, items in model["branches"].items():
        root, label = key.split(":")
        branches[(int(root), bool(int(label)))] = [
            _rule(item) for item in items
        ]
    return initial, roots, branches, result


def paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    if not differences:
        raise ValueError("bootstrap differences are empty")
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return [
        means[int(0.025 * samples)],
        means[min(samples - 1, int(0.975 * samples))],
    ]


def policy_comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    candidate_rows = {
        row["target"]: row for row in candidate["targets"]
    }
    baseline_rows = {
        row["target"]: row for row in baseline["targets"]
    }
    if set(candidate_rows) != set(baseline_rows):
        raise ValueError("policy target sets differ")
    brier_differences = [
        candidate_rows[target]["posterior_predictive_brier"]
        - baseline_rows[target]["posterior_predictive_brier"]
        for target in candidate_rows
    ]
    hamming_differences = [
        candidate_rows[target]["best_hamming_error"]
        - baseline_rows[target]["best_hamming_error"]
        for target in candidate_rows
    ]
    baseline_brier = baseline["mean_posterior_predictive_brier"]
    baseline_hamming = baseline["mean_best_hamming_error"]
    return {
        "candidate_minus_baseline_brier": sum(brier_differences)
        / len(brier_differences),
        "candidate_minus_baseline_brier_95pct_bootstrap": (
            paired_bootstrap_interval(brier_differences)
        ),
        "relative_brier_reduction": (
            (baseline_brier - candidate["mean_posterior_predictive_brier"])
            / baseline_brier
            if baseline_brier > 0.0
            else 0.0
        ),
        "candidate_minus_baseline_hamming": sum(hamming_differences)
        / len(hamming_differences),
        "relative_hamming_reduction": (
            (baseline_hamming - candidate["mean_best_hamming_error"])
            / baseline_hamming
            if baseline_hamming > 0.0
            else 0.0
        ),
        "coverage_difference": (
            candidate["truth_extension_coverage_rate"]
            - baseline["truth_extension_coverage_rate"]
        ),
    }


def aggregate_random_root_control(
    root_results: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    first = next(iter(root_results.values()))
    targets = [row["target"] for row in first["targets"]]
    rows = []
    for target in targets:
        root_rows = [
            next(
                row
                for row in result["targets"]
                if row["target"] == target
            )
            for result in root_results.values()
        ]
        rows.append(
            {
                "target": target,
                "posterior_predictive_brier": sum(
                    row["posterior_predictive_brier"] for row in root_rows
                )
                / len(root_rows),
                "best_hamming_error": sum(
                    row["best_hamming_error"] for row in root_rows
                )
                / len(root_rows),
                "truth_extension_covered": sum(
                    row["truth_extension_covered"] for row in root_rows
                )
                / len(root_rows),
            }
        )
    return {
        "policy": "uniform_random_over_eight_candidate_roots",
        "root": None,
        "mean_posterior_predictive_brier": sum(
            row["posterior_predictive_brier"] for row in rows
        )
        / len(rows),
        "mean_best_hamming_error": sum(
            row["best_hamming_error"] for row in rows
        )
        / len(rows),
        "truth_extension_coverage_rate": sum(
            row["truth_extension_covered"] for row in rows
        )
        / len(rows),
        "targets": rows,
    }


def _adapter(
    *,
    run_id: str,
    output_dir: Path,
) -> DefaultRoutingStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=200.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=1,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return DefaultRoutingStructuredAdapter(
        ModelSpec(
            model=MODEL_ID,
            backend="openrouter",
            max_model_len=65536,
        ),
        config,
    )


def run_holdout(
    *,
    source_model: Path,
    source_result: Path,
    output_dir: Path,
    run_id: str,
    adapter: DefaultRoutingStructuredAdapter | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSE.json"
    initial, roots, branches, original_result = load_source(
        source_model, source_result
    )
    development = predictive_bayes_risk_scores(
        support=initial,
        roots=roots,
        branches=branches,
    )
    predictive_root = choose_predictive_bayes_risk_root(development)
    myopic_root = int(original_result["selection"]["myopic_root"])
    adapter = adapter or _adapter(run_id=run_id, output_dir=output_dir)
    raw = {"target_response": None}
    try:
        response = adapter.chat_complete_messages_batched_structured(
            [initial_messages()],
            temperature=0.0,
            block_size=1,
            response_format=proposal_response_format(),
            max_new_tokens=MAX_TOKENS,
        )[0]
        raw["target_response"] = response
        checkpoint(raw_path, raw)
        targets_list, target_diagnostics = parse_proposals(response)
        targets = {
            f"target_{index:02d}_{hypothesis.name}": hypothesis
            for index, hypothesis in enumerate(targets_list)
        }
        initial_extensions = {hypothesis.extension for hypothesis in initial}
        novel_targets = {
            name: hypothesis
            for name, hypothesis in targets.items()
            if hypothesis.extension not in initial_extensions
        }
        policy_roots = {
            "predictive_bayes_risk": predictive_root,
            "myopic_eig": myopic_root,
            "fixed_support_depth_two": int(
                original_result["selection"]["fixed_depth_two_root"]
            ),
        }
        endpoint = {
            policy: evaluate_policy_root(
                policy=policy,
                root=root,
                targets=targets,
                branches=branches,
            )
            for policy, root in policy_roots.items()
        }
        per_root = {
            root: evaluate_policy_root(
                policy=f"root_{root}",
                root=root,
                targets=targets,
                branches=branches,
            )
            for root in roots
        }
        endpoint["uniform_random_candidate_root"] = (
            aggregate_random_root_control(per_root)
        )
        novel_endpoint = {
            policy: evaluate_policy_root(
                policy=policy,
                root=root,
                targets=novel_targets,
                branches=branches,
            )
            for policy, root in policy_roots.items()
        }
        comparison = policy_comparison(
            endpoint["predictive_bayes_risk"],
            endpoint["myopic_eig"],
        )
        random_comparison = policy_comparison(
            endpoint["predictive_bayes_risk"],
            endpoint["uniform_random_candidate_root"],
        )
        novel_comparison = policy_comparison(
            novel_endpoint["predictive_bayes_risk"],
            novel_endpoint["myopic_eig"],
        )
        usage = adapter.usage_snapshot()
        source_predictive = development[predictive_root]
        source_myopic = development[myopic_root]
        source_comparison = policy_comparison(
            source_predictive, source_myopic
        )
        gates = {
            "exact_one_accepted_request": (
                usage["adapter_requests"] == EXPECTED_REQUESTS
            ),
            "transport_attempt_accounting_exact": (
                usage["http_attempts"]
                == usage["adapter_requests"] + usage["retry_count"]
            ),
            "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
            "zero_forced_exits": usage["forced_exits"] == 0,
            "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
            "at_least_16_valid_unique_targets": (
                len(targets) >= MIN_VALID_TARGETS
            ),
            "at_least_8_targets_novel_to_planning_support": (
                len(novel_targets) >= MIN_NOVEL_TARGETS
            ),
            "frozen_predictive_risk_root_is_34": (
                predictive_root == EXPECTED_PREDICTIVE_RISK_ROOT
            ),
            "source_current_particle_brier_gain_at_least_10_percent": (
                source_comparison["relative_brier_reduction"] >= 0.10
            ),
            "fresh_target_brier_gain_at_least_5_percent": (
                comparison["relative_brier_reduction"] >= 0.05
            ),
            "fresh_target_brier_paired_ci_is_below_zero": (
                comparison[
                    "candidate_minus_baseline_brier_95pct_bootstrap"
                ][1]
                < 0.0
            ),
            "fresh_target_hamming_gain_at_least_5_percent": (
                comparison["relative_hamming_reduction"] >= 0.05
            ),
            "fresh_target_has_no_coverage_loss": (
                comparison["coverage_difference"] >= 0.0
            ),
            "fresh_target_brier_beats_uniform_random_by_5_percent": (
                random_comparison["relative_brier_reduction"] >= 0.05
            ),
            "novel_target_brier_and_hamming_are_directionally_better": (
                novel_comparison["relative_brier_reduction"] > 0.0
                and novel_comparison["relative_hamming_reduction"] > 0.0
            ),
        }
        public_targets = {
            "schema_version": SCHEMA_VERSION,
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "model": MODEL_ID,
                "reasoning": "disabled",
                "temperature": 0.0,
                "source_model_sha256": SOURCE_MODEL_SHA256,
                "source_result_sha256": SOURCE_RESULT_SHA256,
            },
            "raw_response_sha256": sha256_file(raw_path),
            "targets": [
                {
                    **hypothesis.public_dict(),
                    "novel_to_planning_support": (
                        hypothesis.extension not in initial_extensions
                    ),
                }
                for hypothesis in targets_list
            ],
        }
        checkpoint(output_dir / "TARGETS.json", public_targets)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if all(gates.values()) else "holdout_failed",
            "protocol": public_targets["protocol"],
            "gates": gates,
            "target_diagnostics": {
                **target_diagnostics,
                "novel_unique_count": len(novel_targets),
            },
            "selection": {
                "predictive_bayes_risk_root": predictive_root,
                "myopic_root": myopic_root,
                "fixed_support_depth_two_root": policy_roots[
                    "fixed_support_depth_two"
                ],
            },
            "source_development": {
                str(root): {
                    key: value
                    for key, value in development[root].items()
                    if key != "targets"
                }
                for root in roots
            },
            "source_comparison": source_comparison,
            "endpoint": endpoint,
            "comparison_vs_myopic": comparison,
            "comparison_vs_uniform_random": random_comparison,
            "novel_endpoint": novel_endpoint,
            "novel_comparison_vs_myopic": novel_comparison,
            "usage": usage,
            "targets_sha256": sha256_file(output_dir / "TARGETS.json"),
            "raw_response_sha256": public_targets[
                "raw_response_sha256"
            ],
        }
        checkpoint(output_dir / "RESULT.json", result)
        return result
    except Exception as exc:
        checkpoint(raw_path, raw)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "model": MODEL_ID,
            },
            "error": f"{type(exc).__name__}: {exc}",
            "usage": adapter.usage_snapshot(),
            "raw_response_sha256": sha256_file(raw_path),
        }
        checkpoint(output_dir / "FAILURE.json", failure)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run_holdout(
        source_model=args.source_model.resolve(),
        source_result=args.source_result.resolve(),
        output_dir=args.output_dir.resolve(),
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
