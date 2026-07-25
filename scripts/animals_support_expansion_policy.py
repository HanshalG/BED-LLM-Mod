#!/usr/bin/env python3
"""Evaluate non-myopic support expansion in aligned semantic Animals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.animals.questions import (
    evaluate_candidate_coverage_dynamics,
)
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.animals_cabed_aligned_v12 import (
    AlignedSemanticAnimalsEnvironment,
)
from scripts.animals_cabed_aligned_v13 import (
    OversampledAlignedAnimalsEnvironment,
    RecordingBatchedSemanticModel,
)
from scripts.animals_cabed_shared_tree_v10 import (
    PREHISTORY_QUESTIONS,
    _generate_questions,
)
from scripts.animals_coverage_dynamics import (
    _history_messages,
    _serialize_dynamics,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "animals-support-expansion-aligned-1"
TARGET_POOL_SHA256 = (
    "f1357649054e41150580201b8ad2318e30fb752ae7adb647e85873f7115a6715"
)
CONFIG_SHA256 = (
    "d600a6b326f083a0ebed569aa151edb642dbb484e261753f96cbd1e671e2316b"
)
CANDIDATE_WIDTH = 3
RANDOM_SEED = 24_291
BOOTSTRAP_SEED = 24_292
BOOTSTRAP_SAMPLES = 20_000
STAGE_COSTS = {
    "serving_smoke": (0.12, 0.50),
    "development": (0.75, 2.00),
    "confirmation": (2.00, 4.00),
}


class SupportExpansionError(RuntimeError):
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_target_pool(path: Path) -> dict[str, Any]:
    if sha256_file(path) != TARGET_POOL_SHA256:
        raise ValueError("Animals target-pool hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "development_targets": 20,
        "holdout_targets": 60,
        "unused_validated_targets": 46,
    }
    for key, count in expected.items():
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"Animals target-pool {key} changed")
    all_targets = [
        *payload["development_targets"],
        *payload["holdout_targets"],
        *payload["unused_validated_targets"],
    ]
    if len(all_targets) != 126 or len(set(all_targets)) != 126:
        raise ValueError("Animals target prior must contain 126 unique names")
    return payload


def stage_targets(pool: Mapping[str, Any], stage: str) -> list[str]:
    development = list(pool["development_targets"])
    if stage == "serving_smoke":
        return development[:2]
    if stage == "development":
        return development
    if stage == "confirmation":
        raise ValueError(
            "confirmation is sealed pending a separate holdout preregistration"
        )
    raise ValueError("unknown stage")


def expected_support_size(candidate: Mapping[str, Any]) -> float:
    return (
        float(candidate["p_yes"]) * int(candidate["support_size_if_yes"])
        + float(candidate["p_no"]) * int(candidate["support_size_if_no"])
    )


def branch_union_size(candidate: Mapping[str, Any]) -> int:
    return len(
        {
            str(value).strip().casefold()
            for value in (
                list(candidate["support_if_yes"])
                + list(candidate["support_if_no"])
            )
        }
    )


def _argmax(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("cannot select from an empty score vector")
    return max(range(len(values)), key=lambda index: (values[index], -index))


def selector_indices(
    candidates: Sequence[Mapping[str, Any]],
    *,
    random_index: int,
) -> dict[str, int]:
    if len(candidates) != CANDIDATE_WIDTH:
        raise ValueError("support-expansion policy requires exactly 3 candidates")
    if not 0 <= random_index < len(candidates):
        raise ValueError("random selector index is invalid")
    return {
        "support_expansion": _argmax(
            [expected_support_size(candidate) for candidate in candidates]
        ),
        "immediate_eig": _argmax(
            [float(candidate["immediate_eig"]) for candidate in candidates]
        ),
        "support_retention": _argmax(
            [
                float(candidate["expected_current_support_retention"])
                for candidate in candidates
            ]
        ),
        "branch_union": _argmax(
            [float(branch_union_size(candidate)) for candidate in candidates]
        ),
        "random": random_index,
    }


def realized_candidate_endpoint(
    candidate: Mapping[str, Any],
    answer: str,
) -> dict[str, float | int]:
    if answer == "Yes":
        covered = bool(candidate["truth_covered_if_yes"])
        support_size = int(candidate["support_size_if_yes"])
    elif answer == "No":
        covered = bool(candidate["truth_covered_if_no"])
        support_size = int(candidate["support_size_if_no"])
    else:
        raise ValueError("realized semantic answer must be Yes or No")
    return {
        "truth_covered": int(covered),
        "support_size": support_size,
        "uniform_truth_probability": (
            1.0 / support_size if covered and support_size > 0 else 0.0
        ),
    }


def _target_answer(
    env: AlignedSemanticAnimalsEnvironment,
    question: str,
    target: str,
    *,
    seed: int,
) -> str:
    answer = env.observe(
        question,
        target,
        np.random.default_rng(seed),
    )
    if answer not in {"Yes", "No"}:
        raise ValueError("aligned answer must be Yes or No")
    return answer


def run_state(
    env: OversampledAlignedAnimalsEnvironment,
    model: RecordingBatchedSemanticModel,
    config: Config,
    *,
    prior_targets: Sequence[str],
    target: str,
    state_index: int,
) -> dict[str, Any]:
    belief_state = env.initial_belief_state(model, config)
    if belief_state.support_size == 0:
        raise ValueError("initial generated belief is empty")

    bootstrap_question = PREHISTORY_QUESTIONS[
        state_index % len(PREHISTORY_QUESTIONS)
    ]
    env.semantic_yes_probabilities_many(
        prior_targets,
        [bootstrap_question],
    )
    bootstrap_answer = _target_answer(
        env,
        bootstrap_question,
        target,
        seed=RANDOM_SEED + state_index * 100,
    )
    history = [(bootstrap_question, bootstrap_answer)]
    belief_state = env.update_belief_state(
        belief_state,
        history,
        model,
        config,
    )
    if belief_state.support_size == 0:
        raise ValueError("post-bootstrap generated belief is empty")

    candidates = _generate_questions(
        env,
        belief_state,
        history,
        model,
        config,
        width=CANDIDATE_WIDTH,
    )
    env.semantic_yes_probabilities_many(prior_targets, candidates)
    dynamics = evaluate_candidate_coverage_dynamics(
        belief_state,
        _history_messages(history),
        list(candidates),
        target,
        deterministic=False,
        questioner=model,
        config=config,
        exact_current_support=True,
    )
    serialized = [_serialize_dynamics(candidate) for candidate in dynamics]
    if len(serialized) != CANDIDATE_WIDTH:
        raise ValueError("counterfactual evaluator lost a candidate")

    answers = [
        _target_answer(
            env,
            candidate["question"],
            target,
            seed=RANDOM_SEED + state_index * 100 + index + 1,
        )
        for index, candidate in enumerate(serialized)
    ]
    realized = [
        realized_candidate_endpoint(candidate, answer)
        for candidate, answer in zip(serialized, answers, strict=True)
    ]
    random_index = int(
        np.random.default_rng(RANDOM_SEED + state_index).integers(
            0,
            CANDIDATE_WIDTH,
        )
    )
    selectors = selector_indices(serialized, random_index=random_index)
    target_key = target.strip().casefold()
    return {
        "state_index": state_index,
        "target_measurement_only": target,
        "bootstrap_question": bootstrap_question,
        "bootstrap_answer": bootstrap_answer,
        "truth_covered_before_counterfactuals": any(
            hypothesis.strip().casefold() == target_key
            for hypothesis in belief_state.hypotheses
        ),
        "belief_support_size": belief_state.support_size,
        "candidate_dynamics": [
            {
                **candidate,
                "expected_support_size": expected_support_size(candidate),
                "branch_union_size": branch_union_size(candidate),
                "realized_answer": answer,
                "realized_endpoint": endpoint,
            }
            for candidate, answer, endpoint in zip(
                serialized,
                answers,
                realized,
                strict=True,
            )
        ],
        "selector_indices": selectors,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[start]]
        ):
            end += 1
        rank = (start + end + 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _rank_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_values = np.asarray(_average_ranks(left), dtype=float)
    right_values = np.asarray(_average_ranks(right), dtype=float)
    left_values -= left_values.mean()
    right_values -= right_values.mean()
    denominator = float(
        np.linalg.norm(left_values) * np.linalg.norm(right_values)
    )
    if denominator == 0.0:
        return None
    return float(np.dot(left_values, right_values) / denominator)


def _pairwise_accuracy(
    scores: Sequence[float],
    endpoints: Sequence[float],
) -> tuple[float, int]:
    points = 0.0
    pairs = 0
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            endpoint_delta = endpoints[left] - endpoints[right]
            if endpoint_delta == 0.0:
                continue
            score_delta = scores[left] - scores[right]
            pairs += 1
            if score_delta == 0.0:
                points += 0.5
            elif (score_delta > 0.0) == (endpoint_delta > 0.0):
                points += 1.0
    return (points / pairs if pairs else 0.5, pairs)


def _bootstrap_interval(
    differences: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    values = np.asarray(differences, dtype=float)
    if len(values) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_SAMPLES, len(values)),
    )
    means = values[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _sign_test_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    return min(
        1.0,
        sum(
            math.comb(trials, successes)
            for successes in range(wins, trials + 1)
        )
        / (2**trials),
    )


def summarize(
    records: Sequence[dict[str, Any]],
    *,
    stage: str,
    usage: Mapping[str, Any],
) -> dict[str, Any]:
    selector_names = (
        "support_expansion",
        "immediate_eig",
        "support_retention",
        "branch_union",
        "random",
    )
    selector_metrics: dict[str, dict[str, float]] = {}
    for selector in selector_names:
        coverages = []
        expected_coverages = []
        truth_probabilities = []
        support_sizes = []
        for record in records:
            index = int(record["selector_indices"][selector])
            candidate = record["candidate_dynamics"][index]
            endpoint = candidate["realized_endpoint"]
            coverages.append(float(endpoint["truth_covered"]))
            expected_coverages.append(
                float(candidate["expected_truth_coverage"])
            )
            truth_probabilities.append(
                float(endpoint["uniform_truth_probability"])
            )
            support_sizes.append(float(endpoint["support_size"]))
        selector_metrics[selector] = {
            "mean_realized_truth_coverage": float(np.mean(coverages)),
            "mean_expected_truth_coverage": float(
                np.mean(expected_coverages)
            ),
            "mean_uniform_truth_probability": float(
                np.mean(truth_probabilities)
            ),
            "mean_realized_support_size": float(np.mean(support_sizes)),
        }

    expansion_differences = []
    expected_differences = []
    wins = ties = losses = 0
    selector_changes = 0
    dynamic_states = 0
    recovered_after_omission = 0
    support_scores = []
    eig_scores = []
    realized_candidate_coverages = []
    expected_candidate_coverages = []
    for record in records:
        expansion_index = int(
            record["selector_indices"]["support_expansion"]
        )
        eig_index = int(record["selector_indices"]["immediate_eig"])
        selector_changes += expansion_index != eig_index
        candidates = record["candidate_dynamics"]
        state_support_scores = [
            float(candidate["expected_support_size"])
            for candidate in candidates
        ]
        dynamic_states += (
            max(state_support_scores) - min(state_support_scores) > 1.0e-12
        )
        expansion = candidates[expansion_index]
        eig = candidates[eig_index]
        difference = (
            float(expansion["realized_endpoint"]["truth_covered"])
            - float(eig["realized_endpoint"]["truth_covered"])
        )
        expansion_differences.append(difference)
        expected_differences.append(
            float(expansion["expected_truth_coverage"])
            - float(eig["expected_truth_coverage"])
        )
        wins += difference > 0.0
        ties += difference == 0.0
        losses += difference < 0.0
        recovered_after_omission += (
            not record["truth_covered_before_counterfactuals"]
            and bool(
                expansion["realized_endpoint"]["truth_covered"]
            )
        )
        support_scores.extend(state_support_scores)
        eig_scores.extend(
            float(candidate["immediate_eig"])
            for candidate in candidates
        )
        realized_candidate_coverages.extend(
            float(candidate["realized_endpoint"]["truth_covered"])
            for candidate in candidates
        )
        expected_candidate_coverages.extend(
            float(candidate["expected_truth_coverage"])
            for candidate in candidates
        )

    support_pairwise = _pairwise_accuracy(
        support_scores,
        realized_candidate_coverages,
    )
    eig_pairwise = _pairwise_accuracy(
        eig_scores,
        realized_candidate_coverages,
    )
    support_expected_rho = _rank_correlation(
        support_scores,
        expected_candidate_coverages,
    )
    eig_expected_rho = _rank_correlation(
        eig_scores,
        expected_candidate_coverages,
    )
    comparison = {
        "mean_realized_coverage_gain": float(
            np.mean(expansion_differences)
        ),
        "mean_expected_coverage_gain": float(
            np.mean(expected_differences)
        ),
        "realized_gain_bootstrap_95": _bootstrap_interval(
            expansion_differences
        ),
        "wins_ties_losses": [wins, ties, losses],
        "exact_one_sided_sign_p": _sign_test_pvalue(wins, losses),
        "selector_changes": selector_changes,
        "support_score_dynamic_states": dynamic_states,
        "recovered_after_initial_omission": recovered_after_omission,
        "support_score_realized_pairwise_accuracy": {
            "accuracy": support_pairwise[0],
            "pairs": support_pairwise[1],
        },
        "eig_realized_pairwise_accuracy": {
            "accuracy": eig_pairwise[0],
            "pairs": eig_pairwise[1],
        },
        "support_score_expected_coverage_spearman": support_expected_rho,
        "eig_expected_coverage_spearman": eig_expected_rho,
    }
    reasoning_tokens = int(usage.get("adapter_reasoning_tokens", 0))
    forced_exits = int(usage.get("forced_exits", 0))
    cost = float(usage.get("adapter_cost_usd", 0.0))
    gates = {
        "all_states_completed": len(records)
        == {"serving_smoke": 2, "development": 20, "confirmation": 60}[
            stage
        ],
        "all_states_have_three_candidates": all(
            len(record["candidate_dynamics"]) == CANDIDATE_WIDTH
            for record in records
        ),
        "zero_reasoning_tokens": reasoning_tokens == 0,
        "zero_forced_exits": forced_exits == 0,
        "within_stage_cost_cap": cost <= STAGE_COSTS[stage][1],
    }
    if stage == "serving_smoke":
        gates.update(
            {
                "support_score_dynamic_on_at_least_one_state": (
                    dynamic_states >= 1
                ),
                "all_semantic_answers_binary": all(
                    candidate["realized_answer"] in {"Yes", "No"}
                    for record in records
                    for candidate in record["candidate_dynamics"]
                ),
            }
        )
    elif stage == "development":
        eig_coverage = selector_metrics["immediate_eig"][
            "mean_realized_truth_coverage"
        ]
        gates.update(
            {
                "support_score_dynamic_on_at_least_ten_states": (
                    dynamic_states >= 10
                ),
                "selector_changes_on_at_least_five_states": (
                    selector_changes >= 5
                ),
                "eig_endpoint_not_saturated": 0.05 <= eig_coverage <= 0.95,
                "realized_coverage_gain_positive": (
                    comparison["mean_realized_coverage_gain"] > 0.0
                ),
                "expected_coverage_gain_positive": (
                    comparison["mean_expected_coverage_gain"] > 0.0
                ),
                "realized_wins_exceed_losses": wins > losses,
                "support_pairwise_accuracy_above_eig": (
                    support_pairwise[0] > eig_pairwise[0]
                ),
                "at_least_one_recovery_after_initial_omission": (
                    recovered_after_omission >= 1
                ),
            }
        )
    gates["all_pass"] = all(gates.values())
    return {
        "selector_metrics": selector_metrics,
        "support_expansion_vs_eig": comparison,
        "gates": gates,
    }


def run_stage(
    config: Config,
    *,
    stage: str,
    pool: Mapping[str, Any],
    model: RecordingBatchedSemanticModel,
) -> dict[str, Any]:
    targets = stage_targets(pool, stage)
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
        raise SupportExpansionError(
            f"{type(exc).__name__}: {exc}",
            usage=model.usage_snapshot(),
            records=records,
        ) from exc
    usage = model.usage_snapshot()
    summary = summarize(records, stage=stage, usage=usage)
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "passed"
            if summary["gates"]["all_pass"]
            else f"{stage}_gate_failed"
        ),
        "protocol": {
            "stage": stage,
            "target_pool_sha256": TARGET_POOL_SHA256,
            "config_sha256": CONFIG_SHA256,
            "target_count": len(targets),
            "prior_target_count": len(prior_targets),
            "candidate_width": CANDIDATE_WIDTH,
            "candidate_oversample": 2,
            "fixed_prehistory_cycle": list(PREHISTORY_QUESTIONS),
            "selector": "expected_regenerated_support_size",
            "shared_candidates_and_counterfactual_branches": True,
            "compute_matched_immediate_eig_control": True,
            "semantic_table_is_target_blind": True,
            "semantic_table_defines_likelihood_and_realized_answer": True,
            "independent_answer_calls": False,
            "policy_generation_prompts_receive_no_target_field": True,
            "reasoning_requested": False,
            "random_seed": RANDOM_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def _private_payload(
    model: RecordingBatchedSemanticModel,
) -> dict[str, Any]:
    return {
        "semantic_classifications": model.classification_records,
        "generation_records": model.generation_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-pool", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "development", "confirmation"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    if sha256_file(args.config) != CONFIG_SHA256:
        raise ValueError("Animals support-expansion config hash changed")
    pool = load_target_pool(args.target_pool)
    config = load_config(str(args.config))
    config.run_id = args.run_id
    projected, cap = STAGE_COSTS[args.stage]
    config.openrouter_projected_cost_usd = projected
    config.openrouter_run_budget_usd = cap
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"

    delegate = build_model_adapter(config.model_pairs[0].questioner, config)
    model = RecordingBatchedSemanticModel(delegate, config)
    public_name = {
        "serving_smoke": "SERVING_SMOKE.json",
        "development": "DEVELOPMENT.json",
        "confirmation": "CONFIRMATION.json",
    }[args.stage]
    private_path = private_dir / "RAW_RESPONSES.json"
    try:
        payload = run_stage(
            config,
            stage=args.stage,
            pool=pool,
            model=model,
        )
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
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
            "usage": getattr(exc, "usage", model.usage_snapshot()),
            "completed_records": getattr(exc, "records", []),
            "private_raw_sha256": sha256_file(private_path),
        }
        (args.output_dir / f"{args.stage.upper()}_FAILURE.json").write_text(
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
    payload["protocol"]["private_raw_sha256"] = sha256_file(private_path)
    (args.output_dir / public_name).write_text(
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
