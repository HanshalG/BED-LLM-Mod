#!/usr/bin/env python3
"""Run the matched history-blind Qwen support-generation control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_dynamic_support_quality96 as quality
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)
from scripts.number_game_generator_aware_bed import (
    MAX_TOKENS,
    RuleHypothesis,
    best_query,
    initial_messages,
    proposal_response_format,
)
from scripts.number_game_pooled_support import (
    encode_pooled_responses,
    parse_pooled_proposals,
)
from scripts.number_game_qwen_history_blind_serving_smoke import (
    INTERFACE_VERSION as SMOKE_INTERFACE_VERSION,
    MODEL_ID,
    PerRequestSeedStructuredAdapter,
    build_adapter,
    sha256_file,
)
from scripts.number_game_ranking_fidelity_audit import spearman_correlation
from scripts.number_game_predictive_risk_replication import TEMPERATURE
from scripts.number_game_retained_depth_three import (
    _first_branches,
    _rule,
    _second_branches,
)
from scripts.number_game_depth_three_development import (
    retain_parent_hypotheses,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-history-blind-matched32-1"
SOURCE_RESULT = quality.SOURCE_RESULT
SOURCE_TREES = quality.SOURCE_TREES
SOURCE_TARGETS = quality.SOURCE_TARGETS
SOURCE_RESULT_SHA256 = quality.SOURCE_RESULT_SHA256
SOURCE_TREES_SHA256 = quality.SOURCE_TREES_SHA256
SOURCE_TARGETS_SHA256 = quality.SOURCE_TARGETS_SHA256
TREE_COUNT = 32
SOURCE_TREE_SEEDS = tuple(range(80_000, 80_032))
ROOTS_PER_TREE = 8
FIRST_SLOTS_PER_TREE = 16
SECOND_SLOTS_PER_TREE = 32
SLOTS_PER_TREE = FIRST_SLOTS_PER_TREE + SECOND_SLOTS_PER_TREE
DRAWS_PER_SLOT = 2
EXPECTED_REQUESTS = TREE_COUNT * SLOTS_PER_TREE * DRAWS_PER_SLOT
CONTROL_SEED_START = 9_000_000
CONCURRENCY = 256
RUN_BUDGET_USD = 4.25
MIN_STARTING_BALANCE_USD = 5.00
MAX_RETRIES = 32
MIN_DRAW_VALID = 16
MIN_POOL_VALID = 24
MIN_SECOND_DRAW_NOVEL = 2
BOOTSTRAP_SEED = 9_100_000
BOOTSTRAP_SAMPLES = 20_000
MIN_CHANGED_ROOT_TREES = 20
SMOKE_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_history_blind_serving_smoke"
    / "number-game-qwen-history-blind-serving-smoke-20260730T150000Z"
    / "RESULT.json"
)
# Bound after the fresh serving smoke passes.
SMOKE_RESULT_SHA256 = ""


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} requirement"
        )


def validate_smoke_result(path: Path = SMOKE_RESULT) -> dict[str, Any]:
    if not SMOKE_RESULT_SHA256:
        raise ValueError("serving smoke hash has not been bound")
    if sha256_file(path) != SMOKE_RESULT_SHA256:
        raise ValueError("history-blind serving smoke hash changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    if result.get("status") != "passed":
        raise ValueError("history-blind serving smoke did not pass")
    if protocol.get("interface_version") != SMOKE_INTERFACE_VERSION:
        raise ValueError("history-blind smoke interface changed")
    if protocol.get("model") != MODEL_ID:
        raise ValueError("history-blind smoke model changed")
    if protocol.get("expected_requests") != 10:
        raise ValueError("history-blind smoke request count changed")
    if protocol.get("efficacy_used_for_authorization") is not False:
        raise ValueError("history-blind smoke used efficacy")
    if not all((result.get("gates") or {}).values()):
        raise ValueError("history-blind smoke has a failed gate")
    return result


def _key_text(stage: str, key: Sequence[Any]) -> str:
    return ":".join(
        [stage, *[str(int(value)) for value in key]]
    )


def _decode_key(text: str) -> tuple[str, tuple[Any, ...]]:
    parts = text.split(":")
    stage = parts[0]
    values = tuple(int(value) for value in parts[1:])
    if stage == "first" and len(values) == 2:
        return stage, (values[0], bool(values[1]))
    if stage == "second" and len(values) == 4:
        return stage, (
            values[0],
            bool(values[1]),
            values[2],
            bool(values[3]),
        )
    raise ValueError(f"invalid control branch key {text!r}")


def branch_slots(
    public_tree: dict[str, Any],
) -> list[dict[str, Any]]:
    roots = [int(root) for root in public_tree["roots"]]
    if len(roots) != ROOTS_PER_TREE or len(set(roots)) != ROOTS_PER_TREE:
        raise ValueError("source tree candidate roots changed")
    dynamic_first = _first_branches(public_tree)
    dynamic_second = _second_branches(
        public_tree,
        key="second_branches",
    )
    slots = []
    for root in roots:
        for first_label in (False, True):
            slots.append(
                {
                    "stage": "first",
                    "key": (root, first_label),
                    "observations": ((root, first_label),),
                }
            )
    for root in roots:
        for first_label in (False, True):
            second_query, _ = best_query(
                dynamic_first[(root, first_label)],
                excluded=(root,),
            )
            for second_label in (False, True):
                key = (
                    root,
                    first_label,
                    second_query,
                    second_label,
                )
                if key not in dynamic_second:
                    raise ValueError(f"stored dynamic branch missing: {key}")
                slots.append(
                    {
                        "stage": "second",
                        "key": key,
                        "observations": (
                            (root, first_label),
                            (second_query, second_label),
                        ),
                    }
                )
    if len(slots) != SLOTS_PER_TREE:
        raise ValueError("source tree branch-slot count changed")
    return slots


def control_seed(
    tree_index: int,
    history_index: int,
    draw_index: int,
) -> int:
    if not 0 <= tree_index < TREE_COUNT:
        raise ValueError("tree index is outside frozen cohort")
    if not 0 <= history_index < SLOTS_PER_TREE:
        raise ValueError("history index is outside frozen schedule")
    if not 0 <= draw_index < DRAWS_PER_SLOT:
        raise ValueError("draw index is outside frozen pool")
    return (
        CONTROL_SEED_START
        + 1000 * tree_index
        + 2 * history_index
        + draw_index
    )


def request_manifest(
    public_trees: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(public_trees) != TREE_COUNT:
        raise ValueError("formal source cohort must contain 32 trees")
    requests = []
    for tree_index, public_tree in enumerate(public_trees):
        if int(public_tree["tree_index"]) != tree_index:
            raise ValueError("source tree indices changed")
        if int(public_tree["tree_seed"]) != SOURCE_TREE_SEEDS[tree_index]:
            raise ValueError("source tree seed schedule changed")
        for history_index, slot in enumerate(branch_slots(public_tree)):
            for draw_index in range(DRAWS_PER_SLOT):
                requests.append(
                    {
                        "tree_index": tree_index,
                        "tree_seed": int(public_tree["tree_seed"]),
                        "history_index": history_index,
                        "draw_index": draw_index,
                        "stage": slot["stage"],
                        "key": slot["key"],
                        "observations": slot["observations"],
                        "seed": control_seed(
                            tree_index,
                            history_index,
                            draw_index,
                        ),
                    }
                )
    if len(requests) != EXPECTED_REQUESTS:
        raise ValueError("formal request manifest is not exact")
    seeds = [request["seed"] for request in requests]
    if len(set(seeds)) != EXPECTED_REQUESTS:
        raise ValueError("formal control seeds are not unique")
    return requests


def parse_control_responses(
    *,
    public_trees: Sequence[dict[str, Any]],
    requests: Sequence[dict[str, Any]],
    responses: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(requests) != EXPECTED_REQUESTS or len(responses) != EXPECTED_REQUESTS:
        raise ValueError("formal response count is not exact")
    controls = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "source_trees_sha256": SOURCE_TREES_SHA256,
        "model": MODEL_ID,
        "prompt": "initial Number Game prompt with no observations",
        "trees": [],
    }
    pool_rows = []
    offset = 0
    for tree_index, public_tree in enumerate(public_trees):
        branches = {}
        for history_index, slot in enumerate(branch_slots(public_tree)):
            pair_requests = requests[offset : offset + 2]
            pair_responses = responses[offset : offset + 2]
            offset += 2
            expected_seeds = [
                control_seed(tree_index, history_index, draw_index)
                for draw_index in range(2)
            ]
            if [item["seed"] for item in pair_requests] != expected_seeds:
                raise ValueError("response order differs from frozen schedule")
            pooled_response = encode_pooled_responses(pair_responses)
            support, diagnostic = parse_pooled_proposals(
                pooled_response,
                observations=(),
            )
            key_text = _key_text(slot["stage"], slot["key"])
            branches[key_text] = {
                "seeds": expected_seeds,
                "support": [
                    hypothesis.public_dict() for hypothesis in support
                ],
                "diagnostic": diagnostic,
            }
            pool_rows.append(
                {
                    "tree_index": tree_index,
                    "history_index": history_index,
                    "stage": slot["stage"],
                    "key": key_text,
                    "valid_unique_count": len(support),
                    "diagnostic": diagnostic,
                }
            )
        controls["trees"].append(
            {
                "tree_index": tree_index,
                "tree_seed": int(public_tree["tree_seed"]),
                "branches": branches,
            }
        )
    if offset != EXPECTED_REQUESTS:
        raise ValueError("not all formal responses were consumed")
    return controls, pool_rows


def mechanics_gates(
    *,
    usage: dict[str, Any],
    controls: dict[str, Any],
    pool_rows: Sequence[dict[str, Any]],
) -> dict[str, bool]:
    draw_diagnostics = [
        draw
        for row in pool_rows
        for draw in row["diagnostic"]["draw_diagnostics"]
    ]
    gates = {
        "exactly_thirty_two_source_trees": (
            len(controls["trees"]) == TREE_COUNT
        ),
        "exactly_1536_control_branch_slots": (
            len(pool_rows) == TREE_COUNT * SLOTS_PER_TREE
        ),
        "accepted_request_count_exact": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "retries_within_cap": usage["retry_count"] <= MAX_RETRIES,
        "provider_error_retries_within_cap": (
            usage["provider_error_retries"] <= MAX_RETRIES
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "all_3072_draws_are_strict_json": (
            len(draw_diagnostics) == EXPECTED_REQUESTS
            and all(
                diagnostic["codec_mode"] == "strict_json"
                for diagnostic in draw_diagnostics
            )
        ),
        "every_draw_has_at_least_sixteen_valid": all(
            diagnostic["valid_unique_count"] >= MIN_DRAW_VALID
            for diagnostic in draw_diagnostics
        ),
        "every_pool_has_at_least_twenty_four_valid": all(
            row["valid_unique_count"] >= MIN_POOL_VALID
            for row in pool_rows
        ),
        "every_second_draw_adds_at_least_two_extensions": all(
            row["diagnostic"]["draw_novel_contributions"][1]
            >= MIN_SECOND_DRAW_NOVEL
            for row in pool_rows
        ),
    }
    return gates


def _control_supports(
    public_tree: dict[str, Any],
    control_tree: dict[str, Any],
) -> tuple[
    dict[tuple[int, bool], list[RuleHypothesis]],
    dict[tuple[int, bool, int, bool], list[RuleHypothesis]],
]:
    initial = [_rule(item) for item in public_tree["initial"]]
    generated_first = {}
    generated_second = {}
    for key_text, branch in control_tree["branches"].items():
        stage, key = _decode_key(key_text)
        support = [_rule(item) for item in branch["support"]]
        if stage == "first":
            generated_first[key] = support
        else:
            generated_second[key] = support
    first = {}
    for key, generated in generated_first.items():
        first[key], _ = retain_parent_hypotheses(
            parent_support=initial,
            generated_support=generated,
            query=key[0],
            label=key[1],
        )
    second = {}
    for key, generated in generated_second.items():
        second[key], _ = retain_parent_hypotheses(
            parent_support=first[(key[0], key[1])],
            generated_support=generated,
            query=key[2],
            label=key[3],
        )
    if len(first) != FIRST_SLOTS_PER_TREE:
        raise ValueError("control first-stage support count changed")
    if len(second) != SECOND_SLOTS_PER_TREE:
        raise ValueError("control second-stage support count changed")
    return first, second


def _average_target_metrics(
    rows: Sequence[dict[str, float]],
) -> dict[str, float]:
    return quality._average_metrics(rows)


def evaluate_root(
    *,
    root: int,
    canonical_targets: Sequence[RuleHypothesis],
    conditional_first: dict[
        tuple[int, bool], Sequence[RuleHypothesis]
    ],
    conditional_second: dict[
        tuple[int, bool, int, bool], Sequence[RuleHypothesis]
    ],
    control_first: dict[
        tuple[int, bool], Sequence[RuleHypothesis]
    ],
    control_second: dict[
        tuple[int, bool, int, bool], Sequence[RuleHypothesis]
    ],
) -> dict[str, Any]:
    target_rows = []
    for truth in canonical_targets:
        first_label = truth.extension[root]
        first_observations = ((root, first_label),)
        exact_first = quality._consistent(
            canonical_targets,
            first_observations,
        )
        conditional_first_support = list(
            conditional_first[(root, first_label)]
        )
        control_first_support = list(control_first[(root, first_label)])
        second_query, _ = best_query(
            conditional_first_support,
            excluded=(root,),
        )
        second_label = truth.extension[second_query]
        second_observations = (
            (root, first_label),
            (second_query, second_label),
        )
        exact_second = quality._consistent(
            canonical_targets,
            second_observations,
        )
        branch = (root, first_label, second_query, second_label)
        target_rows.append(
            {
                "first": {
                    "conditional": quality.support_metric(
                        support=conditional_first_support,
                        exact_support=exact_first,
                        truth=truth,
                        queried=(root,),
                    ),
                    "history_blind": quality.support_metric(
                        support=control_first_support,
                        exact_support=exact_first,
                        truth=truth,
                        queried=(root,),
                    ),
                },
                "second": {
                    "conditional": quality.support_metric(
                        support=conditional_second[branch],
                        exact_support=exact_second,
                        truth=truth,
                        queried=(root, second_query),
                    ),
                    "history_blind": quality.support_metric(
                        support=control_second[branch],
                        exact_support=exact_second,
                        truth=truth,
                        queried=(root, second_query),
                    ),
                },
            }
        )
    stages = {}
    for stage in ("first", "second"):
        stages[stage] = {
            arm: _average_target_metrics(
                [row[stage][arm] for row in target_rows]
            )
            for arm in ("conditional", "history_blind")
        }
        stages[stage]["differences"] = {
            "conditional_minus_history_blind_predictive_mse": (
                stages[stage]["conditional"]["posterior_predictive_mse"]
                - stages[stage]["history_blind"][
                    "posterior_predictive_mse"
                ]
            ),
            "conditional_minus_history_blind_truth_coverage": (
                stages[stage]["conditional"]["truth_extension_coverage"]
                - stages[stage]["history_blind"][
                    "truth_extension_coverage"
                ]
            ),
            "conditional_minus_history_blind_support_size": (
                stages[stage]["conditional"]["support_size"]
                - stages[stage]["history_blind"]["support_size"]
            ),
        }
    return {
        "root": root,
        "canonical_target_path_count": len(target_rows),
        "stages": stages,
    }


def score_tree(
    *,
    public_tree: dict[str, Any],
    scored_tree: dict[str, Any],
    control_tree: dict[str, Any],
    canonical_targets: Sequence[RuleHypothesis],
) -> dict[str, Any]:
    tree_seed = int(public_tree["tree_seed"])
    if (
        tree_seed != int(scored_tree["tree_seed"])
        or tree_seed != int(control_tree["tree_seed"])
    ):
        raise ValueError("source, score, and control tree seeds differ")
    roots = [int(root) for root in public_tree["roots"]]
    conditional_first = _first_branches(public_tree)
    conditional_second = _second_branches(
        public_tree,
        key="second_branches",
    )
    control_first, control_second = _control_supports(
        public_tree,
        control_tree,
    )
    root_rows = {
        root: evaluate_root(
            root=root,
            canonical_targets=canonical_targets,
            conditional_first=conditional_first,
            conditional_second=conditional_second,
            control_first=control_first,
            control_second=control_second,
        )
        for root in roots
    }
    tree_mean = {}
    for stage in ("first", "second"):
        tree_mean[stage] = {
            arm: {
                metric: quality._mean(
                    [
                        root_rows[root]["stages"][stage][arm][metric]
                        for root in roots
                    ]
                )
                for metric in (
                    "posterior_predictive_mse",
                    "truth_extension_coverage",
                    "support_size",
                )
            }
            for arm in ("conditional", "history_blind")
        }
        tree_mean[stage]["differences"] = {
            key: quality._mean(
                [
                    root_rows[root]["stages"][stage]["differences"][key]
                    for root in roots
                ]
            )
            for key in root_rows[roots[0]]["stages"][stage][
                "differences"
            ]
        }

    selection = scored_tree["selection"]
    dynamic_root = int(selection["crossfit_depth_three_root"])
    fixed_root = int(selection["fixed_support_depth_three_root"])

    def prompt_benefit(root: int) -> float:
        stage = root_rows[root]["stages"]["second"]
        return (
            stage["history_blind"]["posterior_predictive_mse"]
            - stage["conditional"]["posterior_predictive_mse"]
        )

    realized = {
        int(root): float(value)
        for root, value in scored_tree["per_root_endpoint_brier"].items()
    }
    roots_differ = dynamic_root != fixed_root
    selected = {
        "dynamic_root": dynamic_root,
        "fixed_root": fixed_root,
        "roots_differ": roots_differ,
        "dynamic_root_prompt_conditioning_benefit": prompt_benefit(
            dynamic_root
        ),
        "fixed_root_prompt_conditioning_benefit": prompt_benefit(fixed_root),
        "dynamic_minus_fixed_root_prompt_conditioning_benefit": (
            prompt_benefit(dynamic_root) - prompt_benefit(fixed_root)
        ),
        "realized_advantage": (
            realized[fixed_root] - realized[dynamic_root]
            if roots_differ
            else 0.0
        ),
    }
    return {
        "tree_index": int(public_tree["tree_index"]),
        "tree_seed": tree_seed,
        "tree_mean": tree_mean,
        "selected_roots": selected,
        "root_rows": {str(root): root_rows[root] for root in roots},
    }


def _bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    samples: int,
) -> dict[str, list[float]]:
    changed = [
        row for row in rows if row["selected_roots"]["roots_differ"]
    ]
    if not changed:
        raise ValueError("source subset has no changed-root trees")
    rng = random.Random(seed)
    replicates = {
        "second_conditional_minus_history_blind_predictive_mse_95pct": [],
        "second_conditional_minus_history_blind_truth_coverage_95pct": [],
        "changed_root_prompt_benefit_contrast_95pct": [],
        "prompt_benefit_contrast_to_realized_spearman_95pct": [],
    }
    for _ in range(samples):
        sampled = [rng.choice(rows) for _ in rows]
        sampled_changed = [rng.choice(changed) for _ in changed]
        replicates[
            "second_conditional_minus_history_blind_predictive_mse_95pct"
        ].append(
            quality._mean(
                [
                    row["tree_mean"]["second"]["differences"][
                        "conditional_minus_history_blind_predictive_mse"
                    ]
                    for row in sampled
                ]
            )
        )
        replicates[
            "second_conditional_minus_history_blind_truth_coverage_95pct"
        ].append(
            quality._mean(
                [
                    row["tree_mean"]["second"]["differences"][
                        "conditional_minus_history_blind_truth_coverage"
                    ]
                    for row in sampled
                ]
            )
        )
        contrasts = [
            row["selected_roots"][
                "dynamic_minus_fixed_root_prompt_conditioning_benefit"
            ]
            for row in sampled_changed
        ]
        advantages = [
            row["selected_roots"]["realized_advantage"]
            for row in sampled_changed
        ]
        replicates[
            "changed_root_prompt_benefit_contrast_95pct"
        ].append(quality._mean(contrasts))
        replicates[
            "prompt_benefit_contrast_to_realized_spearman_95pct"
        ].append(spearman_correlation(contrasts, advantages))
    return {
        key: quality._interval(values)
        for key, values in replicates.items()
    }


def summarize_scores(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if len(rows) != TREE_COUNT:
        raise ValueError("scored control cohort must contain 32 trees")
    stages = {}
    for stage in ("first", "second"):
        stages[stage] = {
            arm: {
                metric: quality._mean(
                    [
                        row["tree_mean"][stage][arm][metric]
                        for row in rows
                    ]
                )
                for metric in (
                    "posterior_predictive_mse",
                    "truth_extension_coverage",
                    "support_size",
                )
            }
            for arm in ("conditional", "history_blind")
        }
        stages[stage]["differences"] = {
            key: quality._mean(
                [
                    row["tree_mean"][stage]["differences"][key]
                    for row in rows
                ]
            )
            for key in rows[0]["tree_mean"][stage]["differences"]
        }
    changed = [
        row for row in rows if row["selected_roots"]["roots_differ"]
    ]
    contrasts = [
        row["selected_roots"][
            "dynamic_minus_fixed_root_prompt_conditioning_benefit"
        ]
        for row in changed
    ]
    advantages = [
        row["selected_roots"]["realized_advantage"] for row in changed
    ]
    selected = {
        "changed_root_tree_count": len(changed),
        "mean_dynamic_root_prompt_conditioning_benefit": quality._mean(
            [
                row["selected_roots"][
                    "dynamic_root_prompt_conditioning_benefit"
                ]
                for row in changed
            ]
        ),
        "mean_fixed_root_prompt_conditioning_benefit": quality._mean(
            [
                row["selected_roots"][
                    "fixed_root_prompt_conditioning_benefit"
                ]
                for row in changed
            ]
        ),
        "mean_dynamic_minus_fixed_root_prompt_conditioning_benefit": (
            quality._mean(contrasts)
        ),
        "prompt_benefit_contrast_to_realized_advantage_spearman": (
            spearman_correlation(contrasts, advantages)
        ),
    }
    bootstrap = _bootstrap(
        rows,
        seed=BOOTSTRAP_SEED,
        samples=bootstrap_samples,
    )
    gates = {
        "at_least_twenty_changed_root_trees": (
            len(changed) >= MIN_CHANGED_ROOT_TREES
        ),
        "second_stage_conditional_mse_ci_below_history_blind": (
            bootstrap[
                "second_conditional_minus_history_blind_predictive_mse_95pct"
            ][1]
            < 0.0
        ),
        "second_stage_conditional_coverage_ci_not_below_history_blind": (
            bootstrap[
                "second_conditional_minus_history_blind_truth_coverage_95pct"
            ][0]
            >= 0.0
        ),
        "changed_root_prompt_benefit_contrast_ci_above_zero": (
            bootstrap[
                "changed_root_prompt_benefit_contrast_95pct"
            ][0]
            > 0.0
        ),
    }
    return {
        "stages": stages,
        "selected_root_prompt_conditioning": selected,
        "bootstrap": bootstrap,
        "scientific_gates": gates,
        "directionally_coherent": all(gates.values()),
    }


def run_formal(
    *,
    output_dir: Path,
    run_id: str,
    smoke_result_path: Path = SMOKE_RESULT,
    adapter: PerRequestSeedStructuredAdapter | None = None,
    remaining_credit: float | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    source_result, source_trees, canonical_targets = (
        quality.load_and_validate_sources(
            result_path=SOURCE_RESULT,
            trees_path=SOURCE_TREES,
            targets_path=SOURCE_TARGETS,
        )
    )
    public_trees = source_trees["trees"][:TREE_COUNT]
    scored_source = source_result["trees"][:TREE_COUNT]
    manifest = request_manifest(public_trees)
    smoke = validate_smoke_result(smoke_result_path)
    available = (
        openrouter_remaining_credit()
        if remaining_credit is None
        else remaining_credit
    )
    require_starting_balance(available)

    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    controls_path = output_dir / "CONTROLS.json"
    adapter = adapter or build_adapter(
        run_id=run_id,
        output_dir=output_dir,
        concurrency=CONCURRENCY,
        run_budget_usd=RUN_BUDGET_USD,
        projected_cost_usd=RUN_BUDGET_USD,
    )
    seeds = [request["seed"] for request in manifest]
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        [initial_messages() for _ in manifest],
        seeds,
        temperature=TEMPERATURE,
        response_format=proposal_response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    checkpoint(
        raw_path,
        {
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "model": MODEL_ID,
                "source_trees_sha256": SOURCE_TREES_SHA256,
            },
            "requests": [
                {
                    **request,
                    "key": [int(value) for value in request["key"]],
                    "observations": [
                        [int(query), bool(label)]
                        for query, label in request["observations"]
                    ],
                    "response": response,
                }
                for request, response in zip(
                    manifest,
                    responses,
                    strict=True,
                )
            ],
        },
    )
    controls, pool_rows = parse_control_responses(
        public_trees=public_trees,
        requests=manifest,
        responses=responses,
    )
    checkpoint(controls_path, controls)
    usage = summarize_usage(adapter.usage_snapshot())
    mechanics = mechanics_gates(
        usage=usage,
        controls=controls,
        pool_rows=pool_rows,
    )
    controls_sha256 = sha256_file(controls_path)
    if not all(mechanics.values()):
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "mechanics_failed",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "source_result_sha256": SOURCE_RESULT_SHA256,
                "source_trees_sha256": SOURCE_TREES_SHA256,
                "source_targets_sha256": SOURCE_TARGETS_SHA256,
                "smoke_result_sha256": SMOKE_RESULT_SHA256,
                "controls_sha256": controls_sha256,
                "endpoint_accessed": False,
            },
            "usage": usage,
            "mechanics_gates": mechanics,
        }
        checkpoint(output_dir / "FAILURE.json", failure)
        raise RuntimeError("formal history-blind mechanics gates failed")

    scored_rows = [
        score_tree(
            public_tree=public_tree,
            scored_tree=scored_tree,
            control_tree=control_tree,
            canonical_targets=canonical_targets,
        )
        for public_tree, scored_tree, control_tree in zip(
            public_trees,
            scored_source,
            controls["trees"],
            strict=True,
        )
    ]
    analysis = summarize_scores(
        scored_rows,
        bootstrap_samples=bootstrap_samples,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "passed"
            if analysis["directionally_coherent"]
            else "gated_null"
        ),
        "protocol": {
            "analysis_was_preregistered": True,
            "source_status_unchanged": True,
            "cannot_rescue_or_relabel_source": True,
            "candidate_planning_model": MODEL_ID,
            "reasoning": False,
            "prompt_conditioning_control": (
                "fresh initial no-observation prompt per branch slot"
            ),
            "tree_indices": list(range(TREE_COUNT)),
            "tree_seeds": list(SOURCE_TREE_SEEDS),
            "tree_count": TREE_COUNT,
            "candidate_roots_per_tree": ROOTS_PER_TREE,
            "branch_slots_per_tree": SLOTS_PER_TREE,
            "draws_per_slot": DRAWS_PER_SLOT,
            "expected_requests": EXPECTED_REQUESTS,
            "target_generation_requests": 0,
            "validation_generation_requests": 0,
            "control_seed_start": CONTROL_SEED_START,
            "concurrency": CONCURRENCY,
            "run_budget_usd": RUN_BUDGET_USD,
            "starting_balance_usd": available,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_trees_sha256": SOURCE_TREES_SHA256,
            "source_targets_sha256": SOURCE_TARGETS_SHA256,
            "smoke_result_sha256": SMOKE_RESULT_SHA256,
            "controls_sha256": controls_sha256,
            "raw_responses_sha256": sha256_file(raw_path),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": bootstrap_samples,
            "serving_smoke": {
                "status": smoke["status"],
                "usage": smoke["usage"],
            },
        },
        "usage": usage,
        "mechanics_gates": mechanics,
        "analysis": analysis,
        "trees": scored_rows,
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--smoke-result",
        type=Path,
        default=SMOKE_RESULT,
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    try:
        result = run_formal(
            output_dir=args.output_dir,
            run_id=args.run_id,
            smoke_result_path=args.smoke_result,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_dir / "RUNNER_FAILURE.json"
        if not failure_path.exists():
            checkpoint(
                failure_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    print(json.dumps(result["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
