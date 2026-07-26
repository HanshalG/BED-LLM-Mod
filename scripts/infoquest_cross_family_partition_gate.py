#!/usr/bin/env python3
"""Score frozen GPT supports with cross-family semantic likelihoods."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts import infoquest_cached_answer_ranking_diagnostic as diagnostic
from scripts import infoquest_cached_partition_eig_gate as cached
from scripts import infoquest_discrete_action_causal_gate as discrete
from scripts import infoquest_partition_eig_causal_gate as partition
from scripts import infoquest_refresh_continuation_gate as refresh
from scripts import infoquest_support_causal_link_gate as base


INTERFACE_VERSION = "infoquest-cross-family-partition-1"
INTERFACE_VERSION_V2 = "infoquest-cross-family-partition-2"
EXPECTED_SERVING_REQUESTS = 3
EXPECTED_MECHANICS_REQUESTS = 66
SERVING_MAX_COST_USD = 0.03
MECHANICS_MAX_COST_USD = 0.15

ModelBundle = refresh.ModelBundle
GateExecutionError = refresh.GateExecutionError


class DeterministicCrossFamilyScorer(base.DeterministicFixtureModel):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses = []
        for messages in batch_messages:
            request = json.loads(messages[-1]["content"])
            digest = hashlib.sha256(
                json.dumps(request, sort_keys=True).encode()
            ).digest()
            selected = (
                3
                if "SUPPORT=FIXED" in messages[0]["content"]
                else digest[0] % 4
            )
            profiles = []
            for action_index in range(4):
                if action_index == selected:
                    profiles.append([0, 0, 1, 1, 2, 2, 3, 3])
                elif action_index == (selected + 1) % 4:
                    profiles.append([0, 0, 0, 0, 1, 1, 1, 1])
                else:
                    profiles.append([0] * base.SUPPORT_SIZE)
            responses.append(
                json.dumps(
                    {
                        "w": [10] * base.SUPPORT_SIZE,
                        **{
                            action: profiles[index]
                            for index, action in enumerate(
                                partition.ACTION_KEYS
                            )
                        },
                    },
                    separators=(",", ":"),
                )
            )
        self.requests += len(responses)
        return responses


def scoring_messages(
    fixture: base.WorldFixture,
    initial: base.InitialPolicy,
    root_index: int,
    root_answer: str,
    hypotheses: Sequence[str],
    *,
    support_kind: str,
) -> list[dict[str, str]]:
    if support_kind not in {"DYNAMIC", "FIXED"}:
        raise ValueError("support kind must be DYNAMIC or FIXED")
    candidates = discrete.candidate_bank(initial, root_index)
    request = {
        "ambiguous_seed_message": fixture.seed_message,
        "observed_question": initial.roots[root_index],
        "observed_answer": root_answer,
        "hypotheses": list(hypotheses),
        "candidate_actions": {
            label: question
            for label, question in zip(discrete.CHOICE_LABELS, candidates)
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "STAGE=CROSS_FAMILY_PARTITION. "
                f"SUPPORT={support_kind}. Treat the supplied eight hypotheses "
                "as the complete support. Using the observed history, assign "
                "posterior plausibilities in an eight-integer w array, each "
                "from 1 to 100. For each candidate A-D, predict the answer "
                "under every hypothesis and cluster semantically "
                "indistinguishable answers using local integer labels 0..7 "
                "in the corresponding eight-value arrays a, b, c, and d. Do "
                "not choose an action and do not rewrite hypotheses. Output "
                "only one JSON object with exactly fields w,a,b,c,d."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(request, separators=(",", ":")),
        },
    ]


def parse_scored_belief(
    response: str,
    hypotheses: Sequence[str],
    initial: base.InitialPolicy,
    root_index: int,
) -> partition.PartitionBelief:
    value = base._parse_exact_object(
        response,
        {"w", *partition.ACTION_KEYS},
    )
    for key in ("w", *partition.ACTION_KEYS):
        if not isinstance(value[key], list):
            raise ValueError(f"{key} is not an array")
        if len(value[key]) != base.SUPPORT_SIZE:
            raise ValueError(f"{key} array does not have eight values")
    enriched = {
        "h": list(hypotheses),
        **value,
    }
    return partition.parse_partition_belief(
        json.dumps(enriched, separators=(",", ":")),
        initial,
        root_index,
        require_fixed_support=False,
        max_cluster_label=7,
        compact_arrays=True,
    )


def _score_many(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    root_answers: Sequence[Sequence[str]],
    source: Sequence[Sequence[partition.PartitionBelief]],
    scorer: Any,
    *,
    support_kind: str,
    raw_path: Path,
    raw: dict[str, Any],
) -> list[list[partition.PartitionBelief]]:
    requests = []
    supports = []
    for fixture_index, fixture in enumerate(fixtures):
        initial = initials[fixture.record_id]
        for root_index in range(base.ROOT_COUNT):
            hypotheses = source[fixture_index][root_index].hypotheses
            supports.append((hypotheses, initial, root_index))
            requests.append(
                scoring_messages(
                    fixture,
                    initial,
                    root_index,
                    root_answers[fixture_index][root_index],
                    hypotheses,
                    support_kind=support_kind,
                )
            )
    responses = refresh._complete(
        scorer,
        requests,
        max_new_tokens=500,
    )
    refresh._checkpoint_stage(
        raw_path,
        raw,
        f"{support_kind.lower()}_scores",
        responses,
    )
    parsed = [
        parse_scored_belief(response, *support)
        for response, support in zip(responses, supports)
    ]
    return diagnostic._reshape(parsed)


def run_serving_gate(
    models: ModelBundle,
    *,
    raw_path: Path,
    interface_version: str = INTERFACE_VERSION,
) -> dict[str, Any]:
    initial = cached._synthetic_initial()
    fixture = base.WorldFixture(
        fixture_id="SYNTHETIC",
        record_id=-1,
        world=1,
        seed_message="Synthetic ambiguous request.",
        simulator_system="Unused.",
        truth_packet={},
        checklist=tuple(f"Checklist item {index}" for index in range(5)),
    )
    raw: dict[str, Any] = {"interface_version": interface_version}
    try:
        score_responses = refresh._complete(
            models.simulator,
            [
                scoring_messages(
                    fixture,
                    initial,
                    0,
                    "Synthetic root answer.",
                    initial.hypotheses,
                    support_kind=kind,
                )
                for kind in ("DYNAMIC", "FIXED")
            ],
            max_new_tokens=500,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "partition_scores",
            score_responses,
        )
        beliefs = [
            parse_scored_belief(
                response,
                initial.hypotheses,
                initial,
                0,
            )
            for response in score_responses
        ]
        common_answer = "Cached counterfactual answer."
        checklist_response = refresh._complete(
            models.checklist_judge,
            [
                diagnostic.checklist_messages(
                    fixture,
                    initial,
                    ["Initial answer."] * base.ROOT_COUNT,
                    [beliefs[0]] * base.ROOT_COUNT,
                    [common_answer] * base.ROOT_COUNT,
                    [beliefs[0]] * base.ROOT_COUNT,
                    [common_answer] * base.ROOT_COUNT,
                )
            ],
            max_new_tokens=320,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "checklist_judgment",
            checklist_response,
        )
        judgment = base.parse_checklist_judgment(checklist_response[0])
        diagnostic._validate_identical_paths(
            [[beliefs[0]] * base.ROOT_COUNT],
            [[common_answer] * base.ROOT_COUNT],
            [[beliefs[0]] * base.ROOT_COUNT],
            [[common_answer] * base.ROOT_COUNT],
            [judgment],
        )
        usage = refresh.aggregate_usage(models)
    except Exception as exc:
        base._checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            refresh.aggregate_usage(models),
        ) from exc
    gates = {
        "exact_3_physical_requests": (
            usage["physical_requests"] == EXPECTED_SERVING_REQUESTS
        ),
        "exact_3_http_attempts": (
            usage["http_attempts"] == EXPECTED_SERVING_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_03": (
            usage["adapter_cost_usd"] <= SERVING_MAX_COST_USD
        ),
        "all_stage_parsers_pass": True,
        "identical_paths_score_identically": True,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": interface_version,
            "stage": "serving",
            "expected_requests": EXPECTED_SERVING_REQUESTS,
            "support_generator": refresh.GENERATOR_MODEL_ID,
            "cross_family_likelihood_model": refresh.SIMULATOR_MODEL_ID,
            "checklist_judge": refresh.CHECKLIST_JUDGE_MODEL_ID,
            "scientific_endpoint_evaluated": False,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "all_batches_checkpointed_before_parse": True,
        },
        "gates": gates,
        "usage": usage,
    }


def run_mechanics_gate(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    root_answers: Sequence[Sequence[str]],
    source_dynamic: Sequence[Sequence[partition.PartitionBelief]],
    source_fixed: Sequence[Sequence[partition.PartitionBelief]],
    models: ModelBundle,
    *,
    raw_path: Path,
    interface_version: str = INTERFACE_VERSION,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "interface_version": interface_version,
        "source_v3_public_sha256": diagnostic.V3_PUBLIC_SHA256,
        "source_v3_raw_sha256": diagnostic.V3_RAW_SHA256,
        "cached_public_sha256": cached.CACHED_PUBLIC_SHA256,
        "cached_raw_sha256": cached.CACHED_RAW_SHA256,
    }
    try:
        dynamic = _score_many(
            fixtures,
            initials,
            root_answers,
            source_dynamic,
            models.simulator,
            support_kind="DYNAMIC",
            raw_path=raw_path,
            raw=raw,
        )
        fixed = _score_many(
            fixtures,
            initials,
            root_answers,
            source_fixed,
            models.simulator,
            support_kind="FIXED",
            raw_path=raw_path,
            raw=raw,
        )
        dynamic_answers = diagnostic.cached_answers_for_choices(
            fixtures,
            initials,
            root_answers,
            dynamic,
        )
        fixed_answers = diagnostic.cached_answers_for_choices(
            fixtures,
            initials,
            root_answers,
            fixed,
        )
        checklist_responses = refresh._complete(
            models.checklist_judge,
            [
                diagnostic.checklist_messages(
                    fixture,
                    initials[fixture.record_id],
                    root_answers[fixture_index],
                    dynamic[fixture_index],
                    dynamic_answers[fixture_index],
                    fixed[fixture_index],
                    fixed_answers[fixture_index],
                )
                for fixture_index, fixture in enumerate(fixtures)
            ],
            max_new_tokens=320,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "checklist_judgments",
            checklist_responses,
        )
        judgments = [
            base.parse_checklist_judgment(response)
            for response in checklist_responses
        ]
        identical = diagnostic._validate_identical_paths(
            dynamic,
            dynamic_answers,
            fixed,
            fixed_answers,
            judgments,
        )
        usage = refresh.aggregate_usage(models)
        metrics, fixture_metrics, scientific_gates = partition._public_metrics(
            fixtures,
            initials,
            root_answers,
            dynamic,
            fixed,
            dynamic_answers,
            fixed_answers,
            judgments,
        )
    except Exception as exc:
        base._checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            refresh.aggregate_usage(models),
        ) from exc
    metrics["identical_dynamic_fixed_paths"] = identical
    accounting_gates = {
        "exact_66_physical_requests": (
            usage["physical_requests"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "exact_66_http_attempts": (
            usage["http_attempts"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_15": (
            usage["adapter_cost_usd"] <= MECHANICS_MAX_COST_USD
        ),
        "identical_paths_score_identically": True,
    }
    gates = {**accounting_gates, **scientific_gates}
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": interface_version,
            "stage": "mechanics",
            "expected_requests": EXPECTED_MECHANICS_REQUESTS,
            "source_v3_public_sha256": diagnostic.V3_PUBLIC_SHA256,
            "source_v3_raw_sha256": diagnostic.V3_RAW_SHA256,
            "cached_public_sha256": cached.CACHED_PUBLIC_SHA256,
            "cached_raw_sha256": cached.CACHED_RAW_SHA256,
            "fresh_support_calls": 0,
            "fresh_likelihood_calls": 60,
            "fresh_simulator_calls": 0,
            "fresh_checklist_calls": 6,
            "support_generator": refresh.GENERATOR_MODEL_ID,
            "cross_family_likelihood_model": refresh.SIMULATOR_MODEL_ID,
            "checklist_judge": refresh.CHECKLIST_JUDGE_MODEL_ID,
            "development_only_if_passed": True,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "all_batches_checkpointed_before_parse": True,
        },
        "metrics": metrics,
        "fixture_metrics": fixture_metrics,
        "gates": gates,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("serving", "mechanics"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cached-raw", type=Path)
    parser.add_argument("--cached-public", type=Path)
    parser.add_argument("--v3-raw", type=Path)
    parser.add_argument("--v3-public", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--budget-amendment-v2", action="store_true")
    args = parser.parse_args()
    interface_version = (
        INTERFACE_VERSION_V2
        if args.budget_amendment_v2
        else INTERFACE_VERSION
    )

    fixtures, public_fixture = base.build_fixtures(
        settings_path=args.settings,
        baseline_path=args.baseline,
    )
    loaded = None
    if args.stage == "mechanics":
        required = (
            args.cached_raw,
            args.cached_public,
            args.v3_raw,
            args.v3_public,
        )
        if any(path is None for path in required):
            parser.error(
                "mechanics requires cached and V3 raw/public paths"
            )
        initials, root_answers = cached.load_cached_histories(
            fixtures,
            raw_path=args.cached_raw,
            public_path=args.cached_public,
        )
        source_dynamic, source_fixed = diagnostic.load_v3_beliefs(
            fixtures,
            initials,
            raw_path=args.v3_raw,
            public_path=args.v3_public,
        )
        loaded = (
            initials,
            root_answers,
            source_dynamic,
            source_fixed,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = (
        0.02 if args.stage == "serving" else 0.08
    )
    config.openrouter_run_budget_usd = (
        SERVING_MAX_COST_USD
        if args.stage == "serving"
        else MECHANICS_MAX_COST_USD
    )
    config.openrouter_concurrency = (
        EXPECTED_SERVING_REQUESTS
        if args.stage == "serving"
        else 30
    )
    config.openrouter_max_retries = 0
    config.openrouter_max_output_tokens = 500
    config.log_path = args.output_dir / "run.log"
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    models = (
        ModelBundle(
            generator=base.DeterministicFixtureModel("generator"),
            simulator=DeterministicCrossFamilyScorer("simulator"),
            checklist_judge=diagnostic.DeterministicChecklistJudge(
                "checklist_judge"
            ),
        )
        if args.dry_run
        else refresh._build_models(config)
    )

    try:
        if args.stage == "serving":
            result = run_serving_gate(
                models,
                raw_path=raw_path,
                interface_version=interface_version,
            )
        else:
            assert loaded is not None
            result = run_mechanics_gate(
                fixtures,
                loaded[0],
                loaded[1],
                loaded[2],
                loaded[3],
                models,
                raw_path=raw_path,
                interface_version=interface_version,
            )
        result["protocol"]["private_raw_sha256"] = diagnostic._sha256_path(
            raw_path
        )
        result["protocol"]["fixture_sha256"] = public_fixture[
            "private_fixture_sha256"
        ]
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": interface_version,
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = diagnostic._sha256_path(raw_path)
        base._checkpoint(args.output_dir / "GATE_FAILURE.json", failure)
        raise

    output_name = (
        "SERVING.json" if args.stage == "serving" else "MECHANICS.json"
    )
    base._checkpoint(args.output_dir / output_name, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gates"]["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
