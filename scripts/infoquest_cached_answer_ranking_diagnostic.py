#!/usr/bin/env python3
"""Re-score frozen InfoQuest choices with cached counterfactual answers."""

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
from scripts import infoquest_cached_partition_eig_gate as cached
from scripts import infoquest_partition_eig_causal_gate as partition
from scripts import infoquest_refresh_continuation_gate as refresh
from scripts import infoquest_support_causal_link_gate as base


INTERFACE_VERSION = "infoquest-cached-answer-ranking-diagnostic-1"
V3_PUBLIC_SHA256 = (
    "0cfbe3d1590001af508d351d131d12d747f2fcadf0448e2e7868809f40d968f7"
)
V3_RAW_SHA256 = (
    "134a9659dc5f86df3dba6e18fdc8c72a79f3524b166d2cd52d91a7fe88747e1e"
)
EXPECTED_SERVING_REQUESTS = 1
EXPECTED_MECHANICS_REQUESTS = 6
SERVING_MAX_COST_USD = 0.02
MECHANICS_MAX_COST_USD = 0.05

ModelBundle = refresh.ModelBundle
GateExecutionError = refresh.GateExecutionError


class DeterministicChecklistJudge(base.DeterministicFixtureModel):
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
            lines = []
            for index, transcript in enumerate(
                request["transcripts"],
                start=1,
            ):
                same = (
                    base._normalize(transcript["dynamic_followup"])
                    == base._normalize(transcript["fixed_followup"])
                    and transcript["dynamic_answer"]
                    == transcript["fixed_answer"]
                )
                dynamic = "10000" if same else "11000"
                lines.append(f"Q{index}|00000|{dynamic}|10000")
            responses.append("\n".join(lines))
        self.requests += len(responses)
        return responses


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reshape(
    values: Sequence[Any],
) -> list[list[Any]]:
    if len(values) != len(base.MECHANICS_IDS) * 2 * base.ROOT_COUNT:
        raise ValueError("partition response count is not thirty")
    return [
        list(values[index : index + base.ROOT_COUNT])
        for index in range(0, len(values), base.ROOT_COUNT)
    ]


def load_v3_beliefs(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    *,
    raw_path: Path,
    public_path: Path,
    expected_raw_sha256: str = V3_RAW_SHA256,
    expected_public_sha256: str = V3_PUBLIC_SHA256,
) -> tuple[
    list[list[partition.PartitionBelief]],
    list[list[partition.PartitionBelief]],
]:
    if _sha256_path(raw_path) != expected_raw_sha256:
        raise ValueError("V3 private raw SHA-256 mismatch")
    if _sha256_path(public_path) != expected_public_sha256:
        raise ValueError("V3 public mechanics SHA-256 mismatch")
    raw = json.loads(raw_path.read_text())
    public = json.loads(public_path.read_text())
    protocol = public.get("protocol", {})
    if raw.get("interface_version") != cached.INTERFACE_VERSION_V3:
        raise ValueError("V3 private interface version mismatch")
    if protocol.get("interface_version") != cached.INTERFACE_VERSION_V3:
        raise ValueError("V3 public interface version mismatch")
    if protocol.get("private_raw_sha256") != expected_raw_sha256:
        raise ValueError("V3 public artifact does not bind private raw")
    if protocol.get("cached_public_sha256") != cached.CACHED_PUBLIC_SHA256:
        raise ValueError("V3 public cached-public binding mismatch")
    if protocol.get("cached_raw_sha256") != cached.CACHED_RAW_SHA256:
        raise ValueError("V3 public cached-raw binding mismatch")
    if [item.get("fixture_id") for item in raw["private_fixtures"]] != [
        fixture.fixture_id for fixture in fixtures
    ]:
        raise ValueError("V3 fixture order mismatch")

    def parse_many(
        responses: Sequence[str],
        *,
        require_fixed_support: bool,
    ) -> list[list[partition.PartitionBelief]]:
        parsed = []
        for response_index, response in enumerate(responses):
            fixture_index = response_index // base.ROOT_COUNT
            root_index = response_index % base.ROOT_COUNT
            initial = initials[fixtures[fixture_index].record_id]
            parsed.append(
                partition.parse_partition_belief(
                    response,
                    initial,
                    root_index,
                    require_fixed_support=require_fixed_support,
                    max_cluster_label=7,
                    compact_arrays=True,
                )
            )
        return _reshape(parsed)

    return (
        parse_many(
            raw["dynamic_partitions"],
            require_fixed_support=False,
        ),
        parse_many(
            raw["fixed_partitions"],
            require_fixed_support=True,
        ),
    )


def cached_answers_for_choices(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    root_answers: Sequence[Sequence[str]],
    beliefs: Sequence[Sequence[partition.PartitionBelief]],
) -> list[list[str]]:
    rows = []
    for fixture_index, fixture in enumerate(fixtures):
        initial = initials[fixture.record_id]
        root_lookup = {
            base._normalize(root): index
            for index, root in enumerate(initial.roots)
        }
        if len(root_lookup) != base.ROOT_COUNT:
            raise ValueError("initial roots are not distinct")
        row = []
        for belief in beliefs[fixture_index]:
            root_index = root_lookup.get(
                base._normalize(belief.selected_question)
            )
            if root_index is None:
                raise ValueError("selected question is outside cached roots")
            row.append(root_answers[fixture_index][root_index])
        rows.append(row)
    return rows


def checklist_messages(
    fixture: base.WorldFixture,
    initial: base.InitialPolicy,
    root_answers: Sequence[str],
    dynamic: Sequence[partition.PartitionBelief],
    dynamic_answers: Sequence[str],
    fixed: Sequence[partition.PartitionBelief],
    fixed_answers: Sequence[str],
) -> list[dict[str, str]]:
    messages = base.checklist_judge_messages(
        fixture,
        initial,
        root_answers,
        [belief.as_policy() for belief in dynamic],
        dynamic_answers,
        [belief.selected_question for belief in fixed],
        fixed_answers,
    )
    messages[0]["content"] += (
        " If dynamic and fixed followup question-answer pairs are identical, "
        "their five-bit strings must be identical."
    )
    return messages


def _validate_identical_paths(
    dynamic: Sequence[Sequence[partition.PartitionBelief]],
    dynamic_answers: Sequence[Sequence[str]],
    fixed: Sequence[Sequence[partition.PartitionBelief]],
    fixed_answers: Sequence[Sequence[str]],
    judgments: Sequence[base.ChecklistJudgment],
) -> int:
    identical = 0
    for fixture_index in range(len(dynamic)):
        for root_index in range(base.ROOT_COUNT):
            same = (
                base._normalize(
                    dynamic[fixture_index][root_index].selected_question
                )
                == base._normalize(
                    fixed[fixture_index][root_index].selected_question
                )
                and dynamic_answers[fixture_index][root_index]
                == fixed_answers[fixture_index][root_index]
            )
            if same:
                identical += 1
                if (
                    judgments[fixture_index].dynamic[root_index]
                    != judgments[fixture_index].fixed[root_index]
                ):
                    raise ValueError(
                        "checklist judge scored an identical path differently"
                    )
    if identical == 0:
        raise ValueError("diagnostic has no identical paths")
    return identical


def run_serving_gate(
    models: ModelBundle,
    *,
    raw_path: Path,
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
    belief = partition.PartitionBelief(
        hypotheses=initial.hypotheses,
        weights=(10,) * base.SUPPORT_SIZE,
        profiles=((0,) * base.SUPPORT_SIZE,) * 4,
        eig_scores=(0.0,) * 4,
        selected_action_index=0,
        selected_question=initial.roots[1],
    )
    raw: dict[str, Any] = {"interface_version": INTERFACE_VERSION}
    try:
        responses = refresh._complete(
            models.checklist_judge,
            [
                checklist_messages(
                    fixture,
                    initial,
                    ["Initial answer."] * base.ROOT_COUNT,
                    [belief] * base.ROOT_COUNT,
                    ["Cached answer."] * base.ROOT_COUNT,
                    [belief] * base.ROOT_COUNT,
                    ["Cached answer."] * base.ROOT_COUNT,
                )
            ],
            max_new_tokens=320,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "checklist_judgment",
            responses,
        )
        judgment = base.parse_checklist_judgment(responses[0])
        identical = _validate_identical_paths(
            [[belief] * base.ROOT_COUNT],
            [["Cached answer."] * base.ROOT_COUNT],
            [[belief] * base.ROOT_COUNT],
            [["Cached answer."] * base.ROOT_COUNT],
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
        "exact_1_physical_request": usage["physical_requests"] == 1,
        "exact_1_http_attempt": usage["http_attempts"] == 1,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_02": (
            usage["adapter_cost_usd"] <= SERVING_MAX_COST_USD
        ),
        "all_stage_parsers_pass": True,
        "identical_paths_score_identically": identical == base.ROOT_COUNT,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": "serving",
            "expected_requests": EXPECTED_SERVING_REQUESTS,
            "cached_answer_replay": True,
            "scientific_endpoint_evaluated": False,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "all_batches_checkpointed_before_parse": True,
            "models": {
                "checklist_judge": refresh.CHECKLIST_JUDGE_MODEL_ID,
            },
        },
        "gates": gates,
        "usage": usage,
    }


def run_mechanics_gate(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    root_answers: Sequence[Sequence[str]],
    dynamic: Sequence[Sequence[partition.PartitionBelief]],
    fixed: Sequence[Sequence[partition.PartitionBelief]],
    models: ModelBundle,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    dynamic_answers = cached_answers_for_choices(
        fixtures,
        initials,
        root_answers,
        dynamic,
    )
    fixed_answers = cached_answers_for_choices(
        fixtures,
        initials,
        root_answers,
        fixed,
    )
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "source_v3_public_sha256": V3_PUBLIC_SHA256,
        "source_v3_raw_sha256": V3_RAW_SHA256,
        "cached_public_sha256": cached.CACHED_PUBLIC_SHA256,
        "cached_raw_sha256": cached.CACHED_RAW_SHA256,
    }
    try:
        responses = refresh._complete(
            models.checklist_judge,
            [
                checklist_messages(
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
            responses,
        )
        judgments = [
            base.parse_checklist_judgment(response) for response in responses
        ]
        identical = _validate_identical_paths(
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
        "exact_6_physical_requests": (
            usage["physical_requests"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "exact_6_http_attempts": (
            usage["http_attempts"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_05": (
            usage["adapter_cost_usd"] <= MECHANICS_MAX_COST_USD
        ),
        "identical_paths_score_identically": True,
        "exact_18_identical_paths": identical == 18,
    }
    gates = {**accounting_gates, **scientific_gates}
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": "mechanics",
            "expected_requests": EXPECTED_MECHANICS_REQUESTS,
            "source_v3_public_sha256": V3_PUBLIC_SHA256,
            "source_v3_raw_sha256": V3_RAW_SHA256,
            "cached_public_sha256": cached.CACHED_PUBLIC_SHA256,
            "cached_raw_sha256": cached.CACHED_RAW_SHA256,
            "cached_answer_replay": True,
            "new_partition_or_simulator_calls": 0,
            "development_post_hoc": True,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "all_batches_checkpointed_before_parse": True,
            "models": {
                "checklist_judge": refresh.CHECKLIST_JUDGE_MODEL_ID,
            },
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
    args = parser.parse_args()

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
        dynamic, fixed = load_v3_beliefs(
            fixtures,
            initials,
            raw_path=args.v3_raw,
            public_path=args.v3_public,
        )
        loaded = (initials, root_answers, dynamic, fixed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.01
    config.openrouter_run_budget_usd = (
        SERVING_MAX_COST_USD
        if args.stage == "serving"
        else MECHANICS_MAX_COST_USD
    )
    config.openrouter_concurrency = (
        EXPECTED_SERVING_REQUESTS
        if args.stage == "serving"
        else EXPECTED_MECHANICS_REQUESTS
    )
    config.openrouter_max_retries = 0
    config.openrouter_max_output_tokens = 400
    config.log_path = args.output_dir / "run.log"
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    models = (
        ModelBundle(
            generator=base.DeterministicFixtureModel("generator"),
            simulator=base.DeterministicFixtureModel("simulator"),
            checklist_judge=DeterministicChecklistJudge(
                "checklist_judge"
            ),
        )
        if args.dry_run
        else refresh._build_models(config)
    )

    try:
        if args.stage == "serving":
            result = run_serving_gate(models, raw_path=raw_path)
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
            )
        result["protocol"]["private_raw_sha256"] = _sha256_path(raw_path)
        result["protocol"]["fixture_sha256"] = public_fixture[
            "private_fixture_sha256"
        ]
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = _sha256_path(raw_path)
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
