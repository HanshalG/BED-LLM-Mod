#!/usr/bin/env python3
"""Audit InfoQuest EIG rankings against frozen checklist-bit target gain."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import infoquest_cached_answer_ranking_diagnostic as diagnostic
from scripts import infoquest_cached_partition_eig_gate as cached
from scripts import infoquest_support_causal_link_gate as base


INTERFACE_VERSION = "infoquest-target-alignment-audit-1"
DIAGNOSTIC_PUBLIC_SHA256 = (
    "9855e57d3a00afcd074e15812aa14bb987ede1bc8b09d97a9382ae8b901451f2"
)
DIAGNOSTIC_RAW_SHA256 = (
    "e5a128d92fa48693e3b6f28b216d3b8c314bb3c67081ad1ba6cdb485e932dc9a"
)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[cursor]]
        ):
            end += 1
        rank = (cursor + 1 + end) / 2
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def spearman(values: Sequence[float], targets: Sequence[float]) -> float | None:
    if len(values) != len(targets) or len(values) < 2:
        raise ValueError("Spearman inputs have incompatible lengths")
    if len(set(values)) == 1 or len(set(targets)) == 1:
        return None
    x = average_ranks(values)
    y = average_ranks(targets)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    )
    denominator = math.sqrt(
        sum((value - x_mean) ** 2 for value in x)
        * sum((value - y_mean) ** 2 for value in y)
    )
    return numerator / denominator


def additive_gain(
    current: Sequence[int],
    candidate: Sequence[int],
) -> int:
    if len(current) != 5 or len(candidate) != 5:
        raise ValueError("checklist vectors must have five bits")
    return sum(
        current_bit == 0 and candidate_bit == 1
        for current_bit, candidate_bit in zip(current, candidate)
    )


def union_bits(
    first: Sequence[int],
    second: Sequence[int],
) -> tuple[int, ...]:
    if len(first) != 5 or len(second) != 5:
        raise ValueError("checklist vectors must have five bits")
    return tuple(max(a, b) for a, b in zip(first, second))


def load_diagnostic_judgments(
    *,
    raw_path: Path,
    public_path: Path,
) -> tuple[
    list[base.ChecklistJudgment],
    list[str],
]:
    if _sha256_path(raw_path) != DIAGNOSTIC_RAW_SHA256:
        raise ValueError("diagnostic private raw SHA-256 mismatch")
    if _sha256_path(public_path) != DIAGNOSTIC_PUBLIC_SHA256:
        raise ValueError("diagnostic public SHA-256 mismatch")
    raw = json.loads(raw_path.read_text())
    public = json.loads(public_path.read_text())
    protocol = public.get("protocol", {})
    if raw.get("interface_version") != diagnostic.INTERFACE_VERSION:
        raise ValueError("diagnostic raw interface mismatch")
    if protocol.get("interface_version") != diagnostic.INTERFACE_VERSION:
        raise ValueError("diagnostic public interface mismatch")
    if protocol.get("private_raw_sha256") != DIAGNOSTIC_RAW_SHA256:
        raise ValueError("diagnostic public artifact does not bind raw")
    if protocol.get("source_v3_public_sha256") != (
        diagnostic.V3_PUBLIC_SHA256
    ):
        raise ValueError("diagnostic V3 public binding mismatch")
    if protocol.get("source_v3_raw_sha256") != diagnostic.V3_RAW_SHA256:
        raise ValueError("diagnostic V3 raw binding mismatch")
    responses = raw.get("checklist_judgments")
    if not isinstance(responses, list) or len(responses) != 6:
        raise ValueError("diagnostic does not contain six judgments")
    judgments = [
        base.parse_checklist_judgment(response) for response in responses
    ]
    fixture_ids = [
        item.get("fixture_id") for item in public.get("fixture_metrics", [])
    ]
    if len(fixture_ids) != 6 or any(not value for value in fixture_ids):
        raise ValueError("diagnostic fixture IDs are malformed")
    return judgments, fixture_ids


def _selected_root_index(
    initial: base.InitialPolicy,
    question: str,
) -> int:
    normalized = base._normalize(question)
    matches = [
        index
        for index, root in enumerate(initial.roots)
        if base._normalize(root) == normalized
    ]
    if len(matches) != 1:
        raise ValueError("selected question does not map to one root")
    return matches[0]


def audit(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    dynamic: Sequence[Sequence[Any]],
    fixed: Sequence[Sequence[Any]],
    judgments: Sequence[base.ChecklistJudgment],
    fixture_ids: Sequence[str],
) -> dict[str, Any]:
    if [fixture.fixture_id for fixture in fixtures] != list(fixture_ids):
        raise ValueError("fixture order differs across artifacts")
    dynamic_regrets: list[int] = []
    fixed_regrets: list[int] = []
    dynamic_optimal = 0
    fixed_optimal = 0
    dynamic_selected_gains: list[int] = []
    fixed_selected_gains: list[int] = []
    oracle_gains: list[int] = []
    dynamic_rhos: list[float] = []
    fixed_rhos: list[float] = []
    spread_cells = 0
    selected_agreements = 0
    selected_total_bits = 0
    wins = ties = losses = 0
    fixture_metrics = []

    for fixture_index, fixture in enumerate(fixtures):
        initial = initials[fixture.record_id]
        judgment = judgments[fixture_index]
        fixture_dynamic = []
        fixture_fixed = []
        fixture_oracle = []
        fixture_dynamic_regret = []
        fixture_fixed_regret = []
        for root_index in range(base.ROOT_COUNT):
            candidate_indices = [
                index
                for index in range(base.ROOT_COUNT)
                if index != root_index
            ]
            current_bits = judgment.immediate[root_index]
            candidate_gains = [
                additive_gain(current_bits, judgment.immediate[index])
                for index in candidate_indices
            ]
            oracle = max(candidate_gains)
            spread_cells += max(candidate_gains) > min(candidate_gains)

            dynamic_belief = dynamic[fixture_index][root_index]
            fixed_belief = fixed[fixture_index][root_index]
            dynamic_index = _selected_root_index(
                initial,
                dynamic_belief.selected_question,
            )
            fixed_index = _selected_root_index(
                initial,
                fixed_belief.selected_question,
            )
            dynamic_action = candidate_indices.index(dynamic_index)
            fixed_action = candidate_indices.index(fixed_index)
            dynamic_gain = candidate_gains[dynamic_action]
            fixed_gain = candidate_gains[fixed_action]
            dynamic_regret = oracle - dynamic_gain
            fixed_regret = oracle - fixed_gain

            dynamic_selected_gains.append(dynamic_gain)
            fixed_selected_gains.append(fixed_gain)
            oracle_gains.append(oracle)
            dynamic_regrets.append(dynamic_regret)
            fixed_regrets.append(fixed_regret)
            dynamic_optimal += dynamic_regret == 0
            fixed_optimal += fixed_regret == 0
            fixture_dynamic.append(dynamic_gain)
            fixture_fixed.append(fixed_gain)
            fixture_oracle.append(oracle)
            fixture_dynamic_regret.append(dynamic_regret)
            fixture_fixed_regret.append(fixed_regret)

            if dynamic_gain > fixed_gain:
                wins += 1
            elif dynamic_gain == fixed_gain:
                ties += 1
            else:
                losses += 1

            dynamic_rho = spearman(
                dynamic_belief.eig_scores,
                candidate_gains,
            )
            fixed_rho = spearman(
                fixed_belief.eig_scores,
                candidate_gains,
            )
            if dynamic_rho is not None:
                dynamic_rhos.append(dynamic_rho)
            if fixed_rho is not None:
                fixed_rhos.append(fixed_rho)

            for selected_index, observed in (
                (dynamic_index, judgment.dynamic[root_index]),
                (fixed_index, judgment.fixed[root_index]),
            ):
                expected = union_bits(
                    current_bits,
                    judgment.immediate[selected_index],
                )
                selected_agreements += sum(
                    expected_bit == observed_bit
                    for expected_bit, observed_bit in zip(expected, observed)
                )
                selected_total_bits += 5

        fixture_metrics.append(
            {
                "fixture_id": fixture.fixture_id,
                "dynamic_target_gains": fixture_dynamic,
                "fixed_target_gains": fixture_fixed,
                "oracle_target_gains": fixture_oracle,
                "dynamic_target_regrets": fixture_dynamic_regret,
                "fixed_target_regrets": fixture_fixed_regret,
            }
        )

    cells = len(oracle_gains)
    agreement = selected_agreements / selected_total_bits
    metrics = {
        "fixtures": len(fixtures),
        "root_world_cells": cells,
        "candidate_actions": cells * 4,
        "selected_path_bits_checked": selected_total_bits,
        "selected_path_additive_bit_agreement": agreement,
        "cells_with_candidate_target_gain_spread": spread_cells,
        "mean_oracle_target_gain": sum(oracle_gains) / cells,
        "mean_dynamic_selected_target_gain": (
            sum(dynamic_selected_gains) / cells
        ),
        "mean_fixed_selected_target_gain": sum(fixed_selected_gains) / cells,
        "mean_oracle_minus_dynamic_regret": sum(dynamic_regrets) / cells,
        "mean_oracle_minus_fixed_regret": sum(fixed_regrets) / cells,
        "dynamic_target_optimal_cells": dynamic_optimal,
        "fixed_target_optimal_cells": fixed_optimal,
        "dynamic_vs_fixed_target_wins_ties_losses": [wins, ties, losses],
        "dynamic_defined_within_cell_rhos": len(dynamic_rhos),
        "fixed_defined_within_cell_rhos": len(fixed_rhos),
        "mean_dynamic_eig_target_gain_spearman": (
            sum(dynamic_rhos) / len(dynamic_rhos)
            if dynamic_rhos
            else None
        ),
        "mean_fixed_eig_target_gain_spearman": (
            sum(fixed_rhos) / len(fixed_rhos)
            if fixed_rhos
            else None
        ),
    }
    dynamic_rho_value = metrics["mean_dynamic_eig_target_gain_spearman"]
    gates = {
        "exact_6_fixtures_30_cells_120_actions": (
            metrics["fixtures"] == 6
            and metrics["root_world_cells"] == 30
            and metrics["candidate_actions"] == 120
        ),
        "selected_path_additive_agreement_at_least_0_90": agreement >= 0.90,
        "at_least_15_cells_have_target_gain_spread": spread_cells >= 15,
        "oracle_has_at_least_0_15_gain_headroom_over_fixed": (
            metrics["mean_oracle_minus_fixed_regret"] >= 0.15
        ),
        "dynamic_mean_target_spearman_at_least_0_20": (
            dynamic_rho_value is not None and dynamic_rho_value >= 0.20
        ),
        "dynamic_regret_no_worse_than_fixed": (
            metrics["mean_oracle_minus_dynamic_regret"]
            <= metrics["mean_oracle_minus_fixed_regret"]
        ),
        "zero_llm_calls": True,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "v3_public_sha256": diagnostic.V3_PUBLIC_SHA256,
            "v3_raw_sha256": diagnostic.V3_RAW_SHA256,
            "diagnostic_public_sha256": DIAGNOSTIC_PUBLIC_SHA256,
            "diagnostic_raw_sha256": DIAGNOSTIC_RAW_SHA256,
            "target_gain_proxy": (
                "new immediate checklist bits from candidate root answer"
            ),
            "selected_path_proxy_validation": True,
            "post_hoc_development_audit": True,
            "private_semantic_text_emitted": False,
            "llm_calls": 0,
            "cost_usd": 0.0,
        },
        "metrics": metrics,
        "fixture_metrics": fixture_metrics,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cached-raw", type=Path, required=True)
    parser.add_argument("--cached-public", type=Path, required=True)
    parser.add_argument("--v3-raw", type=Path, required=True)
    parser.add_argument("--v3-public", type=Path, required=True)
    parser.add_argument("--diagnostic-raw", type=Path, required=True)
    parser.add_argument("--diagnostic-public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixtures, fixture_public = base.build_fixtures(
        settings_path=args.settings,
        baseline_path=args.baseline,
    )
    initials, _root_answers = cached.load_cached_histories(
        fixtures,
        raw_path=args.cached_raw,
        public_path=args.cached_public,
    )
    dynamic, fixed = diagnostic.load_v3_beliefs(
        fixtures,
        initials,
        raw_path=args.v3_raw,
        public_path=args.v3_public,
    )
    judgments, fixture_ids = load_diagnostic_judgments(
        raw_path=args.diagnostic_raw,
        public_path=args.diagnostic_public,
    )
    result = audit(
        fixtures,
        initials,
        dynamic,
        fixed,
        judgments,
        fixture_ids,
    )
    result["protocol"]["fixture_sha256"] = fixture_public[
        "private_fixture_sha256"
    ]
    base._checkpoint(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gates"]["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
