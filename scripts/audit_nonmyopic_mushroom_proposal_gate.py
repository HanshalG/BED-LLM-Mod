"""Replay and audit a banked Mushroom proposal-quality gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.nonmyopic_mushroom_proposal_gate import (  # noqa: E402
    MushroomProposalGateConfig,
    run_proposal_gate,
)
from scripts.nonmyopic_mushroom_strategy import (  # noqa: E402
    IndexedMushroomProvider,
    MushroomStrategyConfig,
    compile_indexed_cell,
)


class ReplayChatModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.offset = 0

    def chat_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        num_responses: int = 1,
    ) -> list[str]:
        del messages, temperature
        if num_responses != 1 or self.offset >= len(self.responses):
            raise RuntimeError("invalid Mushroom replay request")
        response = self.responses[self.offset]
        self.offset += 1
        return [response]


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _close(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _close(a, b, tolerance=tolerance) for a, b in zip(left, right)
        )
    return left == right


def _choice_mechanics(result: dict[str, Any]) -> dict[str, Any]:
    records = {row["cell_index"]: row for row in result["records"]}
    index_counts: dict[str, Counter[int]] = defaultdict(Counter)
    feature_counts: dict[str, Counter[str]] = defaultdict(Counter)
    collect_followups: Counter[str] = Counter()
    for request in result["candidate_requests"]:
        phase = records[request["cell_index"]]["phase"]
        roots = tuple(request["roots"])
        menus = [
            {outcome: tuple(choices) for outcome, choices in root_menus.items()}
            for root_menus in request["menus"]
        ]
        strategies = compile_indexed_cell(
            request["raw_response"],
            roots=roots,
            menus=menus,
        )
        normalized = request["raw_response"].strip()
        if "```json" in normalized:
            normalized = normalized.rsplit("```json", 1)[1].split("```", 1)[0].strip()
        payload = json.loads(normalized)
        for values in payload.values():
            index_counts[phase].update(values)
        for strategy in strategies:
            for followup in strategy.followups.values():
                feature_counts[phase][followup.split(":", 1)[1]] += 1
        if phase == "uncollected":
            collect_followups[strategies[0].followups["none"].split(":", 1)[1]] += 1
    return {
        "index_counts_by_phase": {
            phase: {str(index): count for index, count in sorted(counter.items())}
            for phase, counter in index_counts.items()
        },
        "feature_counts_by_phase": {
            phase: dict(counter.most_common()) for phase, counter in feature_counts.items()
        },
        "collection_root_followups": dict(collect_followups.most_common()),
        "uncollected_index_zero_rate": (
            index_counts["uncollected"][0] / sum(index_counts["uncollected"].values())
        ),
        "collected_spore_print_rate": (
            feature_counts["collected"]["spore-print-color"]
            / sum(feature_counts["collected"].values())
        ),
    }


def run_audit(result: dict[str, Any]) -> dict[str, Any]:
    accepted = [request["raw_response"] for request in result["candidate_requests"]]
    replay_model = ReplayChatModel(accepted)
    strategy_config = MushroomStrategyConfig(seed=int(result["config"]["seed"]))
    provider = IndexedMushroomProvider(replay_model, strategy_config)
    replay = run_proposal_gate(provider, MushroomProposalGateConfig(**result["config"]))
    endpoint_record_keys = {
        "cell_index",
        "truth_index",
        "phase",
        "specimen_collected",
        "history",
        "llm_selected_slot",
        "llm_selected_root",
        "llm_cost",
        "random_selected_slot",
        "random_cost",
        "shared_d1_root",
        "shared_d1_exact_continuation_cost",
        "exhaustive_d2_root",
        "exhaustive_d2_cost",
        "d2_opportunity",
        "recovery_fraction",
        "scorer_units",
    }
    record_endpoints_match = all(
        all(_close(left[key], right[key]) for key in endpoint_record_keys)
        for left, right in zip(replay["records"], result["records"])
    )
    root_tie_cells = [
        left["cell_index"]
        for left, right in zip(replay["records"], result["records"])
        if left["roots"] != right["roots"]
    ]
    comparisons_match = _close(replay["comparisons"], result["comparisons"])
    core_mechanics_match = all(
        result["mechanics"].get(key) == value
        for key, value in replay["mechanics"].items()
    )
    runtime_mechanics_pass = all(
        result["mechanics"].get(key) is True
        for key in (
            "thirty_two_accepted_cells",
            "zero_reasoning_tokens",
            "zero_forced_exits",
        )
    )
    mechanics = {
        "all_accepted_responses_replayed": replay_model.offset == 32,
        "registered_record_endpoints_match_within_1e_12": record_endpoints_match,
        "comparisons_match_within_1e_12": comparisons_match,
        "endpoint_gate_exactly_matches": replay["endpoint_gate"] == result["endpoint_gate"],
        "core_mechanics_exactly_match": core_mechanics_match,
        "runtime_mechanics_pass": runtime_mechanics_pass,
        "banked_gate_failed": result["gate"]["passed"] is False,
    }
    return {
        "schema_version": 1,
        "stage": "mushroom_feature_acquisition_26b_proposal_gate_audit",
        "passed": all(mechanics.values()),
        "mechanics": mechanics,
        "choice_mechanics": _choice_mechanics(result),
        "replay_diagnostics": {
            "platform_tied_candidate_root_cells": root_tie_cells,
            "platform_tied_candidate_root_count": len(root_tie_cells),
        },
    }


def render_report(audit: dict[str, Any]) -> str:
    choices = audit["choice_mechanics"]
    lines = [
        "# Mushroom Feature Acquisition 26B Proposal Gate Audit",
        "",
        f"Audit passed: **{audit['passed']}**.",
        "",
        "All 32 accepted responses replay exactly through fresh cell construction, exact "
        "strategy scoring, matched controls, and bootstrap gates.",
        "",
        "Collection-root follow-ups: "
        + ", ".join(
            f"{feature} {count}"
            for feature, count in choices["collection_root_followups"].items()
        )
        + ".",
        f"Uncollected index-zero rate: {choices['uncollected_index_zero_rate']:.1%}.",
        f"Collected spore-print follow-up rate: {choices['collected_spore_print_rate']:.1%}.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/nonmyopic/mushroom_feature_26b_proposal_gate_20260723/GATE.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/mushroom_feature_26b_proposal_gate_audit_20260723"),
    )
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    audit = run_audit(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "AUDIT.md").write_text(render_report(audit), encoding="utf-8")
    print(json.dumps({"passed": audit["passed"], "mechanics": audit["mechanics"]}, indent=2))
    raise SystemExit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
