"""Replay and audit a banked Cleveland Heart proposal-quality gate."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.nonmyopic_heart_workup_proposal_gate import (  # noqa: E402
    HeartProposalGateConfig,
    run_proposal_gate,
)
from scripts.nonmyopic_heart_workup_strategy import (  # noqa: E402
    HeartStrategyConfig,
    IndexedHeartProvider,
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
            raise RuntimeError("invalid Heart replay request")
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
    selected_roots: dict[str, Counter[str]] = {
        "unworked": Counter(),
        "worked": Counter(),
    }
    unworked_order_followups: Counter[str] = Counter()
    unworked_other_followups: Counter[str] = Counter()
    worked_followups: Counter[str] = Counter()
    unworked_order_index_zero = 0
    unworked_other_total = 0
    unworked_other_workup = 0
    for request in result["candidate_requests"]:
        phase = records[request["cell_index"]]["phase"]
        selected_roots[phase][records[request["cell_index"]]["llm_selected_root"]] += 1
        roots = tuple(request["roots"])
        menus = [
            {outcome: tuple(choices) for outcome, choices in root_menus.items()}
            for root_menus in request["menus"]
        ]
        strategies = compile_indexed_cell(
            request["raw_response"], roots=roots, menus=menus
        )
        if phase == "unworked":
            order_followup = strategies[0].followups["none"]
            unworked_order_followups[order_followup] += 1
            first_choice = next(iter(menus[0].values()))[0]
            unworked_order_index_zero += int(order_followup == first_choice)
            for strategy in strategies[1:]:
                unworked_other_followups.update(strategy.followups.values())
                unworked_other_total += len(strategy.followups)
                unworked_other_workup += sum(
                    followup == "order:clinical-workup"
                    for followup in strategy.followups.values()
                )
        else:
            for strategy in strategies:
                worked_followups.update(strategy.followups.values())
    return {
        "selected_roots_by_phase": {
            phase: dict(counter.most_common()) for phase, counter in selected_roots.items()
        },
        "unworked_order_root_followups": dict(unworked_order_followups.most_common()),
        "unworked_other_root_followups": dict(unworked_other_followups.most_common()),
        "worked_followups": dict(worked_followups.most_common()),
        "unworked_order_root_index_zero_rate": unworked_order_index_zero / 16,
        "unworked_other_branch_workup_rate": (
            unworked_other_workup / unworked_other_total
        ),
    }


def run_audit(result: dict[str, Any]) -> dict[str, Any]:
    accepted = [request["raw_response"] for request in result["candidate_requests"]]
    replay_model = ReplayChatModel(accepted)
    strategy_config = HeartStrategyConfig(seed=int(result["config"]["seed"]))
    provider = IndexedHeartProvider(replay_model, strategy_config)
    replay = run_proposal_gate(provider, HeartProposalGateConfig(**result["config"]))
    record_keys = {
        "cell_index",
        "truth_index",
        "phase",
        "workup_ordered",
        "history",
        "roots",
        "llm_costs",
        "llm_selected_slot",
        "llm_selected_root",
        "llm_cost",
        "random_costs",
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
    records_match = all(
        all(_close(left[key], right[key]) for key in record_keys)
        for left, right in zip(replay["records"], result["records"])
    )
    runtime_mechanics = all(
        result["mechanics"].get(key) is True
        for key in (
            "thirty_two_accepted_cells",
            "zero_reasoning_tokens",
            "zero_forced_exits",
        )
    )
    mechanics = {
        "all_accepted_responses_replayed": replay_model.offset == 32,
        "registered_records_match_within_1e_12": records_match,
        "comparisons_match_within_1e_12": _close(
            replay["comparisons"], result["comparisons"]
        ),
        "endpoint_gate_exactly_matches": replay["endpoint_gate"] == result["endpoint_gate"],
        "core_mechanics_exactly_match": all(
            result["mechanics"].get(key) == value
            for key, value in replay["mechanics"].items()
        ),
        "runtime_mechanics_pass": runtime_mechanics,
        "banked_gate_failed": result["gate"]["passed"] is False,
    }
    return {
        "schema_version": 1,
        "stage": "cleveland_heart_workup_26b_proposal_gate_audit",
        "passed": all(mechanics.values()),
        "mechanics": mechanics,
        "choice_mechanics": _choice_mechanics(result),
    }


def render_report(audit: dict[str, Any]) -> str:
    choices = audit["choice_mechanics"]
    return "\n".join(
        [
            "# Cleveland Heart Workup 26B Proposal Gate Audit",
            "",
            f"Audit passed: **{audit['passed']}**.",
            "",
            "All 32 accepted responses replay exactly through fresh cell construction, "
            "exact strategy scoring, matched controls, and bootstrap gates.",
            "",
            "Unworked order-root follow-ups: "
            + ", ".join(
                f"{action} {count}"
                for action, count in choices["unworked_order_root_followups"].items()
            )
            + ".",
            "Unworked order-root index-zero rate: "
            f"{choices['unworked_order_root_index_zero_rate']:.1%}.",
            "Unworked ordinary-root branches choosing workup: "
            f"{choices['unworked_other_branch_workup_rate']:.1%}.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/nonmyopic/heart_workup_26b_proposal_gate_20260723/GATE.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nonmyopic/heart_workup_26b_proposal_gate_audit_20260723"),
    )
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    audit = run_audit(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "AUDIT.md").write_text(
        render_report(audit), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
