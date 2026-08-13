#!/usr/bin/env python3
"""Localize a banked Number Game control failure without scoring endpoints."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_PROTOCOL_SHA256 = (
    "6c94b4b5609dc5d9aa3913f7b11c1d989b7303460870eb86643ef329079b02c8"
)
EXPECTED_FAILURE_SHA256 = (
    "bf2a119f23e8241a7c2f694a3d3009104b0f3151073d37407cb6ddd7a917c244"
)
EXPECTED_CONTROLS_SHA256 = (
    "513def73b8e75f31eca123e593c2e9b0deab715fbd52ed6d3833020564237672"
)
EXPECTED_RAW_SHA256 = (
    "f8201be3cd199f895a6d971d4fa713beaee01660d4419a341c167a986cded241"
)
EXPECTED_CONTROL_KEYS = {
    "schema_version",
    "interface_version",
    "source_trees_sha256",
    "model",
    "prompt",
    "trees",
}
EXPECTED_TREE_KEYS = {"tree_index", "local_tree_index", "tree_seed", "branches"}
EXPECTED_BRANCH_KEYS = {"seeds", "support", "diagnostic"}
EXPECTED_REJECTION_KEYS = {
    "duplicate_extension",
    "inconsistent",
    "invalid_expression",
    "invalid_name",
    "missing_or_incomplete",
    "wrong_fields",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: Sequence[int], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires nonempty values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def failure_attribution(
    *, draw_diagnostics: Sequence[Mapping[str, Any]], pool_count: int
) -> str:
    if any(
        draw.get("codec_mode") != "strict_json"
        or int((draw.get("rejected") or {}).get("missing_or_incomplete", 0)) > 0
        for draw in draw_diagnostics
    ):
        return "codec_or_incomplete"
    failing_draws = [
        draw for draw in draw_diagnostics if int(draw["valid_unique_count"]) < 16
    ]
    if failing_draws:
        totals = Counter()
        for draw in failing_draws:
            totals.update(draw["rejected"])
        active = {name for name, count in totals.items() if count > 0}
        invalid = {"invalid_expression", "invalid_name", "wrong_fields"}
        if active and active <= invalid:
            return "invalid_rule"
        if active == {"duplicate_extension"}:
            return "duplicate_collapse"
        if active:
            return "mixed_rule_rejections"
        return "unexplained"
    if pool_count < 24:
        return "cross_draw_overlap"
    return "unexplained"


def localize(
    *,
    failure_path: Path,
    controls_path: Path,
    raw_path: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    bindings = {
        "protocol": sha256_file(protocol_path),
        "failure": sha256_file(failure_path),
        "controls": sha256_file(controls_path),
        "raw_responses": sha256_file(raw_path),
    }
    expected = {
        "protocol": EXPECTED_PROTOCOL_SHA256,
        "failure": EXPECTED_FAILURE_SHA256,
        "controls": EXPECTED_CONTROLS_SHA256,
        "raw_responses": EXPECTED_RAW_SHA256,
    }
    if bindings != expected:
        raise RuntimeError("frozen mechanics-localization binding changed")

    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if failure.get("status") != "mechanics_failed":
        raise RuntimeError("banked control is not the frozen mechanics failure")
    if failure.get("protocol", {}).get("endpoint_accessed") is not False:
        raise RuntimeError("banked policy endpoint was not sealed")

    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    if set(controls) != EXPECTED_CONTROL_KEYS:
        raise RuntimeError("control top-level schema changed")
    trees = controls["trees"]
    if not isinstance(trees, list) or len(trees) != 32:
        raise RuntimeError("control tree count changed")

    draw_counts: list[int] = []
    pool_counts: list[int] = []
    novel_counts: list[int] = []
    rejection_totals = Counter()
    failing_slots: list[dict[str, Any]] = []
    slot_count = 0
    for tree in trees:
        if set(tree) != EXPECTED_TREE_KEYS:
            raise RuntimeError("control tree schema changed")
        local_tree_index = int(tree["local_tree_index"])
        branches = tree["branches"]
        if not isinstance(branches, dict) or len(branches) != 48:
            raise RuntimeError("control branch count changed")
        for history_index, branch in enumerate(branches.values()):
            if set(branch) != EXPECTED_BRANCH_KEYS:
                raise RuntimeError("control branch schema changed")
            diagnostic = branch["diagnostic"]
            draws = diagnostic.get("draw_diagnostics")
            if not isinstance(draws, list) or len(draws) != 2:
                raise RuntimeError("pooled draw diagnostic count changed")
            for draw in draws:
                rejected = draw.get("rejected")
                if not isinstance(rejected, dict) or set(rejected) != EXPECTED_REJECTION_KEYS:
                    raise RuntimeError("draw rejection schema changed")
                count = int(draw["valid_unique_count"])
                draw_counts.append(count)
                rejection_totals.update({key: int(value) for key, value in rejected.items()})
            pool_count = int(diagnostic["valid_unique_count"])
            novelty = [int(value) for value in diagnostic["draw_novel_contributions"]]
            if len(novelty) != 2:
                raise RuntimeError("draw novelty diagnostic count changed")
            pool_counts.append(pool_count)
            novel_counts.append(novelty[1])
            slot_count += 1
            if min(int(draw["valid_unique_count"]) for draw in draws) < 16 or pool_count < 24:
                failing_slots.append(
                    {
                        "local_tree_index": local_tree_index,
                        "history_index": history_index,
                        "stage": "first" if history_index < 16 else "second",
                        "draw_valid_counts": [int(draw["valid_unique_count"]) for draw in draws],
                        "pool_valid_count": pool_count,
                        "draw_novel_contributions": novelty,
                        "rejection_counts": {
                            key: sum(int(draw["rejected"][key]) for draw in draws)
                            for key in sorted(EXPECTED_REJECTION_KEYS)
                        },
                        "attribution": failure_attribution(
                            draw_diagnostics=draws, pool_count=pool_count
                        ),
                    }
                )

    if slot_count != 1536 or len(draw_counts) != 3072:
        raise RuntimeError("frozen control diagnostic population changed")
    attribution_counts = Counter(row["attribution"] for row in failing_slots)
    isolated = (
        len(failing_slots) <= 3
        and set(attribution_counts) <= {
            "duplicate_collapse",
            "invalid_rule",
            "mixed_rule_rejections",
        }
        and quantile(draw_counts, 0.01) >= 18
        and quantile(pool_counts, 0.01) >= 25
    )
    return {
        "schema_version": 1,
        "interface_version": "number-game-fresh-control-mechanics-localization-v1",
        "status": "diagnostic_complete",
        "decision": (
            "authorize_fresh_value_free_support_mechanics_design"
            if isolated
            else "do_not_repeat_same_family_full_control"
        ),
        "authorizes_paid_calls": False,
        "bindings": bindings,
        "counts": {
            "trees": len(trees),
            "slots": slot_count,
            "draws": len(draw_counts),
            "draws_below_16": sum(value < 16 for value in draw_counts),
            "pools_below_24": sum(value < 24 for value in pool_counts),
            "failing_slots": len(failing_slots),
        },
        "draw_valid_counts": {
            "minimum": min(draw_counts),
            "p01": quantile(draw_counts, 0.01),
            "median": quantile(draw_counts, 0.5),
            "maximum": max(draw_counts),
            "histogram": dict(sorted(Counter(draw_counts).items())),
        },
        "pool_valid_counts": {
            "minimum": min(pool_counts),
            "p01": quantile(pool_counts, 0.01),
            "median": quantile(pool_counts, 0.5),
            "maximum": max(pool_counts),
            "histogram": dict(sorted(Counter(pool_counts).items())),
        },
        "second_draw_novel_contributions": {
            "minimum": min(novel_counts),
            "p01": quantile(novel_counts, 0.01),
            "median": quantile(novel_counts, 0.5),
            "maximum": max(novel_counts),
            "histogram": dict(sorted(Counter(novel_counts).items())),
        },
        "rejection_totals": dict(sorted(rejection_totals.items())),
        "attribution_counts": dict(sorted(attribution_counts.items())),
        "failing_slots": failing_slots,
        "privacy": {
            "raw_response_file_hashed_but_not_deserialized": True,
            "support_payloads_not_deserialized": True,
            "source_trees_or_targets_read": 0,
            "policy_endpoints_scored": 0,
            "model_calls": 0,
            "openrouter_cost_usd": 0.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = localize(
        failure_path=args.failure,
        controls_path=args.controls,
        raw_path=args.raw,
        protocol_path=args.protocol,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
