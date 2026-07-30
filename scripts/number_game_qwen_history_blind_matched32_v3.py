#!/usr/bin/env python3
"""Run the fresh-seed V3 matched history-blind Qwen control."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_qwen_history_blind_matched32 as base
from scripts.number_game_qwen_history_blind_serving_smoke import sha256_file


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-history-blind-matched32-v3-1"
CONTROL_SEED_START = 9_400_000
ORIGINAL_BASE_MECHANICS_GATES = base.mechanics_gates

V1_FAILURE = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_history_blind_matched32"
    / "number-game-qwen-history-blind-matched32-20260730T153000Z"
    / "FAILURE.json"
)
V1_FAILURE_SHA256 = (
    "f394c889a56fdb20854d28cda0fa2927ae1a8d40e02c54f0f400f3f2757fdcb3"
)
V1_CONTROLS_SHA256 = (
    "3696476ea4aabd17f92ba06fb5142411369c11cc9aabb91f0c2feeac25437f89"
)
V2_RUNNER_FAILURE = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_history_blind_matched32_v2"
    / "number-game-qwen-history-blind-matched32-v2-20260730T163000Z"
    / "V2_RUNNER_FAILURE.json"
)
V2_RUNNER_FAILURE_SHA256 = (
    "4b3abc38059f11dceffb82ba1996d11ec7ecc1eae51616f4a2d63c895da2abd6"
)
V2_CONTROLS_SHA256 = (
    "f2c8c345eb0b1c3d7658372b5f9cb78d8b78ccd5b1b23b61ff96425d4a252e15"
)
V2_PRIVATE_RAW_RESPONSES_SHA256 = (
    "3057d850408338efdad1fe344d9210ebfd4fdbaefe174e65b3832113b4086994"
)
V2_ENDPOINT_ACCESSED = False


def validate_v1_failure(path: Path = V1_FAILURE) -> dict[str, Any]:
    if sha256_file(path) != V1_FAILURE_SHA256:
        raise ValueError("V1 failure artifact hash changed")
    failure = json.loads(path.read_text(encoding="utf-8"))
    protocol = failure.get("protocol") or {}
    if failure.get("status") != "mechanics_failed":
        raise ValueError("V1 did not fail mechanics")
    if protocol.get("endpoint_accessed") is not False:
        raise ValueError("V1 failure accessed the endpoint")
    if protocol.get("controls_sha256") != V1_CONTROLS_SHA256:
        raise ValueError("V1 controls binding changed")
    gates = failure.get("mechanics_gates") or {}
    failed = {name for name, value in gates.items() if not value}
    if failed != {"every_second_draw_adds_at_least_two_extensions"}:
        raise ValueError("V1 failure does not isolate the frozen novelty gate")
    return failure


def validate_v2_failure(
    path: Path = V2_RUNNER_FAILURE,
) -> dict[str, Any]:
    if sha256_file(path) != V2_RUNNER_FAILURE_SHA256:
        raise ValueError("V2 runner failure artifact hash changed")
    failure = json.loads(path.read_text(encoding="utf-8"))
    if failure.get("status") != "failed_closed":
        raise ValueError("V2 runner did not fail closed")
    if failure.get("error_type") != "RecursionError":
        raise ValueError("V2 failure type changed")
    if failure.get("error") != "maximum recursion depth exceeded":
        raise ValueError("V2 failure message changed")
    if V2_ENDPOINT_ACCESSED is not False:
        raise ValueError("V2 endpoint-access binding changed")
    if not V2_CONTROLS_SHA256 or not V2_PRIVATE_RAW_RESPONSES_SHA256:
        raise ValueError("V2 response bindings are incomplete")
    return failure


def mechanics_gates(
    *,
    usage: dict[str, Any],
    controls: dict[str, Any],
    pool_rows: Sequence[dict[str, Any]],
) -> dict[str, bool]:
    gates = ORIGINAL_BASE_MECHANICS_GATES(
        usage=usage,
        controls=controls,
        pool_rows=pool_rows,
    )
    gates.pop("every_second_draw_adds_at_least_two_extensions")
    return gates


def second_draw_novelty(
    controls: dict[str, Any],
) -> dict[str, Any]:
    values = [
        int(branch["diagnostic"]["draw_novel_contributions"][1])
        for tree in controls["trees"]
        for branch in tree["branches"].values()
    ]
    if len(values) != base.TREE_COUNT * base.SLOTS_PER_TREE:
        raise ValueError("V3 novelty descriptives have the wrong slot count")
    distribution = Counter(values)
    return {
        "branch_slot_count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
        "fraction_at_least_two": sum(value >= 2 for value in values)
        / len(values),
        "distribution": {
            str(value): distribution[value] for value in sorted(distribution)
        },
        "used_as_gate": False,
    }


@contextmanager
def configured_base() -> Iterator[None]:
    overrides = {
        "INTERFACE_VERSION": INTERFACE_VERSION,
        "CONTROL_SEED_START": CONTROL_SEED_START,
        "mechanics_gates": mechanics_gates,
    }
    originals = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def run_formal(
    *,
    output_dir: Path,
    run_id: str,
    smoke_result_path: Path = base.SMOKE_RESULT,
    adapter=None,
    remaining_credit: float | None = None,
    bootstrap_samples: int = base.BOOTSTRAP_SAMPLES,
    v1_failure_path: Path = V1_FAILURE,
    v2_failure_path: Path = V2_RUNNER_FAILURE,
) -> dict[str, Any]:
    validate_v1_failure(v1_failure_path)
    validate_v2_failure(v2_failure_path)
    with configured_base():
        result = base.run_formal(
            output_dir=output_dir,
            run_id=run_id,
            smoke_result_path=smoke_result_path,
            adapter=adapter,
            remaining_credit=remaining_credit,
            bootstrap_samples=bootstrap_samples,
        )
    controls_path = output_dir / "CONTROLS.json"
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    result["protocol"].update(
        {
            "interface_version": INTERFACE_VERSION,
            "v1_failure_sha256": V1_FAILURE_SHA256,
            "v1_controls_sha256": V1_CONTROLS_SHA256,
            "v1_endpoint_accessed": False,
            "v1_responses_reused": False,
            "v2_runner_failure_sha256": V2_RUNNER_FAILURE_SHA256,
            "v2_controls_sha256": V2_CONTROLS_SHA256,
            "v2_private_raw_responses_sha256": (
                V2_PRIVATE_RAW_RESPONSES_SHA256
            ),
            "v2_endpoint_accessed": V2_ENDPOINT_ACCESSED,
            "v2_responses_reused": False,
            "v3_change": (
                "capture and call the immutable original base mechanics "
                "function while second-draw novelty remains descriptive"
            ),
        }
    )
    result["second_draw_novelty"] = second_draw_novelty(controls)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--smoke-result",
        type=Path,
        default=base.SMOKE_RESULT,
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
        failure_path = args.output_dir / "V3_RUNNER_FAILURE.json"
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
