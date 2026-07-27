#!/usr/bin/env python3
"""Audit paired replay after deterministically seeding DiscoveryWorld objects."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any, Callable

import numpy as np


DISCOVERYWORLD_COMMIT = "fd591323920be0d3786ef350955de1945aa571e5"
INTERFACE_VERSION = "discoveryworld-deterministic-replay-audit-1"
SCIENTIFIC_THEMES = (
    "Combinatorial Chemistry",
    "Archaeology Dating",
    "Plant Nutrients",
    "Reactor Lab",
    "Lost in Translation",
    "Space Sick",
    "Proteomics",
    "It's (not) Rocket Science!",
)


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def stable_object_seed(world_seed: int | None, object_uuid: int) -> int:
    return stable_seed(INTERFACE_VERSION, "object", world_seed, object_uuid)


def patch_object_rng(object_class: type) -> Callable[..., None]:
    existing = getattr(object_class, "_bed_original_init", None)
    if existing is not None:
        return existing

    original_init = object_class.__init__

    def deterministic_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        world = args[0] if args else kwargs["world"]
        self.rng.seed(stable_object_seed(world.randomSeed, self.uuid))

    object_class._bed_original_init = original_init
    object_class.__init__ = deterministic_init
    return original_init


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_commit(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def reset_process_rngs(*, theme: str, difficulty: str, scenario_seed: int) -> int:
    seed = stable_seed(
        INTERFACE_VERSION,
        "process",
        theme,
        difficulty,
        scenario_seed,
    )
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return seed


def build_action_schedule(
    teleport_locations: list[str],
    *,
    num_steps: int,
) -> list[dict[str, Any]]:
    if teleport_locations:
        return [
            {
                "action": "TELEPORT_TO_LOCATION",
                "arg1": teleport_locations[index % len(teleport_locations)],
            }
            for index in range(num_steps)
        ]
    directions = ("north", "east", "south", "west")
    return [
        {
            "action": "MOVE_DIRECTION",
            "arg1": directions[index % len(directions)],
        }
        for index in range(num_steps)
    ]


def compare_trace_records(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    fields = (
        "process_seed",
        "actions",
        "action_results",
        "observation_hashes",
        "scorecard_hashes",
        "final_score_normalized",
        "completed",
        "completed_successfully",
    )
    mismatched = [field for field in fields if left[field] != right[field]]
    return {
        "exact_match": not mismatched,
        "mismatched_fields": mismatched,
        "num_observations": len(left["observation_hashes"]),
        "num_actions": len(left["actions"]),
    }


def _load_discoveryworld(source_root: Path) -> type:
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    object_module = importlib.import_module("discoveryworld.objects.Object")
    patch_object_rng(object_module.Object)
    api_module = importlib.import_module("discoveryworld.DiscoveryWorldAPI")
    return api_module.DiscoveryWorldAPI


def run_trace(
    api_class: type,
    *,
    theme: str,
    difficulty: str,
    scenario_seed: int,
    num_steps: int,
    thread_id: int,
    frame_root: Path,
) -> dict[str, Any]:
    process_seed = reset_process_rngs(
        theme=theme,
        difficulty=difficulty,
        scenario_seed=scenario_seed,
    )
    api = api_class(threadID=thread_id)
    api.FRAME_DIR = str(frame_root / f"thread-{thread_id}") + "/"
    loaded = api.loadScenario(
        scenarioName=theme,
        difficultyStr=difficulty,
        randomSeed=scenario_seed,
        numUserAgents=1,
    )
    if not loaded:
        raise RuntimeError(f"could not load {theme} / {difficulty}")

    observation_hashes = [
        canonical_sha256(api.getAgentObservation(agentIdx=0)["ui"])
    ]
    scorecard_hashes = [canonical_sha256(api.getTaskScorecard())]
    teleport_locations = sorted(api.listTeleportLocationsDict())
    actions = build_action_schedule(
        teleport_locations,
        num_steps=num_steps,
    )
    action_results = []
    for action in actions:
        action_results.append(api.performAgentAction(0, action))
        api.tick()
        observation_hashes.append(
            canonical_sha256(api.getAgentObservation(agentIdx=0)["ui"])
        )
        scorecard_hashes.append(canonical_sha256(api.getTaskScorecard()))

    final_scorecard = api.getTaskScorecard()[0]
    return {
        "process_seed": process_seed,
        "teleport_locations": teleport_locations,
        "actions": actions,
        "action_results": action_results,
        "observation_hashes": observation_hashes,
        "scorecard_hashes": scorecard_hashes,
        "final_score_normalized": float(final_scorecard["scoreNormalized"]),
        "completed": bool(final_scorecard["completed"]),
        "completed_successfully": bool(
            final_scorecard["completedSuccessfully"]
        ),
    }


def run_audit(
    *,
    source_root: Path,
    themes: tuple[str, ...] = SCIENTIFIC_THEMES,
    difficulty: str = "Challenge",
    scenario_seed: int = 0,
    num_steps: int = 5,
) -> dict[str, Any]:
    commit = source_commit(source_root)
    if commit != DISCOVERYWORLD_COMMIT:
        raise ValueError(
            f"DiscoveryWorld commit changed: {commit} != {DISCOVERYWORLD_COMMIT}"
        )
    api_class = _load_discoveryworld(source_root)
    results = {}
    with tempfile.TemporaryDirectory(prefix="discoveryworld-replay-") as raw:
        frame_root = Path(raw)
        for index, theme in enumerate(themes):
            left = run_trace(
                api_class,
                theme=theme,
                difficulty=difficulty,
                scenario_seed=scenario_seed,
                num_steps=num_steps,
                thread_id=1000 + 2 * index,
                frame_root=frame_root,
            )
            right = run_trace(
                api_class,
                theme=theme,
                difficulty=difficulty,
                scenario_seed=scenario_seed,
                num_steps=num_steps,
                thread_id=1001 + 2 * index,
                frame_root=frame_root,
            )
            comparison = compare_trace_records(left, right)
            results[theme] = {
                "comparison": comparison,
                "trace": left,
            }

    passed = all(
        item["comparison"]["exact_match"] for item in results.values()
    )
    return {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "deterministic_replay_pass"
            if passed
            else "deterministic_replay_failure"
        ),
        "source": {
            "repository": "https://github.com/allenai/discoveryworld",
            "commit": commit,
        },
        "protocol": {
            "difficulty": difficulty,
            "scenario_seed": scenario_seed,
            "num_steps": num_steps,
            "num_themes": len(themes),
            "object_rng_seed": "sha256(interface,world_seed,object_uuid)",
            "process_rngs_reset": ["random", "numpy"],
            "new_model_calls": 0,
            "openrouter_cost_usd": 0.0,
            "oatml_jobs": 0,
        },
        "summary": {
            "passed": passed,
            "num_exact_themes": sum(
                item["comparison"]["exact_match"]
                for item in results.values()
            ),
            "num_themes": len(results),
        },
        "themes": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--difficulty", default="Challenge")
    parser.add_argument("--scenario-seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=5)
    args = parser.parse_args()

    payload = run_audit(
        source_root=args.source_root,
        difficulty=args.difficulty,
        scenario_seed=args.scenario_seed,
        num_steps=args.num_steps,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    if not payload["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
