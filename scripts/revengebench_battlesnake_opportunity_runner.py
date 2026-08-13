#!/usr/bin/env python3
"""Run the frozen RevengeBench BattleSnake V2 opportunity cohort locally."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from scripts import revengebench_execution_opportunity_math as opportunity_math
except ImportError:  # Direct script execution adds scripts/ rather than the repo root.
    import revengebench_execution_opportunity_math as opportunity_math


PROTOCOL_VERSION = "revengebench-battlesnake-opportunity-v3"
IMAGE = "bed-revengebench-battlesnake-v3:20260813"
IMAGE_ID = "sha256:9313f9a8f0afca2dc313ce9293d865d4eaa3b982367ad038a2af0aaf7a9bd1bc"
SERVER_SHA256 = "1464edfb6a18f8c92bbaebe4c963822ca5e9e3b0f38e05ad6af2552e07c3af87"
MAX_DECISIONS = 64
DIRECTIONS = {(0, 1): "up", (0, -1): "down", (1, 0): "right", (-1, 0): "left"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def run(command: list[str], *, timeout: float = 120.0, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout={result.stdout[-1000:]}\nstderr={result.stderr[-1000:]}"
        )
    return result


def _copy_policy(source: Path, destination: Path, server_path: Path) -> None:
    shutil.copytree(source, destination)
    shutil.copy2(server_path, destination / "server.py")


def _wait_server(container: str) -> None:
    command = [
        "docker",
        "exec",
        container,
        "python",
        "-c",
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/', timeout=1).read()",
    ]
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        result = run(command, timeout=3.0, check=False)
        if result.returncode == 0:
            return
        time.sleep(0.1)
    logs = run(["docker", "logs", container], check=False).stdout
    raise RuntimeError(f"server {container} did not become ready: {logs[-1000:]}")


def _start_policy(container: str, network: str, policy_dir: Path, seed: int, seed_shim: Path) -> None:
    run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container,
            "--network",
            network,
            "-e",
            "PORT=8000",
            "-e",
            f"REVENGEBENCH_REPLAY_SEED={seed}",
            "-e",
            f"PYTHONHASHSEED={seed}",
            "-e",
            "PYTHONPATH=/seed",
            "-v",
            f"{policy_dir}:/bot:ro",
            "-v",
            f"{seed_shim}:/seed/sitecustomize.py:ro",
            "-w",
            "/bot",
            IMAGE,
            "python",
            "main.py",
        ]
    )
    _wait_server(container)


def run_game(
    target_source: Path,
    probe_source: Path,
    server_path: Path,
    seed_shim: Path,
    seed: int,
    output: Path,
) -> None:
    target_source = target_source.resolve()
    probe_source = probe_source.resolve()
    server_path = server_path.resolve()
    seed_shim = seed_shim.resolve()
    output = output.resolve()
    token = uuid.uuid4().hex[:12]
    network = f"rb-bs-{token}"
    target_container = f"rb-bs-target-{token}"
    probe_container = f"rb-bs-probe-{token}"
    run(["docker", "network", "create", network])
    try:
        with tempfile.TemporaryDirectory(prefix="revengebench-bs-policy-") as temp:
            temp_root = Path(temp)
            target_dir = temp_root / "target"
            probe_dir = temp_root / "probe"
            _copy_policy(target_source, target_dir, server_path)
            _copy_policy(probe_source, probe_dir, server_path)
            _start_policy(target_container, network, target_dir, seed, seed_shim)
            _start_policy(probe_container, network, probe_dir, seed, seed_shim)
            output.parent.mkdir(parents=True, exist_ok=True)
            run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    network,
                    "-v",
                    f"{output.parent}:/out",
                    IMAGE,
                    "battlesnake",
                    "play",
                    "--sequential",
                    "--url",
                    f"http://{probe_container}:8000",
                    "--name",
                    "probe",
                    "--url",
                    f"http://{target_container}:8000",
                    "--name",
                    "target",
                    "--width",
                    "11",
                    "--height",
                    "11",
                    "--seed",
                    str(seed),
                    "--output",
                    f"/out/{output.name}",
                ],
                timeout=180.0,
            )
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError("BattleSnake engine produced no trajectory")
    finally:
        run(["docker", "rm", "-f", target_container, probe_container], check=False)
        run(["docker", "network", "rm", network], check=False)


def load_records(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def target_states_and_actions(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    turns = []
    for record in records:
        if not isinstance(record.get("turn"), int) or not isinstance(record.get("board"), dict):
            continue
        turns.append(copy.deepcopy(record))
    turns.sort(key=lambda item: item["turn"])
    if len({item["turn"] for item in turns}) != len(turns):
        raise ValueError("duplicate target turns")
    actions = []
    retained_states = []
    for current, following in zip(turns, turns[1:]):
        target = next(
            (snake for snake in current["board"].get("snakes", []) if snake.get("name") == "target"),
            None,
        )
        next_target = next(
            (snake for snake in following["board"].get("snakes", []) if snake.get("name") == "target"),
            None,
        )
        # Match the release parser: an eliminated target contributes no further
        # state-action pair, while the surviving probe may continue the game.
        if target is None or next_target is None:
            continue
        current["you"] = copy.deepcopy(target)
        head = target["head"]
        next_head = next_target["head"]
        action = DIRECTIONS.get((next_head["x"] - head["x"], next_head["y"] - head["y"]))
        if action is None:
            raise ValueError(f"turn {current['turn']}: invalid head displacement")
        retained_states.append(current)
        actions.append(action)
    return retained_states[:MAX_DECISIONS], actions[:MAX_DECISIONS]


def canonical_replay_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove only the runtime identity fields admitted by the replay protocol."""
    result = copy.deepcopy(states)
    for state in result:
        if isinstance(state.get("game"), dict):
            state["game"].pop("id", None)
        board = state.get("board", {})
        snakes = board.get("snakes", []) if isinstance(board, dict) else []
        for snake in snakes:
            snake.pop("id", None)
            snake.pop("latency", None)
        snakes.sort(key=lambda snake: snake.get("name", ""))
        for field in ("food", "hazards"):
            if isinstance(board.get(field), list):
                board[field].sort(key=lambda point: (point.get("x"), point.get("y")))
        if isinstance(state.get("you"), dict):
            state["you"].pop("id", None)
            state["you"].pop("latency", None)
    return result


@contextlib.contextmanager
def _policy_import_path(source: Path):
    old_path = list(sys.path)
    sys.path.insert(0, str(source))
    before = set(sys.modules)
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name in set(sys.modules) - before:
            module = sys.modules.get(name)
            module_file = getattr(module, "__file__", "") if module is not None else ""
            if module_file and str(source) in str(module_file):
                sys.modules.pop(name, None)


def candidate_actions(source: Path, states: list[dict[str, Any]], seed: int) -> list[str]:
    random.seed(seed)
    module_name = f"revengebench_bs_{uuid.uuid4().hex}"
    with _policy_import_path(source):
        spec = importlib.util.spec_from_file_location(module_name, source / "main.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        actions = []
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if states and hasattr(module, "start"):
                module.start(copy.deepcopy(states[0]))
            for state in states:
                value = module.move(copy.deepcopy(state))
                if isinstance(value, dict):
                    value = value.get("move")
                if value not in DIRECTIONS.values():
                    raise ValueError(f"{source.name}: invalid move {value!r}")
                actions.append(value)
            if states and hasattr(module, "end"):
                module.end(copy.deepcopy(states[-1]))
    return actions


def trajectory_summary(
    trajectory: Path, candidate_sources: list[Path], seed: int
) -> dict[str, Any]:
    records = load_records(trajectory)
    states, target_actions = target_states_and_actions(records)
    if not 3 <= len(states) <= MAX_DECISIONS:
        raise ValueError(f"invalid retained decision count {len(states)}")
    distances = []
    candidate_hashes = []
    for source in candidate_sources:
        actions = candidate_actions(source, states, seed)
        if len(actions) != len(target_actions):
            raise ValueError("candidate action length mismatch")
        values = [0.0 if left == right else 1.0 for left, right in zip(actions, target_actions)]
        distances.append(math.fsum(values) / len(values))
        candidate_hashes.append(sha256_json(actions))
    terminal = records[-1]
    terminal_result = {"winner_name": terminal.get("winnerName"), "is_draw": terminal.get("isDraw")}
    return {
        "decision_count": len(states),
        "state_hash": sha256_json(canonical_replay_states(states)),
        "target_action_hash": sha256_json(target_actions),
        "candidate_action_hashes": candidate_hashes,
        "candidate_mean_distances": distances,
        "terminal_hash": sha256_json(terminal_result),
    }


def evaluate(
    source_root: Path,
    manifest_path: Path,
    server_path: Path,
    seed_shim: Path,
    private_root: Path,
    *,
    hypothesis_indices: list[int] | None = None,
    probe_indices: list[int] | None = None,
    seed_indices: list[int] | None = None,
    arm_count: int = 2,
) -> dict[str, Any]:
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    arena = manifest["arenas"]["battlesnake"]
    hypotheses = arena["hypotheses"]
    probes = arena["probes"]
    seeds = arena["probe_seeds"]
    hypothesis_indices = hypothesis_indices if hypothesis_indices is not None else list(range(3))
    probe_indices = probe_indices if probe_indices is not None else list(range(3))
    seed_indices = seed_indices if seed_indices is not None else list(range(3))
    if sha256_bytes(server_path.read_bytes()) != SERVER_SHA256:
        raise ValueError("public BattleSnake server wrapper hash mismatch")
    if arm_count < 2:
        raise ValueError("arm_count must be at least two")
    image_id = run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"]).stdout.strip()
    if image_id != IMAGE_ID:
        raise ValueError("repaired BattleSnake image ID mismatch")
    candidate_sources = [source_root / "data/targets/battlesnake" / item["target_id"] for item in hypotheses]

    summaries = []
    for h_index in hypothesis_indices:
        target_source = candidate_sources[h_index]
        for q_index in probe_indices:
            probe_source = source_root / "data/targets/battlesnake" / probes[q_index]["target_id"]
            for s_index in seed_indices:
                seed = seeds[s_index]
                arms = []
                for arm in range(arm_count):
                    trajectory = private_root / f"h{h_index}_q{q_index}_s{s_index}_a{arm}.jsonl"
                    run_game(target_source, probe_source, server_path, seed_shim, seed, trajectory)
                    arms.append(trajectory_summary(trajectory, candidate_sources, seed))
                exact = all(arm == arms[0] for arm in arms[1:])
                self_distance = arms[0]["candidate_mean_distances"][h_index]
                summaries.append(
                    {
                        "hypothesis_index": h_index,
                        "probe_index": q_index,
                        "seed_index": s_index,
                        "fresh_arms_exact": exact,
                        "decision_count": arms[0]["decision_count"],
                        "self_distance": self_distance,
                        "candidate_mean_distances": arms[0]["candidate_mean_distances"],
                        "canonical_hash": sha256_json(arms[0]),
                    }
                )
                if not exact or self_distance > 1e-9:
                    raise RuntimeError(
                        f"BattleSnake replay/self gate failed h={h_index} q={q_index} s={s_index}: "
                        f"exact={exact} self_distance={self_distance}"
                    )

    complete = len(summaries) == 27
    distance_matrices = []
    if complete:
        for q_index in range(3):
            matrix = []
            for observed_h in range(3):
                rows = [
                    row["candidate_mean_distances"]
                    for row in summaries
                    if row["probe_index"] == q_index and row["hypothesis_index"] == observed_h
                ]
                matrix.append([math.fsum(row[column] for row in rows) / len(rows) for column in range(3)])
            distance_matrices.append(matrix)
    policy_reports = {}
    if complete:
        for beta in manifest["betas"]:
            likelihoods = [
                opportunity_math.likelihood_from_distances(matrix, beta) for matrix in distance_matrices
            ]
            policy_reports[str(beta)] = opportunity_math.evaluate_policies(likelihoods)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "pass" if complete else "smoke_pass",
        "arena": "battlesnake",
        "bindings": {
            "manifest_sha256": sha256_bytes(manifest_raw),
            "server_sha256": SERVER_SHA256,
            "image": IMAGE,
            "image_id": IMAGE_ID,
        },
        "runs": summaries,
        "distance_matrices": distance_matrices,
        "policy_reports": policy_reports,
        "gates": {
            "all_fresh_arms_exact": all(row["fresh_arms_exact"] for row in summaries),
            "all_self_distances_zero": all(row["self_distance"] <= 1e-9 for row in summaries),
            "decision_counts_valid": all(3 <= row["decision_count"] <= MAX_DECISIONS for row in summaries),
            "complete_frozen_cohort": complete,
        },
        "privacy": {"raw_trajectories_serialized": False, "source_content_serialized": False},
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
    }


def parse_indices(value: str) -> list[int]:
    result = [int(item) for item in value.split(",")]
    if not result or any(item not in (0, 1, 2) for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("indices must be unique members of 0,1,2")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--seed-shim", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hypothesis-indices", type=parse_indices)
    parser.add_argument("--probe-indices", type=parse_indices)
    parser.add_argument("--seed-indices", type=parse_indices)
    parser.add_argument("--arm-count", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate(
        args.source_root,
        args.manifest,
        args.server,
        args.seed_shim,
        args.private_root,
        hypothesis_indices=args.hypothesis_indices,
        probe_indices=args.probe_indices,
        seed_indices=args.seed_indices,
        arm_count=args.arm_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run_count": len(result["runs"]), "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
