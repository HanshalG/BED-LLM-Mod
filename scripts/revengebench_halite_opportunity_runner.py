#!/usr/bin/env python3
"""Run the frozen RevengeBench Halite V3 opportunity cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import select
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "revengebench-halite-opportunity-v3"
IMAGE = "bed-revengebench-halite:20260813"
IMAGE_ID = "sha256:a253fa6291603798bb11c20f6c76cedeba9322bd56d6a46b43c0ea8743bd299d"
MAX_DECISIONS = 64


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def run(command: list[str], *, timeout: float = 180.0, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout={result.stdout[-1000:]}\nstderr={result.stderr[-1000:]}"
        )
    return result


def encode_productions(data: dict[str, Any]) -> str:
    return " ".join(
        str(data["productions"][y][x]) for y in range(data["height"]) for x in range(data["width"])
    )


def encode_frame(frame: list, width: int, height: int) -> str:
    rle: list[str] = []
    current = None
    count = 0
    for y in range(height):
        for x in range(width):
            owner = frame[y][x][0]
            if owner == current:
                count += 1
            else:
                if current is not None:
                    rle.extend((str(count), str(current)))
                current = owner
                count = 1
    if current is not None:
        rle.extend((str(count), str(current)))
    strengths = [str(frame[y][x][1]) for y in range(height) for x in range(width)]
    return " ".join((*rle, *strengths))


def decode_moves(line: str) -> list[list[int]]:
    tokens = line.split()
    if len(tokens) % 3:
        raise ValueError("malformed Halite move line")
    return sorted(
        [int(tokens[index + 1]), int(tokens[index]), int(tokens[index + 2])]
        for index in range(0, len(tokens), 3)
        if int(tokens[index + 2]) != 0
    )


def target_actions(data: dict[str, Any], player_tag: int) -> list[list[list[int]]]:
    actions = []
    for index, grid in enumerate(data["moves"][:MAX_DECISIONS]):
        frame = data["frames"][index]
        actions.append(
            sorted(
                [row, column, grid[row][column]]
                for row in range(data["height"])
                for column in range(data["width"])
                if frame[row][column][0] == player_tag
            )
        )
    return actions


def decision_frames(data: dict[str, Any]) -> list[list]:
    """Frames consumed by GetFrame before each recorded action."""
    return data["frames"][: min(len(data["moves"]), MAX_DECISIONS)]


def query_bot(executable: Path, data: dict[str, Any], player_tag: int) -> list[list[list[int]]]:
    width, height = data["width"], data["height"]
    frames = data["frames"]
    query_frames = decision_frames(data)
    process = subprocess.Popen(
        [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        process.stdin.write(
            f"{player_tag} {width} {height}\n{encode_productions(data)}\n"
            f"{encode_frame(frames[0], width, height)}\n"
        )
        process.stdin.flush()
        process.stdout.readline()
        result = []
        for index, decision_frame in enumerate(query_frames):
            # GetInit consumed the initialization map. The bot then calls
            # GetFrame before every action, so replay frame i must be sent here;
            # frame 0 is intentionally sent once in init and once for turn 0.
            process.stdin.write(encode_frame(decision_frame, width, height) + "\n")
            process.stdin.flush()
            if not select.select([process.stdout], [], [], 10.0)[0]:
                raise TimeoutError(f"candidate timed out at Halite turn {index}")
            moves = decode_moves(process.stdout.readline())
            moved = {(row, column) for row, column, _ in moves}
            frame = frames[index]
            for row in range(height):
                for column in range(width):
                    if frame[row][column][0] == player_tag and (row, column) not in moved:
                        moves.append([row, column, 0])
            result.append(sorted(moves))
        return result
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()


def action_distance(left: list[list[int]], right: list[list[int]]) -> float:
    left_map = {(row, column): move for row, column, move in left}
    right_map = {(row, column): move for row, column, move in right}
    cells = set(left_map) | set(right_map)
    if not cells:
        return 0.0
    return sum(left_map.get(cell) != right_map.get(cell) for cell in cells) / len(cells)


def query_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-replay", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--player-tag", type=int, required=True)
    args = parser.parse_args()
    data = json.loads(args.query_replay.read_text())
    actions = query_bot(args.executable, data, args.player_tag)
    print(json.dumps(actions, separators=(",", ":")))
    return 0


def compile_policy(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    run(
        [
            "docker", "run", "--rm", "-v", f"{source.resolve()}:/src:ro", "-v",
            f"{destination.resolve()}:/out", IMAGE, "sh", "-c",
            "cp /src/main.c /out/main.c && cp /workspace/submission/hlt.h /out/hlt.h && "
            "gcc /out/main.c -O2 -o /out/bot",
        ]
    )


def compile_shim(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(
        ["docker", "run", "--rm", "-v", f"{source.resolve()}:/src.c:ro", "-v",
         f"{destination.parent.resolve()}:/out", IMAGE, "gcc", "-shared", "-fPIC", "/src.c", "-o", f"/out/{destination.name}"]
    )


def run_game(probe: Path, target: Path, seed: int, shim: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(output_dir.glob("*.hlt"))
    run(
        [
            "docker", "run", "--rm", "-e", f"REVENGEBENCH_REPLAY_EPOCH={seed}",
            "-e", "LD_PRELOAD=/shim/replay.so", "-v", f"{probe.parent.resolve()}:/probe:ro",
            "-v", f"{target.parent.resolve()}:/target:ro", "-v", f"{shim.parent.resolve()}:/shim:ro",
            "-v", f"{output_dir.resolve()}:/out", IMAGE, "/workspace/environment/halite",
            "--replaydirectory", "/out", "--seed", str(seed), "--dimensions", "30 30", "--timeout",
            "/probe/bot", "/target/bot",
        ],
        timeout=240.0,
    )
    created = set(output_dir.glob("*.hlt")) - before
    if len(created) != 1:
        raise RuntimeError(f"expected one new Halite replay, found {len(created)}")
    return created.pop()


def query_in_container(script: Path, replay: Path, executable: Path, player_tag: int) -> list:
    result = run(
        [
            "docker", "run", "--rm", "-v", f"{script.resolve()}:/runner.py:ro",
            "-v", f"{replay.parent.resolve()}:/replay:ro", "-v", f"{executable.parent.resolve()}:/bot:ro",
            IMAGE, "python", "/runner.py", "--query-replay", f"/replay/{replay.name}",
            "--executable", "/bot/bot", "--player-tag", str(player_tag),
        ], timeout=120.0
    )
    return json.loads(result.stdout)


def evaluate(source_root: Path, manifest_path: Path, shim_source: Path, private_root: Path) -> dict[str, Any]:
    try:
        from scripts import revengebench_execution_opportunity_math as opportunity_math
    except ImportError:
        import revengebench_execution_opportunity_math as opportunity_math
    image_id = run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"]).stdout.strip()
    if image_id != IMAGE_ID:
        raise ValueError("Halite image ID mismatch")
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    arena = manifest["arenas"]["halite"]
    policy_root = private_root / "compiled"
    hypothesis_bins, probe_bins = [], []
    for role, items, result in (("h", arena["hypotheses"], hypothesis_bins), ("q", arena["probes"], probe_bins)):
        for index, item in enumerate(items):
            out = policy_root / f"{role}{index}"
            compile_policy(source_root / "data/targets/halite" / item["target_id"], out)
            result.append(out / "bot")
    shim = private_root / "shim" / "replay.so"
    compile_shim(shim_source, shim)
    script = Path(__file__)
    rows = []
    for h_index, target in enumerate(hypothesis_bins):
        for q_index, probe in enumerate(probe_bins):
            for s_index, seed in enumerate(arena["probe_seeds"]):
                arms = []
                for arm in range(2):
                    replay = run_game(probe, target, seed, shim, private_root / "replays" / f"h{h_index}q{q_index}s{s_index}a{arm}")
                    data = json.loads(replay.read_text())
                    if len(data.get("player_names", [])) != 2:
                        raise RuntimeError("Halite replay missing two player names")
                    tag = 2
                    truth = target_actions(data, tag)
                    candidate_actions = [query_in_container(script, replay, executable, tag) for executable in hypothesis_bins]
                    distances = [math.fsum(action_distance(a, b) for a, b in zip(actions, truth)) / len(truth) for actions in candidate_actions]
                    arms.append({
                        "replay_hash": sha256_json(data), "decision_count": len(truth),
                        "target_action_hash": sha256_json(truth),
                        "candidate_action_hashes": [sha256_json(item) for item in candidate_actions],
                        "candidate_mean_distances": distances,
                    })
                exact = arms[0] == arms[1]
                self_distance = arms[0]["candidate_mean_distances"][h_index]
                if not exact or self_distance > 1e-9 or not 3 <= arms[0]["decision_count"] <= MAX_DECISIONS:
                    raise RuntimeError(f"Halite gate failed h={h_index} q={q_index} s={s_index}: exact={exact} self={self_distance}")
                rows.append({"hypothesis_index": h_index, "probe_index": q_index, "seed_index": s_index,
                             "fresh_arms_exact": exact, "decision_count": arms[0]["decision_count"],
                             "self_distance": self_distance, "candidate_mean_distances": arms[0]["candidate_mean_distances"],
                             "canonical_hash": sha256_json(arms[0])})
    matrices = []
    for q_index in range(3):
        matrix = []
        for observed in range(3):
            samples = [row["candidate_mean_distances"] for row in rows if row["probe_index"] == q_index and row["hypothesis_index"] == observed]
            matrix.append([math.fsum(sample[column] for sample in samples) / len(samples) for column in range(3)])
        matrices.append(matrix)
    reports = {str(beta): opportunity_math.evaluate_policies([opportunity_math.likelihood_from_distances(matrix, beta) for matrix in matrices]) for beta in manifest["betas"]}
    return {"protocol_version": PROTOCOL_VERSION, "status": "pass", "arena": "halite",
            "bindings": {"manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(), "image": IMAGE, "image_id": IMAGE_ID},
            "runs": rows, "distance_matrices": matrices, "policy_reports": reports,
            "gates": {"all_fresh_arms_exact": True, "all_self_distances_zero": True,
                      "decision_counts_valid": True, "complete_frozen_cohort": len(rows) == 27},
            "privacy": {"raw_trajectories_serialized": False, "source_content_serialized": False},
            "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shim-source", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    if "--query-replay" in sys.argv:
        return query_main()
    args = parse_args()
    result = evaluate(args.source_root, args.manifest, args.shim_source, args.private_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_count": len(result["runs"]), "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
