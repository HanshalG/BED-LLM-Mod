from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


EXPECTED_COMMIT = "351a5a7c2671150bae44c8bc46d7115ec996615f"
EXPECTED_TREE = "8991d42d09f3f8fb095580d87a68ba21b7dc6f5c"
EXPECTED_CODECLASH_COMMIT = "a66d63eee9a1f4f5bcda7ca753c404dcfdb63e92"
SPLIT_SALT = "revengebench-bed-20260813:"
ARENA_ENTRYPOINTS = {
    "battlesnake": "main.py",
    "halite": "main.c",
    "huskybench": "player.py",
    "robocode": "MyTank.java",
    "robotrumble": "robot.js",
}
SPLIT_SIZES = (("mechanics", 1), ("opportunity", 3), ("development", 4), ("confirmation", 4))
ALLOWED_FILES = (
    Path("README.md"),
    Path("LICENSE"),
    Path("pyproject.toml"),
    Path(".gitmodules"),
    Path("src/revenge_bench/arenas/arena.py"),
    Path("src/revenge_bench/arenas/battlesnake/battlesnake.py"),
    Path("src/revenge_bench/tournaments/inverse_strategy.py"),
    Path("src/revenge_bench/tournaments/inverse_strategy_interventionist.py"),
    Path("src/revenge_bench/tournaments/bayesian_program_inference.py"),
    Path("src/revenge_bench/strategy_pool.py"),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=source_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def split_target_ids(arena: str, target_ids: list[str]) -> dict[str, list[str]]:
    ordered = sorted(
        target_ids,
        key=lambda target_id: (
            sha256_bytes(f"{SPLIT_SALT}{arena}:{target_id}".encode("utf-8")),
            target_id,
        ),
    )
    splits: dict[str, list[str]] = {}
    offset = 0
    for name, size in SPLIT_SIZES:
        splits[name] = ordered[offset : offset + size]
        offset += size
    splits["reserve"] = ordered[offset:]
    return splits


def inventory_targets(source_root: Path) -> dict[str, Any]:
    root = source_root / "data" / "targets"
    arenas: dict[str, Any] = {}
    for arena, entrypoint in ARENA_ENTRYPOINTS.items():
        arena_root = root / arena
        target_dirs = sorted(path for path in arena_root.iterdir() if path.is_dir())
        target_ids = [path.name for path in target_dirs]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError(f"duplicate target IDs in {arena}")
        missing_entrypoints = [path.name for path in target_dirs if not (path / entrypoint).is_file()]
        splits = split_target_ids(arena, target_ids)
        arenas[arena] = {
            "entrypoint": entrypoint,
            "target_count": len(target_ids),
            "all_entrypoints_present": not missing_entrypoints,
            "missing_entrypoint_ids": missing_entrypoints,
            "ordered_target_id_sha256": sha256_bytes(canonical_json(target_ids).encode("ascii")),
            "splits": splits,
            "split_ordered_id_sha256": {
                name: sha256_bytes(canonical_json(ids).encode("ascii"))
                for name, ids in splits.items()
            },
        }
    return {
        "arenas": arenas,
        "arena_count": len(arenas),
        "target_count": sum(value["target_count"] for value in arenas.values()),
        "split_counts": {
            name: sum(len(value["splits"][name]) for value in arenas.values())
            for name in ("mechanics", "opportunity", "development", "confirmation", "reserve")
        },
    }


def normalized_condition_config(text: str) -> str:
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^name:\s", stripped):
            continue
        if re.match(r"^max_probes_per_round:\s", stripped):
            continue
        if stripped.startswith("<<: !include prompts/game/"):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept) + "\n"


def condition_parity(source_root: Path) -> dict[str, Any]:
    active_root = source_root / "configs" / "benchmark"
    no_probe_root = source_root / "configs" / "conditions" / "no_probe"
    common_models = sorted(
        path.name for path in active_root.iterdir() if path.is_dir() and (no_probe_root / path.name).is_dir()
    )
    comparisons = []
    for model in common_models:
        for arena in ARENA_ENTRYPOINTS:
            active_matches = sorted((active_root / model).glob(f"*_{arena}.yaml"))
            no_probe_matches = sorted((no_probe_root / model).glob(f"*_{arena}.yaml"))
            if len(active_matches) != 1 or len(no_probe_matches) != 1:
                comparisons.append({"model": model, "arena": arena, "matched": False, "reason": "config_count"})
                continue
            active = normalized_condition_config(active_matches[0].read_text(encoding="utf-8"))
            no_probe = normalized_condition_config(no_probe_matches[0].read_text(encoding="utf-8"))
            comparisons.append(
                {
                    "model": model,
                    "arena": arena,
                    "matched": active == no_probe,
                    "active_normalized_sha256": sha256_bytes(active.encode("utf-8")),
                    "no_probe_normalized_sha256": sha256_bytes(no_probe.encode("utf-8")),
                }
            )
    return {
        "common_models": common_models,
        "comparison_count": len(comparisons),
        "all_matched_modulo_probe_fields": bool(comparisons) and all(item["matched"] for item in comparisons),
        "comparisons": comparisons,
    }


def source_contracts(source_root: Path) -> dict[str, Any]:
    readme = (source_root / "README.md").read_text(encoding="utf-8")
    license_text = (source_root / "LICENSE").read_text(encoding="utf-8")
    inverse = (source_root / "src/revenge_bench/tournaments/inverse_strategy.py").read_text(encoding="utf-8")
    intervention = (
        source_root / "src/revenge_bench/tournaments/inverse_strategy_interventionist.py"
    ).read_text(encoding="utf-8")
    arena = (source_root / "src/revenge_bench/arenas/arena.py").read_text(encoding="utf-8")
    battlesnake = (
        source_root / "src/revenge_bench/arenas/battlesnake/battlesnake.py"
    ).read_text(encoding="utf-8")
    bpi_configs = list((source_root / "configs/baselines/bpi").glob("*/*.yaml"))
    return {
        "mit_license": "MIT License" in license_text and "Permission is hereby granted" in license_text,
        "probe_is_normal_gameplay_opponent": (
            re.search(r"probe opponents\*?: runnable policies", readme, re.IGNORECASE) is not None
            and re.search(
                r"no\s+privileged\s+access\s+to\s+target\s+source\s+or\s+internal\s+state",
                readme,
                re.IGNORECASE,
            )
            is not None
            and "_execute_probe" in intervention
        ),
        "heldout_executable_action_distance_endpoint": (
            "offline evaluation" in inverse.lower()
            and "mean action distance" in inverse.lower()
            and "actions_distance(learner_action, target_action)" in inverse
        ),
        "learner_and_target_are_separate_agents": (
            "Learner does NOT play in simulations" in inverse
            and "self.target_agent" in inverse
            and "self.learner_agent" in inverse
        ),
        "bpi_baseline_config_count": len(bpi_configs),
        "bpi_baseline_released": len(bpi_configs) >= 5,
        "unseeded_agent_order_shuffle": "random.shuffle(agents)" in arena,
        "unseeded_battlesnake_player_shuffle": "random.shuffle(players)" in battlesnake,
        "battlesnake_command_has_explicit_seed": "--seed" in battlesnake,
    }


def source_binding(source_root: Path) -> dict[str, Any]:
    submodule_status = git_output(source_root, "submodule", "status")
    return {
        "repository": "https://github.com/bethgelab/revenge-bench",
        "commit": git_output(source_root, "rev-parse", "HEAD"),
        "tree": git_output(source_root, "rev-parse", "HEAD^{tree}"),
        "working_tree_clean": git_output(source_root, "status", "--short") == "",
        "codeclash_commit": git_output(source_root / "vendor" / "codeclash", "rev-parse", "HEAD"),
        "submodule_status": submodule_status,
        "submodule_clean": not submodule_status.startswith(("-", "+", "U")),
    }


def audit(source_root: Path, protocol_path: Path) -> dict[str, Any]:
    binding = source_binding(source_root)
    missing = [str(path) for path in ALLOWED_FILES if not (source_root / path).is_file()]
    if missing:
        raise FileNotFoundError("missing expected RevengeBench files: " + ", ".join(missing))
    if binding["commit"] != EXPECTED_COMMIT or binding["tree"] != EXPECTED_TREE:
        raise ValueError("RevengeBench source binding mismatch")
    if binding["codeclash_commit"] != EXPECTED_CODECLASH_COMMIT:
        raise ValueError("CodeClash submodule binding mismatch")
    if not binding["working_tree_clean"] or not binding["submodule_clean"]:
        raise ValueError("RevengeBench checkout or submodule is dirty")

    targets = inventory_targets(source_root)
    parity = condition_parity(source_root)
    contracts = source_contracts(source_root)
    split_counts = targets["split_counts"]
    exact_replay_exposed = not (
        contracts["unseeded_agent_order_shuffle"]
        or contracts["unseeded_battlesnake_player_shuffle"]
        or not contracts["battlesnake_command_has_explicit_seed"]
    )
    gates = {
        "immutable_clean_source_and_submodule": True,
        "research_permissive_license": contracts["mit_license"],
        "at_least_four_arenas_sixty_targets_ten_each": (
            targets["arena_count"] >= 4
            and targets["target_count"] >= 60
            and all(value["target_count"] >= 10 for value in targets["arenas"].values())
        ),
        "unique_targets_and_entrypoints": all(
            value["all_entrypoints_present"] for value in targets["arenas"].values()
        ),
        "active_no_probe_condition_parity": parity["all_matched_modulo_probe_fields"],
        "probe_is_ordinary_executable_opponent": contracts["probe_is_normal_gameplay_opponent"],
        "external_executable_action_distance_endpoint": contracts[
            "heldout_executable_action_distance_endpoint"
        ],
        "exact_common_random_number_replay_exposed": exact_replay_exposed,
        "target_and_policy_visibility_separable": contracts["learner_and_target_are_separate_agents"],
        "frozen_split_sufficient": (
            split_counts["mechanics"] == len(ARENA_ENTRYPOINTS)
            and split_counts["opportunity"] == 3 * len(ARENA_ENTRYPOINTS)
            and split_counts["development"] >= 20
            and split_counts["confirmation"] >= 20
            and split_counts["reserve"] > 0
        ),
        "released_bpi_control": contracts["bpi_baseline_released"],
    }
    non_replay_gates = {key: value for key, value in gates.items() if key != "exact_common_random_number_replay_exposed"}
    all_non_replay = all(non_replay_gates.values())
    if all(gates.values()):
        status = "source_gate_passed"
        decision = "continue_to_structural_opportunity_audit"
    elif all_non_replay and not exact_replay_exposed:
        status = "pending_replay"
        decision = "run_zero_call_deterministic_replay_audit_only"
    else:
        status = "source_gate_failed"
        decision = "gate_failed_before_target_content"

    return {
        "schema_version": 1,
        "status": status,
        "decision": decision,
        "source": binding,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "split_salt": SPLIT_SALT,
        },
        "allowed_file_hashes": {str(path): sha256_file(source_root / path) for path in ALLOWED_FILES},
        "targets": targets,
        "condition_parity": parity,
        "contracts": contracts,
        "gates": {**gates, "all_non_replay_source_gates_pass": all_non_replay, "all_source_gates_pass": all(gates.values())},
        "privacy": {
            "target_policy_contents_opened": False,
            "target_provenance_contents_opened": False,
            "released_messages_opened": False,
            "released_simulations_opened": False,
            "released_tournaments_opened": False,
            "released_per_target_scores_opened": False,
            "development_or_confirmation_observations_opened": False,
            "endpoint_actions_opened": False,
            "serialized_target_fields": ["arena", "target_id", "entrypoint_exists"],
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "cluster_use": 0},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/nonmyopic/REVENGEBENCH_SOURCE_ADMISSION_PROTOCOL_20260813.md"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nonmyopic/revengebench_source_admission/AUDIT.json"),
    )
    args = parser.parse_args()
    result = audit(args.source_root.resolve(), args.protocol.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json({"status": result["status"], "decision": result["decision"]}))
    return 0 if result["status"] in {"source_gate_passed", "pending_replay"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
