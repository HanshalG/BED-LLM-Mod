#!/usr/bin/env python3
"""Run the frozen zero-call compositional NeuronBench opportunity screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.neuronbench_compose.mechanics import (
    CompositionalBank,
    CompositionalPlanner,
    MECHANISM_NAMES,
    candidate_masks,
    compare_losses,
    deterministic_random_action,
    mask_name,
    policy_divergence,
    truth_masks,
)


SCHEMA_VERSION = "neuronbench-compositional-mopen-opportunity-v1"
PROTOCOL_PATH = Path(
    "results/nonmyopic/NEURONBENCH_COMPOSITIONAL_MOPEN_OPPORTUNITY_PROTOCOL_20260815.md"
)
PROTOCOL_SHA256 = "e693587d96f5e557089d3438430cc0ee3a6406d59b69dd44187be1a589525ba5"
SOURCE_COMMIT = "c354622458c460b419cab821d482c879f0578377"
SOURCE_TREE = "4b0fba903168abcdd85db9336a37944670a990c9"
SOURCE_HASHES = {
    "neuronbench/worlds.py": "d462834c969cb5e20b103a14971e3ecc9db49cc90696af478e90d4d2f5b95d64",
    "neuronbench/stochastic.py": "20f897c97521fe30e8157ec31a227428fc4af826d94b5bb0a595aa98d0879b96",
    "neuronbench/protocols.py": "38a38610a7269236486abea159da706148cd62b52b52b175136a2a8aa29e9b04",
    "neuronbench/features.py": "7d4f161e3dd4cec1e0dd56a44c11cc6f7687a892932d3fdb43f0fb6ced801aea",
    "neuronbench/evaluator.py": "7e559789b33c1e6412e443affb3b5de65fdedc124facd302a3372d4de7849ad7",
}
EXECUTION_BUDGET = 4
OBSERVATION_SD = 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, arguments: Sequence[str]) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_protocol() -> dict[str, str]:
    actual = sha256(REPO_ROOT / PROTOCOL_PATH)
    if actual != PROTOCOL_SHA256:
        raise RuntimeError(f"protocol hash mismatch: {actual} != {PROTOCOL_SHA256}")
    return {"path": str(PROTOCOL_PATH), "sha256": actual}


def require_pushed_commit(required_commit: str) -> str:
    head = git_value(REPO_ROOT, ("rev-parse", "HEAD"))
    resolved = git_value(REPO_ROOT, ("rev-parse", required_commit))
    if head != resolved:
        raise RuntimeError(f"required implementation commit is not HEAD: {resolved} != {head}")
    remote = "origin/codex/location-finding-llmstrategy"
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", head, remote],
        check=True,
        capture_output=True,
        text=True,
    )
    return head


def verify_source(source_root: Path) -> dict[str, Any]:
    root = source_root.resolve()
    commit = git_value(root, ("rev-parse", "HEAD"))
    tree = git_value(root, ("rev-parse", "HEAD^{tree}"))
    if commit != SOURCE_COMMIT or tree != SOURCE_TREE:
        raise RuntimeError(f"NeuronBench source binding changed: {commit}/{tree}")
    files = {}
    for relative, expected in SOURCE_HASHES.items():
        actual = sha256(root / relative)
        if actual != expected:
            raise RuntimeError(f"source hash mismatch for {relative}: {actual} != {expected}")
        files[relative] = actual
    return {
        "repository": "https://github.com/murphyk/neuronbench",
        "commit": commit,
        "tree": tree,
        "files": files,
    }


def load_worlds(source_root: Path) -> Any:
    root = str(source_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    loaded = sys.modules.get("neuronbench.worlds")
    if loaded is not None:
        loaded_path = Path(loaded.__file__).resolve()
        if source_root.resolve() not in loaded_path.parents:
            raise RuntimeError("a different neuronbench package is already imported")
        return loaded
    return importlib.import_module("neuronbench.worlds")


def canonical_segments(segments: Iterable[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(duration), float(amplitude)) for duration, amplitude in segments)


def build_query_battery(worlds: Any) -> tuple[tuple[tuple[float, float], ...], ...]:
    action_segments = {canonical_segments(segments) for _, segments in worlds.POOL}
    seen: set[tuple[tuple[float, float], ...]] = set()
    queries = []
    for spec in worlds.WORLDS.values():
        for _, segments in spec["test"]:
            canonical = canonical_segments(segments)
            if canonical in action_segments or canonical in seen:
                continue
            seen.add(canonical)
            queries.append(canonical)
    if not queries:
        raise RuntimeError("held-out query battery is empty")
    return tuple(queries)


def mechanism_kwargs(worlds: Any, mask: int) -> dict[str, Any]:
    channels = (worlds.Z, worlds.IH, None, worlds.MT, worlds.ID, worlds.MC)
    extra = [channel for index, channel in enumerate(channels) if channel is not None and mask & (1 << index)]
    return {"extra": extra, "slow_na": bool(mask & (1 << 2))}


def simulate_count(worlds: Any, mask: int, segments: Sequence[Sequence[float]]) -> int:
    current, test_start = worlds.build_I(segments, dt=0.01)
    value = worlds.simulate(
        current,
        test_start,
        **mechanism_kwargs(worlds, mask),
        dt=0.01,
    )
    if not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise RuntimeError("NeuronBench returned an invalid spike count")
    return int(value)


def build_response_bank(worlds: Any) -> tuple[CompositionalBank, dict[str, Any]]:
    masks = candidate_masks()
    actions = tuple(canonical_segments(segments) for _, segments in worlds.POOL)
    action_names = tuple(str(label) for label, _ in worlds.POOL)
    queries = build_query_battery(worlds)
    action_counts = np.asarray(
        [[simulate_count(worlds, mask, segments) for segments in actions] for mask in masks],
        dtype=float,
    )
    query_counts = np.asarray(
        [[simulate_count(worlds, mask, segments) for segments in queries] for mask in masks],
        dtype=float,
    )
    bank = CompositionalBank(
        action_counts,
        query_counts,
        action_names,
        tuple(f"query_{index:03d}" for index in range(len(queries))),
        observation_sd=OBSERVATION_SD,
    )
    payload = {
        "candidate_masks": list(masks),
        "candidate_names": [mask_name(mask) for mask in masks],
        "truth_masks": list(truth_masks()),
        "action_names": list(action_names),
        "action_segments": [[list(segment) for segment in action] for action in actions],
        "query_segments": [[list(segment) for segment in query] for query in queries],
        "action_counts": action_counts.astype(int).tolist(),
        "query_counts": query_counts.astype(int).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return bank, payload


def random_dynamic_control(bank: CompositionalBank, *, seed: int = 2026081501) -> dict[str, Any]:
    losses = []
    action_rows = []
    for truth_mask in bank.truths:
        state = bank.initial_state()
        available = tuple(range(bank.num_actions))
        actions = []
        for _ in range(EXECUTION_BUDGET):
            action = deterministic_random_action(state, available, seed)
            truth_index = bank.mask_to_index[truth_mask]
            observation = int(bank.action_counts[truth_index, action])
            state = bank.transition(state, action, observation, proposal_mode="oracle")
            available = tuple(item for item in available if item != action)
            actions.append(action)
        losses.append(bank.truth_loss(state, truth_mask))
        action_rows.append(actions)
    return {
        "seed": seed,
        "expected_terminal_mse": float(np.mean(losses)),
        "truth_losses": [float(value) for value in losses],
        "truth_actions": action_rows,
    }


def apply_gate(
    bank: CompositionalBank,
    dynamic: dict[str, dict[str, Any]],
    full_support_d1: dict[str, Any],
    divergences: dict[str, dict[str, Any]],
    runtime_checks: dict[str, bool],
) -> dict[str, Any]:
    d1 = dynamic["d1"]["truth_losses"]
    d2 = dynamic["d2"]["truth_losses"]
    d3 = dynamic["d3"]["truth_losses"]
    d2_comparison = compare_losses(d1, d2)
    d3_comparison = compare_losses(d2, d3)
    d31_comparison = compare_losses(d1, d3)
    calibrated = all(
        abs(item["planned_value"] - item["expected_terminal_mse"]) <= 1e-10
        for item in dynamic.values()
    )
    conditions = {
        "runtime_checks_pass": all(runtime_checks.values()),
        "d2_mean_reduction_at_least_5pct": d2_comparison["relative_reduction"] >= 0.05,
        "d3_mean_reduction_at_least_5pct": d3_comparison["relative_reduction"] >= 0.05,
        "d2_paired_majority": d2_comparison["wins"] > d2_comparison["losses"],
        "d3_paired_majority": d3_comparison["wins"] > d3_comparison["losses"],
        "d3_beats_d1_on_at_least_12_truths": d31_comparison["wins"] >= 12,
        "d3_within_10pct_of_full_support_d1": dynamic["d3"]["expected_terminal_mse"]
        <= 1.1 * full_support_d1["expected_terminal_mse"],
        "d2_behaviorally_distinct": divergences["d2_vs_d1"]["root_changed"]
        or divergences["d2_vs_d1"]["fraction_changed"] >= 0.2,
        "d3_behaviorally_distinct": divergences["d3_vs_d2"]["root_changed"]
        or divergences["d3_vs_d2"]["fraction_changed"] >= 0.2,
        "planned_truth_replay_calibrated": calibrated,
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "d2_vs_d1": d2_comparison,
        "d3_vs_d2": d3_comparison,
        "d3_vs_d1": d31_comparison,
        "num_truths": len(bank.truths),
    }


def run(source_root: Path, implementation_commit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = verify_protocol()
    source = verify_source(source_root)
    worlds = load_worlds(source_root)
    bank, response_bank = build_response_bank(worlds)
    runtime_checks = {
        "model_calls_zero": True,
        "cost_zero": True,
        "candidate_count_22": len(bank.masks) == 22,
        "truth_count_15": len(bank.truths) == 15,
        "action_count_9": bank.num_actions == 9,
        "query_battery_nonempty": bank.query_counts.shape[1] > 0,
        "responses_finite": bool(
            np.isfinite(bank.action_counts).all() and np.isfinite(bank.query_counts).all()
        ),
        "primitive_order_exact": MECHANISM_NAMES
        == ("z_rebound", "h_sag", "na_fatigue", "ca_rebound", "d_type", "textbook_M"),
    }
    dynamic_planner = CompositionalPlanner(bank, proposal_mode="oracle")
    dynamic = {
        f"d{level}": dynamic_planner.evaluate(level, execution_budget=EXECUTION_BUDGET)
        for level in (1, 2, 3)
    }
    full_support_d1 = CompositionalPlanner(bank, proposal_mode="fixed").evaluate(
        1,
        execution_budget=EXECUTION_BUDGET,
        initial_support=bank.masks,
    )
    plain_fixed_d1 = CompositionalPlanner(bank, proposal_mode="fixed").evaluate(
        1,
        execution_budget=EXECUTION_BUDGET,
        initial_support=(0,),
    )
    history_blind_d3 = CompositionalPlanner(bank, proposal_mode="history_blind").evaluate(
        3,
        execution_budget=EXECUTION_BUDGET,
        initial_support=(0,),
    )
    divergences = {
        "d2_vs_d1": policy_divergence(
            bank, dynamic_planner, 1, dynamic_planner, 2, execution_budget=EXECUTION_BUDGET
        ),
        "d3_vs_d2": policy_divergence(
            bank, dynamic_planner, 2, dynamic_planner, 3, execution_budget=EXECUTION_BUDGET
        ),
    }
    controls = {
        "full_support_d1": full_support_d1,
        "plain_fixed_d1": plain_fixed_d1,
        "history_blind_d3": history_blind_d3,
        "random_dynamic": random_dynamic_control(bank),
    }
    gate = apply_gate(bank, dynamic, full_support_d1, divergences, runtime_checks)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gate["passed"] else "failed_closed",
        "model_calls": 0,
        "cost_usd": 0.0,
        "implementation_commit": implementation_commit,
        "protocol": protocol,
        "source": source,
        "response_bank_sha256": response_bank["sha256"],
        "mechanism_names": list(MECHANISM_NAMES),
        "num_candidates": len(bank.masks),
        "num_truths": len(bank.truths),
        "num_actions": bank.num_actions,
        "num_queries": bank.query_counts.shape[1],
        "execution_budget": EXECUTION_BUDGET,
        "observation_sd": OBSERVATION_SD,
        "runtime_checks": runtime_checks,
        "dynamic": dynamic,
        "controls": controls,
        "divergences": divergences,
        "gate": gate,
        "interpretation": (
            "Zero-call compositional horizon opportunity passed; this authorizes only the frozen small "
            "LLM edit gate."
            if gate["passed"]
            else "The exact compositional deterministic opportunity formulation is closed; no LLM edit gate opens."
        ),
    }
    return result, response_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    implementation_commit = require_pushed_commit(args.implementation_commit)
    result, response_bank = run(args.source_root, implementation_commit)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "RESULT.json"
    bank_path = args.output_dir / "RESPONSE_BANK.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    bank_path.write_text(json.dumps(response_bank, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "result": str(result_path),
        "response_bank": str(bank_path),
        "gate": result["gate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
