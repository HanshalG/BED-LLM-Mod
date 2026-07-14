#!/usr/bin/env python3
"""Reanalyze the banked animals depth sweep with trial-level uncertainty."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


NUM_ROUNDS = 20
Q_THRESHOLD = 0.80
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
DEFAULT_BOOTSTRAP_SEED = 1304

CANONICAL_RUNS: dict[str, tuple[int, ...]] = {
    "naive_nonthinking": (83349,),
    "naive_thinking": (83389,),
    "eig_depth_1": (83419,),
    "eig_depth_2": (83484, 83485, 83486, 83487),
    "eig_depth_3": (85570, 85571, 85572, 85573),
}

TRACE_RE = re.compile(r"^Running accuracy trace:\s*(\[[^\n]*\])\s*$", re.MULTILINE)
TARGET_RE = re.compile(
    r"^\[categorical\] Sampled answerer target sequence from prior:\s*(\[[^\n]*\])\s*$",
    re.MULTILINE,
)
PRIOR_RE = re.compile(
    r"^\[categorical\] Randomized answerer prior summaries by trial:\s*(\[[^\n]*\])\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class RunBlock:
    job_id: int
    seed: int
    source_path: Path
    source_sha256: str
    source_mtime_utc: str
    config: dict[str, Any]
    targets: tuple[str, ...]
    prior_summaries: tuple[str, ...]
    trials: tuple[tuple[int, ...], ...]
    aggregate_curve: tuple[float, ...]


@dataclass(frozen=True)
class Arm:
    name: str
    blocks: tuple[RunBlock, ...]
    targets: tuple[str, ...]
    prior_summaries: tuple[str, ...]
    trials: tuple[tuple[int, ...], ...]


def extract_embedded_config(text: str) -> dict[str, Any]:
    """Decode the JSON config printed near the beginning of a run log."""
    marker = "Config parameters:"
    marker_index = text.find(marker)
    if marker_index < 0:
        raise ValueError("Log does not contain an embedded config")
    object_index = text.find("{", marker_index + len(marker))
    if object_index < 0:
        raise ValueError("Embedded config marker is not followed by a JSON object")
    config, _ = json.JSONDecoder().raw_decode(text[object_index:])
    if not isinstance(config, dict):
        raise ValueError("Embedded config is not a JSON object")
    return config


def recover_trial_accuracy(
    cumulative_traces: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-7,
) -> tuple[tuple[int, ...], ...]:
    """Invert cumulative per-trial means into binary per-trial outcomes."""
    if not cumulative_traces:
        raise ValueError("No cumulative accuracy traces found")
    rounds = len(cumulative_traces[0])
    if rounds == 0 or any(len(trace) != rounds for trace in cumulative_traces):
        raise ValueError("Cumulative traces must have one common, non-zero length")

    previous = [0.0] * rounds
    recovered: list[tuple[int, ...]] = []
    for trial_index, trace in enumerate(cumulative_traces, start=1):
        row: list[int] = []
        for round_index, current_mean in enumerate(trace):
            value = trial_index * float(current_mean) - (trial_index - 1) * previous[round_index]
            rounded = int(round(value))
            if rounded not in (0, 1) or abs(value - rounded) > tolerance:
                raise ValueError(
                    "Could not recover a binary outcome at "
                    f"trial {trial_index}, round {round_index + 1}: {value}"
                )
            row.append(rounded)
        recovered.append(tuple(row))
        previous = [float(value) for value in trace]

    reconstructed = mean_curve(recovered)
    if any(
        abs(reconstructed[index] - float(cumulative_traces[-1][index])) > tolerance
        for index in range(rounds)
    ):
        raise ValueError("Recovered trials do not reconstruct the final aggregate trace")
    return tuple(recovered)


def mean_curve(trials: Sequence[Sequence[int]]) -> tuple[float, ...]:
    if not trials:
        raise ValueError("At least one trial is required")
    rounds = len(trials[0])
    if rounds == 0 or any(len(trial) != rounds for trial in trials):
        raise ValueError("Trials must have one common, non-zero length")
    return tuple(sum(trial[index] for trial in trials) / len(trials) for index in range(rounds))


def accuracy_auc(trials: Sequence[Sequence[int]]) -> float:
    curve = mean_curve(trials)
    return sum(curve) / len(curve)


def q_at_threshold(
    curve: Sequence[float],
    *,
    threshold: float = Q_THRESHOLD,
    tolerance: float = 1e-12,
) -> int:
    """Return the first one-indexed threshold crossing, censoring at rounds + 1."""
    for index, value in enumerate(curve, start=1):
        if float(value) + tolerance >= threshold:
            return index
    return len(curve) + 1


def _parse_literal_list(pattern: re.Pattern[str], text: str, label: str) -> list[Any]:
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Log does not contain {label}")
    value = ast.literal_eval(match.group(1))
    if not isinstance(value, list):
        raise ValueError(f"{label} is not a list")
    return value


def load_run_block(log_path: Path, *, config_path: Path | None = None) -> RunBlock:
    text = log_path.read_text(errors="replace")
    embedded_config = extract_embedded_config(text)
    if config_path is not None:
        resolved_config = json.loads(config_path.read_text())
        for key in (
            "answerer_prior_seed",
            "answerer_num_prior_trials",
            "search_depth",
            "method_names",
        ):
            if embedded_config.get(key) != resolved_config.get(key):
                raise ValueError(f"Embedded and resolved configs disagree on {key}: {log_path}")
        config = resolved_config
    else:
        config = embedded_config

    traces = [ast.literal_eval(match) for match in TRACE_RE.findall(text)]
    trials = recover_trial_accuracy(traces)
    targets = tuple(str(item) for item in _parse_literal_list(TARGET_RE, text, "targets"))
    priors = tuple(str(item) for item in _parse_literal_list(PRIOR_RE, text, "prior summaries"))
    expected_trials = int(config["answerer_num_prior_trials"])
    if len(trials) != expected_trials:
        raise ValueError(
            f"Expected {expected_trials} trials but found {len(trials)} in {log_path}"
        )
    if len(targets) != expected_trials or len(priors) != expected_trials:
        raise ValueError(f"Target/prior metadata length mismatch in {log_path}")
    if any(len(trial) != NUM_ROUNDS for trial in trials):
        raise ValueError(f"Expected {NUM_ROUNDS} rounds in {log_path}")

    stat = log_path.stat()
    return RunBlock(
        job_id=int(str(config.get("run_id") or log_path.name).split("_")[0]),
        seed=int(config["answerer_prior_seed"]),
        source_path=log_path,
        source_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
        source_mtime_utc=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        config=config,
        targets=targets,
        prior_summaries=priors,
        trials=trials,
        aggregate_curve=mean_curve(trials),
    )


def combine_arm(name: str, blocks: Iterable[RunBlock]) -> Arm:
    ordered = tuple(sorted(blocks, key=lambda block: (block.seed, block.job_id)))
    if not ordered:
        raise ValueError(f"Arm {name} has no run blocks")
    return Arm(
        name=name,
        blocks=ordered,
        targets=tuple(target for block in ordered for target in block.targets),
        prior_summaries=tuple(prior for block in ordered for prior in block.prior_summaries),
        trials=tuple(trial for block in ordered for trial in block.trials),
    )


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one match for {root / pattern}, found {len(matches)}")
    return matches[0]


def load_canonical_arms(old_log_dir: Path, depth3_dir: Path) -> dict[str, Arm]:
    arms: dict[str, Arm] = {}
    for name, job_ids in CANONICAL_RUNS.items():
        blocks: list[RunBlock] = []
        for job_id in job_ids:
            if name == "eig_depth_3":
                run_dir = _find_one(depth3_dir, f"{job_id}_config*")
                blocks.append(
                    load_run_block(
                        run_dir / "run.log",
                        config_path=run_dir / "config.resolved.json",
                    )
                )
            else:
                blocks.append(load_run_block(_find_one(old_log_dir, f"{job_id}_*.log")))
        arms[name] = combine_arm(name, blocks)
    return arms


def _stable_seed(seed: int, label: str) -> int:
    suffix = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return seed + suffix


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot take a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _discrete_percentile(values: Sequence[int], probability: float) -> int:
    if not values:
        raise ValueError("Cannot take a percentile of an empty sequence")
    ordered = sorted(int(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def bootstrap_arm(
    trials: Sequence[Sequence[int]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    rng = random.Random(seed)
    n = len(trials)
    rounds = len(trials[0])
    trial_aucs = [sum(trial) / rounds for trial in trials]
    auc_samples: list[float] = []
    q_samples: list[int] = []
    for _ in range(replicates):
        indices = [rng.randrange(n) for _ in range(n)]
        auc_samples.append(sum(trial_aucs[index] for index in indices) / n)
        curve = [
            sum(trials[index][round_index] for index in indices) / n
            for round_index in range(rounds)
        ]
        q_samples.append(q_at_threshold(curve))
    curve = mean_curve(trials)
    return {
        "num_trials": n,
        "mean_accuracy_curve": list(curve),
        "accuracy_auc": accuracy_auc(trials),
        "accuracy_auc_ci95": [
            _percentile(auc_samples, 0.025),
            _percentile(auc_samples, 0.975),
        ],
        "q80": q_at_threshold(curve),
        "q80_ci95": [
            _discrete_percentile(q_samples, 0.025),
            _discrete_percentile(q_samples, 0.975),
        ],
        "round20_accuracy": curve[-1],
        "per_trial_accuracy_auc": trial_aucs,
    }


def _resampled_metrics(
    trials: Sequence[Sequence[int]], indices: Sequence[int]
) -> tuple[float, int]:
    selected = [trials[index] for index in indices]
    curve = mean_curve(selected)
    return sum(curve) / len(curve), q_at_threshold(curve)


def bootstrap_comparison(
    better_trials: Sequence[Sequence[int]],
    baseline_trials: Sequence[Sequence[int]],
    *,
    paired: bool,
    replicates: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    if paired and len(better_trials) != len(baseline_trials):
        raise ValueError("Paired comparisons require equal trial counts")
    rng = random.Random(_stable_seed(seed, label))
    better_n = len(better_trials)
    baseline_n = len(baseline_trials)
    auc_deltas: list[float] = []
    q80_deltas: list[float] = []
    for _ in range(replicates):
        better_indices = [rng.randrange(better_n) for _ in range(better_n)]
        baseline_indices = (
            better_indices
            if paired
            else [rng.randrange(baseline_n) for _ in range(baseline_n)]
        )
        better_auc, better_q80 = _resampled_metrics(better_trials, better_indices)
        baseline_auc, baseline_q80 = _resampled_metrics(
            baseline_trials, baseline_indices
        )
        auc_deltas.append(better_auc - baseline_auc)
        q80_deltas.append(float(better_q80 - baseline_q80))

    better_auc = accuracy_auc(better_trials)
    baseline_auc = accuracy_auc(baseline_trials)
    better_q80 = q_at_threshold(mean_curve(better_trials))
    baseline_q80 = q_at_threshold(mean_curve(baseline_trials))
    result: dict[str, Any] = {
        "label": label,
        "paired": paired,
        "better_num_trials": better_n,
        "baseline_num_trials": baseline_n,
        "accuracy_auc_delta": better_auc - baseline_auc,
        "accuracy_auc_delta_ci95": [
            _percentile(auc_deltas, 0.025),
            _percentile(auc_deltas, 0.975),
        ],
        "q80_delta": better_q80 - baseline_q80,
        "q80_delta_ci95": [
            _percentile(q80_deltas, 0.025),
            _percentile(q80_deltas, 0.975),
        ],
    }
    if paired:
        better_trial_aucs = [sum(trial) / len(trial) for trial in better_trials]
        baseline_trial_aucs = [sum(trial) / len(trial) for trial in baseline_trials]
        differences = [
            better - baseline
            for better, baseline in zip(better_trial_aucs, baseline_trial_aucs)
        ]
        result.update(
            {
                "wins": sum(value > 1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
                "losses": sum(value < -1e-12 for value in differences),
                "per_trial_accuracy_auc_deltas": differences,
            }
        )
    return result


def _pairing_matches(first: Arm, second: Arm) -> bool:
    return (
        first.targets == second.targets
        and first.prior_summaries == second.prior_summaries
    )


def _scientific_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    pair = config["model_pairs"][0]
    return {
        "animals": config["animals"],
        "questioner_model": pair["questioner"]["model"],
        "questioner_thinking": pair["questioner"].get("thinking"),
        "answerer_model": pair["answerer"]["model"],
        "answerer_thinking": pair["answerer"].get("thinking"),
        "belief_state_mode": config.get("belief_state_mode"),
        "belief_prior_mode": config.get("belief_prior_mode"),
        "belief_prior_exponential_rate": config.get("belief_prior_exponential_rate"),
        "belief_generation_enabled": config.get("belief_generation_enabled"),
        "belief_filtering_enabled": config.get("belief_filtering_enabled"),
        "belief_guess_threshold": config.get("belief_guess_threshold"),
        "num_mc_samples": config.get("num_mc_samples"),
        "target_num_questions": config.get("target_num_questions"),
        "generation_temperature_diverse": config.get("generation_temperature_diverse"),
        "generation_temperature_simple": config.get("generation_temperature_simple"),
        "answer_temperature": config.get("answer_temperature"),
        "answerer_sample_from_prior": config.get("answerer_sample_from_prior"),
        "answerer_randomize_prior_order_per_trial": config.get(
            "answerer_randomize_prior_order_per_trial"
        ),
    }


def _source_record(block: RunBlock, base_dir: Path) -> dict[str, Any]:
    try:
        display_path = str(block.source_path.relative_to(base_dir))
    except ValueError:
        display_path = str(block.source_path)
    return {
        "job_id": block.job_id,
        "seed": block.seed,
        "num_trials": len(block.trials),
        "path": display_path,
        "sha256": block.source_sha256,
        "mtime_utc": block.source_mtime_utc,
        "method_names": block.config.get("method_names"),
        "search_depth": block.config.get("search_depth"),
        "batched_block_size": block.config.get("batched_block_size"),
    }


def analyze(
    old_log_dir: Path,
    depth3_dir: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    arms = load_canonical_arms(old_log_dir, depth3_dir)
    base_dir = base_dir or Path.cwd()

    for name in ("naive_nonthinking", "naive_thinking", "eig_depth_1"):
        if len(arms[name].trials) != 40:
            raise ValueError(f"{name} must contain 40 trials")
    for name in ("eig_depth_2", "eig_depth_3"):
        if [block.seed for block in arms[name].blocks] != [12345, 12346, 12347, 12348]:
            raise ValueError(f"{name} does not contain the expected seed blocks")
        if any(len(block.trials) != 10 for block in arms[name].blocks):
            raise ValueError(f"{name} blocks must contain ten trials each")

    expected_depths = {"eig_depth_1": 1, "eig_depth_2": 2, "eig_depth_3": 3}
    for name, expected_depth in expected_depths.items():
        if any(block.config.get("search_depth") != expected_depth for block in arms[name].blocks):
            raise ValueError(f"{name} has an unexpected search depth")

    eig_fingerprints = {
        name: _scientific_fingerprint(arms[name].blocks[0].config)
        for name in expected_depths
    }
    scientific_match = len(
        {json.dumps(value, sort_keys=True) for value in eig_fingerprints.values()}
    ) == 1
    if not scientific_match:
        raise ValueError("Depth-1/2/3 scientific configurations are not matched")

    pairings = {
        "naive_nonthinking_vs_naive_thinking_all40": _pairing_matches(
            arms["naive_nonthinking"], arms["naive_thinking"]
        ),
        "eig_depth_1_vs_naive_nonthinking_all40": _pairing_matches(
            arms["eig_depth_1"], arms["naive_nonthinking"]
        ),
        "eig_depth_1_vs_naive_thinking_all40": _pairing_matches(
            arms["eig_depth_1"], arms["naive_thinking"]
        ),
        "eig_depth_2_vs_eig_depth_3_all40": _pairing_matches(
            arms["eig_depth_2"], arms["eig_depth_3"]
        ),
        "eig_depth_1_first10_vs_eig_depth_2_seed12345": (
            arms["eig_depth_1"].targets[:10] == arms["eig_depth_2"].targets[:10]
            and arms["eig_depth_1"].prior_summaries[:10]
            == arms["eig_depth_2"].prior_summaries[:10]
        ),
        "eig_depth_1_first10_vs_eig_depth_3_seed12345": (
            arms["eig_depth_1"].targets[:10] == arms["eig_depth_3"].targets[:10]
            and arms["eig_depth_1"].prior_summaries[:10]
            == arms["eig_depth_3"].prior_summaries[:10]
        ),
    }
    if not all(pairings.values()):
        raise ValueError("One or more expected target/prior pairings failed")

    arm_summaries = {
        name: bootstrap_arm(
            arm.trials,
            replicates=bootstrap_replicates,
            seed=_stable_seed(bootstrap_seed, f"arm:{name}"),
        )
        for name, arm in arms.items()
    }

    d1_first10 = arms["eig_depth_1"].trials[:10]
    d2_first10 = arms["eig_depth_2"].blocks[0].trials
    d3_first10 = arms["eig_depth_3"].blocks[0].trials
    comparisons = {
        "eig_depth_1_vs_naive_nonthinking_paired40": bootstrap_comparison(
            arms["eig_depth_1"].trials,
            arms["naive_nonthinking"].trials,
            paired=True,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            label="depth 1 EIG - naive non-thinking (paired 40)",
        ),
        "eig_depth_1_vs_naive_thinking_paired40": bootstrap_comparison(
            arms["eig_depth_1"].trials,
            arms["naive_thinking"].trials,
            paired=True,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            label="depth 1 EIG - naive thinking (paired 40)",
        ),
        "eig_depth_2_vs_eig_depth_1_unpaired40": bootstrap_comparison(
            arms["eig_depth_2"].trials,
            arms["eig_depth_1"].trials,
            paired=False,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            label="depth 2 EIG - depth 1 EIG (unpaired 40)",
        ),
        "eig_depth_2_vs_eig_depth_1_paired_seed12345": bootstrap_comparison(
            d2_first10,
            d1_first10,
            paired=True,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            label="depth 2 EIG - depth 1 EIG (paired seed 12345, n=10)",
        ),
        "eig_depth_3_vs_eig_depth_2_paired40": bootstrap_comparison(
            arms["eig_depth_3"].trials,
            arms["eig_depth_2"].trials,
            paired=True,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            label="depth 3 EIG - depth 2 EIG (paired 40)",
        ),
        "eig_depth_3_vs_eig_depth_1_paired_seed12345": bootstrap_comparison(
            d3_first10,
            d1_first10,
            paired=True,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            label="depth 3 EIG - depth 1 EIG (paired seed 12345, n=10)",
        ),
    }

    depth2_blocks = {
        str(block.seed): {
            "num_trials": len(block.trials),
            "accuracy_auc": accuracy_auc(block.trials),
            "q80": q_at_threshold(mean_curve(block.trials)),
        }
        for block in arms["eig_depth_2"].blocks
    }
    depth3_blocks = {
        str(block.seed): {
            "num_trials": len(block.trials),
            "accuracy_auc": accuracy_auc(block.trials),
            "q80": q_at_threshold(mean_curve(block.trials)),
        }
        for block in arms["eig_depth_3"].blocks
    }

    direct_depth2 = comparisons["eig_depth_2_vs_eig_depth_1_paired_seed12345"]
    depth3_vs_depth2 = comparisons["eig_depth_3_vs_eig_depth_2_paired40"]
    status = (
        "holds"
        if direct_depth2["accuracy_auc_delta"] > 0.0
        and depth3_vs_depth2["accuracy_auc_delta"] >= 0.0
        else "collapses"
    )

    return {
        "analysis": "banked animals non-myopic depth reanalysis",
        "status": status,
        "decision": {
            "track1_non_myopic_spine_supported": status == "holds",
            "paper_implication": (
                "Animals can carry the non-myopic claim."
                if status == "holds"
                else "Animals supports one-step BED transfer, but not a defensible "
                "non-myopic gain; MediQ must carry the non-myopic claim."
            ),
            "reason": (
                "The only directly paired depth-2 versus depth-1 block reverses the "
                "small full-sample unpaired gain, and depth 3 is materially worse than depth 2."
            ),
        },
        "metric_definitions": {
            "accuracy_auc": "mean exact-guess accuracy over questions 1-20 (range 0-1)",
            "q80": "first question with mean exact-guess accuracy >= 0.80",
            "q80_censor_value": NUM_ROUNDS + 1,
            "q80_censor_display": f">{NUM_ROUNDS}",
            "bootstrap": {
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "interval": "95% percentile",
                "paired_resampling": "shared trial indices for paired comparisons",
            },
        },
        "comparability": {
            "scientific_depth_config_match": scientific_match,
            "scientific_fingerprint": eig_fingerprints["eig_depth_1"],
            "pairings": pairings,
            "operational_differences": {
                "eig_depth_1": {
                    "blocks": 1,
                    "trials_per_block": [40],
                    "seeds": [12345],
                    "batched_block_size": [1000],
                },
                "eig_depth_2": {
                    "blocks": 4,
                    "trials_per_block": [10, 10, 10, 10],
                    "seeds": [12345, 12346, 12347, 12348],
                    "batched_block_size": [2000],
                },
                "eig_depth_3": {
                    "blocks": 4,
                    "trials_per_block": [10, 10, 10, 10],
                    "seeds": [12345, 12346, 12347, 12348],
                    "batched_block_size": [2000],
                },
            },
            "code_provenance": {
                "logged_git_commit_available": False,
                "strictly_code_matched": False,
                "note": (
                    "The logs do not record Git commits. Depth-1 and depth-2 logs are from "
                    "May 4, while depth-3 logs are from May 18-20 after commit 9c2261f "
                    "rewrote recursive forward search. Treat depth-3 as configuration-matched, "
                    "not bitwise implementation-matched."
                ),
            },
        },
        "arms": arm_summaries,
        "block_diagnostics": {
            "eig_depth_1_first10": {
                "accuracy_auc": accuracy_auc(d1_first10),
                "q80": q_at_threshold(mean_curve(d1_first10)),
            },
            "eig_depth_1_remaining30": {
                "accuracy_auc": accuracy_auc(arms["eig_depth_1"].trials[10:]),
                "q80": q_at_threshold(mean_curve(arms["eig_depth_1"].trials[10:])),
            },
            "eig_depth_2_by_seed": depth2_blocks,
            "eig_depth_3_by_seed": depth3_blocks,
        },
        "comparisons": comparisons,
        "sources": {
            name: [_source_record(block, base_dir) for block in arm.blocks]
            for name, arm in arms.items()
        },
        "recovered_trials": {
            name: {
                "targets": list(arm.targets),
                "prior_summaries": list(arm.prior_summaries),
                "accuracy": [list(trial) for trial in arm.trials],
            }
            for name, arm in arms.items()
        },
    }


def _format_q(value: int) -> str:
    return f">{NUM_ROUNDS}" if value > NUM_ROUNDS else str(value)


def _format_ci(values: Sequence[float], digits: int = 3) -> str:
    return f"[{values[0]:.{digits}f}, {values[1]:.{digits}f}]"


def render_markdown(result: dict[str, Any]) -> str:
    arms = result["arms"]
    comparisons = result["comparisons"]
    display_names = {
        "naive_nonthinking": "Naive, non-thinking",
        "naive_thinking": "Naive, thinking",
        "eig_depth_1": "EIG depth 1",
        "eig_depth_2": "EIG depth 2",
        "eig_depth_3": "EIG depth 3",
    }
    lines = [
        "# Animals Depth Reanalysis",
        "",
        "## Decision",
        "",
        "**Track 1 collapses as a credible non-myopic paper spine.** The banked results "
        "still show that one-step EIG beats both naive baselines, but they do not show a "
        "defensible benefit from deeper planning. The full 40-trial depth-2 point estimate "
        "is slightly above depth 1; however, those 40 trials are not paired. On the only "
        "directly paired seed block, the sign reverses. Depth 3 is then materially worse "
        "than depth 2.",
        "",
        "Paper implication: use animals as evidence that LLM-generated queries can support "
        "one-step BED, not as the non-myopic result. MediQ must carry the non-myopic claim.",
        "",
        "## Metrics",
        "",
        "- **Accuracy-AUC:** mean exact-guess accuracy over questions 1-20, normalized to [0, 1].",
        "- **Q@80:** first question where mean exact-guess accuracy reaches 0.80. `>20` means it never reaches 0.80.",
        f"- Intervals are deterministic 95% percentile bootstrap CIs ({result['metric_definitions']['bootstrap']['replicates']:,} replicates; seed {result['metric_definitions']['bootstrap']['seed']}).",
        "",
        "## Arm Results",
        "",
        "| Arm | n | Accuracy-AUC | 95% CI | Q@80 | Q@80 95% CI | Accuracy at Q20 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in display_names:
        summary = arms[name]
        q_ci = summary["q80_ci95"]
        lines.append(
            f"| {display_names[name]} | {summary['num_trials']} | "
            f"{summary['accuracy_auc']:.3f} | {_format_ci(summary['accuracy_auc_ci95'])} | "
            f"{_format_q(summary['q80'])} | "
            f"[{_format_q(q_ci[0])}, {_format_q(q_ci[1])}] | "
            f"{summary['round20_accuracy']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            "Positive Accuracy-AUC deltas favor the first method. Negative Q@80 deltas mean it reaches 80% in fewer questions.",
            "",
            "| Comparison | Pairing | AUC delta | 95% CI | Q@80 delta | 95% CI | W/T/L |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    comparison_order = (
        "eig_depth_1_vs_naive_nonthinking_paired40",
        "eig_depth_1_vs_naive_thinking_paired40",
        "eig_depth_2_vs_eig_depth_1_unpaired40",
        "eig_depth_2_vs_eig_depth_1_paired_seed12345",
        "eig_depth_3_vs_eig_depth_2_paired40",
        "eig_depth_3_vs_eig_depth_1_paired_seed12345",
    )
    for key in comparison_order:
        comparison = comparisons[key]
        pairing = (
            f"paired n={comparison['better_num_trials']}"
            if comparison["paired"]
            else f"unpaired {comparison['better_num_trials']} vs {comparison['baseline_num_trials']}"
        )
        wtl = (
            f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']}"
            if comparison["paired"]
            else "-"
        )
        lines.append(
            f"| {comparison['label']} | {pairing} | "
            f"{comparison['accuracy_auc_delta']:+.3f} | "
            f"{_format_ci(comparison['accuracy_auc_delta_ci95'])} | "
            f"{comparison['q80_delta']:+d} | "
            f"{_format_ci(comparison['q80_delta_ci95'], digits=1)} | {wtl} |"
        )

    d2_blocks = result["block_diagnostics"]["eig_depth_2_by_seed"]
    d3_blocks = result["block_diagnostics"]["eig_depth_3_by_seed"]
    lines.extend(
        [
            "",
            "## Why The Apparent Depth-2 Gain Does Not Hold",
            "",
            "Depth 1 used one 40-trial stream at seed 12345. Depths 2 and 3 used four independently restarted 10-trial blocks at seeds 12345-12348. Only the first ten depth-1 trials are exactly paired with the seed-12345 blocks; the remaining 30 depth-1 trials do not share targets and randomized prior orders with the deeper runs.",
            "",
            f"The full, unpaired depth-2 estimate is {arms['eig_depth_2']['accuracy_auc']:.3f} versus {arms['eig_depth_1']['accuracy_auc']:.3f} for depth 1, a delta of {comparisons['eig_depth_2_vs_eig_depth_1_unpaired40']['accuracy_auc_delta']:+.3f}. But on the only direct pair, depth 2 is {result['block_diagnostics']['eig_depth_2_by_seed']['12345']['accuracy_auc']:.3f} versus {result['block_diagnostics']['eig_depth_1_first10']['accuracy_auc']:.3f} for depth 1, a delta of {comparisons['eig_depth_2_vs_eig_depth_1_paired_seed12345']['accuracy_auc_delta']:+.3f} with W/T/L {comparisons['eig_depth_2_vs_eig_depth_1_paired_seed12345']['wins']}/{comparisons['eig_depth_2_vs_eig_depth_1_paired_seed12345']['ties']}/{comparisons['eig_depth_2_vs_eig_depth_1_paired_seed12345']['losses']}.",
            "",
            "The depth-2 block AUCs are "
            + ", ".join(f"{seed}: {row['accuracy_auc']:.3f}" for seed, row in d2_blocks.items())
            + ". This large block spread explains how the unpaired aggregate can look favorable while the matched block reverses.",
            "",
            "Depth 3 does not rescue the trend. Its paired 40-trial AUC delta versus depth 2 is "
            f"{comparisons['eig_depth_3_vs_eig_depth_2_paired40']['accuracy_auc_delta']:+.3f} "
            f"with W/T/L {comparisons['eig_depth_3_vs_eig_depth_2_paired40']['wins']}/{comparisons['eig_depth_3_vs_eig_depth_2_paired40']['ties']}/{comparisons['eig_depth_3_vs_eig_depth_2_paired40']['losses']}. "
            "Its block AUCs are "
            + ", ".join(f"{seed}: {row['accuracy_auc']:.3f}" for seed, row in d3_blocks.items())
            + ".",
            "",
            "## Comparability Audit",
            "",
            "The EIG arms match on the scientifically relevant configuration: Gemma 4 E4B non-thinking questioner, Gemma 4 31B non-thinking answerer, categorical belief, exponential-rank prior (rate 0.12), no belief generation or filtering, guess threshold 0.99, 40 Monte Carlo samples, generation temperatures 1.3/1.0, answer temperature 0.7, and the same 40-animal pool.",
            "",
            "Operationally, depth 1 used `batched_block_size=1000` in one 40-trial run; depths 2 and 3 used `batched_block_size=2000` in four 10-trial runs. This does not itself change the intended policy, but the seed-block design prevents a paired full-sample depth-1 comparison.",
            "",
            "Code provenance is weaker than configuration provenance. The logs do not record Git commits. Depth-1 and depth-2 logs are from May 4; depth-3 logs are from May 18-20, after commit `9c2261f` rewrote recursive forward search. Therefore depth 3 is configuration-matched but not demonstrably implementation-identical to the earlier runs.",
            "",
            "## Bottom Line",
            "",
            "1. One-step EIG is the stable positive result in this bank: it substantially outperforms both naive baselines on 40 exactly paired trials.",
            "2. The small depth-2 advantage in the old aggregate is not credible causal evidence for planning depth because the direct paired subset reverses it.",
            "3. Depth 3 is worse than depth 2 on all 40 paired trials in aggregate, with more trial-level losses than wins.",
            "4. Do not rerun Paprika or build another animals variant. Proceed to the pre-registered MediQ non-myopic test, while retaining this animals result as one-step BED transfer evidence.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python scripts/analyze_animals_depth_reanalysis.py",
            "pytest tests/test_analyze_animals_depth_reanalysis.py -q",
            "```",
            "",
            "The JSON artifact records source SHA-256 hashes, recovered binary trial matrices, target/prior pairing metadata, curves, intervals, and comparison diagnostics. Raw logs remain untracked.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-log-dir",
        type=Path,
        default=Path("tmp/animals_reanalysis/source/old"),
    )
    parser.add_argument(
        "--depth3-dir",
        type=Path,
        default=Path("tmp/animals_reanalysis/source/depth3"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/path_e/animals_reanalysis/ANIMALS_REANALYSIS.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/path_e/ANIMALS_REANALYSIS.md"),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = analyze(
        args.old_log_dir,
        args.depth3_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output_md.write_text(render_markdown(result))
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Decision: {result['status']}")


if __name__ == "__main__":
    main()
