#!/usr/bin/env python3
"""Audit released Battleship question programs for non-myopic opportunity."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import subprocess
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


DEFAULT_EXTERNAL_ROOT = Path("external/battleship")
DEFAULT_OUTPUT_ROOT = Path(
    "results/nonmyopic/battleship_llm_native_opportunity_audit"
)
EXPECTED_COMMIT = "b98a4ba1c55be1bd5aa8038d42ad0070c41cabcb"
EXPECTED_TRAJECTORY_SHA256 = (
    "c39aa87d888fb98d5c6750ce4a3d70da6757adab9a5d4a69a44801f298290e94"
)
EXPECTED_STAGE_ZERO_PROGRAMS = 39
DEFAULT_SEEDS = (39600, 39601)
DEFAULT_SAMPLES = 4096
DEFAULT_EPSILON = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def extract_stage_zero_programs(payload: dict) -> list[dict]:
    programs = []
    for game in payload["games"]:
        for event in game["events"]:
            if (
                event.get("stage") == 0
                and event.get("question")
                and event.get("fn_str")
            ):
                programs.append(
                    {
                        "model": game["captain_llm"],
                        "captain_type": game["captain_type"],
                        "board_id": game["board_id"],
                        "question": event["question"]["text"],
                        "fn_str": event["fn_str"],
                    }
                )
    return programs


def compile_program(fn_text: str) -> Callable[[np.ndarray, np.ndarray], bool]:
    namespace: dict = {}
    exec(
        fn_text,
        {"np": np, "__builtins__": __builtins__},
        namespace,
    )
    answer = namespace.get("answer")
    if not callable(answer):
        raise ValueError("program does not define callable answer")
    return answer


def evaluate_program(
    answer: Callable[[np.ndarray, np.ndarray], bool],
    boards: np.ndarray,
    partial_board: np.ndarray,
) -> np.ndarray:
    values = []
    for board in boards:
        value = answer(board, partial_board)
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"program returned {type(value).__name__}, not bool")
        values.append(bool(value))
    return np.asarray(values, dtype=np.uint8)


def dedupe_by_joint_behavior(
    programs: list[dict],
    outcomes_by_program: list[list[np.ndarray]],
) -> tuple[list[dict], list[list[np.ndarray]]]:
    kept_programs = []
    kept_outcomes = []
    seen = set()
    for program, outcomes in zip(programs, outcomes_by_program):
        key = b"".join(outcome.tobytes() for outcome in outcomes)
        if key in seen:
            continue
        seen.add(key)
        kept_programs.append(program)
        kept_outcomes.append(outcomes)
    return kept_programs, kept_outcomes


def binary_entropy_bits(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (
        1.0 - probability
    ) * math.log2(1.0 - probability)


@dataclass(frozen=True)
class RootEvaluation:
    values: np.ndarray
    best_indices: tuple[int, ...]
    best_value: float


class FiniteHorizonQuestionPlanner:
    """Exact planning on a finite weighted board-particle approximation."""

    def __init__(
        self,
        outcomes: np.ndarray,
        occupancy: np.ndarray,
        *,
        epsilon: float,
    ) -> None:
        self.outcomes = np.asarray(outcomes, dtype=np.uint8)
        self.occupancy = np.asarray(occupancy, dtype=np.float64)
        self.epsilon = float(epsilon)
        self.num_questions, self.num_particles = self.outcomes.shape
        if self.occupancy.shape[0] != self.num_particles:
            raise ValueError("outcome and occupancy particle counts differ")
        self.full_mask = (1 << self.num_questions) - 1
        self.initial_weights = np.full(
            self.num_particles,
            1.0 / self.num_particles,
            dtype=np.float64,
        )

    def terminal_hit_probability(self, weights: np.ndarray) -> float:
        return float(np.max(weights @ self.occupancy))

    def observation_probability(
        self, weights: np.ndarray, question_index: int
    ) -> float:
        latent_true = float(self.outcomes[question_index] @ weights)
        return self.epsilon + (1.0 - 2.0 * self.epsilon) * latent_true

    def posterior(
        self,
        weights: np.ndarray,
        question_index: int,
        observed_answer: int,
    ) -> np.ndarray:
        likelihood = np.where(
            self.outcomes[question_index] == observed_answer,
            1.0 - self.epsilon,
            self.epsilon,
        )
        posterior = weights * likelihood
        posterior /= posterior.sum()
        return posterior

    @functools.lru_cache(maxsize=None)
    def _optimal(
        self,
        weight_bytes: bytes,
        available_mask: int,
        depth: int,
    ) -> tuple[float, int]:
        weights = np.frombuffer(weight_bytes, dtype=np.float64)
        if depth == 0 or available_mask == 0:
            return self.terminal_hit_probability(weights), -1

        best_value = -math.inf
        best_question = -1
        for question_index in range(self.num_questions):
            question_bit = 1 << question_index
            if not available_mask & question_bit:
                continue
            probability_yes = self.observation_probability(
                weights, question_index
            )
            value = 0.0
            for answer, probability in (
                (1, probability_yes),
                (0, 1.0 - probability_yes),
            ):
                posterior = self.posterior(weights, question_index, answer)
                child_value, _ = self._optimal(
                    posterior.tobytes(),
                    available_mask ^ question_bit,
                    depth - 1,
                )
                value += probability * child_value
            if value > best_value + 1e-15:
                best_value = value
                best_question = question_index
        return best_value, best_question

    def root_evaluation(self, depth: int) -> RootEvaluation:
        values = np.empty(self.num_questions, dtype=np.float64)
        weights = self.initial_weights
        for question_index in range(self.num_questions):
            question_bit = 1 << question_index
            probability_yes = self.observation_probability(
                weights, question_index
            )
            value = 0.0
            for answer, probability in (
                (1, probability_yes),
                (0, 1.0 - probability_yes),
            ):
                posterior = self.posterior(weights, question_index, answer)
                if depth == 1:
                    child_value = self.terminal_hit_probability(posterior)
                else:
                    child_value, _ = self._optimal(
                        posterior.tobytes(),
                        self.full_mask ^ question_bit,
                        depth - 1,
                    )
                value += probability * child_value
            values[question_index] = value
        best_value = float(np.max(values))
        best_indices = tuple(
            np.flatnonzero(
                np.isclose(values, best_value, rtol=0.0, atol=1e-12)
            ).tolist()
        )
        return RootEvaluation(values, best_indices, best_value)

    def receding_horizon_value(
        self,
        *,
        total_questions: int,
        planning_horizon: int,
    ) -> float:
        @functools.lru_cache(maxsize=None)
        def evaluate(
            weight_bytes: bytes,
            available_mask: int,
            questions_left: int,
        ) -> float:
            weights = np.frombuffer(weight_bytes, dtype=np.float64)
            if questions_left == 0 or available_mask == 0:
                return self.terminal_hit_probability(weights)
            _, question_index = self._optimal(
                weight_bytes,
                available_mask,
                min(planning_horizon, questions_left),
            )
            question_bit = 1 << question_index
            probability_yes = self.observation_probability(
                weights, question_index
            )
            value = 0.0
            for answer, probability in (
                (1, probability_yes),
                (0, 1.0 - probability_yes),
            ):
                posterior = self.posterior(weights, question_index, answer)
                value += probability * evaluate(
                    posterior.tobytes(),
                    available_mask ^ question_bit,
                    questions_left - 1,
                )
            return value

        return evaluate(
            self.initial_weights.tobytes(),
            self.full_mask,
            total_questions,
        )

    def greedy_eig_root(self) -> tuple[int, float]:
        channel_entropy = binary_entropy_bits(self.epsilon)
        values = []
        for question_index in range(self.num_questions):
            probability_yes = self.observation_probability(
                self.initial_weights, question_index
            )
            values.append(
                binary_entropy_bits(probability_yes) - channel_entropy
            )
        best_index = int(np.argmax(values))
        return best_index, float(values[best_index])

    def greedy_eig_terminal_value(self, *, total_questions: int) -> float:
        channel_entropy = binary_entropy_bits(self.epsilon)

        @functools.lru_cache(maxsize=None)
        def evaluate(
            weight_bytes: bytes,
            available_mask: int,
            questions_left: int,
        ) -> float:
            weights = np.frombuffer(weight_bytes, dtype=np.float64)
            if questions_left == 0 or available_mask == 0:
                return self.terminal_hit_probability(weights)

            best_eig = -math.inf
            question_index = -1
            for candidate in range(self.num_questions):
                if not available_mask & (1 << candidate):
                    continue
                probability_yes = self.observation_probability(
                    weights, candidate
                )
                eig = (
                    binary_entropy_bits(probability_yes) - channel_entropy
                )
                if eig > best_eig + 1e-15:
                    best_eig = eig
                    question_index = candidate

            question_bit = 1 << question_index
            probability_yes = self.observation_probability(
                weights, question_index
            )
            value = 0.0
            for answer, probability in (
                (1, probability_yes),
                (0, 1.0 - probability_yes),
            ):
                posterior = self.posterior(weights, question_index, answer)
                value += probability * evaluate(
                    posterior.tobytes(),
                    available_mask ^ question_bit,
                    questions_left - 1,
                )
            return value

        return evaluate(
            self.initial_weights.tobytes(),
            self.full_mask,
            total_questions,
        )


def _install_ipython_display_stub() -> None:
    if "IPython.display" in sys.modules:
        return
    try:
        __import__("IPython.display")
        return
    except ImportError:
        pass
    ipython = types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")
    display_module.display = lambda *args, **kwargs: None
    ipython.display = display_module
    sys.modules["IPython"] = ipython
    sys.modules["IPython.display"] = display_module


def sample_official_prior(
    external_root: Path,
    *,
    seeds: Iterable[int],
    num_samples: int,
) -> list[np.ndarray]:
    _install_ipython_display_stub()
    sys.path.insert(0, str(external_root.resolve()))
    try:
        from battleship.board import Board
        from battleship.fast_sampler import FastSampler
    finally:
        sys.path.pop(0)

    blocks = []
    ship_tracker = [(2, None), (3, None), (4, None), (5, None)]
    for seed in seeds:
        sampler = FastSampler(
            Board.hidden_board(8),
            ship_tracker,
            seed=seed,
        )
        weighted = sampler.get_weighted_samples(n_samples=num_samples)
        if len(weighted) != num_samples:
            raise RuntimeError(
                f"sampled {len(weighted)}/{num_samples} boards for seed {seed}"
            )
        blocks.append(np.stack([board.board for board, _ in weighted]))
    return blocks


def run_audit(
    *,
    external_root: Path,
    seeds: tuple[int, ...],
    num_samples: int,
    epsilon: float,
) -> dict:
    trajectory_path = (
        external_root / "docs/static/data/trajectory_samples.json"
    )
    commit = git_commit(external_root)
    trajectory_sha256 = sha256_file(trajectory_path)
    payload = json.loads(trajectory_path.read_text())
    source_programs = extract_stage_zero_programs(payload)
    blocks = sample_official_prior(
        external_root,
        seeds=seeds,
        num_samples=num_samples,
    )
    partial_board = np.full((8, 8), -1, dtype=int)

    valid_programs = []
    outcomes_by_program = []
    failures = []
    for source_index, program in enumerate(source_programs):
        try:
            answer = compile_program(program["fn_str"])
            outcomes = [
                evaluate_program(answer, boards, partial_board)
                for boards in blocks
            ]
            if any(
                int(outcome.sum()) in (0, len(outcome))
                for outcome in outcomes
            ):
                raise ValueError("program is constant on a posterior block")
            valid_programs.append(
                {"source_index": source_index, **program}
            )
            outcomes_by_program.append(outcomes)
        except Exception as exc:
            failures.append(
                {
                    "source_index": source_index,
                    "question": program["question"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    programs, deduped_outcomes = dedupe_by_joint_behavior(
        valid_programs, outcomes_by_program
    )
    block_results = []
    root_questions_by_depth: dict[int, list[str]] = {
        1: [],
        2: [],
        3: [],
    }
    for block_index, (seed, boards) in enumerate(zip(seeds, blocks)):
        outcomes = np.stack(
            [values[block_index] for values in deduped_outcomes]
        )
        occupancy = (boards > 0).reshape(len(boards), -1)
        planner = FiniteHorizonQuestionPlanner(
            outcomes,
            occupancy,
            epsilon=epsilon,
        )
        depth_results = {}
        for depth in (1, 2, 3):
            root = planner.root_evaluation(depth)
            root_questions = [
                programs[index]["question"] for index in root.best_indices
            ]
            root_questions_by_depth[depth].extend(root_questions)
            depth_results[str(depth)] = {
                "best_value": root.best_value,
                "best_indices": list(root.best_indices),
                "best_questions": root_questions,
                "top_five": [
                    {
                        "index": int(index),
                        "value": float(root.values[index]),
                        "question": programs[index]["question"],
                    }
                    for index in np.argsort(-root.values)[:5]
                ],
            }

        receding_values = {
            str(depth): planner.receding_horizon_value(
                total_questions=3,
                planning_horizon=depth,
            )
            for depth in (1, 2, 3)
        }
        greedy_index, greedy_eig = planner.greedy_eig_root()
        block_results.append(
            {
                "block_index": block_index,
                "seed": seed,
                "num_samples": len(boards),
                "base_hit_probability": planner.terminal_hit_probability(
                    planner.initial_weights
                ),
                "depth": depth_results,
                "three_question_receding_hit_probability": receding_values,
                "greedy_eig": {
                    "root_index": greedy_index,
                    "root_question": programs[greedy_index]["question"],
                    "root_eig_bits": greedy_eig,
                    "three_question_hit_probability": (
                        planner.greedy_eig_terminal_value(
                            total_questions=3
                        )
                    ),
                },
            }
        )

    stable_roots = {
        str(depth): len(set(root_questions_by_depth[depth])) == 1
        for depth in (1, 2, 3)
    }
    disjoint_roots = []
    monotonic_values = []
    minimum_step_gain = []
    for block in block_results:
        roots = {
            depth: set(block["depth"][str(depth)]["best_questions"])
            for depth in (1, 2, 3)
        }
        disjoint_roots.append(
            roots[1].isdisjoint(roots[2])
            and roots[2].isdisjoint(roots[3])
        )
        values = block["three_question_receding_hit_probability"]
        gains = (
            values["2"] - values["1"],
            values["3"] - values["2"],
        )
        monotonic_values.append(gains[0] > 0 and gains[1] > 0)
        minimum_step_gain.append(min(gains))

    gates = {
        "source_commit_matches": commit == EXPECTED_COMMIT,
        "trajectory_sha256_matches": (
            trajectory_sha256 == EXPECTED_TRAJECTORY_SHA256
        ),
        "stage_zero_program_count_exact": (
            len(source_programs) == EXPECTED_STAGE_ZERO_PROGRAMS
        ),
        "all_stage_zero_programs_valid_nonconstant": (
            len(valid_programs) == len(source_programs)
        ),
        "at_least_20_unique_behaviors": len(programs) >= 20,
        "stable_root_per_depth_across_blocks": all(
            stable_roots.values()
        ),
        "pairwise_depth_roots_disjoint_each_block": all(disjoint_roots),
        "three_question_value_strictly_monotonic_each_block": all(
            monotonic_values
        ),
        "minimum_adjacent_horizon_gain_at_least_0_01": (
            min(minimum_step_gain) >= 0.01
        ),
    }
    public_programs = [
        {
            key: value
            for key, value in program.items()
            if key != "fn_str"
        }
        for program in programs
    ]
    return {
        "status": "passed" if all(gates.values()) else "gate_failed",
        "source": {
            "repository": "https://github.com/gabegrand/battleship",
            "commit": commit,
            "trajectory_path": str(trajectory_path),
            "trajectory_sha256": trajectory_sha256,
        },
        "config": {
            "seeds": list(seeds),
            "num_samples_per_block": num_samples,
            "epsilon": epsilon,
            "execution_question_budget": 3,
        },
        "counts": {
            "released_games": len(payload["games"]),
            "stage_zero_programs": len(source_programs),
            "valid_nonconstant_programs": len(valid_programs),
            "unique_joint_behaviors": len(programs),
            "program_failures": len(failures),
        },
        "program_failures": failures,
        "question_bank": public_programs,
        "blocks": block_results,
        "gates": gates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=DEFAULT_EXTERNAL_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if not 0.0 <= args.epsilon < 0.5:
        raise ValueError("--epsilon must be in [0, 0.5)")
    if len(args.seeds) < 2:
        raise ValueError("--seeds requires at least two blocks")

    result = run_audit(
        external_root=args.external_root,
        seeds=tuple(args.seeds),
        num_samples=args.samples,
        epsilon=args.epsilon,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"battleship-llm-native-opportunity-{timestamp}"
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    result["run_id"] = run_id
    result_path = run_dir / "RESULT.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    latest_path = args.output_root / "LATEST_RUN_ID"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(run_id + "\n")
    print(json.dumps(result, indent=2))
    print(f"RESULT_PATH={result_path}")


if __name__ == "__main__":
    main()

