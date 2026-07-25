#!/usr/bin/env python3
"""Matched-tree LLM-native BED on the DiscoverPhysics simulator."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from openrouter_model import OpenRouterAdapter


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-llm-bed-1"
DISCOVERPHYSICS_COMMIT = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
MODEL_ID = "openai/gpt-5.4"
WORLD = "extra_dimensions"
SEED = 24365
NUM_HYPOTHESES = 5
NUM_CANDIDATES = 4
NUM_BRANCHES = 2
NUM_EXPERIMENT_ROUNDS = 2
NOISE_STD = 0.01
TREE_MAX_TOKENS = 4600
FINAL_MAX_TOKENS = 5200
RUN_BUDGET_USD = 0.50
PROJECTED_COST_USD = 0.42

EXPERIMENT_KEYS = frozenset(
    {"p1", "p2", "pos2", "velocity2", "measurement_times"}
)
TREE_KEYS = frozenset({"prior", "candidates"})
BELIEF_KEYS = frozenset({"hypotheses", "probabilities"})
CANDIDATE_KEYS = frozenset({"id", "experiment", "branches"})
BRANCH_KEYS = frozenset(
    {
        "probability",
        "observation",
        "posterior_probabilities",
        "continuation_experiment",
        "terminal_belief",
    }
)
FINAL_LAW_PATTERN = re.compile(
    r"<final_law>\s*(.*?)\s*</final_law>", re.DOTALL
)
EXPLANATION_PATTERN = re.compile(
    r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL
)

# Hidden from the policy. These augment the benchmark's default two cases with
# the short/long-distance crossover that defines the public world.
STRESS_TEST_CASES = (
    {
        "p1": 0.15,
        "p2": 1.0,
        "pos2": [0.4, 0.0],
        "velocity2": [0.0, 0.15],
        "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    },
    {
        "p1": 0.3,
        "p2": 1.5,
        "pos2": [0.75, 0.0],
        "velocity2": [0.0, 0.1],
        "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    },
    {
        "p1": 0.8,
        "p2": 2.0,
        "pos2": [1.25, 0.0],
        "velocity2": [0.0, 0.2],
        "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    },
    {
        "p1": 1.0,
        "p2": 1.0,
        "pos2": [3.0, 0.0],
        "velocity2": [0.0, 0.3],
        "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    },
    {
        "p1": 2.0,
        "p2": 3.0,
        "pos2": [6.0, 0.0],
        "velocity2": [0.0, 0.4],
        "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    },
)


class NonReasoningOpenRouterAdapter(OpenRouterAdapter):
    """Force the environment policy onto the explicit non-reasoning route."""

    def _payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        *,
        disable_reasoning: bool = False,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages,
            temperature,
            n,
            max_tokens,
            disable_reasoning=disable_reasoning,
        )
        payload["reasoning"] = {"enabled": False, "exclude": True}
        return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


def _probabilities(
    values: Any,
    *,
    count: int,
    label: str,
) -> list[float]:
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        )
    ):
        raise ValueError(f"{label} must contain exactly {count} numbers")
    probabilities = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError(f"{label} contains an invalid probability")
    total = sum(probabilities)
    if abs(total - 1.0) > 0.02:
        raise ValueError(f"{label} sums to {total}, not one")
    if total <= 0.0:
        raise ValueError(f"{label} has zero total mass")
    return [value / total for value in probabilities]


def validate_belief(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != BELIEF_KEYS:
        raise ValueError(f"{label} has the wrong fields")
    hypotheses = value["hypotheses"]
    if (
        not isinstance(hypotheses, list)
        or len(hypotheses) != NUM_HYPOTHESES
        or any(
            not isinstance(hypothesis, str) or not hypothesis.strip()
            for hypothesis in hypotheses
        )
    ):
        raise ValueError(
            f"{label}.hypotheses must contain {NUM_HYPOTHESES} strings"
        )
    normalized = [" ".join(hypothesis.casefold().split()) for hypothesis in hypotheses]
    if len(set(normalized)) != NUM_HYPOTHESES:
        raise ValueError(f"{label} contains duplicate hypotheses")
    return {
        "hypotheses": [hypothesis.strip() for hypothesis in hypotheses],
        "probabilities": _probabilities(
            value["probabilities"],
            count=NUM_HYPOTHESES,
            label=f"{label}.probabilities",
        ),
    }


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_experiment(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != EXPERIMENT_KEYS:
        raise ValueError(f"{label} has the wrong experiment fields")
    p1 = _finite_number(value["p1"], label=f"{label}.p1")
    p2 = _finite_number(value["p2"], label=f"{label}.p2")
    if not (0.1 <= p1 <= 10.0 and 0.1 <= p2 <= 10.0):
        raise ValueError(f"{label} particle properties are out of range")

    def vector(name: str, bound: float) -> list[float]:
        raw = value[name]
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"{label}.{name} must be a length-two list")
        parsed = [
            _finite_number(component, label=f"{label}.{name}") for component in raw
        ]
        if any(abs(component) > bound for component in parsed):
            raise ValueError(f"{label}.{name} is out of range")
        return parsed

    pos2 = vector("pos2", 10.0)
    if math.hypot(*pos2) < 0.35:
        raise ValueError(f"{label}.pos2 is too close to the singular source")
    velocity2 = vector("velocity2", 5.0)
    times = value["measurement_times"]
    if (
        not isinstance(times, list)
        or len(times) != 10
        or any(
            isinstance(time, bool) or not isinstance(time, (int, float))
            for time in times
        )
    ):
        raise ValueError(f"{label}.measurement_times must contain 10 numbers")
    parsed_times = [float(time) for time in times]
    if (
        any(not math.isfinite(time) for time in parsed_times)
        or parsed_times != sorted(parsed_times)
        or len(set(parsed_times)) != len(parsed_times)
        or parsed_times[0] < 0.0
        or parsed_times[-1] < 5.0
        or parsed_times[-1] > 10.0
    ):
        raise ValueError(f"{label}.measurement_times are invalid")
    return {
        "p1": p1,
        "p2": p2,
        "pos2": pos2,
        "velocity2": velocity2,
        "measurement_times": parsed_times,
    }


def parse_tree(response: str) -> dict[str, Any]:
    try:
        value, end = json.JSONDecoder().raw_decode(response.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"tree response is not exact JSON: {exc}") from exc
    if response.strip()[end:].strip():
        raise ValueError("tree response has trailing content")
    if not isinstance(value, dict) or set(value) != TREE_KEYS:
        raise ValueError("tree response has the wrong top-level fields")
    prior = validate_belief(value["prior"], label="prior")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) != NUM_CANDIDATES:
        raise ValueError(f"tree must contain exactly {NUM_CANDIDATES} candidates")
    parsed_candidates = []
    candidate_ids: set[str] = set()
    experiment_keys: set[str] = set()
    for candidate_index, candidate in enumerate(candidates):
        label = f"candidates[{candidate_index}]"
        if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_KEYS:
            raise ValueError(f"{label} has the wrong fields")
        candidate_id = candidate["id"]
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or candidate_id in candidate_ids
        ):
            raise ValueError(f"{label}.id is invalid or duplicated")
        candidate_ids.add(candidate_id)
        experiment = validate_experiment(
            candidate["experiment"], label=f"{label}.experiment"
        )
        experiment_key = json.dumps(experiment, sort_keys=True)
        if experiment_key in experiment_keys:
            raise ValueError("candidate root experiments are duplicated")
        experiment_keys.add(experiment_key)
        branches = candidate["branches"]
        if not isinstance(branches, list) or len(branches) != NUM_BRANCHES:
            raise ValueError(f"{label} must contain {NUM_BRANCHES} branches")
        branch_probabilities = _probabilities(
            [branch.get("probability") if isinstance(branch, dict) else None for branch in branches],
            count=NUM_BRANCHES,
            label=f"{label}.branch_probabilities",
        )
        parsed_branches = []
        observations: set[str] = set()
        for branch_index, (branch, branch_probability) in enumerate(
            zip(branches, branch_probabilities, strict=True)
        ):
            branch_label = f"{label}.branches[{branch_index}]"
            if not isinstance(branch, dict) or set(branch) != BRANCH_KEYS:
                raise ValueError(f"{branch_label} has the wrong fields")
            observation = branch["observation"]
            if not isinstance(observation, str) or not observation.strip():
                raise ValueError(f"{branch_label}.observation is invalid")
            observation_key = " ".join(observation.casefold().split())
            if observation_key in observations:
                raise ValueError(f"{label} has duplicate observation branches")
            observations.add(observation_key)
            posterior = _probabilities(
                branch["posterior_probabilities"],
                count=NUM_HYPOTHESES,
                label=f"{branch_label}.posterior_probabilities",
            )
            continuation = validate_experiment(
                branch["continuation_experiment"],
                label=f"{branch_label}.continuation_experiment",
            )
            if json.dumps(continuation, sort_keys=True) == experiment_key:
                raise ValueError(f"{branch_label} repeats the root experiment")
            terminal = validate_belief(
                branch["terminal_belief"],
                label=f"{branch_label}.terminal_belief",
            )
            parsed_branches.append(
                {
                    "probability": branch_probability,
                    "observation": observation.strip(),
                    "posterior_probabilities": posterior,
                    "continuation_experiment": continuation,
                    "terminal_belief": terminal,
                }
            )
        parsed_candidates.append(
            {
                "id": candidate_id.strip(),
                "experiment": experiment,
                "branches": parsed_branches,
            }
        )
    tree = {"prior": prior, "candidates": parsed_candidates}
    return score_tree(tree)


def score_tree(tree: dict[str, Any]) -> dict[str, Any]:
    prior_entropy = entropy(tree["prior"]["probabilities"])
    for candidate in tree["candidates"]:
        immediate_entropy = sum(
            branch["probability"]
            * entropy(branch["posterior_probabilities"])
            for branch in candidate["branches"]
        )
        terminal_entropy = sum(
            branch["probability"]
            * entropy(branch["terminal_belief"]["probabilities"])
            for branch in candidate["branches"]
        )
        candidate["immediate_expected_entropy"] = immediate_entropy
        candidate["terminal_expected_entropy"] = terminal_entropy
        candidate["immediate_eig"] = prior_entropy - immediate_entropy
        candidate["terminal_eig"] = prior_entropy - terminal_entropy
    tree["prior_entropy"] = prior_entropy
    return tree


def select_candidate(tree: dict[str, Any], mode: str) -> int:
    if mode not in {"myopic", "nonmyopic"}:
        raise ValueError(f"unknown selection mode {mode!r}")
    key = "immediate_eig" if mode == "myopic" else "terminal_eig"
    return max(
        range(len(tree["candidates"])),
        key=lambda index: (tree["candidates"][index][key], -index),
    )


def tree_messages(history: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    schema = {
        "prior": {
            "hypotheses": ["exactly 5 concise, distinct force-law hypotheses"],
            "probabilities": ["exactly 5 numbers summing to 1"],
        },
        "candidates": [
            {
                "id": "E1",
                "experiment": {
                    "p1": 1.0,
                    "p2": 1.0,
                    "pos2": [3.0, 0.0],
                    "velocity2": [0.0, 0.0],
                    "measurement_times": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                },
                "branches": [
                    {
                        "probability": 0.5,
                        "observation": "concise predicted qualitative outcome",
                        "posterior_probabilities": ["5 numbers over prior hypotheses"],
                        "continuation_experiment": "same experiment object schema",
                        "terminal_belief": {
                            "hypotheses": ["5 hypotheses regenerated after root and continuation"],
                            "probabilities": ["5 numbers summing to 1"],
                        },
                    }
                ],
            }
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are the belief and experimental-design component of a "
                "scientific discovery agent. Return exact JSON only. Do not "
                "include chain-of-thought, markdown, comments, or extra fields."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    "Infer an unknown force law in a two-particle 2D simulator.",
                    "Particle 1 is fixed at [0,0]. You control scalar properties p1,p2,",
                    "the probe's initial position/velocity, and 10 measurement times.",
                    "Reported trajectories contain small unknown Gaussian noise.",
                    "The hidden physics may be noncanonical. Do not assume the relevant",
                    "law family is listed anywhere; generate semantic hypotheses yourself.",
                    "",
                    f"Return exactly {NUM_HYPOTHESES} prior hypotheses and exactly "
                    f"{NUM_CANDIDATES} distinct root experiments.",
                    f"Each candidate must have exactly {NUM_BRANCHES} mutually exclusive "
                    "qualitative outcome branches with probabilities summing to one.",
                    "posterior_probabilities are over the current prior hypotheses.",
                    "For every branch, design the best next experiment, then regenerate",
                    "a fresh five-hypothesis terminal belief under that two-step history.",
                    "Candidate roots should test genuinely different physical features.",
                    "Use exactly one experiment per root/continuation, not a batch.",
                    "Every measurement_times list has exactly 10 increasing values,",
                    "starts at or above 0, ends between 5 and 10, and has no duplicates.",
                    "Use p1,p2 in [0.1,10], positions in [-10,10]^2 at radius >=0.35,",
                    "and velocities in [-5,5]^2.",
                    "",
                    "Return this exact field structure, expanding lists to the required",
                    "counts and replacing the illustrative strings/values:",
                    json.dumps(schema, separators=(",", ":")),
                    "",
                    "REAL_HISTORY=" + json.dumps(list(history), separators=(",", ":")),
                ]
            ),
        },
    ]


def final_messages(
    history: Sequence[dict[str, Any]],
    final_belief: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the final scientific inference component. Do not expose "
                "chain-of-thought. Return exactly one <explanation> block followed "
                "by one <final_law> block and no other text."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    "Infer the hidden law from the two completed experiments below.",
                    "The world is a two-particle 2D simulator. Particle 1 is fixed at",
                    "[0,0]. Write a concise conceptual explanation and executable Python.",
                    "The code may import numpy. It must simulate from t=0 to duration and",
                    "return (final_pos2, final_vel2). Use numerically stable integration.",
                    "Unknown continuous constants may be exposed through **params and an",
                    "optional fit_parameters() returning init/bounds dictionaries.",
                    "",
                    "<explanation>2-4 sentences</explanation>",
                    "<final_law>",
                    "def discovered_law(pos1, pos2, p1, p2, velocity2, duration, **params):",
                    "    # executable implementation",
                    "    return final_pos2, final_vel2",
                    "</final_law>",
                    "",
                    "Do not name a benchmark or world. Do not use hidden test cases.",
                    "REAL_HISTORY=" + json.dumps(list(history), separators=(",", ":")),
                    "FINAL_GENERATED_BELIEF="
                    + json.dumps(final_belief, separators=(",", ":")),
                ]
            ),
        },
    ]


def parse_final(response: str) -> dict[str, str]:
    explanation_matches = EXPLANATION_PATTERN.findall(response)
    law_matches = FINAL_LAW_PATTERN.findall(response)
    if len(explanation_matches) != 1 or len(law_matches) != 1:
        raise ValueError("final response must contain one explanation and one law")
    stripped = FINAL_LAW_PATTERN.sub("", EXPLANATION_PATTERN.sub("", response)).strip()
    if stripped:
        raise ValueError("final response has content outside the two required tags")
    explanation = explanation_matches[0].strip()
    law_source = law_matches[0].strip()
    if not explanation or "def discovered_law" not in law_source:
        raise ValueError("final response is missing explanation or discovered_law")
    return {"explanation": explanation, "law_source": law_source}


def concept_score_extra_dimensions(text: str) -> dict[str, Any]:
    normalized = " ".join(text.casefold().replace("²", "^2").split())
    features = {
        "extra_dimension": bool(
            re.search(
                r"extra(?: spatial)?[- ]dimension|additional spatial dimension",
                normalized,
            )
        ),
        "compactification": "compact" in normalized,
        "long_range_inverse_r": bool(
            re.search(r"(long[- ]range.{0,80}(1/r|inverse[- ]distance))|"
                      r"((1/r|inverse[- ]distance).{0,80}long[- ]range)", normalized)
        ),
        "short_range_inverse_square": bool(
            re.search(r"(short[- ]range.{0,80}(1/r\^?2|inverse[- ]square))|"
                      r"((1/r\^?2|inverse[- ]square).{0,80}short[- ]range)", normalized)
        ),
        "crossover_scale": bool(
            re.search(
                r"cross(?:es|ing|ed)?(?:[- ]?over)?|transition|"
                r"two regimes|distance scale",
                normalized,
            )
        ),
    }
    return {"score": sum(features.values()), "features": features}


def _load_discoverphysics(root: Path) -> dict[str, Any]:
    commit = (
        __import__("subprocess")
        .run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    if commit != DISCOVERPHYSICS_COMMIT:
        raise ValueError(
            f"DiscoverPhysics is at {commit}, expected {DISCOVERPHYSICS_COMMIT}"
        )
    science_agent = root / "ScienceAgent"
    if str(science_agent) not in sys.path:
        sys.path.insert(0, str(science_agent))
    worlds = importlib.import_module("scienceagent.worlds")
    evaluator = importlib.import_module("scienceagent.evaluator")
    return {
        "get_world": worlds.get_world,
        "Evaluator": evaluator.Evaluator,
        "extract_training": evaluator._extract_training_trajectories,
    }


def _history_entry(
    experiment: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    return {"experiment": experiment, "observation": observation}


def _conversation_log(history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "action": "experiment",
            "experiment_input": [entry["experiment"]],
            "experiment_output": [entry["observation"]],
        }
        for entry in history
    ]


def evaluate_law(
    modules: dict[str, Any],
    *,
    executor: Any,
    law_source: str,
    history: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    training = modules["extract_training"](_conversation_log(history))
    official = modules["Evaluator"](executor).evaluate(
        law_source,
        verbose=False,
        training_trajectories=training,
    )
    stress = modules["Evaluator"](
        executor, test_cases=list(STRESS_TEST_CASES)
    ).evaluate(
        law_source,
        verbose=False,
        training_trajectories=training,
    )
    return {"official_default": official, "crossover_stress": stress}


def _adapter(
    *,
    run_id: str,
    output_dir: Path,
) -> NonReasoningOpenRouterAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=100.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=2,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=max(TREE_MAX_TOKENS, FINAL_MAX_TOKENS),
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    spec = ModelSpec(
        model=MODEL_ID,
        backend="openrouter",
        max_model_len=65536,
    )
    return NonReasoningOpenRouterAdapter(spec, config)


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_smoke(
    *,
    discoverphysics_root: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    modules = _load_discoverphysics(discoverphysics_root)
    adapter = _adapter(run_id=run_id, output_dir=output_dir)
    raw_path = output_dir / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {"initial_tree": None, "round2_trees": {}, "finals": {}}

    initial_response = adapter.chat_complete_messages_batched(
        [tree_messages([])],
        temperature=0.0,
        block_size=1,
        max_new_tokens=TREE_MAX_TOKENS,
    )[0]
    raw["initial_tree"] = initial_response
    _checkpoint(raw_path, raw)
    initial_tree = parse_tree(initial_response)
    initial_indices = {
        "myopic": select_candidate(initial_tree, "myopic"),
        "nonmyopic": select_candidate(initial_tree, "nonmyopic"),
    }

    worlds = {
        mode: modules["get_world"](
            WORLD,
            engine="nbody",
            noise_std=NOISE_STD,
            noise_seed=SEED,
        )
        for mode in ("myopic", "nonmyopic")
    }
    histories: dict[str, list[dict[str, Any]]] = {
        "myopic": [],
        "nonmyopic": [],
    }
    for mode in ("myopic", "nonmyopic"):
        experiment = initial_tree["candidates"][
            initial_indices[mode]
        ]["experiment"]
        observation = worlds[mode]["executor"].run([experiment])[0]
        histories[mode].append(_history_entry(experiment, observation))

    round2_responses = adapter.chat_complete_messages_batched(
        [tree_messages(histories[mode]) for mode in ("myopic", "nonmyopic")],
        temperature=0.0,
        block_size=2,
        max_new_tokens=TREE_MAX_TOKENS,
    )
    round2_trees: dict[str, dict[str, Any]] = {}
    for mode, response in zip(
        ("myopic", "nonmyopic"), round2_responses, strict=True
    ):
        raw["round2_trees"][mode] = response
        round2_trees[mode] = parse_tree(response)
    _checkpoint(raw_path, raw)

    # One action remains, so both policies use the one-step criterion.
    for mode in ("myopic", "nonmyopic"):
        candidate_index = select_candidate(round2_trees[mode], "myopic")
        experiment = round2_trees[mode]["candidates"][candidate_index][
            "experiment"
        ]
        observation = worlds[mode]["executor"].run([experiment])[0]
        histories[mode].append(_history_entry(experiment, observation))

    final_responses = adapter.chat_complete_messages_batched(
        [
            final_messages(histories[mode], round2_trees[mode]["prior"])
            for mode in ("myopic", "nonmyopic")
        ],
        temperature=0.0,
        block_size=2,
        max_new_tokens=FINAL_MAX_TOKENS,
    )
    finals: dict[str, dict[str, str]] = {}
    for mode, response in zip(
        ("myopic", "nonmyopic"), final_responses, strict=True
    ):
        raw["finals"][mode] = response
        finals[mode] = parse_final(response)
    _checkpoint(raw_path, raw)

    evaluations = {
        mode: evaluate_law(
            modules,
            executor=worlds[mode]["executor"],
            law_source=finals[mode]["law_source"],
            history=histories[mode],
        )
        for mode in ("myopic", "nonmyopic")
    }
    concepts = {
        mode: concept_score_extra_dimensions(
            finals[mode]["explanation"] + "\n" + finals[mode]["law_source"]
        )
        for mode in ("myopic", "nonmyopic")
    }
    usage = adapter.usage_snapshot()
    myopic_root = initial_tree["candidates"][initial_indices["myopic"]]
    nonmyopic_root = initial_tree["candidates"][initial_indices["nonmyopic"]]
    gates = {
        "exact_5_requests": usage["adapter_requests"] == 5,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "within_0_50_run_cap": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "initial_roots_differ": (
            initial_indices["myopic"] != initial_indices["nonmyopic"]
        ),
        "nonmyopic_sacrifices_immediate_eig": (
            nonmyopic_root["immediate_eig"] < myopic_root["immediate_eig"]
        ),
        "nonmyopic_has_higher_terminal_eig": (
            nonmyopic_root["terminal_eig"] > myopic_root["terminal_eig"]
        ),
        "realized_histories_differ": (
            histories["myopic"][0]["experiment"]
            != histories["nonmyopic"][0]["experiment"]
        ),
        "both_laws_finite_on_official_cases": all(
            math.isfinite(
                evaluations[mode]["official_default"]["mean_pos_error"]
            )
            for mode in ("myopic", "nonmyopic")
        ),
        "both_laws_finite_on_crossover_stress": all(
            math.isfinite(
                evaluations[mode]["crossover_stress"]["mean_pos_error"]
            )
            for mode in ("myopic", "nonmyopic")
        ),
    }
    gates["all_mechanics_pass"] = all(gates.values())
    scientific = {
        "nonmyopic_lower_crossover_mse": (
            evaluations["nonmyopic"]["crossover_stress"]["mean_pos_error"]
            < evaluations["myopic"]["crossover_stress"]["mean_pos_error"]
        ),
        "nonmyopic_concept_score_not_lower": (
            concepts["nonmyopic"]["score"] >= concepts["myopic"]["score"]
        ),
    }
    scientific["all_pass"] = all(scientific.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "passed"
            if gates["all_mechanics_pass"] and scientific["all_pass"]
            else "gate_failed"
        ),
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "discoverphysics_commit": DISCOVERPHYSICS_COMMIT,
            "model": MODEL_ID,
            "reasoning_enabled": False,
            "world_hidden_from_policy": True,
            "world": WORLD,
            "seed": SEED,
            "noise_std": NOISE_STD,
            "num_hypotheses": NUM_HYPOTHESES,
            "num_candidates": NUM_CANDIDATES,
            "num_branches": NUM_BRANCHES,
            "num_experiment_rounds": NUM_EXPERIMENT_ROUNDS,
            "run_budget_usd": RUN_BUDGET_USD,
            "stress_test_cases_sha256": hashlib.sha256(
                json.dumps(
                    STRESS_TEST_CASES,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        },
        "initial_tree": initial_tree,
        "initial_selection": initial_indices,
        "round2_trees": round2_trees,
        "histories": histories,
        "finals": finals,
        "evaluations": evaluations,
        "concepts": concepts,
        "gates": gates,
        "scientific_gates": scientific,
        "usage": usage,
        "raw_responses_sha256": sha256_file(raw_path),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discoverphysics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "SMOKE.json"
    failure_path = args.output_dir / "FAILURE.json"
    try:
        payload = run_smoke(
            discoverphysics_root=args.discoverphysics_root,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        failure_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "initial_selection": payload["initial_selection"],
                "gates": payload["gates"],
                "scientific_gates": payload["scientific_gates"],
                "mse": {
                    mode: {
                        endpoint: payload["evaluations"][mode][endpoint][
                            "mean_pos_error"
                        ]
                        for endpoint in ("official_default", "crossover_stress")
                    }
                    for mode in ("myopic", "nonmyopic")
                },
                "concept_scores": {
                    mode: payload["concepts"][mode]["score"]
                    for mode in ("myopic", "nonmyopic")
                },
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
