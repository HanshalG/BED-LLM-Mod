#!/usr/bin/env python3
"""Gate LLM-native lookahead over regenerated DiscoverPhysics beliefs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from openrouter_model import OpenRouterAdapter


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-oscillator-belief-compiler-1"
DISCOVERPHYSICS_COMMIT = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
MODEL_ID = "openai/gpt-5.4"
WORLD = "oscillator"
SEED = 24368
NUM_HYPOTHESES = 5
NUM_ROOTS = 4
NUM_BRANCHES = 2
EXPECTED_REQUESTS = 10
TREE_MAX_TOKENS = 3600
REFRESH_MAX_TOKENS = 1500
SCORER_MAX_TOKENS = 900
RUN_BUDGET_USD = 0.50
PROJECTED_COST_USD = 0.22
MIN_IMMEDIATE_SACRIFICE_NATS = 0.01
MIN_EXPECTED_COVERAGE_GAIN = 0.05
MIN_SCORER_GAIN = 5

ACTION_FIELDS = frozenset(
    {
        "separation",
        "initial_motion",
        "start_phase",
        "source_strength",
        "probe_inertia",
    }
)
ACTION_OPTIONS = {
    "separation": ("near", "medium", "far"),
    "initial_motion": ("rest", "tangential_slow", "radial_outward"),
    "start_phase": ("phase_0", "phase_1", "phase_2", "phase_3"),
    "source_strength": ("low", "high"),
    "probe_inertia": ("light", "heavy"),
}
TREE_FIELDS = frozenset({"prior", "candidates"})
BELIEF_FIELDS = frozenset({"hypotheses", "probabilities"})
CANDIDATE_FIELDS = frozenset({"id", "action", "branches"})
BRANCH_FIELDS = frozenset(
    {"probability", "observation", "posterior_probabilities"}
)
REFRESH_FIELDS = frozenset(
    {
        "hypotheses",
        "probabilities",
        "coverage_probability",
        "continuation_action",
        "expected_learning",
    }
)
SCORER_FIELDS = frozenset({"root_scores"})
MEASUREMENT_TIMES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


class NonReasoningOpenRouterAdapter(OpenRouterAdapter):
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


class SmokeExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


def canonical_text(value: str) -> str:
    return " ".join(value.casefold().split())


def strict_json_object(response: str, *, label: str) -> dict[str, Any]:
    stripped = response.strip()
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not exact JSON: {exc}") from exc
    if stripped[end:].strip():
        raise ValueError(f"{label} has trailing content")
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


def probabilities(value: Any, *, count: int, label: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value
        )
    ):
        raise ValueError(f"{label} must contain exactly {count} numbers")
    parsed = [float(item) for item in value]
    if any(not math.isfinite(item) or item < 0.0 for item in parsed):
        raise ValueError(f"{label} contains an invalid probability")
    total = sum(parsed)
    if total <= 0.0 or abs(total - 1.0) > 0.02:
        raise ValueError(f"{label} sums to {total}, not one")
    return [item / total for item in parsed]


def bounded_probability(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be in [0,1]")
    return parsed


def validate_hypotheses(
    value: Any,
    *,
    label: str,
) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != NUM_HYPOTHESES
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(
            f"{label} must contain exactly {NUM_HYPOTHESES} strings"
        )
    parsed = [item.strip() for item in value]
    if len({canonical_text(item) for item in parsed}) != NUM_HYPOTHESES:
        raise ValueError(f"{label} contains duplicate hypotheses")
    return parsed


def validate_belief(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != BELIEF_FIELDS:
        raise ValueError(f"{label} has the wrong fields")
    return {
        "hypotheses": validate_hypotheses(
            value["hypotheses"], label=f"{label}.hypotheses"
        ),
        "probabilities": probabilities(
            value["probabilities"],
            count=NUM_HYPOTHESES,
            label=f"{label}.probabilities",
        ),
    }


def validate_action(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != ACTION_FIELDS:
        raise ValueError(f"{label} has the wrong fields")
    parsed: dict[str, str] = {}
    for field in sorted(ACTION_FIELDS):
        option = value[field]
        if option not in ACTION_OPTIONS[field]:
            raise ValueError(f"{label}.{field} has invalid option {option!r}")
        parsed[field] = option
    return parsed


def compile_action(action: dict[str, str]) -> dict[str, Any]:
    separation = {"near": 0.75, "medium": 2.0, "far": 5.0}[
        action["separation"]
    ]
    velocity = {
        "rest": [0.0, 0.0],
        "tangential_slow": [0.0, 0.25],
        "radial_outward": [0.25, 0.0],
    }[action["initial_motion"]]
    return {
        "p1": {"low": 0.5, "high": 2.0}[action["source_strength"]],
        "p2": {"light": 0.5, "heavy": 2.0}[action["probe_inertia"]],
        "pos2": [separation, 0.0],
        "velocity2": velocity,
        "measurement_times": list(MEASUREMENT_TIMES),
        "start_time": {
            "phase_0": 0.0,
            "phase_1": 1.0,
            "phase_2": 2.0,
            "phase_3": 3.0,
        }[action["start_phase"]],
    }


def parse_tree(response: str) -> dict[str, Any]:
    value = strict_json_object(response, label="initial tree")
    if set(value) != TREE_FIELDS:
        raise ValueError("initial tree has the wrong top-level fields")
    prior = validate_belief(value["prior"], label="prior")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) != NUM_ROOTS:
        raise ValueError(f"tree must contain exactly {NUM_ROOTS} roots")
    parsed_candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_actions: set[str] = set()
    for root_index, candidate in enumerate(candidates):
        label = f"candidates[{root_index}]"
        if not isinstance(candidate, dict) or set(candidate) != CANDIDATE_FIELDS:
            raise ValueError(f"{label} has the wrong fields")
        candidate_id = candidate["id"]
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or candidate_id.strip() in seen_ids
        ):
            raise ValueError(f"{label}.id is invalid or duplicated")
        candidate_id = candidate_id.strip()
        seen_ids.add(candidate_id)
        action = validate_action(candidate["action"], label=f"{label}.action")
        action_key = json.dumps(action, sort_keys=True)
        if action_key in seen_actions:
            raise ValueError("root actions are duplicated")
        seen_actions.add(action_key)
        branches = candidate["branches"]
        if not isinstance(branches, list) or len(branches) != NUM_BRANCHES:
            raise ValueError(f"{label} must contain exactly two branches")
        branch_weights = probabilities(
            [
                branch.get("probability")
                if isinstance(branch, dict)
                else None
                for branch in branches
            ],
            count=NUM_BRANCHES,
            label=f"{label}.branch_probabilities",
        )
        parsed_branches: list[dict[str, Any]] = []
        observations: set[str] = set()
        for branch_index, (branch, weight) in enumerate(
            zip(branches, branch_weights, strict=True)
        ):
            branch_label = f"{label}.branches[{branch_index}]"
            if not isinstance(branch, dict) or set(branch) != BRANCH_FIELDS:
                raise ValueError(f"{branch_label} has the wrong fields")
            observation = branch["observation"]
            if not isinstance(observation, str) or not observation.strip():
                raise ValueError(f"{branch_label}.observation is empty")
            observation = observation.strip()
            observation_key = canonical_text(observation)
            if observation_key in observations:
                raise ValueError(f"{label} has duplicate observations")
            observations.add(observation_key)
            parsed_branches.append(
                {
                    "probability": weight,
                    "observation": observation,
                    "posterior_probabilities": probabilities(
                        branch["posterior_probabilities"],
                        count=NUM_HYPOTHESES,
                        label=f"{branch_label}.posterior_probabilities",
                    ),
                }
            )
        expected_entropy = sum(
            branch["probability"]
            * entropy(branch["posterior_probabilities"])
            for branch in parsed_branches
        )
        parsed_candidates.append(
            {
                "id": candidate_id,
                "action": action,
                "compiled_experiment": compile_action(action),
                "branches": parsed_branches,
                "immediate_eig": entropy(prior["probabilities"])
                - expected_entropy,
            }
        )
    return {
        "prior": prior,
        "prior_entropy": entropy(prior["probabilities"]),
        "candidates": parsed_candidates,
    }


def parse_refresh(
    response: str,
    *,
    root_action: dict[str, str],
    label: str,
) -> dict[str, Any]:
    value = strict_json_object(response, label=label)
    if set(value) != REFRESH_FIELDS:
        raise ValueError(f"{label} has the wrong fields")
    continuation = validate_action(
        value["continuation_action"],
        label=f"{label}.continuation_action",
    )
    if continuation == root_action:
        raise ValueError(f"{label} repeats the root action")
    expected_learning = value["expected_learning"]
    if (
        not isinstance(expected_learning, str)
        or not expected_learning.strip()
    ):
        raise ValueError(f"{label}.expected_learning is empty")
    return {
        "hypotheses": validate_hypotheses(
            value["hypotheses"], label=f"{label}.hypotheses"
        ),
        "probabilities": probabilities(
            value["probabilities"],
            count=NUM_HYPOTHESES,
            label=f"{label}.probabilities",
        ),
        "coverage_probability": bounded_probability(
            value["coverage_probability"],
            label=f"{label}.coverage_probability",
        ),
        "continuation_action": continuation,
        "compiled_continuation": compile_action(continuation),
        "expected_learning": expected_learning.strip(),
    }


def parse_scorer(response: str) -> dict[str, Any]:
    value = strict_json_object(response, label="root scorer")
    if set(value) != SCORER_FIELDS:
        raise ValueError("root scorer has the wrong fields")
    scores = value["root_scores"]
    if (
        not isinstance(scores, list)
        or len(scores) != NUM_ROOTS
        or any(
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= 100
            for score in scores
        )
    ):
        raise ValueError("root_scores must be four integers in [0,100]")
    return {"root_scores": list(scores)}


def action_schema_example() -> dict[str, str]:
    return {
        "separation": "near|medium|far",
        "initial_motion": "rest|tangential_slow|radial_outward",
        "start_phase": "phase_0|phase_1|phase_2|phase_3",
        "source_strength": "low|high",
        "probe_inertia": "light|heavy",
    }


def apparatus_prompt() -> str:
    return "\n".join(
        [
            "The unknown environment is a two-particle 2D simulator.",
            "Particle 1 is fixed at the origin and particle 2 is mobile.",
            "The hidden law may be noncanonical, time-varying, or depend on",
            "source strength, inertia, distance, velocity, or absolute start time.",
            "Each experiment reports ten noisy positions and velocities from",
            "relative times 0.5 through 5.0. A compiler handles numeric values.",
            "The start_phase options shift the experiment's absolute starting clock",
            "by 0, 1, 2, or 3 time units while preserving its initial configuration.",
            "Do not name or guess a benchmark/world. Infer semantic law families.",
        ]
    )


def tree_messages() -> list[dict[str, str]]:
    schema = {
        "prior": {
            "hypotheses": ["exactly five distinct semantic force laws"],
            "probabilities": ["exactly five numbers summing to one"],
        },
        "candidates": [
            {
                "id": "R1",
                "action": action_schema_example(),
                "branches": [
                    {
                        "probability": 0.5,
                        "observation": "qualitatively distinct predicted trajectory",
                        "posterior_probabilities": [
                            "five numbers over the initial hypotheses"
                        ],
                    }
                ],
            }
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a scientific hypothesis and experimental-design model. "
                "Return one exact JSON object only, without markdown, comments, "
                "reasoning, or extra fields."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    apparatus_prompt(),
                    "",
                    "Generate exactly five current law hypotheses and probabilities.",
                    "Generate exactly four distinct conceptual root experiments.",
                    "Each root has exactly two mutually exclusive qualitative outcome",
                    "branches, probabilities summing to one, and a posterior over the",
                    "same five initial hypotheses. Roots should test different physical",
                    "mechanisms, not superficial numeric variants.",
                    "Use only the listed action option strings.",
                    "Return exactly this structure, expanding required lists:",
                    json.dumps(schema, separators=(",", ":")),
                ]
            ),
        },
    ]


def refresh_messages(
    tree: dict[str, Any],
    *,
    root_index: int,
    branch_index: int,
) -> list[dict[str, str]]:
    root = tree["candidates"][root_index]
    branch = root["branches"][branch_index]
    schema = {
        "hypotheses": ["exactly five freshly regenerated semantic laws"],
        "probabilities": ["exactly five numbers summing to one"],
        "coverage_probability": 0.65,
        "continuation_action": action_schema_example(),
        "expected_learning": "one concise sentence",
    }
    return [
        {
            "role": "system",
            "content": (
                "You regenerate a scientific belief state after a hypothetical "
                "experiment outcome. Return one exact JSON object only. Do not "
                "include markdown, comments, reasoning, or extra fields."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    apparatus_prompt(),
                    "",
                    "CURRENT_BELIEF="
                    + json.dumps(tree["prior"], separators=(",", ":")),
                    "ROOT_ACTION="
                    + json.dumps(root["action"], separators=(",", ":")),
                    "HYPOTHETICAL_OUTCOME=" + branch["observation"],
                    "POSTERIOR_OVER_OLD_SUPPORT="
                    + json.dumps(
                        branch["posterior_probabilities"],
                        separators=(",", ":"),
                    ),
                    "",
                    "Regenerate the law support from the complete hypothetical history;",
                    "do not merely rename or reorder the old hypotheses. Then choose the",
                    "single best different continuation experiment.",
                    "coverage_probability is your calibrated probability that the actual",
                    "unknown law is represented well enough by at least one of your five",
                    "fresh hypotheses to support a correct final executable law after the",
                    "continuation. It is not the probability of your top hypothesis.",
                    "Use only listed action strings. Return exactly:",
                    json.dumps(schema, separators=(",", ":")),
                ]
            ),
        },
    ]


def scorer_messages(
    tree: dict[str, Any],
    refreshes: list[list[dict[str, Any]]],
) -> list[dict[str, str]]:
    roots = []
    for root_index, root in enumerate(tree["candidates"]):
        roots.append(
            {
                "label": chr(ord("A") + root_index),
                "root_action": root["action"],
                "branches": [
                    {
                        "probability": branch["probability"],
                        "predicted_observation": branch["observation"],
                        "refreshed_hypotheses": refreshes[root_index][
                            branch_index
                        ]["hypotheses"],
                        "refreshed_probabilities": refreshes[root_index][
                            branch_index
                        ]["probabilities"],
                        "coverage_probability": refreshes[root_index][
                            branch_index
                        ]["coverage_probability"],
                        "continuation_action": refreshes[root_index][
                            branch_index
                        ]["continuation_action"],
                        "expected_learning": refreshes[root_index][
                            branch_index
                        ]["expected_learning"],
                    }
                    for branch_index, branch in enumerate(root["branches"])
                ],
            }
        )
    return [
        {
            "role": "system",
            "content": (
                "You score complete two-experiment scientific belief pathways. "
                "Return one exact JSON object only, without explanation, markdown, "
                "reasoning, or extra fields."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    apparatus_prompt(),
                    "",
                    "Score each root A-D from 0 to 100 for the probability that its",
                    "outcome-conditioned continuation will leave the agent able to state",
                    "and execute the correct hidden law. Judge semantic coverage, branch",
                    "robustness, and whether the continuation resolves remaining",
                    "ambiguity. Do not favor the first label or average the scores.",
                    "You are not shown any immediate information-gain score.",
                    "Return exactly {\"root_scores\":[A,B,C,D]} with integers.",
                    "PATHWAYS=" + json.dumps(roots, separators=(",", ":")),
                ]
            ),
        },
    ]


def verify_discoverphysics(root: Path) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != DISCOVERPHYSICS_COMMIT:
        raise ValueError(
            f"DiscoverPhysics is at {commit}, expected {DISCOVERPHYSICS_COMMIT}"
        )
    return commit


def _adapter(
    *,
    run_id: str,
    output_dir: Path,
) -> NonReasoningOpenRouterAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=105.38480269545715,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=8,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=TREE_MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    spec = ModelSpec(
        model=MODEL_ID,
        backend="openrouter",
        max_model_len=65536,
    )
    return NonReasoningOpenRouterAdapter(spec, config)


def checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_smoke(
    *,
    discoverphysics_root: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    commit = verify_discoverphysics(discoverphysics_root)
    adapter = _adapter(run_id=run_id, output_dir=output_dir)
    raw_path = output_dir / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {
        "initial_tree": None,
        "refreshes": [],
        "scorer": None,
    }
    try:
        initial_response = adapter.chat_complete_messages_batched(
            [tree_messages()],
            temperature=0.0,
            block_size=1,
            max_new_tokens=TREE_MAX_TOKENS,
        )[0]
        raw["initial_tree"] = initial_response
        checkpoint(raw_path, raw)
        tree = parse_tree(initial_response)

        refresh_requests = [
            refresh_messages(
                tree,
                root_index=root_index,
                branch_index=branch_index,
            )
            for root_index in range(NUM_ROOTS)
            for branch_index in range(NUM_BRANCHES)
        ]
        refresh_responses = adapter.chat_complete_messages_batched(
            refresh_requests,
            temperature=0.0,
            block_size=8,
            max_new_tokens=REFRESH_MAX_TOKENS,
        )
        raw["refreshes"] = list(refresh_responses)
        checkpoint(raw_path, raw)
        parsed_flat = [
            parse_refresh(
                response,
                root_action=tree["candidates"][index // NUM_BRANCHES][
                    "action"
                ],
                label=f"refresh[{index}]",
            )
            for index, response in enumerate(refresh_responses)
        ]
        refreshes = [
            parsed_flat[
                root_index * NUM_BRANCHES : (root_index + 1) * NUM_BRANCHES
            ]
            for root_index in range(NUM_ROOTS)
        ]

        scorer_response = adapter.chat_complete_messages_batched(
            [scorer_messages(tree, refreshes)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=SCORER_MAX_TOKENS,
        )[0]
        raw["scorer"] = scorer_response
        checkpoint(raw_path, raw)
        scorer = parse_scorer(scorer_response)
    except Exception as exc:
        raise SmokeExecutionError(
            f"{type(exc).__name__}: {exc}", adapter.usage_snapshot()
        ) from exc

    immediate_values = [
        candidate["immediate_eig"] for candidate in tree["candidates"]
    ]
    expected_coverages = [
        sum(
            branch["probability"]
            * refreshes[root_index][branch_index]["coverage_probability"]
            for branch_index, branch in enumerate(candidate["branches"])
        )
        for root_index, candidate in enumerate(tree["candidates"])
    ]
    myopic_root = max(
        range(NUM_ROOTS),
        key=lambda index: (immediate_values[index], -index),
    )
    root_scores = scorer["root_scores"]
    model_root = max(
        range(NUM_ROOTS),
        key=lambda index: (root_scores[index], -index),
    )
    score_maximizers = [
        index for index, score in enumerate(root_scores) if score == max(root_scores)
    ]
    initial_support = {
        canonical_text(hypothesis) for hypothesis in tree["prior"]["hypotheses"]
    }
    refreshed_supports = [
        {canonical_text(hypothesis) for hypothesis in refresh["hypotheses"]}
        for pair in refreshes
        for refresh in pair
    ]
    changed_from_initial = sum(
        support != initial_support for support in refreshed_supports
    )
    branch_distinct_roots = sum(
        refreshed_supports[2 * root_index]
        != refreshed_supports[2 * root_index + 1]
        for root_index in range(NUM_ROOTS)
    )
    usage = adapter.usage_snapshot()
    gates = {
        "exact_10_requests": usage["adapter_requests"] == EXPECTED_REQUESTS,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "all_compiled_experiments_legal": all(
            len(candidate["compiled_experiment"]["measurement_times"]) == 10
            and candidate["compiled_experiment"]["measurement_times"][-1] >= 5.0
            and all(
                len(refresh["compiled_continuation"]["measurement_times"])
                == 10
                and refresh["compiled_continuation"]["measurement_times"][-1]
                >= 5.0
                for refresh in refreshes[root_index]
            )
            for root_index, candidate in enumerate(tree["candidates"])
        ),
        "at_least_6_refreshes_change_support": changed_from_initial >= 6,
        "at_least_3_roots_have_branch_distinct_supports": (
            branch_distinct_roots >= 3
        ),
        "coverage_signal_varies_by_at_least_0_05": (
            max(expected_coverages) - min(expected_coverages)
            >= MIN_EXPECTED_COVERAGE_GAIN
        ),
        "scorer_scores_vary": len(set(root_scores)) >= 2,
        "scorer_has_unique_maximum": len(score_maximizers) == 1,
    }
    scientific_gates = {
        "model_aware_root_differs_from_myopic": model_root != myopic_root,
        "model_aware_sacrifices_immediate_eig": (
            immediate_values[myopic_root] - immediate_values[model_root]
            >= MIN_IMMEDIATE_SACRIFICE_NATS
        ),
        "model_aware_expected_coverage_gain_at_least_0_05": (
            expected_coverages[model_root] - expected_coverages[myopic_root]
            >= MIN_EXPECTED_COVERAGE_GAIN
        ),
        "model_aware_scorer_gain_at_least_5": (
            root_scores[model_root] - root_scores[myopic_root]
            >= MIN_SCORER_GAIN
        ),
    }
    all_gates_pass = all(gates.values()) and all(scientific_gates.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all_gates_pass else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "discoverphysics_commit": commit,
            "world": WORLD,
            "world_hidden_from_model": True,
            "model": MODEL_ID,
            "reasoning_enabled": False,
            "seed": SEED,
            "num_hypotheses": NUM_HYPOTHESES,
            "num_roots": NUM_ROOTS,
            "num_branches": NUM_BRANCHES,
            "expected_requests": EXPECTED_REQUESTS,
            "run_budget_usd": RUN_BUDGET_USD,
            "projected_cost_usd": PROJECTED_COST_USD,
            "minimum_immediate_sacrifice_nats": (
                MIN_IMMEDIATE_SACRIFICE_NATS
            ),
            "minimum_expected_coverage_gain": MIN_EXPECTED_COVERAGE_GAIN,
            "minimum_scorer_gain": MIN_SCORER_GAIN,
            "action_options": ACTION_OPTIONS,
            "measurement_times": list(MEASUREMENT_TIMES),
        },
        "tree": tree,
        "refreshes": refreshes,
        "scorer": scorer,
        "selection": {
            "myopic_root": myopic_root,
            "model_aware_root": model_root,
            "immediate_eig_nats": immediate_values,
            "expected_coverage": expected_coverages,
            "root_scores": root_scores,
            "immediate_sacrifice_nats": (
                immediate_values[myopic_root] - immediate_values[model_root]
            ),
            "expected_coverage_gain": (
                expected_coverages[model_root]
                - expected_coverages[myopic_root]
            ),
            "scorer_gain": (
                root_scores[model_root] - root_scores[myopic_root]
            ),
        },
        "mechanism": {
            "refreshes_changed_from_initial": changed_from_initial,
            "roots_with_branch_distinct_supports": branch_distinct_roots,
        },
        "gates": gates,
        "scientific_gates": scientific_gates,
        "all_gates_pass": all_gates_pass,
        "simulator_calls": 0,
        "policy_endpoint_exists": False,
        "usage": usage,
        "raw_responses_sha256": sha256_file(raw_path),
    }


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
            discoverphysics_root=args.discoverphysics_root.resolve(),
            output_dir=args.output_dir.resolve(),
            run_id=args.run_id,
        )
    except SmokeExecutionError as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": str(exc),
            "usage": exc.usage,
            "simulator_calls": 0,
            "policy_endpoint_exists": False,
        }
        checkpoint(failure_path, failure)
        raise
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection": payload["selection"],
                "mechanism": payload["mechanism"],
                "gates": payload["gates"],
                "scientific_gates": payload["scientific_gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
