#!/usr/bin/env python3
"""Gate LLM-native semantic lookahead for DiscoverPhysics dark matter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from openrouter_model import OpenRouterAdapter
from scripts.discoverphysics_dark_matter_opportunity import (
    active_probe_actions,
    verify_discoverphysics,
)
from scripts.discoverphysics_oscillator_belief_smoke import (
    SmokeExecutionError,
    canonical_text,
    checkpoint,
    entropy,
    probabilities,
    sha256_file,
    strict_json_object,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "discoverphysics-dark-matter-semantic-smoke-1"
DISCOVERPHYSICS_COMMIT = "33b7fa9df96de9c35744efd181ca7e5a8dd60ad5"
MODEL_ID = "openai/gpt-5.4"
SEED = 24506
NUM_HYPOTHESES = 8
NUM_ROOTS = 4
NUM_BRANCHES = 2
EXPECTED_REQUESTS = 10
TREE_MAX_TOKENS = 5000
REFRESH_MAX_TOKENS = 1800
SCORER_MAX_TOKENS = 1000
RUN_BUDGET_USD = 0.20
PROJECTED_COST_USD = 0.10
MIN_IMMEDIATE_SACRIFICE_NATS = 0.03
MIN_SCORER_GAIN = 5
MIN_READINESS_RANGE = 5.0

ROOTS = (
    {"id": "A", "action_id": "r4.5_a3", "position": [-3.182, 3.182]},
    {"id": "B", "action_id": "center", "position": [0.0, 0.0]},
    {"id": "C", "action_id": "r4.5_a5", "position": [-3.182, -3.182]},
    {"id": "D", "action_id": "r4.5_a1", "position": [3.182, 3.182]},
)
MYOPIC_ROOT_ID = "D"
LOOKAHEAD_ROOT_ID = "B"
BLIND_LABELS = {"A": "K", "B": "M", "C": "Q", "D": "T"}
SCORER_ORDER = ("Q", "K", "T", "M")


class NonReasoningOpenRouterAdapter(OpenRouterAdapter):
    """Force this smoke onto the explicit non-reasoning route."""

    def _payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        *,
        disable_reasoning: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = super()._payload(
            messages,
            temperature,
            n,
            max_tokens,
            disable_reasoning=disable_reasoning,
            response_format=response_format,
        )
        payload["reasoning"] = {"enabled": False, "exclude": True}
        return payload


def root_by_id(root_id: str) -> dict[str, Any]:
    return next(root for root in ROOTS if root["id"] == root_id)


def action_table() -> dict[str, list[float]]:
    points, labels = active_probe_actions()
    return {
        label: [round(float(value), 3) for value in point]
        for label, point in zip(labels, points, strict=True)
    }


def parse_hypotheses(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, list) or len(value) != NUM_HYPOTHESES:
        raise ValueError(f"{label} must contain exactly eight hypotheses")
    descriptions: list[str] = []
    masses: list[float] = []
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(item, dict) or set(item) != {
            "description",
            "probability",
        }:
            raise ValueError(f"{item_label} has the wrong fields")
        description = item["description"]
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{item_label}.description is empty")
        descriptions.append(description.strip())
        masses.append(item["probability"])
    if len({canonical_text(item) for item in descriptions}) != len(
        descriptions
    ):
        raise ValueError(f"{label} contains duplicate descriptions")
    return {
        "descriptions": descriptions,
        "probabilities": probabilities(
            masses,
            count=NUM_HYPOTHESES,
            label=f"{label}.probabilities",
        ),
    }


def parse_tree(response: str) -> dict[str, Any]:
    value = strict_json_object(response, label="initial tree")
    if set(value) != {"hypotheses", "roots"}:
        raise ValueError("initial tree has the wrong top-level fields")
    belief = parse_hypotheses(value["hypotheses"], label="hypotheses")
    roots = value["roots"]
    if not isinstance(roots, list) or len(roots) != NUM_ROOTS:
        raise ValueError("initial tree must contain exactly four roots")
    parsed_roots = []
    seen_ids = set()
    for root_index, root in enumerate(roots):
        label = f"roots[{root_index}]"
        if not isinstance(root, dict) or set(root) != {"id", "branches"}:
            raise ValueError(f"{label} has the wrong fields")
        root_id = root["id"]
        if root_id not in {item["id"] for item in ROOTS}:
            raise ValueError(f"{label}.id is invalid")
        if root_id in seen_ids:
            raise ValueError(f"{label}.id is duplicated")
        seen_ids.add(root_id)
        branches = root["branches"]
        if not isinstance(branches, list) or len(branches) != NUM_BRANCHES:
            raise ValueError(f"{label} must contain exactly two branches")
        branch_probabilities = probabilities(
            [branch.get("probability") for branch in branches],
            count=NUM_BRANCHES,
            label=f"{label}.branch_probabilities",
        )
        observations = set()
        parsed_branches = []
        for branch_index, (branch, branch_probability) in enumerate(
            zip(branches, branch_probabilities, strict=True)
        ):
            branch_label = f"{label}.branches[{branch_index}]"
            if not isinstance(branch, dict) or set(branch) != {
                "probability",
                "observation",
                "posterior_probabilities",
            }:
                raise ValueError(f"{branch_label} has the wrong fields")
            observation = branch["observation"]
            if not isinstance(observation, str) or not observation.strip():
                raise ValueError(f"{branch_label}.observation is empty")
            observation = observation.strip()
            normalized = canonical_text(observation)
            if normalized in observations:
                raise ValueError(f"{label} has duplicate observations")
            observations.add(normalized)
            parsed_branches.append(
                {
                    "probability": branch_probability,
                    "observation": observation,
                    "posterior_probabilities": probabilities(
                        branch["posterior_probabilities"],
                        count=NUM_HYPOTHESES,
                        label=(
                            f"{branch_label}.posterior_probabilities"
                        ),
                    ),
                }
            )
        expected_entropy = sum(
            branch["probability"]
            * entropy(branch["posterior_probabilities"])
            for branch in parsed_branches
        )
        parsed_roots.append(
            {
                **root_by_id(root_id),
                "branches": parsed_branches,
                "immediate_eig_nats": (
                    entropy(belief["probabilities"]) - expected_entropy
                ),
            }
        )
    if seen_ids != {item["id"] for item in ROOTS}:
        raise ValueError("initial tree is missing a root")
    parsed_roots.sort(key=lambda item: item["id"])
    return {
        "belief": belief,
        "prior_entropy_nats": entropy(belief["probabilities"]),
        "roots": parsed_roots,
    }


def parse_refresh(
    response: str,
    *,
    root_action_id: str,
    label: str,
) -> dict[str, Any]:
    value = strict_json_object(response, label=label)
    if set(value) != {
        "hypotheses",
        "continuation_action",
        "expected_map_readiness",
        "expected_learning",
    }:
        raise ValueError(f"{label} has the wrong fields")
    continuation = value["continuation_action"]
    if continuation not in action_table():
        raise ValueError(f"{label}.continuation_action is invalid")
    if continuation == root_action_id:
        raise ValueError(f"{label} repeats the root action")
    readiness = value["expected_map_readiness"]
    if (
        isinstance(readiness, bool)
        or not isinstance(readiness, int)
        or not 0 <= readiness <= 100
    ):
        raise ValueError(
            f"{label}.expected_map_readiness must be an integer in [0,100]"
        )
    expected_learning = value["expected_learning"]
    if (
        not isinstance(expected_learning, str)
        or not expected_learning.strip()
    ):
        raise ValueError(f"{label}.expected_learning is empty")
    return {
        "belief": parse_hypotheses(
            value["hypotheses"],
            label=f"{label}.hypotheses",
        ),
        "continuation_action": continuation,
        "continuation_position": action_table()[continuation],
        "expected_map_readiness": readiness,
        "expected_learning": expected_learning.strip(),
    }


def parse_scorer(response: str) -> dict[str, int]:
    value = strict_json_object(response, label="pathway scorer")
    if set(value) != {"scores"} or not isinstance(value["scores"], dict):
        raise ValueError("pathway scorer has the wrong fields")
    scores = value["scores"]
    if set(scores) != set(BLIND_LABELS.values()):
        raise ValueError("pathway scorer has the wrong blind labels")
    if any(
        isinstance(score, bool)
        or not isinstance(score, int)
        or not 0 <= score <= 100
        for score in scores.values()
    ):
        raise ValueError("pathway scores must be integers in [0,100]")
    return {
        root_id: scores[blind_label]
        for root_id, blind_label in BLIND_LABELS.items()
    }


def apparatus_prompt() -> str:
    return "\n".join(
        [
            "You are designing sequential experiments in an unknown 2D field.",
            "The field contains ten concealed positive sources forming one",
            "compact or elongated halo. The halo lies in exactly one region:",
            "northeast probability .40, northwest .30, southwest .20,",
            "southeast .10. Within a region, location, spread, and orientation",
            "are unknown. Coordinates use +x east and +y north.",
            "One neutral probe starts at rest at a chosen coordinate. At t=.5",
            "you observe its noisy final coordinate. A nearby halo can reveal",
            "fine geometry; a central probe more coarsely reveals direction.",
            "After observing the first probe, one different probe is allowed.",
            "The final goal is to predict unseen probe trajectories through t=5.",
            "Do not name a benchmark or assume access to a finite map bank.",
        ]
    )


def tree_messages() -> list[dict[str, str]]:
    schema = {
        "hypotheses": [
            {
                "description": "free-form semantic halo configuration",
                "probability": 0.125,
            }
        ],
        "roots": [
            {
                "id": "A|B|C|D",
                "branches": [
                    {
                        "probability": 0.5,
                        "observation": "qualitative noisy probe outcome",
                        "posterior_probabilities": [
                            "exactly eight numbers summing to one"
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
                "You generate scientific hypotheses and qualitative "
                "likelihoods. Return one exact JSON object only, without "
                "markdown, comments, reasoning, or extra fields."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    apparatus_prompt(),
                    "",
                    "CANDIDATE_ROOTS="
                    + json.dumps(list(ROOTS), separators=(",", ":")),
                    "",
                    "Generate exactly eight distinct semantic halo-map",
                    "hypotheses spanning region and within-region geometry,",
                    "with probabilities summing to one. For each fixed root,",
                    "predict exactly two mutually exclusive qualitative",
                    "outcomes, their probabilities, and the posterior over the",
                    "same eight hypotheses. Preserve root IDs A-D exactly.",
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
    root = tree["roots"][root_index]
    branch = root["branches"][branch_index]
    schema = {
        "hypotheses": [
            {
                "description": "fresh semantic halo configuration",
                "probability": 0.125,
            }
        ],
        "continuation_action": "one action ID from ACTION_TABLE",
        "expected_map_readiness": 70,
        "expected_learning": "one concise sentence",
    }
    return [
        {
            "role": "system",
            "content": (
                "You regenerate a scientific belief after a hypothetical "
                "probe outcome. Return one exact JSON object only, without "
                "markdown, comments, reasoning, or extra fields."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    apparatus_prompt(),
                    "",
                    "INITIAL_BELIEF="
                    + json.dumps(tree["belief"], separators=(",", ":")),
                    "ROOT="
                    + json.dumps(
                        {
                            key: root[key]
                            for key in ("id", "action_id", "position")
                        },
                        separators=(",", ":"),
                    ),
                    "HYPOTHETICAL_OUTCOME=" + branch["observation"],
                    "POSTERIOR_OVER_OLD_SUPPORT="
                    + json.dumps(
                        branch["posterior_probabilities"],
                        separators=(",", ":"),
                    ),
                    "ACTION_TABLE="
                    + json.dumps(action_table(), separators=(",", ":")),
                    "",
                    "Regenerate exactly eight free-form halo hypotheses from",
                    "the complete hypothetical history; do not just reorder or",
                    "rename the old support. Choose the single best different",
                    "continuation action. expected_map_readiness is the",
                    "probability-like 0-100 readiness to predict held-out",
                    "trajectories after that continuation, accounting for both",
                    "support coverage and localization precision.",
                    "Return exactly:",
                    json.dumps(schema, separators=(",", ":")),
                ]
            ),
        },
    ]


def scorer_messages(
    tree: dict[str, Any],
    refreshes: list[list[dict[str, Any]]],
) -> list[dict[str, str]]:
    by_blind_label = {}
    for root_index, root in enumerate(tree["roots"]):
        blind_label = BLIND_LABELS[root["id"]]
        by_blind_label[blind_label] = {
            "label": blind_label,
            "root_position": root["position"],
            "branches": [
                {
                    "probability": branch["probability"],
                    "predicted_observation": branch["observation"],
                    "refreshed_hypotheses": refreshes[root_index][
                        branch_index
                    ]["belief"],
                    "continuation_position": refreshes[root_index][
                        branch_index
                    ]["continuation_position"],
                    "expected_map_readiness": refreshes[root_index][
                        branch_index
                    ]["expected_map_readiness"],
                    "expected_learning": refreshes[root_index][branch_index][
                        "expected_learning"
                    ],
                }
                for branch_index, branch in enumerate(root["branches"])
            ],
        }
    pathways = [by_blind_label[label] for label in SCORER_ORDER]
    return [
        {
            "role": "system",
            "content": (
                "You compare complete two-probe scientific pathways. Return "
                "one exact JSON object only, without markdown, comments, "
                "reasoning, or extra fields."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    apparatus_prompt(),
                    "",
                    "PATHWAYS="
                    + json.dumps(pathways, separators=(",", ":")),
                    "",
                    "Score each labeled pathway from 0 to 100 for expected",
                    "final readiness to predict held-out trajectories after",
                    "its branch-adaptive continuation. Consider branch",
                    "probability, support quality, and localization precision.",
                    "Do not score immediate information alone.",
                    'Return exactly {"scores":{"K":0,"M":0,"Q":0,"T":0}}',
                    "with integer scores.",
                ]
            ),
        },
    ]


def _adapter(
    *,
    run_id: str,
    output_dir: Path,
) -> NonReasoningOpenRouterAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=140.0,
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


def support_key(belief: dict[str, Any]) -> frozenset[str]:
    return frozenset(
        canonical_text(description)
        for description in belief["descriptions"]
    )


def quadrant(action_id: str) -> tuple[int, int] | None:
    position = action_table()[action_id]
    if math.isclose(position[0], 0.0) or math.isclose(position[1], 0.0):
        return None
    return (
        1 if position[0] > 0.0 else -1,
        1 if position[1] > 0.0 else -1,
    )


def run_smoke(
    *,
    discoverphysics_root: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    commit = verify_discoverphysics(discoverphysics_root)
    if commit != DISCOVERPHYSICS_COMMIT:
        raise ValueError("DiscoverPhysics commit changed after verification")
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
                root_action_id=tree["roots"][index // NUM_BRANCHES][
                    "action_id"
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
        scores = parse_scorer(scorer_response)
    except Exception as exc:
        raise SmokeExecutionError(
            f"{type(exc).__name__}: {exc}",
            adapter.usage_snapshot(),
        ) from exc

    immediate = {
        root["id"]: root["immediate_eig_nats"] for root in tree["roots"]
    }
    myopic_root = max(immediate, key=lambda root_id: (immediate[root_id], root_id))
    lookahead_root = max(scores, key=lambda root_id: (scores[root_id], root_id))
    initial_support = support_key(tree["belief"])
    refresh_supports = [
        support_key(refresh["belief"])
        for root_refreshes in refreshes
        for refresh in root_refreshes
    ]
    changed_supports = sum(
        support != initial_support for support in refresh_supports
    )
    branch_distinct_roots = sum(
        support_key(refreshes[root_index][0]["belief"])
        != support_key(refreshes[root_index][1]["belief"])
        for root_index in range(NUM_ROOTS)
    )
    center_index = next(
        index
        for index, root in enumerate(tree["roots"])
        if root["id"] == LOOKAHEAD_ROOT_ID
    )
    center_continuations = [
        refresh["continuation_action"] for refresh in refreshes[center_index]
    ]
    center_quadrants = [quadrant(action) for action in center_continuations]
    expected_readiness = {
        root["id"]: sum(
            branch["probability"]
            * refreshes[root_index][branch_index][
                "expected_map_readiness"
            ]
            for branch_index, branch in enumerate(root["branches"])
        )
        for root_index, root in enumerate(tree["roots"])
    }
    usage = adapter.usage_snapshot()
    mechanics_gates = {
        "exact_10_requests": usage["adapter_requests"] == EXPECTED_REQUESTS,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "at_least_6_refreshes_change_support": changed_supports >= 6,
        "at_least_3_roots_have_branch_distinct_supports": (
            branch_distinct_roots >= 3
        ),
        "center_branches_choose_distinct_continuations": (
            len(set(center_continuations)) == NUM_BRANCHES
        ),
        "center_continuations_target_distinct_quadrants": (
            None not in center_quadrants
            and len(set(center_quadrants)) == NUM_BRANCHES
        ),
        "expected_readiness_range_at_least_5": (
            max(expected_readiness.values())
            - min(expected_readiness.values())
            >= MIN_READINESS_RANGE
        ),
        "scorer_scores_vary": len(set(scores.values())) >= 2,
        "scorer_has_unique_maximum": (
            sum(score == max(scores.values()) for score in scores.values())
            == 1
        ),
    }
    scientific_gates = {
        "llm_likelihood_myopic_root_is_northeast_target": (
            myopic_root == MYOPIC_ROOT_ID
        ),
        "complete_path_root_is_center_scout": (
            lookahead_root == LOOKAHEAD_ROOT_ID
        ),
        "center_sacrifices_immediate_eig": (
            immediate[MYOPIC_ROOT_ID] - immediate[LOOKAHEAD_ROOT_ID]
            >= MIN_IMMEDIATE_SACRIFICE_NATS
        ),
        "center_scorer_gain_at_least_5": (
            scores[LOOKAHEAD_ROOT_ID] - scores[MYOPIC_ROOT_ID]
            >= MIN_SCORER_GAIN
        ),
    }
    all_gates_pass = all(mechanics_gates.values()) and all(
        scientific_gates.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all_gates_pass else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "discoverphysics_commit": commit,
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
            "minimum_scorer_gain": MIN_SCORER_GAIN,
            "minimum_readiness_range": MIN_READINESS_RANGE,
            "roots": list(ROOTS),
            "region_prior": [0.40, 0.30, 0.20, 0.10],
        },
        "tree": tree,
        "refreshes": refreshes,
        "scores": scores,
        "selection": {
            "myopic_root": myopic_root,
            "lookahead_root": lookahead_root,
            "immediate_eig_nats": immediate,
            "expected_readiness": expected_readiness,
            "pathway_scores": scores,
            "immediate_sacrifice_nats": (
                immediate[MYOPIC_ROOT_ID] - immediate[LOOKAHEAD_ROOT_ID]
            ),
            "scorer_gain": (
                scores[LOOKAHEAD_ROOT_ID] - scores[MYOPIC_ROOT_ID]
            ),
        },
        "mechanism": {
            "refreshes_changed_from_initial": changed_supports,
            "roots_with_branch_distinct_supports": branch_distinct_roots,
            "center_continuations": center_continuations,
            "center_continuation_quadrants": center_quadrants,
        },
        "mechanics_gates": mechanics_gates,
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
    checkpoint(output_path, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selection": payload["selection"],
                "mechanism": payload["mechanism"],
                "mechanics_gates": payload["mechanics_gates"],
                "scientific_gates": payload["scientific_gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
