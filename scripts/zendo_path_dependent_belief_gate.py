#!/usr/bin/env python3
"""Test model-aware lookahead over LLM-revised Zendo rule particles."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Callable, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter


MODEL_ID = "openai/gpt-5.4"
INTERFACE_VERSION = "zendo-path-dependent-belief-1"
SELECTION_SEED = 24349
PARTICLE_COUNT = 12
SCENE_COUNT = 8
ROOT_COUNT = 4
AUDIT_RANDOM_SCENES = 512
OBSERVATION_ACCURACY = 0.95
EXPECTED_REQUESTS = {"smoke": 20, "opportunity": 80}
COST_CAP_USD = {"smoke": 0.75, "opportunity": 2.00}
TASKS = {
    "smoke": ("zeta", "xi"),
    "opportunity": (
        "phi",
        "upsilon",
        "iota",
        "kappa",
        "omega",
        "mu",
        "nu",
        "psi",
    ),
}
RULE_ORDER = (
    "zeta",
    "phi",
    "upsilon",
    "iota",
    "kappa",
    "omega",
    "mu",
    "nu",
    "xi",
    "psi",
)
SOURCE_COMMIT = "af07590c4f4f617a79791e173460e5a4322b727f"
CASES_SHA256 = "6440c543ff491af13606b79c57384281fae4e8bc205366e67be06d1c81fbacbc"
COLORS = ("blue", "red", "green")
SIZES = ("small", "medium", "large")
ORIENTATIONS = ("upright", "left", "right", "strange")
SIZE_ORDER = {"small": 0, "medium": 1, "large": 2}

RULE_DSL = {
    "block_predicate": (
        '{"op":"any"} | {"op":"attribute","attribute":'
        '"color|size|orientation|grounded","value":...} | '
        '{"op":"and|or","args":[P,...]} | {"op":"not","arg":P}'
    ),
    "rule": (
        '{"op":"exists|forall","predicate":P} | '
        '{"op":"count","predicate":P,"comparison":"eq|ge|le","value":0..6} | '
        '{"op":"all_same","attribute":"color|size|orientation|grounded"} | '
        '{"op":"touching","left":P,"right":P} | '
        '{"op":"stacking","upper":P,"lower":P} | '
        '{"op":"largest_all","predicate":P} | '
        '{"op":"and|or","rules":[R,...]} | {"op":"not","rule":R}'
    ),
}


class GateExecutionError(RuntimeError):
    """A failed-closed gate with usage accounting attached."""

    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(source_dir: Path) -> Path:
    cases_path = source_dir / "data" / "zendo_cases.json"
    if _sha256(cases_path) != CASES_SHA256:
        raise ValueError("official Zendo cases hash mismatch")
    commit = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"official Zendo checkout is {commit}, expected {SOURCE_COMMIT}")
    return cases_path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _strict_json_object(text: str) -> dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("response must be one JSON object")
    return payload


def validate_predicate(value: Any, *, depth: int = 0) -> dict[str, Any]:
    if depth > 5 or not isinstance(value, dict):
        raise ValueError("invalid block predicate")
    op = value.get("op")
    if op == "any":
        if set(value) != {"op"}:
            raise ValueError("any predicate has extra fields")
        return {"op": "any"}
    if op == "attribute":
        if set(value) != {"op", "attribute", "value"}:
            raise ValueError("attribute predicate has wrong fields")
        attribute = value["attribute"]
        allowed: dict[str, tuple[Any, ...]] = {
            "color": COLORS,
            "size": SIZES,
            "orientation": ORIENTATIONS,
            "grounded": (True, False),
        }
        if attribute not in allowed or value["value"] not in allowed[attribute]:
            raise ValueError("invalid predicate attribute or value")
        return {
            "op": "attribute",
            "attribute": attribute,
            "value": value["value"],
        }
    if op in {"and", "or"}:
        if set(value) != {"op", "args"}:
            raise ValueError("compound predicate has wrong fields")
        args = value["args"]
        if not isinstance(args, list) or not 2 <= len(args) <= 4:
            raise ValueError("compound predicate needs two to four arguments")
        return {
            "op": op,
            "args": [
                validate_predicate(item, depth=depth + 1) for item in args
            ],
        }
    if op == "not":
        if set(value) != {"op", "arg"}:
            raise ValueError("not predicate has wrong fields")
        return {
            "op": "not",
            "arg": validate_predicate(value["arg"], depth=depth + 1),
        }
    raise ValueError(f"unsupported block predicate op: {op!r}")


def validate_rule(value: Any, *, depth: int = 0) -> dict[str, Any]:
    if depth > 5 or not isinstance(value, dict):
        raise ValueError("invalid rule")
    op = value.get("op")
    if op in {"exists", "forall"}:
        if set(value) != {"op", "predicate"}:
            raise ValueError("quantified rule has wrong fields")
        return {
            "op": op,
            "predicate": validate_predicate(value["predicate"]),
        }
    if op == "count":
        if set(value) != {"op", "predicate", "comparison", "value"}:
            raise ValueError("count rule has wrong fields")
        comparison = value["comparison"]
        count = value["value"]
        if comparison not in {"eq", "ge", "le"}:
            raise ValueError("invalid count comparison")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 6:
            raise ValueError("count value must be an integer from zero to six")
        return {
            "op": "count",
            "predicate": validate_predicate(value["predicate"]),
            "comparison": comparison,
            "value": count,
        }
    if op == "all_same":
        if set(value) != {"op", "attribute"}:
            raise ValueError("all_same rule has wrong fields")
        if value["attribute"] not in {
            "color",
            "size",
            "orientation",
            "grounded",
        }:
            raise ValueError("invalid all_same attribute")
        return {"op": "all_same", "attribute": value["attribute"]}
    if op in {"touching", "stacking"}:
        fields = ("left", "right") if op == "touching" else ("upper", "lower")
        if set(value) != {"op", *fields}:
            raise ValueError(f"{op} rule has wrong fields")
        return {
            "op": op,
            fields[0]: validate_predicate(value[fields[0]]),
            fields[1]: validate_predicate(value[fields[1]]),
        }
    if op == "largest_all":
        if set(value) != {"op", "predicate"}:
            raise ValueError("largest_all rule has wrong fields")
        return {
            "op": "largest_all",
            "predicate": validate_predicate(value["predicate"]),
        }
    if op in {"and", "or"}:
        if set(value) != {"op", "rules"}:
            raise ValueError("compound rule has wrong fields")
        rules = value["rules"]
        if not isinstance(rules, list) or not 2 <= len(rules) <= 4:
            raise ValueError("compound rule needs two to four child rules")
        return {
            "op": op,
            "rules": [validate_rule(rule, depth=depth + 1) for rule in rules],
        }
    if op == "not":
        if set(value) != {"op", "rule"}:
            raise ValueError("not rule has wrong fields")
        return {
            "op": "not",
            "rule": validate_rule(value["rule"], depth=depth + 1),
        }
    raise ValueError(f"unsupported rule op: {op!r}")


def validate_scene(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"blocks"}:
        raise ValueError("scene must contain only blocks")
    raw_blocks = value["blocks"]
    if not isinstance(raw_blocks, list) or not 1 <= len(raw_blocks) <= 6:
        raise ValueError("scene must have one to six blocks")
    blocks = []
    required = {
        "color",
        "size",
        "orientation",
        "grounded",
        "touching",
        "stacking_on",
    }
    for raw in raw_blocks:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("block has wrong fields")
        if raw["color"] not in COLORS:
            raise ValueError("invalid color")
        if raw["size"] not in SIZES:
            raise ValueError("invalid size")
        if raw["orientation"] not in ORIENTATIONS:
            raise ValueError("invalid orientation")
        if not isinstance(raw["grounded"], bool):
            raise ValueError("grounded must be Boolean")
        touching = raw["touching"]
        if (
            not isinstance(touching, list)
            or any(isinstance(item, bool) or not isinstance(item, int) for item in touching)
            or len(touching) != len(set(touching))
        ):
            raise ValueError("touching must be unique integer block ids")
        stacking_on = raw["stacking_on"]
        if stacking_on is not None and (
            isinstance(stacking_on, bool) or not isinstance(stacking_on, int)
        ):
            raise ValueError("stacking_on must be an integer or null")
        blocks.append(
            {
                "color": raw["color"],
                "size": raw["size"],
                "orientation": raw["orientation"],
                "grounded": raw["grounded"],
                "touching": sorted(touching),
                "stacking_on": stacking_on,
            }
        )
    size = len(blocks)
    for index, block in enumerate(blocks, start=1):
        if any(other < 1 or other > size or other == index for other in block["touching"]):
            raise ValueError("touching contains an invalid block id")
        for other in block["touching"]:
            if index not in blocks[other - 1]["touching"]:
                raise ValueError("touching relation must be symmetric")
        target = block["stacking_on"]
        if target is not None and (
            target < 1
            or target > size
            or target == index
            or target not in block["touching"]
        ):
            raise ValueError("stacking_on must identify a touched other block")
    return {"blocks": blocks}


def evaluate_predicate(predicate: dict[str, Any], block: dict[str, Any]) -> bool:
    op = predicate["op"]
    if op == "any":
        return True
    if op == "attribute":
        return block[predicate["attribute"]] == predicate["value"]
    if op == "and":
        return all(evaluate_predicate(item, block) for item in predicate["args"])
    if op == "or":
        return any(evaluate_predicate(item, block) for item in predicate["args"])
    if op == "not":
        return not evaluate_predicate(predicate["arg"], block)
    raise AssertionError(op)


def evaluate_rule(rule: dict[str, Any], scene: dict[str, Any]) -> bool:
    blocks = scene["blocks"]
    op = rule["op"]
    if op == "exists":
        return any(evaluate_predicate(rule["predicate"], block) for block in blocks)
    if op == "forall":
        return all(evaluate_predicate(rule["predicate"], block) for block in blocks)
    if op == "count":
        count = sum(evaluate_predicate(rule["predicate"], block) for block in blocks)
        target = rule["value"]
        return {
            "eq": count == target,
            "ge": count >= target,
            "le": count <= target,
        }[rule["comparison"]]
    if op == "all_same":
        values = {block[rule["attribute"]] for block in blocks}
        return len(values) == 1
    if op == "touching":
        for left_index, left in enumerate(blocks, start=1):
            if not evaluate_predicate(rule["left"], left):
                continue
            for right_index in left["touching"]:
                if right_index != left_index and evaluate_predicate(
                    rule["right"], blocks[right_index - 1]
                ):
                    return True
        return False
    if op == "stacking":
        for upper in blocks:
            target = upper["stacking_on"]
            if (
                target is not None
                and evaluate_predicate(rule["upper"], upper)
                and evaluate_predicate(rule["lower"], blocks[target - 1])
            ):
                return True
        return False
    if op == "largest_all":
        largest = max(SIZE_ORDER[block["size"]] for block in blocks)
        return all(
            evaluate_predicate(rule["predicate"], block)
            for block in blocks
            if SIZE_ORDER[block["size"]] == largest
        )
    if op == "and":
        return all(evaluate_rule(item, scene) for item in rule["rules"])
    if op == "or":
        return any(evaluate_rule(item, scene) for item in rule["rules"])
    if op == "not":
        return not evaluate_rule(rule["rule"], scene)
    raise AssertionError(op)


def parse_hypotheses(text: str) -> list[dict[str, Any]]:
    payload = _strict_json_object(text)
    if set(payload) != {"hypotheses"}:
        raise ValueError("hypothesis response has unexpected fields")
    raw = payload["hypotheses"]
    if not isinstance(raw, list) or len(raw) != PARTICLE_COUNT:
        raise ValueError(f"expected exactly {PARTICLE_COUNT} hypotheses")
    hypotheses = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or set(item) != {"id", "rule_text", "rule"}:
            raise ValueError("hypothesis has wrong fields")
        expected_id = f"H{index:02d}"
        if item["id"] != expected_id:
            raise ValueError(f"expected hypothesis id {expected_id}")
        rule_text = item["rule_text"]
        if not isinstance(rule_text, str) or not rule_text.strip():
            raise ValueError("rule_text must be nonempty")
        hypotheses.append(
            {
                "id": expected_id,
                "rule_text": " ".join(rule_text.split()),
                "rule": validate_rule(item["rule"]),
            }
        )
    if len({_canonical_json(item["rule"]) for item in hypotheses}) != len(hypotheses):
        raise ValueError("hypothesis ASTs must be unique")
    return hypotheses


def parse_scenes(text: str) -> list[dict[str, Any]]:
    payload = _strict_json_object(text)
    if set(payload) != {"scenes"}:
        raise ValueError("scene response has unexpected fields")
    raw = payload["scenes"]
    if not isinstance(raw, list) or len(raw) != SCENE_COUNT:
        raise ValueError(f"expected exactly {SCENE_COUNT} scenes")
    scenes = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or set(item) != {"id", "scene"}:
            raise ValueError("candidate scene has wrong fields")
        expected_id = f"X{index:02d}"
        if item["id"] != expected_id:
            raise ValueError(f"expected scene id {expected_id}")
        scenes.append(validate_scene(item["scene"]))
    if len({_canonical_json(scene) for scene in scenes}) != len(scenes):
        raise ValueError("candidate scenes must be distinct")
    return scenes


def _rotation_orientation(rotation: float) -> str:
    rotation %= 2 * math.pi
    if abs(rotation - math.pi) < math.pi / 6:
        return "upright"
    if abs(rotation - 1.2475) < math.pi / 6:
        return "left"
    if abs(rotation - 5.0375) < math.pi / 6:
        return "right"
    return "strange"


def raw_official_scene(raw: dict[str, Any]) -> dict[str, Any]:
    contacts = (
        list(raw["contact"][0].values())
        if isinstance(raw["contact"][0], dict)
        else raw["contact"]
    )
    blocks = []
    for index, color in enumerate(raw["colours"]):
        contact = contacts[index]
        touching_zero = list(contact) if isinstance(contact, list) else [index]
        touching = sorted(other + 1 for other in touching_zero if other != index)
        orientation = _rotation_orientation(float(raw["rotations"][index]))
        stacking_on = None
        if not raw["grounded"][index] and orientation == "upright":
            for other in touching_zero:
                if other == index:
                    continue
                same_x = abs(float(raw["xpos"][other]) - float(raw["xpos"][index])) < 0.01
                other_upright = _rotation_orientation(
                    float(raw["rotations"][other])
                ) == "upright"
                if raw["grounded"][other] and same_x and other_upright:
                    stacking_on = other + 1
                    break
        blocks.append(
            {
                "color": str(color).lower(),
                "size": SIZES[int(raw["sizes"][index]) - 1],
                "orientation": orientation,
                "grounded": bool(raw["grounded"][index]),
                "touching": touching,
                "stacking_on": stacking_on,
            }
        )
    return validate_scene({"blocks": blocks})


def truth_function(rule_name: str) -> Callable[[dict[str, Any]], bool]:
    def zeta(scene: dict[str, Any]) -> bool:
        return any(block["color"] == "red" for block in scene["blocks"])

    def phi(scene: dict[str, Any]) -> bool:
        return len({block["size"] for block in scene["blocks"]}) == 1

    def upsilon(scene: dict[str, Any]) -> bool:
        return all(block["orientation"] != "upright" for block in scene["blocks"])

    def iota(scene: dict[str, Any]) -> bool:
        return sum(block["color"] == "blue" for block in scene["blocks"]) == 1

    def kappa(scene: dict[str, Any]) -> bool:
        return any(
            block["color"] == "blue" and block["size"] == "small"
            for block in scene["blocks"]
        )

    def omega(scene: dict[str, Any]) -> bool:
        return all(
            block["color"] == "blue" or block["size"] == "small"
            for block in scene["blocks"]
        )

    def mu(scene: dict[str, Any]) -> bool:
        largest = max(SIZE_ORDER[block["size"]] for block in scene["blocks"])
        return all(
            block["color"] == "red"
            for block in scene["blocks"]
            if SIZE_ORDER[block["size"]] == largest
        )

    def nu(scene: dict[str, Any]) -> bool:
        return any(block["touching"] for block in scene["blocks"])

    def xi(scene: dict[str, Any]) -> bool:
        blocks = scene["blocks"]
        return any(
            block["color"] == "blue"
            and any(blocks[other - 1]["color"] == "red" for other in block["touching"])
            for block in blocks
        )

    def psi(scene: dict[str, Any]) -> bool:
        return any(block["stacking_on"] is not None for block in scene["blocks"])

    return {
        "zeta": zeta,
        "phi": phi,
        "upsilon": upsilon,
        "iota": iota,
        "kappa": kappa,
        "omega": omega,
        "mu": mu,
        "nu": nu,
        "xi": xi,
        "psi": psi,
    }[rule_name]


def random_scene(rng: random.Random) -> dict[str, Any]:
    count = rng.randint(1, 6)
    blocks = [
        {
            "color": rng.choice(COLORS),
            "size": rng.choice(SIZES),
            "orientation": rng.choice(ORIENTATIONS),
            "grounded": bool(rng.randrange(2)),
            "touching": [],
            "stacking_on": None,
        }
        for _ in range(count)
    ]
    for left in range(count):
        for right in range(left + 1, count):
            if rng.random() < 0.28:
                blocks[left]["touching"].append(right + 1)
                blocks[right]["touching"].append(left + 1)
    possible_upper = [
        index
        for index, block in enumerate(blocks)
        if block["touching"] and rng.random() < 0.22
    ]
    used_lowers: set[int] = set()
    for upper in possible_upper:
        lowers = [
            other - 1
            for other in blocks[upper]["touching"]
            if other - 1 not in used_lowers
        ]
        if not lowers:
            continue
        lower = rng.choice(lowers)
        used_lowers.add(lower)
        blocks[upper]["stacking_on"] = lower + 1
        blocks[upper]["grounded"] = False
        blocks[upper]["orientation"] = "upright"
        blocks[lower]["orientation"] = "upright"
        blocks[lower]["grounded"] = True
    return validate_scene({"blocks": blocks})


def audit_bank(
    task_index: int,
    official_case: dict[str, Any],
) -> list[dict[str, Any]]:
    rng = random.Random(SELECTION_SEED + 1009 * task_index)
    deduped: dict[str, dict[str, Any]] = {}
    while len(deduped) < AUDIT_RANDOM_SCENES:
        scene = random_scene(rng)
        deduped.setdefault(_canonical_json(scene), scene)
    for raw in [*official_case["t"], *official_case["f"]]:
        scene = raw_official_scene(raw)
        deduped.setdefault(_canonical_json(scene), scene)
    return list(deduped.values())


def _scene_text(scene: dict[str, Any]) -> str:
    return _canonical_json(scene)


def initial_messages(scene: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "observed_history": [{"scene": scene, "is_good": True}],
        "required_output": {
            "hypotheses": [
                {"id": "H01..H12", "rule_text": "one sentence", "rule": "RULE_AST"}
            ]
        },
        "dsl": RULE_DSL,
    }
    return [
        {
            "role": "system",
            "content": (
                "Infer possible hidden rules in a Zendo concept-learning game. "
                "Generate exactly 12 diverse, plausible rules consistent with the "
                "observed labeled scenes. Each rule needs a concise natural-language "
                "description and an exactly equivalent executable AST in the supplied "
                "DSL. Use only listed operators and attribute values. Prefer genuinely "
                "different behavioral rules, including relations when plausible. "
                "Return exactly one JSON object and no markdown or explanation."
            ),
        },
        {"role": "user", "content": _canonical_json(payload)},
    ]


def scene_messages(
    scene: dict[str, Any],
    hypotheses: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = {
        "observed_history": [{"scene": scene, "is_good": True}],
        "current_hypotheses": hypotheses,
        "required_output": {
            "scenes": [{"id": "X01..X08", "scene": "SCENE_OBJECT"}]
        },
        "scene_schema": {
            "blocks": [
                {
                    "color": "blue|red|green",
                    "size": "small|medium|large",
                    "orientation": "upright|left|right|strange",
                    "grounded": "boolean",
                    "touching": "symmetric list of one-based other block ids",
                    "stacking_on": "touched one-based block id or null",
                }
            ],
            "block_count": "1..6",
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "Design exactly eight distinct legal Zendo experiments that "
                "discriminate among the supplied hypotheses. Include simple and "
                "relational scenes and make the first four especially informative. "
                "Touching must be symmetric. A stacking_on id must also appear in "
                "touching. Return exactly one JSON object and no markdown or explanation."
            ),
        },
        {"role": "user", "content": _canonical_json(payload)},
    ]


def refresh_messages(
    initial_scene: dict[str, Any],
    hypotheses: Sequence[dict[str, Any]],
    candidate: dict[str, Any],
    outcome: bool,
) -> list[dict[str, str]]:
    payload = {
        "observed_history": [
            {"scene": initial_scene, "is_good": True},
            {"scene": candidate, "is_good": outcome},
        ],
        "current_hypotheses": hypotheses,
        "required_output": {
            "hypotheses": [
                {"id": "H01..H12", "rule_text": "one sentence", "rule": "RULE_AST"}
            ]
        },
        "dsl": RULE_DSL,
    }
    return [
        {
            "role": "system",
            "content": (
                "Revise a Zendo rule-particle population after a new experiment. "
                "Return exactly 12 diverse hypotheses with equivalent executable ASTs. "
                "Preserve current rules that remain plausible, locally revise or replace "
                "contradicted rules, and allocate the population to explanations of the "
                "complete labeled history. Do not mention a hidden answer or use any "
                "operator outside the supplied DSL. Return exactly one JSON object and "
                "no markdown or explanation."
            ),
        },
        {"role": "user", "content": _canonical_json(payload)},
    ]


def hypothesis_predictions(
    hypotheses: Sequence[dict[str, Any]],
    scenes: Sequence[dict[str, Any]],
) -> list[list[bool]]:
    return [
        [evaluate_rule(hypothesis["rule"], scene) for scene in scenes]
        for hypothesis in hypotheses
    ]


def posterior_weights(
    hypotheses: Sequence[dict[str, Any]],
    history: Sequence[tuple[dict[str, Any], bool]],
) -> list[float]:
    log_weights = [-math.log(len(hypotheses))] * len(hypotheses)
    for scene, outcome in history:
        for index, hypothesis in enumerate(hypotheses):
            matches = evaluate_rule(hypothesis["rule"], scene) == outcome
            log_weights[index] += math.log(
                OBSERVATION_ACCURACY if matches else 1.0 - OBSERVATION_ACCURACY
            )
    maximum = max(log_weights)
    weights = [math.exp(value - maximum) for value in log_weights]
    total = sum(weights)
    return [value / total for value in weights]


def _entropy(weights: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in weights if value > 0.0)


def update_weights(
    hypotheses: Sequence[dict[str, Any]],
    weights: Sequence[float],
    scene: dict[str, Any],
    outcome: bool,
) -> tuple[list[float], float]:
    likelihoods = [
        OBSERVATION_ACCURACY
        if evaluate_rule(hypothesis["rule"], scene) == outcome
        else 1.0 - OBSERVATION_ACCURACY
        for hypothesis in hypotheses
    ]
    evidence = sum(weight * likelihood for weight, likelihood in zip(weights, likelihoods))
    posterior = [
        weight * likelihood / evidence
        for weight, likelihood in zip(weights, likelihoods)
    ]
    return posterior, evidence


def eig(
    hypotheses: Sequence[dict[str, Any]],
    weights: Sequence[float],
    scene: dict[str, Any],
) -> float:
    before = _entropy(weights)
    yes_weights, probability_yes = update_weights(
        hypotheses, weights, scene, True
    )
    no_weights, probability_no = update_weights(
        hypotheses, weights, scene, False
    )
    return before - (
        probability_yes * _entropy(yes_weights)
        + probability_no * _entropy(no_weights)
    )


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def fixed_support_depth_two_scores(
    hypotheses: Sequence[dict[str, Any]],
    initial_weights: Sequence[float],
    scenes: Sequence[dict[str, Any]],
) -> list[float]:
    scores = []
    for root_index, root in enumerate(scenes[:ROOT_COUNT]):
        immediate = eig(hypotheses, initial_weights, root)
        continuation = 0.0
        for outcome in (False, True):
            branch_weights, probability = update_weights(
                hypotheses, initial_weights, root, outcome
            )
            next_values = [
                eig(hypotheses, branch_weights, candidate)
                for index, candidate in enumerate(scenes)
                if index != root_index
            ]
            continuation += probability * max(next_values)
        scores.append(immediate + continuation)
    return scores


def model_aware_depth_two_scores(
    hypotheses: Sequence[dict[str, Any]],
    initial_weights: Sequence[float],
    scenes: Sequence[dict[str, Any]],
    branches: dict[str, list[dict[str, Any]]],
    initial_scene: dict[str, Any],
) -> list[float]:
    scores = []
    for root_index, root in enumerate(scenes[:ROOT_COUNT]):
        immediate = eig(hypotheses, initial_weights, root)
        continuation = 0.0
        for outcome in (False, True):
            _, probability = update_weights(
                hypotheses, initial_weights, root, outcome
            )
            branch = branches[branch_key(root_index, outcome)]
            branch_weights = posterior_weights(
                branch,
                [(initial_scene, True), (root, outcome)],
            )
            next_values = [
                eig(branch, branch_weights, candidate)
                for index, candidate in enumerate(scenes)
                if index != root_index
            ]
            continuation += probability * max(next_values)
        scores.append(immediate + continuation)
    return scores


def branch_key(root_index: int, outcome: bool) -> str:
    return f"root_{root_index + 1}_{'yes' if outcome else 'no'}"


def behavior_agreements(
    hypotheses: Sequence[dict[str, Any]],
    audit: Sequence[dict[str, Any]],
    truth: Callable[[dict[str, Any]], bool],
) -> list[float]:
    truth_values = [truth(scene) for scene in audit]
    return [
        sum(
            evaluate_rule(hypothesis["rule"], scene) == target
            for scene, target in zip(audit, truth_values)
        )
        / len(audit)
        for hypothesis in hypotheses
    ]


def weighted_agreement(
    agreements: Sequence[float],
    weights: Sequence[float],
) -> float:
    return sum(value * weight for value, weight in zip(agreements, weights))


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and math.isclose(
            values[order[end]], values[order[cursor]], abs_tol=1e-12
        ):
            end += 1
        rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    ranks_a = _ranks(values_a)
    ranks_b = _ranks(values_b)
    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)
    centered_a = [value - mean_a for value in ranks_a]
    centered_b = [value - mean_b for value in ranks_b]
    denominator = math.sqrt(
        sum(value * value for value in centered_a)
        * sum(value * value for value in centered_b)
    )
    if denominator == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(centered_a, centered_b)) / denominator


def _signature_count(
    hypotheses: Sequence[dict[str, Any]],
    audit: Sequence[dict[str, Any]],
) -> int:
    return len(
        {
            tuple(evaluate_rule(hypothesis["rule"], scene) for scene in audit)
            for hypothesis in hypotheses
        }
    )


def analyze_task(
    *,
    task_name: str,
    task_index: int,
    official_case: dict[str, Any],
    initial_scene: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    branches: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    audit = audit_bank(task_index, official_case)
    truth = truth_function(task_name)
    initial_history = [(initial_scene, True)]
    initial_weights = posterior_weights(hypotheses, initial_history)
    initial_agreements = behavior_agreements(hypotheses, audit, truth)
    initial_weighted = weighted_agreement(initial_agreements, initial_weights)
    initial_maximum = max(initial_agreements)
    myopic_scores = [
        eig(hypotheses, initial_weights, scene) for scene in scenes[:ROOT_COUNT]
    ]
    fixed_scores = fixed_support_depth_two_scores(
        hypotheses, initial_weights, scenes
    )
    model_scores = model_aware_depth_two_scores(
        hypotheses,
        initial_weights,
        scenes,
        branches,
        initial_scene,
    )
    branch_rows: dict[str, dict[str, Any]] = {}
    root_rows = []
    for root_index, root in enumerate(scenes[:ROOT_COUNT]):
        for outcome in (False, True):
            key = branch_key(root_index, outcome)
            support = branches[key]
            weights = posterior_weights(
                support,
                [(initial_scene, True), (root, outcome)],
            )
            agreements = behavior_agreements(support, audit, truth)
            branch_rows[key] = {
                "outcome": outcome,
                "support_signature_count": _signature_count(support, audit),
                "posterior_weighted_truth_agreement": weighted_agreement(
                    agreements, weights
                ),
                "maximum_truth_agreement": max(agreements),
                "truth_recovered_at_0_95": max(agreements) >= 0.95,
            }
        actual = truth(root)
        actual_row = branch_rows[branch_key(root_index, actual)]
        root_rows.append(
            {
                "root_index": root_index + 1,
                "actual_outcome": actual,
                "myopic_score": myopic_scores[root_index],
                "fixed_depth_two_score": fixed_scores[root_index],
                "model_aware_depth_two_score": model_scores[root_index],
                "realized_weighted_truth_agreement": actual_row[
                    "posterior_weighted_truth_agreement"
                ],
                "realized_maximum_truth_agreement": actual_row[
                    "maximum_truth_agreement"
                ],
                "realized_truth_recovered_at_0_95": actual_row[
                    "truth_recovered_at_0_95"
                ],
            }
        )
    endpoints = [
        row["realized_weighted_truth_agreement"] for row in root_rows
    ]
    myopic_choice = _argmax(myopic_scores)
    fixed_choice = _argmax(fixed_scores)
    model_choice = _argmax(model_scores)
    all_branch_signatures = {
        tuple(
            evaluate_rule(hypothesis["rule"], scene)
            for hypothesis in branches[key]
            for scene in audit
        )
        for key in sorted(branches)
    }
    informative_roots = sum(
        min(
            update_weights(hypotheses, initial_weights, scene, True)[1],
            update_weights(hypotheses, initial_weights, scene, False)[1],
        )
        >= 0.10
        for scene in scenes[:ROOT_COUNT]
    )
    return {
        "task_name": task_name,
        "audit_scene_count": len(audit),
        "audit_sha256": hashlib.sha256(
            _canonical_json(audit).encode("utf-8")
        ).hexdigest(),
        "initial_support_signature_count": _signature_count(hypotheses, audit),
        "informative_root_count": informative_roots,
        "distinct_refreshed_branch_support_count": len(all_branch_signatures),
        "initial_posterior_weighted_truth_agreement": initial_weighted,
        "initial_maximum_truth_agreement": initial_maximum,
        "root_realized_endpoint_range": max(endpoints) - min(endpoints),
        "best_realized_maximum_gain_over_initial": (
            max(row["realized_maximum_truth_agreement"] for row in root_rows)
            - initial_maximum
        ),
        "model_score_realized_endpoint_spearman": spearman(
            model_scores, endpoints
        ),
        "myopic_selected_root": myopic_choice + 1,
        "fixed_depth_two_selected_root": fixed_choice + 1,
        "model_aware_depth_two_selected_root": model_choice + 1,
        "myopic_selected_realized_weighted_truth_agreement": endpoints[
            myopic_choice
        ],
        "fixed_depth_two_selected_realized_weighted_truth_agreement": endpoints[
            fixed_choice
        ],
        "model_aware_selected_realized_weighted_truth_agreement": endpoints[
            model_choice
        ],
        "model_minus_myopic_realized_weighted_truth_agreement": (
            endpoints[model_choice] - endpoints[myopic_choice]
        ),
        "model_minus_fixed_realized_weighted_truth_agreement": (
            endpoints[model_choice] - endpoints[fixed_choice]
        ),
        "root_rows": root_rows,
        "branch_rows": branch_rows,
    }


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot["adapter_requests"]),
        "reasoning_tokens": int(snapshot["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(snapshot["adapter_cost_usd"]),
        "generator": snapshot,
    }


def _build_model(config: Config) -> Any:
    if len(config.model_pairs) != 1:
        raise ValueError("Zendo gate requires one model pair")
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("Zendo gate config selects the wrong model")
    return build_model_adapter(spec, config)


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_task_calls(
    model: Any,
    initial_scene: dict[str, Any],
    raw_task: dict[str, Any],
    *,
    block_size: int,
    max_new_tokens: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    initial_response = model.chat_complete_messages_batched(
        [initial_messages(initial_scene)],
        temperature=0.0,
        block_size=1,
        max_new_tokens=max_new_tokens,
    )[0]
    raw_task["initial_hypotheses"] = initial_response
    hypotheses = parse_hypotheses(initial_response)
    scene_response = model.chat_complete_messages_batched(
        [scene_messages(initial_scene, hypotheses)],
        temperature=0.0,
        block_size=1,
        max_new_tokens=max_new_tokens,
    )[0]
    raw_task["candidate_scenes"] = scene_response
    scenes = parse_scenes(scene_response)
    interventions = [
        (root_index, outcome)
        for root_index in range(ROOT_COUNT)
        for outcome in (False, True)
    ]
    responses = model.chat_complete_messages_batched(
        [
            refresh_messages(
                initial_scene,
                hypotheses,
                scenes[root_index],
                outcome,
            )
            for root_index, outcome in interventions
        ],
        temperature=0.0,
        block_size=block_size,
        max_new_tokens=max_new_tokens,
    )
    raw_task["branch_refreshes"] = {
        branch_key(root_index, outcome): response
        for (root_index, outcome), response in zip(
            interventions, responses, strict=True
        )
    }
    branches = {
        key: parse_hypotheses(response)
        for key, response in raw_task["branch_refreshes"].items()
    }
    return hypotheses, scenes, branches


def smoke_mechanics_gates(
    diagnostics: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, bool]:
    expected = EXPECTED_REQUESTS[stage]
    generator = usage.get("generator", {})
    gates = {
        f"exact_{expected}_physical_requests": (
            int(usage.get("physical_requests", -1)) == expected
        ),
        f"exact_{expected}_http_attempts": (
            int(generator.get("http_attempts", -1)) == expected
        ),
        "zero_transport_retries": int(generator.get("retry_count", -1)) == 0,
        "zero_reasoning_tokens": int(usage.get("reasoning_tokens", -1)) == 0,
        "zero_forced_exits": int(generator.get("forced_exits", -1)) == 0,
        "all_responses_parsed_without_repair": len(diagnostics)
        == len(TASKS[stage]),
        "all_initial_supports_have_six_signatures": all(
            row["initial_support_signature_count"] >= 6 for row in diagnostics
        ),
        "all_tasks_have_three_informative_roots": all(
            row["informative_root_count"] >= 3 for row in diagnostics
        ),
        "all_tasks_have_four_distinct_refreshed_branches": all(
            row["distinct_refreshed_branch_support_count"] >= 4
            for row in diagnostics
        ),
        f"cost_at_most_{COST_CAP_USD[stage]:.2f}": (
            float(usage.get("adapter_cost_usd", float("inf")))
            <= COST_CAP_USD[stage]
        ),
    }
    return gates


def summarize(
    diagnostics: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    mechanics = smoke_mechanics_gates(diagnostics, usage, stage=stage)
    summary: dict[str, Any] = {
        "stage": stage,
        "num_tasks": len(diagnostics),
        "mechanics_gates": mechanics,
    }
    gates = dict(mechanics)
    if stage == "opportunity":
        model_minus_myopic = [
            row["model_minus_myopic_realized_weighted_truth_agreement"]
            for row in diagnostics
        ]
        model_minus_fixed = [
            row["model_minus_fixed_realized_weighted_truth_agreement"]
            for row in diagnostics
        ]
        summary.update(
            {
                "non_saturated_initial_count": sum(
                    row["initial_posterior_weighted_truth_agreement"] < 0.95
                    for row in diagnostics
                ),
                "endpoint_range_at_least_0_10_count": sum(
                    row["root_realized_endpoint_range"] >= 0.10
                    for row in diagnostics
                ),
                "maximum_recovery_gain_at_least_0_10_count": sum(
                    row["best_realized_maximum_gain_over_initial"] >= 0.10
                    for row in diagnostics
                ),
                "model_vs_myopic_win_count": sum(
                    value > 1e-12 for value in model_minus_myopic
                ),
                "model_vs_myopic_loss_count": sum(
                    value < -1e-12 for value in model_minus_myopic
                ),
                "mean_model_minus_myopic": sum(model_minus_myopic)
                / len(model_minus_myopic),
                "model_vs_fixed_win_count": sum(
                    value > 1e-12 for value in model_minus_fixed
                ),
                "model_vs_fixed_loss_count": sum(
                    value < -1e-12 for value in model_minus_fixed
                ),
                "mean_model_minus_fixed": sum(model_minus_fixed)
                / len(model_minus_fixed),
                "mean_model_score_realized_endpoint_spearman": sum(
                    row["model_score_realized_endpoint_spearman"]
                    for row in diagnostics
                )
                / len(diagnostics),
            }
        )
        gates.update(
            {
                "non_saturated_initial_count_at_least_4": (
                    summary["non_saturated_initial_count"] >= 4
                ),
                "endpoint_range_count_at_least_4": (
                    summary["endpoint_range_at_least_0_10_count"] >= 4
                ),
                "maximum_recovery_gain_count_at_least_4": (
                    summary["maximum_recovery_gain_at_least_0_10_count"] >= 4
                ),
                "model_vs_myopic_wins_at_least_3": (
                    summary["model_vs_myopic_win_count"] >= 3
                ),
                "model_vs_myopic_losses_at_most_1": (
                    summary["model_vs_myopic_loss_count"] <= 1
                ),
                "mean_model_minus_myopic_at_least_0_05": (
                    summary["mean_model_minus_myopic"] >= 0.05
                ),
                "model_vs_fixed_wins_at_least_3": (
                    summary["model_vs_fixed_win_count"] >= 3
                ),
                "model_vs_fixed_losses_at_most_1": (
                    summary["model_vs_fixed_loss_count"] <= 1
                ),
                "mean_model_minus_fixed_at_least_0_03": (
                    summary["mean_model_minus_fixed"] >= 0.03
                ),
                "mean_score_endpoint_spearman_at_least_0_30": (
                    summary["mean_model_score_realized_endpoint_spearman"]
                    >= 0.30
                ),
            }
        )
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def public_task_payload(
    task_name: str,
    initial_scene: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    branches: dict[str, list[dict[str, Any]]],
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "initial_observation": {"scene": initial_scene, "is_good": True},
        "initial_hypotheses": hypotheses,
        "candidate_scenes": scenes,
        "refreshed_hypotheses": branches,
        "diagnostic": diagnostic,
    }


def run_gate(
    config: Config,
    *,
    stage: str,
    source_dir: Path,
    raw_checkpoint_path: Path,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    if stage not in TASKS:
        raise ValueError(f"unknown stage {stage!r}")
    cases_path = verify_source(source_dir)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    case_by_name = dict(zip(RULE_ORDER, cases, strict=True))
    model = model_adapter if model_adapter is not None else _build_model(config)
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "stage": stage,
        "tasks": {},
    }
    public_tasks = []
    diagnostics = []
    try:
        for task_name in TASKS[stage]:
            task_index = RULE_ORDER.index(task_name)
            official_case = case_by_name[task_name]
            initial_scene = raw_official_scene(official_case["t"][0])
            raw_task: dict[str, Any] = {}
            raw["tasks"][task_name] = raw_task
            hypotheses, scenes, branches = run_task_calls(
                model,
                initial_scene,
                raw_task,
                block_size=config.batched_block_size,
                max_new_tokens=config.openrouter_max_output_tokens,
            )
            _checkpoint(raw_checkpoint_path, raw)
            diagnostic = analyze_task(
                task_name=task_name,
                task_index=task_index,
                official_case=official_case,
                initial_scene=initial_scene,
                hypotheses=hypotheses,
                scenes=scenes,
                branches=branches,
            )
            diagnostics.append(diagnostic)
            public_tasks.append(
                public_task_payload(
                    task_name,
                    initial_scene,
                    hypotheses,
                    scenes,
                    branches,
                    diagnostic,
                )
            )
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_checkpoint_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc
    summary = summarize(diagnostics, usage, stage=stage)
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": stage,
            "model": MODEL_ID,
            "selection_seed": SELECTION_SEED,
            "source_commit": SOURCE_COMMIT,
            "cases_sha256": CASES_SHA256,
            "task_names": list(TASKS[stage]),
            "particle_count": PARTICLE_COUNT,
            "scene_count": SCENE_COUNT,
            "root_count": ROOT_COUNT,
            "expected_physical_requests": EXPECTED_REQUESTS[stage],
            "observation_accuracy": OBSERVATION_ACCURACY,
            "truth_hidden_until_all_branches_frozen": True,
            "reasoning_requested": False,
            "repairs_or_scientific_retries": 0,
            "official_rules_are_development_only": True,
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "tasks": public_tasks,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(TASKS), required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("external/doing-experiments-and-revising-rules"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = (
        0.30 if args.stage == "smoke" else 1.20
    )
    config.openrouter_run_budget_usd = COST_CAP_USD[args.stage]
    config.openrouter_concurrency = 8
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        payload = run_gate(
            config,
            stage=args.stage,
            source_dir=args.source_dir,
            raw_checkpoint_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = _sha256(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = _sha256(raw_path)
        failure_path = args.output_dir / f"{args.stage.upper()}_FAILURE.json"
        failure_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path = args.output_dir / f"{args.stage.upper()}.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output_path),
                "summary": payload["summary"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
