#!/usr/bin/env python3
"""Shared-tree depth-one/depth-two CA-BED ranking gate for Detective Cases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config


SELECTION_SEED = 24312
RANDOM_CONTROL_SEED = 24313
BOOTSTRAP_SEED = 24314
ROOT_WIDTH = 3
FOLLOWUP_WIDTH = 3
ANSWER_ROLLOUTS = 4
ESTIMATOR_CONFIDENCE = 0.7
BOOTSTRAP_SAMPLES = 10_000
EXPECTED_REQUESTS_PER_CASE = 52
DATA_SHA256 = "049ea3003753b15e3319483d15993591b5d50ac7eedb3187dca6ef3951cd2a57"
DEFAULT_DATA_PATH = Path(
    "external/ca-bed/src/ca_bed/tasks/detective_cases/DetectiveCases.json"
)
SMOKE_CASE_IDS = (27, 94)
RANKING_CASE_IDS = (31, 96, 91, 65, 99, 56, 2, 4, 50, 6, 87, 34)
CONFIRMATION_CASE_IDS = (
    75,
    70,
    89,
    98,
    86,
    88,
    8,
    74,
    82,
    19,
    78,
    72,
    18,
    97,
    76,
    46,
    24,
    67,
    42,
    3,
    80,
    1,
    79,
    14,
)


@dataclass(frozen=True)
class Belief:
    hypotheses: tuple[str, ...]
    probabilities: tuple[float, ...]

    @classmethod
    def uniform(cls, hypotheses: Sequence[str]) -> "Belief":
        if not hypotheses:
            raise ValueError("belief support cannot be empty")
        probability = 1.0 / len(hypotheses)
        return cls(tuple(hypotheses), tuple(probability for _ in hypotheses))

    def entropy(self) -> float:
        return -sum(
            probability * math.log(probability)
            for probability in self.probabilities
            if probability > 0.0
        )

    def truth_log_probability(self, truth: str) -> float:
        try:
            index = self.hypotheses.index(truth)
        except ValueError as exc:
            raise ValueError(f"truth {truth!r} is absent from belief support") from exc
        return math.log(max(self.probabilities[index], 1.0e-300))


@dataclass(frozen=True)
class BranchPlan:
    observation: str
    probability: float
    belief: Belief
    questions: tuple[str, ...]
    likelihoods: tuple[tuple[float, ...], ...]
    eig_scores: tuple[float, ...]
    selected_index: int

    @property
    def selected_question(self) -> str:
        return self.questions[self.selected_index]


@dataclass(frozen=True)
class RootPlan:
    question: str
    likelihoods: tuple[float, ...]
    immediate_eig: float
    depth_two_score: float
    branches: tuple[BranchPlan, BranchPlan]

    def branch_for(self, observation: str) -> BranchPlan:
        for branch in self.branches:
            if branch.observation == observation:
                return branch
        raise ValueError(f"missing branch for observation {observation!r}")


def binary_entropy(probability: float) -> float:
    probability = min(max(float(probability), 0.0), 1.0)
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log(probability) - (
        1.0 - probability
    ) * math.log(1.0 - probability)


def immediate_eig(belief: Belief, yes_probabilities: Sequence[float]) -> float:
    if len(yes_probabilities) != len(belief.probabilities):
        raise ValueError("likelihood row must align with belief support")
    marginal_yes = sum(
        prior * float(yes)
        for prior, yes in zip(
            belief.probabilities,
            yes_probabilities,
            strict=True,
        )
    )
    conditional_entropy = sum(
        prior * binary_entropy(float(yes))
        for prior, yes in zip(
            belief.probabilities,
            yes_probabilities,
            strict=True,
        )
    )
    return binary_entropy(marginal_yes) - conditional_entropy


def bayes_update(
    belief: Belief,
    yes_probabilities: Sequence[float],
    observation: str,
) -> tuple[Belief, float]:
    if observation not in {"Yes", "No"}:
        raise ValueError("observation must be Yes or No")
    if len(yes_probabilities) != len(belief.probabilities):
        raise ValueError("likelihood row must align with belief support")
    likelihoods = (
        [float(value) for value in yes_probabilities]
        if observation == "Yes"
        else [1.0 - float(value) for value in yes_probabilities]
    )
    weights = [
        prior * likelihood
        for prior, likelihood in zip(
            belief.probabilities,
            likelihoods,
            strict=True,
        )
    ]
    marginal = sum(weights)
    if not math.isfinite(marginal) or marginal <= 0.0:
        raise ValueError("Bayesian update has non-positive marginal probability")
    posterior = tuple(weight / marginal for weight in weights)
    return Belief(belief.hypotheses, posterior), marginal


def _canonical_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip()).casefold().rstrip("?.!")


def _target_from_question(question: str, hypotheses: Sequence[str]) -> str:
    match = re.fullmatch(
        r"\s*\[Target:\s*(.*?)\]\s*(.+\?)\s*",
        question,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"question does not use [Target: Name] format: {question!r}")
    target = re.sub(r"\s+", " ", match.group(1).strip()).casefold()
    matches = [
        hypothesis
        for hypothesis in hypotheses
        if re.sub(r"\s+", " ", hypothesis.strip()).casefold() == target
    ]
    if len(matches) != 1:
        raise ValueError(f"question target is not exactly one suspect: {question!r}")
    return matches[0]


def parse_questions(
    response: str,
    *,
    width: int,
    hypotheses: Sequence[str],
    history: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    matches = re.findall(
        r"##Question##[^:]*:[^\[]*(\[Target:.*?\]\s*.*?\?)",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    forbidden = {_canonical_question(question) for question, _answer in history}
    selected: list[str] = []
    seen = set(forbidden)
    for match in matches:
        question = re.sub(r"\s+", " ", match.strip())
        canonical = _canonical_question(question)
        _target_from_question(question, hypotheses)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        selected.append(question)
        if len(selected) == width:
            break
    if len(selected) != width:
        raise ValueError(
            f"required {width} valid unique questions, parsed {len(selected)}"
        )
    return tuple(selected)


_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def parse_likelihoods(
    response: str,
    hypotheses: Sequence[str],
    *,
    confidence: float = ESTIMATOR_CONFIDENCE,
) -> tuple[float, ...]:
    parsed: dict[str, float] = {}
    expected = {
        re.sub(r"\s+", " ", hypothesis.strip()).casefold(): hypothesis
        for hypothesis in hypotheses
    }
    for raw_name, raw_value in re.findall(
        rf"##(.*?)##[^:]*:\s*({_NUMBER_PATTERN})",
        response,
        flags=re.IGNORECASE,
    ):
        canonical = re.sub(r"\s+", " ", raw_name.strip()).casefold()
        if canonical not in expected:
            raise ValueError(f"likelihood response contains unknown row {raw_name!r}")
        if canonical in parsed:
            raise ValueError(f"likelihood response duplicates row {raw_name!r}")
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"likelihood must be finite and in [0, 1], got {value}")
        parsed[canonical] = value
    if set(parsed) != set(expected):
        missing = sorted(set(expected) - set(parsed))
        raise ValueError(f"likelihood response is missing rows: {missing}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("estimator confidence must be in [0, 1]")
    smoothed = tuple(
        confidence
        * parsed[re.sub(r"\s+", " ", hypothesis.strip()).casefold()]
        + (1.0 - confidence) * 0.5
        for hypothesis in hypotheses
    )
    if not all(0.0 < value < 1.0 for value in smoothed):
        raise ValueError("smoothed likelihoods must be strictly between zero and one")
    return smoothed


def parse_answer(response: str) -> str:
    matches = re.findall(
        r"##Answer##[^:]*:\s*['\"]?(Yes|No)['\"]?",
        response,
        flags=re.IGNORECASE,
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one ##Answer## Yes/No marker, got {len(matches)}"
        )
    return matches[0].capitalize()


def case_context(case: dict[str, Any]) -> str:
    victim = case["victim"]
    parts = [
        f"Time: {case['time']}",
        f"Location: {case['location']}",
        (
            f"Victim: {victim['name']} - {victim['introduction']} "
            f"(Cause of death: {victim['cause_of_death']}, "
            f"Weapon: {victim['murder_weapon']})"
        ),
        "",
        "Suspects:",
    ]
    for suspect in case["suspects"]:
        parts.append(
            f"- {suspect['name']}: {suspect['introduction']}. "
            f"Reason at scene: {suspect['reason_at_scene']} "
            f"Testimony: {suspect['testimony']}"
        )
    return "\n".join(parts)


def question_prompt(
    case: dict[str, Any],
    history: Sequence[tuple[str, str]],
    belief: Belief,
    width: int,
    *,
    prompt_style: str = "published",
) -> str:
    possible = [
        hypothesis
        for hypothesis, probability in zip(
            belief.hypotheses,
            belief.probabilities,
            strict=True,
        )
        if probability > 0.0
    ]
    parts = [
        "You are the lead detective trying to solve a murder mystery.",
        "Your goal is to deduce the murderer among the remaining suspects.",
        f"The murderer is currently believed to be one of these: {'; '.join(possible)}.",
        "",
        "### Case Details ###",
        case_context(case),
    ]
    if history:
        parts.extend(
            [
                "",
                "### Interrogation History so far ###",
                *[
                    f"{index}. Q: {question}; A: {answer}"
                    for index, (question, answer) in enumerate(history, start=1)
                ],
            ]
        )
    parts.extend(
        [
            "",
            (
                f"What are {width} excellent yes/no questions that you could ask "
                "to narrow down the suspect list?"
            ),
            (
                "You MUST specify WHICH suspect you are asking each question to "
                "using the [Target: Name] format."
            ),
        ]
    )
    if prompt_style == "published":
        parts.extend(
            [
                "Provide a short explanation, then exactly one marker per question:",
                "##Question##: [Target: <Suspect Name>] <Your yes/no question here>",
            ]
        )
    elif prompt_style == "concise":
        parts.extend(
            [
                "Return no explanation or other text.",
                f"Return exactly {width} rows in this format:",
                "##Question##: [Target: <Suspect Name>] <Your yes/no question here>",
            ]
        )
    else:
        raise ValueError(f"unknown prompt style {prompt_style!r}")
    return "\n".join(parts)


def likelihood_prompt(
    case: dict[str, Any],
    question: str,
    hypotheses: Sequence[str],
    *,
    prompt_style: str = "published",
) -> str:
    parts = [
        "You are analyzing an interrogation in a murder mystery.",
        "",
        "### Case Details ###",
        case_context(case),
        "",
        f'The detective asked the following question: "{question}"',
        "",
        (
            'The possible answers are "Yes" and "No". The possible true '
            f"murderers are: {'; '.join(hypotheses)}."
        ),
        "",
        (
            "For each suspect, ASSUMING THEY ARE THE ACTUAL MURDERER, how "
            'likely is it that the targeted suspect would answer "Yes"?'
        ),
        (
            "Keep in mind that the target might be innocent and telling the "
            "truth, or might be the murderer and lying to protect themselves."
        ),
    ]
    if prompt_style == "published":
        parts.extend(
            [
                "Give a short explanation, then exactly one row per suspect:",
                "##<Suspect Name>##: <a single number between 0 and 1>",
            ]
        )
    elif prompt_style == "concise":
        parts.extend(
            [
                "Return no explanation or other text.",
                f"Return exactly {len(hypotheses)} rows, one per suspect:",
                "##<Suspect Name>##: <a single number between 0 and 1>",
            ]
        )
    else:
        raise ValueError(f"unknown prompt style {prompt_style!r}")
    return "\n".join(parts)


def answer_prompt(
    case: dict[str, Any],
    question: str,
    *,
    assumed_murderer: str | None = None,
) -> str:
    target = _target_from_question(
        question,
        [suspect["name"] for suspect in case["suspects"]],
    )
    suspect = next(item for item in case["suspects"] if item["name"] == target)
    clean_question = re.sub(
        r"\[Target:\s*.*?\]\s*",
        "",
        question,
        count=1,
        flags=re.IGNORECASE,
    )
    is_murderer = (
        bool(suspect["is_murderer"])
        if assumed_murderer is None
        else target.casefold() == assumed_murderer.strip().casefold()
    )
    if assumed_murderer is not None and assumed_murderer not in {
        item["name"] for item in case["suspects"]
    }:
        raise ValueError(f"unknown assumed murderer {assumed_murderer!r}")
    role = (
        "You are the MURDERER. You must lie if necessary to avoid getting caught, "
        "but try to sound consistent with your alibi."
        if is_murderer
        else (
            "You are INNOCENT. Tell the truth based on your knowledge and "
            "relationship to the victim."
        )
    )
    return "\n".join(
        [
            f"You are roleplaying as {suspect['name']} in a murder investigation.",
            "",
            "### Case Details ###",
            case_context(case),
            "",
            "### Your Profile ###",
            f"Name: {suspect['name']}",
            f"Story: {suspect['story']}",
            "",
            "### Your Secret Reality ###",
            role,
            "",
            f"The detective asks you: {clean_question!r}",
            "",
            (
                'Reply strictly with exactly "Yes" or "No" in the following '
                "format:"
            ),
            "##Answer##: <'Yes' or 'No'>",
        ]
    )


def _messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _write_raw(path: Path | None, raw: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _call_batch(
    model: Any,
    config: Config,
    prompts: Sequence[str],
    *,
    temperature: float,
    raw: dict[str, Any],
    raw_key: str,
    raw_path: Path | None,
) -> list[str]:
    responses = model.chat_complete_messages_batched(
        [_messages(prompt) for prompt in prompts],
        temperature=temperature,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    if len(responses) != len(prompts):
        raise ValueError(
            f"{raw_key}: expected {len(prompts)} responses, got {len(responses)}"
        )
    raw[raw_key] = {
        "prompt_sha256": [_prompt_hash(prompt) for prompt in prompts],
        "responses": responses,
    }
    _write_raw(raw_path, raw)
    return responses


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError(
            f"DetectiveCases.json hash mismatch: expected {DATA_SHA256}, got {digest}"
        )
    cases = json.loads(payload)
    if not isinstance(cases, list) or len(cases) != 100:
        raise ValueError("released CA-BED DetectiveCases.json must contain 100 cases")
    for index, case in enumerate(cases):
        suspects = case.get("suspects")
        if case.get("num") != index or not isinstance(suspects, list) or len(suspects) != 4:
            raise ValueError(f"invalid released Detective case at index {index}")
        if sum(bool(suspect.get("is_murderer")) for suspect in suspects) != 1:
            raise ValueError(f"case {index} must contain exactly one murderer")
    return cases


def _argmax(values: Sequence[float]) -> int:
    if not values:
        raise ValueError("cannot take argmax of empty sequence")
    return max(range(len(values)), key=lambda index: (float(values[index]), -index))


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: float(values[index]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and math.isclose(
                float(values[order[start]]),
                float(values[order[end]]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            end += 1
        rank = 0.5 * ((start + 1) + end)
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must align and contain at least two values")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = sum((x - left_mean) ** 2 for x in left_ranks)
    right_scale = sum((y - right_mean) ** 2 for y in right_ranks)
    if left_scale <= 0.0 or right_scale <= 0.0:
        return None
    return numerator / math.sqrt(left_scale * right_scale)


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    confidence: float = 0.90,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap values cannot be empty")
    rng = random.Random(seed)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        samples.append(
            sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        )
    samples.sort()
    tail = (1.0 - confidence) / 2.0
    lower_index = max(0, min(len(samples) - 1, int(tail * len(samples))))
    upper_index = max(
        0,
        min(len(samples) - 1, int((1.0 - tail) * len(samples)) - 1),
    )
    return samples[lower_index], samples[upper_index]


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = getattr(model, "usage_snapshot", None)
    return snapshot() if callable(snapshot) else {}


def run_stage(
    config: Config,
    *,
    stage: str,
    model: Any,
    cases: Sequence[dict[str, Any]],
    prompt_style: str = "published",
    raw_path: Path | None = None,
) -> dict[str, Any]:
    if stage not in {"serving_smoke", "ranking"}:
        raise ValueError("stage must be serving_smoke or ranking")
    if prompt_style not in {"published", "concise"}:
        raise ValueError("prompt_style must be published or concise")
    case_ids = SMOKE_CASE_IDS if stage == "serving_smoke" else RANKING_CASE_IDS
    selected_cases = [cases[index] for index in case_ids]
    raw: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "case_ids": list(case_ids),
    }
    beliefs = [
        Belief.uniform([suspect["name"] for suspect in case["suspects"]])
        for case in selected_cases
    ]
    histories: list[list[tuple[str, str]]] = [[] for _case in selected_cases]

    root_prompts = [
        question_prompt(
            case,
            history,
            belief,
            ROOT_WIDTH,
            prompt_style=prompt_style,
        )
        for case, history, belief in zip(
            selected_cases,
            histories,
            beliefs,
            strict=True,
        )
    ]
    root_responses = _call_batch(
        model,
        config,
        root_prompts,
        temperature=float(config.generation_temperature_diverse),
        raw=raw,
        raw_key="root_questions",
        raw_path=raw_path,
    )
    root_questions = [
        parse_questions(
            response,
            width=ROOT_WIDTH,
            hypotheses=belief.hypotheses,
            history=history,
        )
        for response, belief, history in zip(
            root_responses,
            beliefs,
            histories,
            strict=True,
        )
    ]

    root_keys: list[tuple[int, int]] = []
    root_likelihood_prompts: list[str] = []
    for case_index, (case, belief, questions) in enumerate(
        zip(selected_cases, beliefs, root_questions, strict=True)
    ):
        for root_index, question in enumerate(questions):
            root_keys.append((case_index, root_index))
            root_likelihood_prompts.append(
                likelihood_prompt(
                    case,
                    question,
                    belief.hypotheses,
                    prompt_style=prompt_style,
                )
            )
    root_likelihood_responses = _call_batch(
        model,
        config,
        root_likelihood_prompts,
        temperature=float(config.generation_temperature_simple),
        raw=raw,
        raw_key="root_likelihoods",
        raw_path=raw_path,
    )
    root_likelihoods: dict[tuple[int, int], tuple[float, ...]] = {
        key: parse_likelihoods(response, beliefs[key[0]].hypotheses)
        for key, response in zip(
            root_keys,
            root_likelihood_responses,
            strict=True,
        )
    }

    branch_keys: list[tuple[int, int, str]] = []
    branch_beliefs: dict[tuple[int, int, str], Belief] = {}
    branch_probabilities: dict[tuple[int, int, str], float] = {}
    followup_prompts: list[str] = []
    for case_index, (case, belief, questions) in enumerate(
        zip(selected_cases, beliefs, root_questions, strict=True)
    ):
        for root_index, question in enumerate(questions):
            likelihoods = root_likelihoods[(case_index, root_index)]
            for observation in ("Yes", "No"):
                key = (case_index, root_index, observation)
                posterior, marginal = bayes_update(
                    belief,
                    likelihoods,
                    observation,
                )
                branch_keys.append(key)
                branch_beliefs[key] = posterior
                branch_probabilities[key] = marginal
                followup_prompts.append(
                    question_prompt(
                        case,
                        [(question, observation)],
                        posterior,
                        FOLLOWUP_WIDTH,
                        prompt_style=prompt_style,
                    )
                )
    followup_responses = _call_batch(
        model,
        config,
        followup_prompts,
        temperature=float(config.generation_temperature_diverse),
        raw=raw,
        raw_key="followup_questions",
        raw_path=raw_path,
    )
    followup_questions = {
        key: parse_questions(
            response,
            width=FOLLOWUP_WIDTH,
            hypotheses=branch_beliefs[key].hypotheses,
            history=[(root_questions[key[0]][key[1]], key[2])],
        )
        for key, response in zip(branch_keys, followup_responses, strict=True)
    }

    followup_likelihood_keys: list[tuple[int, int, str, int]] = []
    followup_likelihood_prompts: list[str] = []
    for key in branch_keys:
        case_index, root_index, observation = key
        for followup_index, question in enumerate(followup_questions[key]):
            followup_likelihood_keys.append(
                (case_index, root_index, observation, followup_index)
            )
            followup_likelihood_prompts.append(
                likelihood_prompt(
                    selected_cases[case_index],
                    question,
                    branch_beliefs[key].hypotheses,
                    prompt_style=prompt_style,
                )
            )
    followup_likelihood_responses = _call_batch(
        model,
        config,
        followup_likelihood_prompts,
        temperature=float(config.generation_temperature_simple),
        raw=raw,
        raw_key="followup_likelihoods",
        raw_path=raw_path,
    )
    followup_likelihoods = {
        key: parse_likelihoods(response, beliefs[key[0]].hypotheses)
        for key, response in zip(
            followup_likelihood_keys,
            followup_likelihood_responses,
            strict=True,
        )
    }

    root_plans_many: list[tuple[RootPlan, ...]] = []
    for case_index, (belief, questions) in enumerate(
        zip(beliefs, root_questions, strict=True)
    ):
        plans: list[RootPlan] = []
        for root_index, question in enumerate(questions):
            root_row = root_likelihoods[(case_index, root_index)]
            root_eig = immediate_eig(belief, root_row)
            branches: list[BranchPlan] = []
            continuation = 0.0
            for observation in ("Yes", "No"):
                branch_key = (case_index, root_index, observation)
                rows = tuple(
                    followup_likelihoods[
                        (case_index, root_index, observation, followup_index)
                    ]
                    for followup_index in range(FOLLOWUP_WIDTH)
                )
                eig_scores = tuple(
                    immediate_eig(branch_beliefs[branch_key], row)
                    for row in rows
                )
                selected_index = _argmax(eig_scores)
                probability = branch_probabilities[branch_key]
                continuation += probability * eig_scores[selected_index]
                branches.append(
                    BranchPlan(
                        observation=observation,
                        probability=probability,
                        belief=branch_beliefs[branch_key],
                        questions=followup_questions[branch_key],
                        likelihoods=rows,
                        eig_scores=eig_scores,
                        selected_index=selected_index,
                    )
                )
            plans.append(
                RootPlan(
                    question=question,
                    likelihoods=root_row,
                    immediate_eig=root_eig,
                    depth_two_score=root_eig + continuation,
                    branches=(branches[0], branches[1]),
                )
            )
        root_plans_many.append(tuple(plans))

    root_answer_keys: list[tuple[int, int, int]] = []
    root_answer_prompts: list[str] = []
    for case_index, plans in enumerate(root_plans_many):
        for root_index, plan in enumerate(plans):
            for rollout_index in range(ANSWER_ROLLOUTS):
                root_answer_keys.append((case_index, root_index, rollout_index))
                root_answer_prompts.append(
                    answer_prompt(selected_cases[case_index], plan.question)
                )
    root_answer_responses = _call_batch(
        model,
        config,
        root_answer_prompts,
        temperature=float(config.answer_temperature),
        raw=raw,
        raw_key="root_answers",
        raw_path=raw_path,
    )
    root_answers = {
        key: parse_answer(response)
        for key, response in zip(
            root_answer_keys,
            root_answer_responses,
            strict=True,
        )
    }

    followup_answer_prompts: list[str] = []
    for key in root_answer_keys:
        case_index, root_index, _rollout_index = key
        observation = root_answers[key]
        followup_answer_prompts.append(
            answer_prompt(
                selected_cases[case_index],
                root_plans_many[case_index][root_index]
                .branch_for(observation)
                .selected_question,
            )
        )
    followup_answer_responses = _call_batch(
        model,
        config,
        followup_answer_prompts,
        temperature=float(config.answer_temperature),
        raw=raw,
        raw_key="followup_answers",
        raw_path=raw_path,
    )
    followup_answers = {
        key: parse_answer(response)
        for key, response in zip(
            root_answer_keys,
            followup_answer_responses,
            strict=True,
        )
    }

    records: list[dict[str, Any]] = []
    random_rng = random.Random(RANDOM_CONTROL_SEED)
    for case_index, (case_id, case, belief, plans) in enumerate(
        zip(case_ids, selected_cases, beliefs, root_plans_many, strict=True)
    ):
        truth = next(
            suspect["name"] for suspect in case["suspects"] if suspect["is_murderer"]
        )
        initial_truth_log_probability = belief.truth_log_probability(truth)
        root_records = []
        for root_index, plan in enumerate(plans):
            rollouts = []
            for rollout_index in range(ANSWER_ROLLOUTS):
                key = (case_index, root_index, rollout_index)
                root_observation = root_answers[key]
                branch = plan.branch_for(root_observation)
                after_root, _root_marginal = bayes_update(
                    belief,
                    plan.likelihoods,
                    root_observation,
                )
                followup_observation = followup_answers[key]
                after_followup, _followup_marginal = bayes_update(
                    after_root,
                    branch.likelihoods[branch.selected_index],
                    followup_observation,
                )
                truth_gain = (
                    after_followup.truth_log_probability(truth)
                    - initial_truth_log_probability
                )
                rollouts.append(
                    {
                        "root_answer": root_observation,
                        "followup_question": branch.selected_question,
                        "followup_answer": followup_observation,
                        "truth_log_probability_gain": truth_gain,
                        "final_truth_nll": -after_followup.truth_log_probability(truth),
                        "final_entropy": after_followup.entropy(),
                        "final_probabilities": dict(
                            zip(
                                after_followup.hypotheses,
                                after_followup.probabilities,
                                strict=True,
                            )
                        ),
                    }
                )
            mean_truth_gain = sum(
                item["truth_log_probability_gain"] for item in rollouts
            ) / ANSWER_ROLLOUTS
            mean_final_entropy = sum(
                item["final_entropy"] for item in rollouts
            ) / ANSWER_ROLLOUTS
            root_records.append(
                {
                    "question": plan.question,
                    "yes_probabilities": dict(
                        zip(belief.hypotheses, plan.likelihoods, strict=True)
                    ),
                    "immediate_eig": plan.immediate_eig,
                    "depth_two_score": plan.depth_two_score,
                    "branches": [
                        {
                            "observation": branch.observation,
                            "probability": branch.probability,
                            "questions": list(branch.questions),
                            "eig_scores": list(branch.eig_scores),
                            "selected_index": branch.selected_index,
                            "selected_question": branch.selected_question,
                        }
                        for branch in plan.branches
                    ],
                    "rollouts": rollouts,
                    "mean_realized_truth_log_probability_gain": mean_truth_gain,
                    "mean_final_entropy": mean_final_entropy,
                }
            )
        depth_one_index = _argmax(
            [record["immediate_eig"] for record in root_records]
        )
        depth_two_index = _argmax(
            [record["depth_two_score"] for record in root_records]
        )
        random_index = random_rng.randrange(ROOT_WIDTH)
        realized = [
            record["mean_realized_truth_log_probability_gain"]
            for record in root_records
        ]
        records.append(
            {
                "case_id": case_id,
                "location": case["location"],
                "victim": case["victim"]["name"],
                "truth": truth,
                "hypotheses": list(belief.hypotheses),
                "root_plans": root_records,
                "depth_one_selected_index": depth_one_index,
                "depth_two_selected_index": depth_two_index,
                "random_selected_index": random_index,
                "depth_one_selected_question": root_records[depth_one_index][
                    "question"
                ],
                "depth_two_selected_question": root_records[depth_two_index][
                    "question"
                ],
                "random_selected_question": root_records[random_index]["question"],
                "depth_one_truth_gain_spearman": spearman(
                    [record["immediate_eig"] for record in root_records],
                    realized,
                ),
                "depth_two_truth_gain_spearman": spearman(
                    [record["depth_two_score"] for record in root_records],
                    realized,
                ),
            }
        )

    usage = _usage_snapshot(model)
    expected_requests = EXPECTED_REQUESTS_PER_CASE * len(case_ids)
    summary = summarize_records(
        records,
        stage=stage,
        usage=usage,
        expected_requests=expected_requests,
    )
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "selection_seed": SELECTION_SEED,
            "random_control_seed": RANDOM_CONTROL_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "case_ids": list(case_ids),
            "confirmation_case_ids": list(CONFIRMATION_CASE_IDS),
            "data_sha256": DATA_SHA256,
            "root_width": ROOT_WIDTH,
            "followup_width": FOLLOWUP_WIDTH,
            "answer_rollouts_per_root": ANSWER_ROLLOUTS,
            "estimator_confidence": ESTIMATOR_CONFIDENCE,
            "shared_tree": True,
            "textual_likelihoods": True,
            "prompt_style": prompt_style,
            "raw_reasoning_requested": False,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def summarize_records(
    records: Sequence[dict[str, Any]],
    *,
    stage: str,
    usage: dict[str, Any],
    expected_requests: int,
) -> dict[str, Any]:
    depth_one_correlations = [
        float(record["depth_one_truth_gain_spearman"])
        for record in records
        if record["depth_one_truth_gain_spearman"] is not None
    ]
    depth_two_correlations = [
        float(record["depth_two_truth_gain_spearman"])
        for record in records
        if record["depth_two_truth_gain_spearman"] is not None
    ]
    paired_correlations = [
        (
            float(record["depth_one_truth_gain_spearman"]),
            float(record["depth_two_truth_gain_spearman"]),
        )
        for record in records
        if record["depth_one_truth_gain_spearman"] is not None
        and record["depth_two_truth_gain_spearman"] is not None
    ]
    differences_vs_one = []
    differences_vs_random = []
    entropy_differences = []
    distinct = 0
    wins_vs_one = 0
    for record in records:
        roots = record["root_plans"]
        depth_one = roots[record["depth_one_selected_index"]]
        depth_two = roots[record["depth_two_selected_index"]]
        random_root = roots[record["random_selected_index"]]
        difference_vs_one = (
            depth_two["mean_realized_truth_log_probability_gain"]
            - depth_one["mean_realized_truth_log_probability_gain"]
        )
        difference_vs_random = (
            depth_two["mean_realized_truth_log_probability_gain"]
            - random_root["mean_realized_truth_log_probability_gain"]
        )
        differences_vs_one.append(difference_vs_one)
        differences_vs_random.append(difference_vs_random)
        entropy_differences.append(
            depth_two["mean_final_entropy"] - depth_one["mean_final_entropy"]
        )
        distinct += int(
            record["depth_one_selected_index"] != record["depth_two_selected_index"]
        )
        wins_vs_one += int(difference_vs_one > 0.0)

    def mean(values: Sequence[float] | Any) -> float:
        materialized = list(values)
        return (
            sum(materialized) / len(materialized)
            if materialized
            else float("nan")
        )
    mean_depth_one_correlation = mean(depth_one_correlations)
    mean_depth_two_correlation = mean(depth_two_correlations)
    correlation_advantage = (
        mean(second - first for first, second in paired_correlations)
        if paired_correlations
        else float("nan")
    )
    mean_vs_one = mean(differences_vs_one)
    mean_vs_random = mean(differences_vs_random)
    mean_entropy_difference = mean(entropy_differences)
    ci_vs_one = _bootstrap_mean_ci(
        differences_vs_one,
        seed=BOOTSTRAP_SEED,
    )
    ci_vs_random = _bootstrap_mean_ci(
        differences_vs_random,
        seed=BOOTSTRAP_SEED + 1,
    )
    finite_endpoints = all(
        math.isfinite(value)
        for record in records
        for root in record["root_plans"]
        for value in (
            root["immediate_eig"],
            root["depth_two_score"],
            root["mean_realized_truth_log_probability_gain"],
            root["mean_final_entropy"],
        )
    )
    mechanics = {
        "all_cases_complete": len(records)
        == (len(SMOKE_CASE_IDS) if stage == "serving_smoke" else len(RANKING_CASE_IDS)),
        "all_tree_shapes_complete": all(
            len(record["root_plans"]) == ROOT_WIDTH
            and all(
                len(root["branches"]) == 2
                and all(
                    len(branch["questions"]) == FOLLOWUP_WIDTH
                    for branch in root["branches"]
                )
                and len(root["rollouts"]) == ANSWER_ROLLOUTS
                for root in record["root_plans"]
            )
            for record in records
        ),
        "all_endpoints_finite": finite_endpoints,
        "exact_request_count": int(usage.get("requests", -1)) == expected_requests,
        "zero_reasoning_tokens": int(usage.get("reasoning_tokens", -1)) == 0,
        "zero_forced_exits": int(usage.get("forced_exits", -1)) == 0,
    }
    if stage == "serving_smoke":
        gates = dict(mechanics)
    else:
        gates = {
            **mechanics,
            "at_least_nine_rankable_cases": len(paired_correlations) >= 9,
            "depth_two_distinct_at_least_four": distinct >= 4,
            "depth_two_mean_spearman_at_least_point_two": (
                math.isfinite(mean_depth_two_correlation)
                and mean_depth_two_correlation >= 0.20
            ),
            "depth_two_spearman_advantage_at_least_point_one": (
                math.isfinite(correlation_advantage)
                and correlation_advantage >= 0.10
            ),
            "depth_two_vs_one_mean_at_least_point_zero_two": mean_vs_one >= 0.02,
            "depth_two_vs_one_bootstrap_lower_positive": ci_vs_one[0] > 0.0,
            "depth_two_wins_at_least_seven": wins_vs_one >= 7,
            "depth_two_vs_random_mean_at_least_point_zero_two": (
                mean_vs_random >= 0.02
            ),
            "depth_two_vs_random_bootstrap_lower_positive": (
                ci_vs_random[0] > 0.0
            ),
            "depth_two_entropy_not_worse_than_point_zero_two": (
                mean_entropy_difference <= 0.02
            ),
        }
    gates["all_pass"] = all(gates.values())
    return {
        "num_cases": len(records),
        "expected_requests": expected_requests,
        "rankable_case_count": len(paired_correlations),
        "depth_two_distinct_root_count": distinct,
        "mean_depth_one_truth_gain_spearman": mean_depth_one_correlation,
        "mean_depth_two_truth_gain_spearman": mean_depth_two_correlation,
        "mean_depth_two_spearman_advantage": correlation_advantage,
        "mean_truth_nll_improvement_depth_two_vs_one": mean_vs_one,
        "truth_nll_improvement_depth_two_vs_one_bootstrap_90_ci": list(ci_vs_one),
        "depth_two_win_count_vs_one": wins_vs_one,
        "mean_truth_nll_improvement_depth_two_vs_random": mean_vs_random,
        "truth_nll_improvement_depth_two_vs_random_bootstrap_90_ci": list(
            ci_vs_random
        ),
        "mean_final_entropy_depth_two_minus_one": mean_entropy_difference,
        "gates": gates,
    }


class DeterministicMechanicsModel:
    """No-network model for orchestration and request-accounting tests."""

    def __init__(self) -> None:
        self.requests = 0

    @staticmethod
    def _hypotheses(prompt: str, marker: str) -> list[str]:
        match = re.search(
            rf"{re.escape(marker)}\s*(.*?)\.\s*(?:\n|$)",
            prompt,
            flags=re.DOTALL,
        )
        if match is None:
            raise ValueError(f"deterministic model could not parse {marker!r}")
        return [item.strip() for item in match.group(1).split(";")]

    def _response(self, prompt: str) -> str:
        if "excellent yes/no questions" in prompt:
            hypotheses = self._hypotheses(
                prompt,
                "The murderer is currently believed to be one of these:",
            )
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            bank = (
                "Did you enter the library before the victim?",
                "Did you handle the suspected weapon that evening?",
                "Did you speak privately with the victim?",
                "Did you leave the scene before the body was found?",
                "Did you hear an argument involving the victim?",
                "Did you see another suspect near the victim?",
            )
            offset = digest[0] % len(bank)
            history = {
                _canonical_question(value)
                for value in re.findall(
                    r"\d+\.\s+Q:\s*(.*?);\s+A:\s*(?:Yes|No)",
                    prompt,
                    flags=re.IGNORECASE,
                )
            }
            rows = []
            candidate_index = 0
            while len(rows) < ROOT_WIDTH and candidate_index < 4 * len(bank):
                target = hypotheses[
                    (digest[(candidate_index % 15) + 1] + candidate_index)
                    % len(hypotheses)
                ]
                question = bank[(offset + candidate_index) % len(bank)]
                row = f"[Target: {target}] {question}"
                if _canonical_question(row) not in history and row not in rows:
                    rows.append(row)
                candidate_index += 1
            if len(rows) != ROOT_WIDTH:
                raise ValueError("deterministic question bank was exhausted")
            rows = [f"##Question##: {row}" for row in rows]
            return "\n".join(rows)
        if "ASSUMING THEY ARE THE ACTUAL MURDERER" in prompt:
            hypotheses = self._hypotheses(
                prompt,
                "The possible true murderers are:",
            )
            rows = []
            for hypothesis in hypotheses:
                digest = hashlib.sha256(
                    f"{hypothesis}|{prompt}".encode("utf-8")
                ).digest()
                probability = 0.05 + 0.90 * int.from_bytes(
                    digest[:2],
                    "big",
                ) / 65535.0
                rows.append(f"##{hypothesis}##: {probability:.6f}")
            return "\n".join(rows)
        if "##Answer##" in prompt:
            digest = hashlib.sha256(prompt.encode("utf-8")).digest()
            return f"##Answer##: {'Yes' if digest[0] % 2 == 0 else 'No'}"
        raise ValueError("deterministic model received an unknown prompt")

    def chat_complete_messages_batched(
        self,
        batch_messages: Sequence[Sequence[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        self.requests += len(batch_messages)
        return [self._response(messages[-1]["content"]) for messages in batch_messages]

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "reasoning_tokens": 0,
            "forced_exits": 0,
            "run_cost_usd": 0.0,
        }


def _build_model(config: Config) -> Any:
    from model_factory import build_model_adapter

    if len(config.model_pairs) != 1:
        raise ValueError("Detective gate requires exactly one model pair")
    pair = config.model_pairs[0]
    if pair.questioner != pair.answerer:
        raise ValueError("Detective gate requires one shared questioner/answerer model")
    return build_model_adapter(pair.questioner, config=config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "ranking"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--prompt-style",
        choices=("published", "concise"),
        default="published",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    if args.stage == "serving_smoke" and args.prompt_style == "concise":
        config.openrouter_projected_cost_usd = 0.10
        config.openrouter_run_budget_usd = 0.50
    elif args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.20
        config.openrouter_run_budget_usd = 0.75
    elif args.prompt_style == "concise":
        config.openrouter_projected_cost_usd = 0.60
        config.openrouter_run_budget_usd = 2.00
    else:
        config.openrouter_projected_cost_usd = 1.20
        config.openrouter_run_budget_usd = 3.00

    output_name = "SERVING_SMOKE.json" if args.stage == "serving_smoke" else "GATE.json"
    failure_name = (
        "SERVING_SMOKE_FAILURE.json"
        if args.stage == "serving_smoke"
        else "GATE_FAILURE.json"
    )
    raw_path = args.output_dir / "RAW_RESPONSES.json"
    try:
        cases = load_cases(args.data_path)
        model = DeterministicMechanicsModel() if args.dry_run else _build_model(config)
        payload = run_stage(
            config,
            stage=args.stage,
            model=model,
            cases=cases,
            prompt_style=args.prompt_style,
            raw_path=raw_path,
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        (args.output_dir / failure_name).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
