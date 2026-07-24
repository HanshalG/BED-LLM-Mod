#!/usr/bin/env python3
"""Paired CA-BED depth-one/depth-two ranking gate for Animals."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import BeliefState
from environments.animals import AnimalsBEDEnvironment
from helpers import Config, load_config


SELECTION_SEED = 24310
BOOTSTRAP_SEED = 24311
ROOT_WIDTH = 4
FOLLOWUP_WIDTH = 3
BOOTSTRAP_SAMPLES = 10_000
SMOKE_TARGETS = ("Wombat", "Aardvark")
FORMAL_TARGETS = (
    "Red fox",
    "Reindeer",
    "Bald eagle",
    "Saltwater crocodile",
    "Manatee",
    "Horseshoe crab",
    "Cheetah",
    "Tasmanian devil",
    "Coyote",
    "African grey parrot",
    "Alpaca",
    "Giant panda",
    "African elephant",
    "Green sea turtle",
    "Kangaroo",
    "Praying mantis",
    "Meerkat",
    "Yak",
    "Tiger shark",
    "Bottlenose dolphin",
    "Honey badger",
    "Pangolin",
    "Giant squid",
    "Walrus",
)
PREHISTORY_QUESTIONS = (
    "Is it warm-blooded?",
    "Does it primarily live in water?",
    "Is it typically capable of powered flight?",
    "Is it an invertebrate?",
    "Is it commonly domesticated by humans?",
    "Is it primarily nocturnal?",
    "Is it native to Africa?",
    "Is it usually larger than an adult human?",
)


@dataclass(frozen=True)
class BranchPlan:
    observation: str
    probability: float
    belief_state: BeliefState[str]
    candidates: tuple[str, ...]
    eig_scores: tuple[float, ...]
    selected_question: str
    selected_eig: float


@dataclass(frozen=True)
class RootPlan:
    question: str
    immediate_eig: float
    depth_two_score: float
    yes_probability: float
    branches: tuple[BranchPlan, BranchPlan]

    def branch_for(self, observation: str) -> BranchPlan:
        for branch in self.branches:
            if branch.observation == observation:
                return branch
        raise ValueError(f"no branch plan for observation {observation!r}")


def _binary_entropy(probability: float) -> float:
    probability = min(max(float(probability), 0.0), 1.0)
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    complement = 1.0 - probability
    return -probability * math.log(probability) - complement * math.log(complement)


def immediate_eig(
    belief_state: BeliefState[str],
    yes_probabilities: Sequence[float],
) -> float:
    prior = np.asarray(belief_state.probabilities, dtype=float)
    yes = np.asarray(yes_probabilities, dtype=float)
    if len(prior) != len(yes):
        raise ValueError("semantic likelihood row must match belief support")
    marginal_yes = float(np.dot(prior, yes))
    conditional_entropy = float(
        np.dot(prior, np.asarray([_binary_entropy(value) for value in yes]))
    )
    return _binary_entropy(marginal_yes) - conditional_entropy


def bayes_update(
    belief_state: BeliefState[str],
    yes_probabilities: Sequence[float],
    observation: str,
) -> BeliefState[str]:
    if observation not in {"Yes", "No"}:
        raise ValueError("fixed-support CA-BED observations must be Yes or No")
    prior = np.asarray(belief_state.probabilities, dtype=float)
    yes = np.asarray(yes_probabilities, dtype=float)
    if len(prior) != len(yes):
        raise ValueError("semantic likelihood row must match belief support")
    likelihood = yes if observation == "Yes" else 1.0 - yes
    return BeliefState.from_unnormalized(
        belief_state.hypotheses,
        prior * np.clip(likelihood, 1.0e-12, 1.0),
        fallback_to_uniform=True,
    ).sorted_descending()


def _canonical_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip()).casefold().rstrip("?.!")


def _is_direct_guess(question: str, support: Sequence[str]) -> bool:
    canonical = _canonical_question(question)
    guesses = set()
    for animal in support:
        label = _canonical_question(animal)
        guesses.update(
            {
                f"is it {label}",
                f"is it a {label}",
                f"is it an {label}",
                f"is the animal {label}",
            }
        )
    return canonical in guesses


def select_valid_questions(
    candidates: Sequence[str],
    *,
    width: int,
    history: Sequence[tuple[str, str]],
    support: Sequence[str],
) -> tuple[str, ...]:
    forbidden = {_canonical_question(question) for question, _answer in history}
    selected: list[str] = []
    seen = set(forbidden)
    for candidate in candidates:
        cleaned = re.sub(r"\s+", " ", str(candidate).strip())
        canonical = _canonical_question(cleaned)
        if (
            not canonical
            or canonical in seen
            or _is_direct_guess(cleaned, support)
        ):
            continue
        seen.add(canonical)
        selected.append(cleaned)
        if len(selected) == width:
            break
    if len(selected) != width:
        raise ValueError(
            f"required {width} valid questions, received {len(selected)}"
        )
    return tuple(selected)


def _generate_questions(
    env: AnimalsBEDEnvironment,
    belief_state: BeliefState[str],
    history: Sequence[tuple[str, str]],
    model: Any,
    config: Config,
    *,
    width: int,
) -> tuple[str, ...]:
    original_width = config.target_num_questions
    config.target_num_questions = width
    try:
        candidates = env.generate_candidate_actions(
            belief_state,
            history,
            model,
            config,
        )
    finally:
        config.target_num_questions = original_width
    return select_valid_questions(
        candidates,
        width=width,
        history=history,
        support=belief_state.hypotheses,
    )


def build_shared_tree(
    env: AnimalsBEDEnvironment,
    belief_state: BeliefState[str],
    history: Sequence[tuple[str, str]],
    model: Any,
    config: Config,
) -> tuple[RootPlan, ...]:
    roots = _generate_questions(
        env,
        belief_state,
        history,
        model,
        config,
        width=ROOT_WIDTH,
    )
    root_table = env.semantic_yes_probabilities_many(
        belief_state.hypotheses,
        roots,
    )

    pending: list[
        tuple[
            str,
            float,
            float,
            list[tuple[str, float, BeliefState[str], tuple[str, ...]]],
        ]
    ] = []
    all_followups: list[str] = []
    for root_index, root in enumerate(roots):
        yes_row = root_table[root_index]
        marginal_yes = float(
            np.dot(
                np.asarray(belief_state.probabilities, dtype=float),
                yes_row,
            )
        )
        branches = []
        for observation, branch_probability in (
            ("Yes", marginal_yes),
            ("No", 1.0 - marginal_yes),
        ):
            branch_belief = bayes_update(
                belief_state,
                yes_row,
                observation,
            )
            branch_history = [*history, (root, observation)]
            followups = _generate_questions(
                env,
                branch_belief,
                branch_history,
                model,
                config,
                width=FOLLOWUP_WIDTH,
            )
            all_followups.extend(followups)
            branches.append(
                (
                    observation,
                    branch_probability,
                    branch_belief,
                    followups,
                )
            )
        pending.append(
            (
                root,
                immediate_eig(belief_state, yes_row),
                marginal_yes,
                branches,
            )
        )

    unique_followups = tuple(dict.fromkeys(all_followups))
    env.semantic_yes_probabilities_many(
        belief_state.hypotheses,
        unique_followups,
    )

    plans: list[RootPlan] = []
    for root, root_eig, marginal_yes, pending_branches in pending:
        branch_plans: list[BranchPlan] = []
        future_value = 0.0
        for (
            observation,
            branch_probability,
            branch_belief,
            followups,
        ) in pending_branches:
            followup_table = env.semantic_yes_probabilities_many(
                branch_belief.hypotheses,
                followups,
            )
            eig_scores = tuple(
                immediate_eig(branch_belief, row)
                for row in followup_table
            )
            selected_index = int(np.argmax(eig_scores))
            selected_eig = float(eig_scores[selected_index])
            future_value += branch_probability * selected_eig
            branch_plans.append(
                BranchPlan(
                    observation=observation,
                    probability=float(branch_probability),
                    belief_state=branch_belief,
                    candidates=followups,
                    eig_scores=eig_scores,
                    selected_question=followups[selected_index],
                    selected_eig=selected_eig,
                )
            )
        plans.append(
            RootPlan(
                question=root,
                immediate_eig=float(root_eig),
                depth_two_score=float(root_eig + future_value),
                yes_probability=float(marginal_yes),
                branches=(branch_plans[0], branch_plans[1]),
            )
        )
    return tuple(plans)


def _duplicate_answer(
    env: AnimalsBEDEnvironment,
    question: str,
    target: str,
    *,
    seed: int,
) -> str:
    answers = [
        env.observe(
            question,
            target,
            np.random.default_rng(seed + duplicate_index),
        )
        for duplicate_index in range(2)
    ]
    if answers[0] != answers[1]:
        raise ValueError(
            f"duplicate answer disagreement for {target!r}, {question!r}: "
            f"{answers!r}"
        )
    if answers[0] not in {"Yes", "No"}:
        raise ValueError(
            f"answer must be Yes or No, received {answers[0]!r}"
        )
    return answers[0]


def _truth_probability(
    belief_state: BeliefState[str],
    target: str,
) -> float:
    target_key = target.strip().casefold()
    return belief_state.probability_of(
        lambda hypothesis: hypothesis.strip().casefold() == target_key
    )


def realize_root(
    env: AnimalsBEDEnvironment,
    belief_state: BeliefState[str],
    target: str,
    plan: RootPlan,
    *,
    seed: int,
) -> dict[str, Any]:
    root_answer = _duplicate_answer(
        env,
        plan.question,
        target,
        seed=seed,
    )
    branch = plan.branch_for(root_answer)
    followup_answer = _duplicate_answer(
        env,
        branch.selected_question,
        target,
        seed=seed + 2,
    )
    followup_yes = env.semantic_yes_probabilities_many(
        branch.belief_state.hypotheses,
        [branch.selected_question],
    )[0]
    final_belief = bayes_update(
        branch.belief_state,
        followup_yes,
        followup_answer,
    )
    start_truth_probability = _truth_probability(belief_state, target)
    final_truth_probability = _truth_probability(final_belief, target)
    start_nll = -math.log(max(start_truth_probability, 1.0e-300))
    final_nll = -math.log(max(final_truth_probability, 1.0e-300))
    return {
        "root_answer": root_answer,
        "followup_question": branch.selected_question,
        "followup_answer": followup_answer,
        "start_entropy": belief_state.entropy(),
        "final_entropy": final_belief.entropy(),
        "entropy_drop": belief_state.entropy() - final_belief.entropy(),
        "start_truth_probability": start_truth_probability,
        "final_truth_probability": final_truth_probability,
        "start_truth_nll": start_nll,
        "final_truth_nll": final_nll,
        "truth_log_gain": start_nll - final_nll,
    }


def _prehistory_for_state(state_index: int) -> tuple[str, str]:
    first = (2 * state_index) % len(PREHISTORY_QUESTIONS)
    second = (2 * state_index + 3) % len(PREHISTORY_QUESTIONS)
    return PREHISTORY_QUESTIONS[first], PREHISTORY_QUESTIONS[second]


def run_state(
    env: AnimalsBEDEnvironment,
    questioner: Any,
    config: Config,
    *,
    state_index: int,
    target: str,
) -> dict[str, Any]:
    belief_state = env.initial_belief_state(questioner, config)
    history: list[tuple[str, str]] = []
    prehistory_questions = _prehistory_for_state(state_index)
    env.semantic_yes_probabilities_many(
        belief_state.hypotheses,
        prehistory_questions,
    )
    for question_index, question in enumerate(prehistory_questions):
        answer = _duplicate_answer(
            env,
            question,
            target,
            seed=SELECTION_SEED + state_index * 1000 + question_index * 10,
        )
        history.append((question, answer))
        belief_state = env.update_belief_state(
            belief_state,
            history,
            questioner,
            config,
        )

    plans = build_shared_tree(
        env,
        belief_state,
        history,
        questioner,
        config,
    )
    outcomes = [
        realize_root(
            env,
            belief_state,
            target,
            plan,
            seed=SELECTION_SEED + state_index * 1000 + 100 + root_index * 10,
        )
        for root_index, plan in enumerate(plans)
    ]
    depth_one_index = int(np.argmax([plan.immediate_eig for plan in plans]))
    depth_two_index = int(np.argmax([plan.depth_two_score for plan in plans]))
    random_index = int(
        np.random.default_rng(SELECTION_SEED + state_index).integers(
            0,
            len(plans),
        )
    )
    return {
        "state_index": state_index,
        "target": target,
        "prehistory": [
            {"question": question, "answer": answer}
            for question, answer in history
        ],
        "start_support_size": belief_state.support_size,
        "start_entropy": belief_state.entropy(),
        "start_truth_probability": _truth_probability(belief_state, target),
        "roots": [
            {
                "question": plan.question,
                "immediate_eig": plan.immediate_eig,
                "depth_two_score": plan.depth_two_score,
                "yes_probability": plan.yes_probability,
                "branches": [
                    {
                        "observation": branch.observation,
                        "probability": branch.probability,
                        "candidates": list(branch.candidates),
                        "eig_scores": list(branch.eig_scores),
                        "selected_question": branch.selected_question,
                        "selected_eig": branch.selected_eig,
                    }
                    for branch in plan.branches
                ],
                "realized": outcome,
            }
            for plan, outcome in zip(plans, outcomes)
        ],
        "selections": {
            "depth_one": depth_one_index,
            "depth_two": depth_two_index,
            "random": random_index,
        },
    }


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    ranks_a = _average_ranks(values_a)
    ranks_b = _average_ranks(values_b)
    if np.std(ranks_a) == 0.0 or np.std(ranks_b) == 0.0:
        return 0.0
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.90,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        len(array),
        size=(BOOTSTRAP_SAMPLES, len(array)),
    )
    means = np.mean(array[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


def summarize_records(
    records: Sequence[dict[str, Any]],
    *,
    formal: bool,
) -> dict[str, Any]:
    depth_one_correlations = []
    depth_two_correlations = []
    depth_two_vs_one = []
    depth_two_vs_random = []
    depth_two_entropy_minus_one = []
    distinct_roots = 0
    for record in records:
        roots = record["roots"]
        truth_gains = [root["realized"]["truth_log_gain"] for root in roots]
        depth_one_correlations.append(
            spearman(
                [root["immediate_eig"] for root in roots],
                truth_gains,
            )
        )
        depth_two_correlations.append(
            spearman(
                [root["depth_two_score"] for root in roots],
                truth_gains,
            )
        )
        selections = record["selections"]
        depth_one = roots[selections["depth_one"]]["realized"]
        depth_two = roots[selections["depth_two"]]["realized"]
        random = roots[selections["random"]]["realized"]
        depth_two_vs_one.append(
            depth_one["final_truth_nll"] - depth_two["final_truth_nll"]
        )
        depth_two_vs_random.append(
            random["final_truth_nll"] - depth_two["final_truth_nll"]
        )
        depth_two_entropy_minus_one.append(
            depth_two["final_entropy"] - depth_one["final_entropy"]
        )
        distinct_roots += int(
            selections["depth_one"] != selections["depth_two"]
        )

    mean_depth_one_correlation = float(np.mean(depth_one_correlations))
    mean_depth_two_correlation = float(np.mean(depth_two_correlations))
    mean_depth_two_vs_one = float(np.mean(depth_two_vs_one))
    mean_depth_two_vs_random = float(np.mean(depth_two_vs_random))
    depth_two_vs_one_ci = _bootstrap_mean_ci(depth_two_vs_one)
    depth_two_vs_random_ci = _bootstrap_mean_ci(depth_two_vs_random)
    depth_two_wins = sum(value > 1.0e-12 for value in depth_two_vs_one)
    mean_entropy_difference = float(np.mean(depth_two_entropy_minus_one))

    gates = {
        "all_states_complete": all(
            len(record["roots"]) == ROOT_WIDTH
            and all(
                len(branch["candidates"]) == FOLLOWUP_WIDTH
                for root in record["roots"]
                for branch in root["branches"]
            )
            for record in records
        ),
        "all_recorded_answers_are_binary": all(
            entry["answer"] in {"Yes", "No"}
            for record in records
            for entry in record["prehistory"]
        )
        and all(
            root["realized"]["root_answer"] in {"Yes", "No"}
            and root["realized"]["followup_answer"] in {"Yes", "No"}
            for record in records
            for root in record["roots"]
        ),
        "all_duplicate_answers_agree": True,
    }
    if formal:
        gates.update(
            {
                "exact_formal_state_count": len(records) == len(FORMAL_TARGETS),
                "depth_two_distinct_root_count_at_least_6": distinct_roots >= 6,
                "depth_two_truth_gain_spearman_at_least_0_20": (
                    mean_depth_two_correlation >= 0.20
                ),
                "depth_two_spearman_advantage_at_least_0_10": (
                    mean_depth_two_correlation
                    - mean_depth_one_correlation
                    >= 0.10
                ),
                "depth_two_vs_one_mean_nll_improvement_at_least_0_02": (
                    mean_depth_two_vs_one >= 0.02
                ),
                "depth_two_vs_one_bootstrap_lower_above_zero": (
                    depth_two_vs_one_ci[0] > 0.0
                ),
                "depth_two_wins_at_least_14": depth_two_wins >= 14,
                "depth_two_vs_random_mean_nll_improvement_at_least_0_02": (
                    mean_depth_two_vs_random >= 0.02
                ),
                "depth_two_vs_random_bootstrap_lower_above_zero": (
                    depth_two_vs_random_ci[0] > 0.0
                ),
                "depth_two_entropy_not_worse_by_0_02": (
                    mean_entropy_difference <= 0.02
                ),
            }
        )
    gates["all_pass"] = all(gates.values())
    return {
        "num_states": len(records),
        "mean_depth_one_truth_gain_spearman": mean_depth_one_correlation,
        "mean_depth_two_truth_gain_spearman": mean_depth_two_correlation,
        "mean_depth_two_spearman_advantage": (
            mean_depth_two_correlation - mean_depth_one_correlation
        ),
        "depth_two_distinct_root_count": distinct_roots,
        "mean_truth_nll_improvement_depth_two_vs_one": mean_depth_two_vs_one,
        "truth_nll_improvement_depth_two_vs_one_bootstrap_90_ci": list(
            depth_two_vs_one_ci
        ),
        "depth_two_win_count_vs_one": depth_two_wins,
        "mean_truth_nll_improvement_depth_two_vs_random": (
            mean_depth_two_vs_random
        ),
        "truth_nll_improvement_depth_two_vs_random_bootstrap_90_ci": list(
            depth_two_vs_random_ci
        ),
        "mean_final_entropy_depth_two_minus_one": mean_entropy_difference,
        "gates": gates,
    }


def run_stage(
    config: Config,
    *,
    stage: str,
    questioner: Any,
    answerer: Any,
) -> dict[str, Any]:
    if stage not in {"serving_smoke", "formal"}:
        raise ValueError("stage must be serving_smoke or formal")
    targets = SMOKE_TARGETS if stage == "serving_smoke" else FORMAL_TARGETS
    support = tuple(config.animals[config.version])
    if len(support) != 64 or len(set(support)) != 64:
        raise ValueError("V10 requires the frozen 64-animal support")
    if not set(targets).issubset(support):
        raise ValueError("all V10 targets must be in the fixed support")
    env = AnimalsBEDEnvironment(
        config=config,
        answerer=answerer,
        target_animals=list(targets),
    )
    records = [
        run_state(
            env,
            questioner,
            config,
            state_index=state_index,
            target=target,
        )
        for state_index, target in enumerate(targets)
    ]
    summary = summarize_records(
        records,
        formal=stage == "formal",
    )
    likelihood_cache_valid = all(
        math.isfinite(yes_probability)
        and math.isfinite(no_probability)
        and 0.0 < yes_probability < 1.0
        and 0.0 < no_probability < 1.0
        and math.isclose(
            yes_probability + no_probability,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for yes_probability, no_probability in env._likelihood_cache.values()
    )
    summary["gates"][
        "all_cached_likelihoods_finite_and_complementary"
    ] = likelihood_cache_valid
    summary["gates"]["all_pass"] = all(summary["gates"].values())
    return {
        "schema_version": 10,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "selection_seed": SELECTION_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "root_width": ROOT_WIDTH,
            "followup_width": FOLLOWUP_WIDTH,
            "likelihood_confidence": config.animals_likelihood_confidence,
            "belief_update_mode": config.animals_belief_update_mode,
            "support_size": len(support),
            "targets": list(targets),
            "prehistory_questions": list(PREHISTORY_QUESTIONS),
            "duplicate_answers_required": True,
            "shared_tree_across_controls": True,
            "raw_reasoning_requested": False,
        },
        "summary": summary,
        "records": records,
        "likelihood_cache_entries": len(env._likelihood_cache),
    }


class DeterministicMechanicsModel:
    """No-network model used only to verify V10 orchestration."""

    candidate_bank = (
        "Does it have fur?",
        "Does it lay eggs?",
        "Does it have more than four legs?",
        "Does it mainly eat other animals?",
        "Does it spend most of its life underground?",
        "Does it have a shell?",
        "Does it live mainly in trees?",
        "Is it able to breathe underwater?",
        "Does it have hooves?",
        "Does it migrate seasonally?",
    )

    def __init__(self) -> None:
        self.complete_calls = 0
        self.probability_calls = 0

    def chat_complete(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.0,
        num_responses: int = 1,
    ) -> list[str]:
        self.complete_calls += 1
        system = messages[0]["content"]
        if "Your chosen entity is" in system:
            match = re.search(
                r"Your chosen entity is:\s*(.*?)\s*When asked",
                system,
                flags=re.DOTALL,
            )
            if match is None:
                raise ValueError("could not parse deterministic target")
            target = match.group(1).strip()
            question = messages[-1]["content"].strip()
            digest = hashlib.sha256(
                f"{target}|{question}".encode("utf-8")
            ).digest()
            return ["Yes" if digest[0] % 2 == 0 else "No"]
        return ["\n".join(self.candidate_bank)]

    def chat_probabilities_messages_batched(
        self,
        messages: Sequence[Sequence[dict[str, str]]],
        responses: Sequence[str],
        temperature: float = 0.0,
        block_size: int = 50,
    ) -> list[dict[str, float]]:
        self.probability_calls += 1
        rows = []
        for conversation in messages:
            payload = conversation[-1]["content"]
            digest = hashlib.sha256(payload.encode("utf-8")).digest()
            yes_probability = 0.1 + 0.8 * int.from_bytes(
                digest[:2],
                "big",
            ) / 65535.0
            rows.append(
                {
                    str(responses[0]): yes_probability,
                    str(responses[1]): 1.0 - yes_probability,
                }
            )
        return rows


def _build_models(config: Config) -> tuple[Any, Any]:
    from model import build_model_adapter

    if len(config.model_pairs) != 1:
        raise ValueError("V10 requires exactly one model pair")
    pair = config.model_pairs[0]
    questioner = build_model_adapter(pair.questioner, config=config)
    if pair.answerer == pair.questioner:
        answerer = questioner
    else:
        answerer = build_model_adapter(pair.answerer, config=config)
    return questioner, answerer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("serving_smoke", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    output_name = "SERVING_SMOKE.json" if args.stage == "serving_smoke" else "GATE.json"
    failure_name = (
        "SERVING_SMOKE_FAILURE.json"
        if args.stage == "serving_smoke"
        else "GATE_FAILURE.json"
    )
    if args.dry_run:
        questioner = DeterministicMechanicsModel()
        answerer = questioner
    else:
        questioner, answerer = _build_models(config)
    try:
        payload = run_stage(
            config,
            stage=args.stage,
            questioner=questioner,
            answerer=answerer,
        )
    except Exception as exc:
        failure = {
            "schema_version": 10,
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
