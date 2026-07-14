"""Prompts for finite-target Bayesian design on MediQ."""

from __future__ import annotations

import json
from typing import Sequence

from core import BeliefState

from .types import MediQAction, MediQObservation, MediQTask


def _options_text(task: MediQTask) -> str:
    return "\n".join(f"{label}: {text}" for label, text in task.options)


def _history_text(
    history: Sequence[tuple[MediQAction, MediQObservation]],
) -> str:
    if not history:
        return "None"
    return "\n".join(
        f"Doctor: {action.query}\nPatient: {observation.reply}"
        for action, observation in history
    )


def prior_messages(task: MediQTask) -> list[dict[str, str]]:
    labels = list(task.option_labels)
    return [
        {
            "role": "system",
            "content": (
                "You are a calibrated clinical multiple-choice judge. Estimate a "
                "distribution over the finite answer labels from only the evidence shown. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial patient information:\n{task.initial_info}\n\n"
                f"Clinical question:\n{task.question}\n\nOptions:\n{_options_text(task)}\n\n"
                "Assign probabilities to the answer labels based only on the available "
                "patient information and standard medical knowledge. Do not assume hidden "
                "facts from the original case. Return "
                + json.dumps({"probabilities": {label: 0.0 for label in labels}})
                + ". Values must sum to 1."
            ),
        },
    ]


def posterior_messages(
    task: MediQTask,
    history: Sequence[tuple[MediQAction, MediQObservation]],
) -> list[dict[str, str]]:
    labels = list(task.option_labels)
    return [
        {
            "role": "system",
            "content": (
                "You are a calibrated clinical multiple-choice judge. Estimate a "
                "distribution over the finite answer labels from the complete observed "
                "conversation. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial patient information:\n{task.initial_info}\n\n"
                f"Conversation:\n{_history_text(history)}\n\n"
                f"Clinical question:\n{task.question}\n\nOptions:\n{_options_text(task)}\n\n"
                "Return "
                + json.dumps({"probabilities": {label: 0.0 for label in labels}})
                + ". Values must sum to 1."
            ),
        },
    ]


def candidate_messages(
    task: MediQTask,
    belief_state: BeliefState[str],
    history: Sequence[tuple[MediQAction, MediQObservation]],
    count: int,
    *,
    naive: bool = False,
) -> list[dict[str, str]]:
    prior = "\n".join(
        f"{label}: {task.option_text(label)} (p={probability:.4f})"
        for label, probability in zip(
            belief_state.hypotheses, belief_state.probabilities
        )
    )
    belief_section = "" if naive else f"\nCurrent answer belief:\n{prior}\n"
    purpose = (
        "Ask the single most useful missing clinical question."
        if naive
        else "Propose questions whose possible answers best distinguish the answer options."
    )
    return [
        {
            "role": "system",
            "content": (
                "You generate atomic patient questions for the MediQ interactive medical "
                "benchmark. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial patient information:\n{task.initial_info}\n\n"
                f"Conversation so far:\n{_history_text(history)}\n\n"
                f"Clinical question:\n{task.question}\n\nOptions:\n{_options_text(task)}\n"
                f"{belief_section}\n{purpose} Return exactly {count} candidate(s). Each "
                "query must ask about exactly one observable variable: one patient fact, "
                "symptom, history item, examination, or test result. Never join distinct "
                "variables or qualifiers with 'and', 'or', or 'and/or'. Do not ask the "
                "patient to solve the multiple-choice question or name an option. Do not "
                "repeat an earlier query or ask for information already explicit in the "
                "initial information or conversation. For each query, give 3-5 response "
                "categories that form a mutually exclusive and collectively exhaustive "
                "partition of every possible record-grounded answer. For a numeric variable, "
                "use contiguous non-overlapping intervals covering the full plausible range; "
                "do not leave gaps. Include exactly one unavailable category, written exactly "
                "as 'Information unavailable / not in record', and no synonym such as 'not "
                "recorded' or 'unknown'. "
                "Return {\"candidates\":[{\"query\":\"...\",\"outcomes\":[\"...\"]}]} "
                "and nothing else."
            ),
        },
    ]


def candidate_validation_messages(
    task: MediQTask,
    action: MediQAction,
    history: Sequence[tuple[MediQAction, MediQObservation]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict MediQ candidate-space auditor. Check structural validity, "
                "not whether you personally prefer the clinical question. Return strict "
                "JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial patient information:\n{task.initial_info}\n\n"
                f"Conversation so far:\n{_history_text(history)}\n\n"
                f"Clinical question:\n{task.question}\n\nOptions:\n{_options_text(task)}\n\n"
                f"Proposed doctor query: {action.query}\n"
                f"Proposed response categories: {json.dumps(list(action.outcomes))}\n\n"
                "Mark valid=true only when all conditions hold: the query asks exactly one "
                "observable variable; it contains no answer-option leakage; it is not already "
                "answered above; its non-unavailable categories are mutually exclusive and "
                "collectively exhaustive; numeric ranges have no gaps or overlaps; and there "
                "is exactly one unavailable category. Do not infer missing categories from "
                "intent. Return {\"valid\":true,\"reason\":\"brief structural reason\"}."
            ),
        },
    ]


def likelihood_messages(hypothesis: str, action: MediQAction) -> list[dict[str, str]]:
    task = action.task
    outcomes = list(action.outcomes)
    return [
        {
            "role": "system",
            "content": (
                "You are a calibrated clinical generative model. Estimate how a patient "
                "response category would vary if a specified answer option were correct. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial patient information:\n{task.initial_info}\n\n"
                f"Conversation so far:\n"
                + (
                    "\n".join(
                        f"Doctor: {query}\nPatient: {reply}"
                        for query, reply in action.transcript
                    )
                    or "None"
                )
                + f"\n\nClinical question:\n{task.question}\n\nOptions:\n{_options_text(task)}\n\n"
                f"Assume the correct answer is {hypothesis}: {task.option_text(hypothesis)}.\n"
                f"Next doctor query: {action.query}\nResponse categories: {json.dumps(outcomes)}\n\n"
                "Estimate P(response category | assumed correct answer, available history, "
                "query). Keep uncertainty soft rather than assigning unjustified zeroes. "
                "Return "
                + json.dumps({"probabilities": {outcome: 0.0 for outcome in outcomes}})
                + ". Values must sum to 1."
            ),
        },
    ]


def patient_fact_messages(task: MediQTask, query: str, max_facts: int) -> list[dict[str, str]]:
    facts = "\n".join(f"{index}: {fact}" for index, fact in enumerate(task.facts))
    return [
        {
            "role": "system",
            "content": (
                "You are the official-style MediQ Fact-Select patient. You may reveal only "
                "facts explicitly present in the supplied patient record. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Atomic patient facts:\n{facts}\n\nDoctor question: {query}\n\n"
                f"Select at most {max_facts} fact indices that directly answer the question. "
                "Every requested qualifier must be explicit in the selected facts. Do not "
                "infer or add information: for example, being sexually active does not "
                "establish new partners or unprotected sex, and symptom presence does not "
                "establish its frequency or severity. If the facts only partially or "
                "implicitly address the question, select no indices and set cannot_answer "
                "to true. Return "
                '{"fact_indices":[0],"cannot_answer":false} and nothing else.'
            ),
        },
    ]


def relevance_messages(query: str, reply: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict MediQ explicit-entailment auditor. Judge only whether the "
                "supplied record facts explicitly answer the doctor query. You are not shown "
                "response categories. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Doctor query: {query}\nSelected patient-record facts: {reply}\n\n"
                "Set relevant=true only if the selected text explicitly answers the single "
                "requested variable and every qualifier. Do not use clinical, commonsense, "
                "or demographic inference. A generic fact does not establish a more specific "
                "qualifier, frequency, severity, timing, or numeric range. Return "
                '{"relevant":true,"reason":"brief entailment reason"}.'
            ),
        },
    ]


def mapping_messages(reply: str, action: MediQAction) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Map an already-validated, explicitly relevant MediQ patient response to one "
                "supplied response category. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Doctor query: {action.query}\nPatient response: {reply}\n"
                f"Response categories: {json.dumps(list(action.outcomes))}\n\n"
                "Set clean=true only if the literal response explicitly entails exactly one "
                "category. Do not infer an unstated qualifier, frequency, severity, timing, "
                "or range. If no category or multiple categories fit, set clean=false and "
                "outcome=null. Return "
                '{"clean":true,"outcome":"exact supplied category"}.'
            ),
        },
    ]


def repair_messages(
    messages: Sequence[dict[str, str]],
    response: str,
    error: str,
) -> list[dict[str, str]]:
    return list(messages) + [
        {"role": "assistant", "content": response},
        {
            "role": "user",
            "content": (
                f"That response was invalid: {error}. Return only corrected strict JSON "
                "matching the requested schema."
            ),
        },
    ]
