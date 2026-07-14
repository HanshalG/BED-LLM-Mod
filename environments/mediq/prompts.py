"""Prompts for finite-target Bayesian design on MediQ."""

from __future__ import annotations

import json
from typing import Sequence

from core import BeliefState

from .types import MediQAction, MediQObservation, MediQTask


ANSWERABLE_RECORD_OUTCOME = "Answerable from record"
UNANSWERABLE_RECORD_OUTCOME = "Not answerable from record"


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
                "query must be one medically meaningful yes/no predicate about exactly one "
                "observable patient fact, symptom, history item, examination, or test result. "
                "Start it with Is/Are/Was/Were/Has/Have/Had/Do/Does/Did/Can/Could/Would/Will. "
                "Never join distinct variables or qualifiers with 'and', 'or', or 'and/or'. "
                "Do not ask an open-ended 'what' question. Do not ask the "
                "patient to solve the multiple-choice question or name an option. Do not "
                "repeat an earlier query or ask for information already explicit in the "
                "initial information or conversation; paraphrases count as repeats even when "
                "the earlier answer was unavailable. Ask only for pre-decision patient "
                "evidence. Never ask whether a diagnosis was made, a treatment or drug was "
                "given, a test was ordered or performed, or a management action was chosen. "
                "Never paraphrase an answer option through a drug class, mechanism, diagnosis, "
                "or procedure. The predicate must be directly answerable from one atomic fact "
                "or an explicit numeric comparison; do not ask for a derived judgment such as "
                "'hemodynamically stable'. Phrase numeric questions as explicit "
                "threshold predicates, such as 'Was the glucose above 250 mg/dL?', rather "
                "than asking for an open-ended value. Every candidate must use exactly these "
                "three response categories: 'Yes', 'No', and 'Information unavailable / not "
                "in record'. "
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
                "Mark valid=true only when all conditions hold: the query is a medically "
                "meaningful yes/no predicate about exactly one observable variable; it "
                "contains no answer-option leakage; it is not already answered above; a "
                "patient-record fact could explicitly establish Yes or No without guessing; "
                "and the categories are exactly Yes, No, and one unavailable outcome. Reject "
                "semantic paraphrases of any earlier query, including an unavailable one. "
                "Reject diagnosis, current treatment/drug, test-order/status, or management "
                "questions that reveal or paraphrase the target instead of measuring patient "
                "evidence. Reject invented variables and derived clinical summaries such as "
                "'hemodynamically stable'; direct numeric-threshold predicates are valid. Return "
                "{\"valid\":true,\"reason\":\"brief structural reason\"}."
            ),
        },
    ]


def candidate_set_validation_messages(
    task: MediQTask,
    actions: Sequence[MediQAction],
    history: Sequence[tuple[MediQAction, MediQObservation]],
) -> list[dict[str, str]]:
    candidates = [
        {"index": index, "query": action.query}
        for index, action in enumerate(actions)
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are a strict MediQ candidate-set deduplication auditor. Identify only "
                "queries that ask the same clinical fact or logically equivalent predicate, "
                "including medical synonyms. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial patient information:\n{task.initial_info}\n\n"
                f"Conversation so far:\n{_history_text(history)}\n\n"
                f"Clinical question:\n{task.question}\n\n"
                f"Candidate queries:\n{json.dumps(candidates)}\n\n"
                "Group indices only when the queries are semantic duplicates, such as "
                "'renal calculi', 'kidney stones', and 'nephrolithiasis'. Different symptoms, "
                "different tests, or meaningfully different numeric thresholds are not "
                "duplicates. Each index may appear in at most one group. Return "
                '{"duplicate_groups":[[0,1]],"reason":"brief set-level reason"}. '
                "Use an empty duplicate_groups list when every query is distinct."
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


def record_availability_messages(action: MediQAction) -> list[dict[str, str]]:
    task = action.task
    outcomes = [ANSWERABLE_RECORD_OUTCOME, UNANSWERABLE_RECORD_OUTCOME]
    return [
        {
            "role": "system",
            "content": (
                "You estimate record coverage for the MediQ benchmark. This is a "
                "label-independent missingness model: do not assume any answer option is "
                "correct. Return strict JSON only."
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
                + f"\n\nClinical question:\n{task.question}\n\n"
                f"Doctor's next yes/no query: {action.query}\n\n"
                "Estimate whether the hidden original exam-case record is likely to contain "
                "an explicit fact that directly establishes Yes or No for this exact query. "
                "Do not estimate whether the medical answer is knowable in general. A fact "
                "that would merely be clinically useful, plausible, or inferable does not "
                "make the query answerable from the record. This distribution must not depend "
                "on which multiple-choice option is correct. Return "
                + json.dumps(
                    {"probabilities": {outcome: 0.0 for outcome in outcomes}}
                )
                + ". Values must sum to 1."
            ),
        },
    ]


def factored_likelihood_messages(
    hypothesis: str, action: MediQAction
) -> list[dict[str, str]]:
    task = action.task
    outcomes = ["Yes", "No"]
    return [
        {
            "role": "system",
            "content": (
                "You are a calibrated counterfactual clinical-record model. Estimate a "
                "binary patient finding conditional on an exam answer being correct. Return "
                "strict JSON only."
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
                + f"\n\nClinical question:\n{task.question}\n\n"
                f"Options:\n{_options_text(task)}\n\n"
                f"Assume the exam's correct answer is {hypothesis}: "
                f"{task.option_text(hypothesis)}.\n"
                f"Next doctor query: {action.query}\n\n"
                "Also assume the hidden original record explicitly answers this query, so "
                "the only possible categories here are Yes and No. Infer the likely finding "
                "in counterfactual complete patient records consistent with the initial "
                "information, conversation, and correct exam answer. The answer option may "
                "be a diagnosis, mechanism, next step, or treatment priority. It is not a "
                "mutually exclusive description of the patient: findings associated with "
                "other options may coexist, and a priority answer does not imply that other "
                "abnormalities are absent. Do not choose Yes merely because the query repeats "
                "words or concepts from the assumed option. Return "
                + json.dumps(
                    {"probabilities": {outcome: 0.0 for outcome in outcomes}}
                )
                + ". Values must sum to 1."
            ),
        },
    ]


def data_estimation_outcome_messages(
    action: MediQAction,
) -> list[dict[str, str]]:
    task = action.task
    outcomes = list(action.outcomes)
    return [
        {
            "role": "system",
            "content": (
                "You are a calibrated predictive model for the MediQ patient interface. "
                "Predict the next response category without assuming which exam option is "
                "correct. Return strict JSON only."
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
                + f"\n\nClinical question:\n{task.question}\n\n"
                f"Next doctor query: {action.query}\n"
                f"Response categories: {json.dumps(outcomes)}\n\n"
                "Predict how the official-style Fact-Select patient will categorize its "
                "reply using only the information currently available to the doctor. "
                "Unavailable means the hidden original record contains no explicit fact "
                "that establishes Yes or No for the exact predicate. Do not condition this "
                "prediction on any answer option being correct. Return "
                + json.dumps(
                    {"probabilities": {outcome: 0.0 for outcome in outcomes}}
                )
                + ". Values must sum to 1."
            ),
        },
    ]


def data_estimation_posterior_messages(
    action: MediQAction,
    hypothetical_outcome: str,
) -> list[dict[str, str]]:
    task = action.task
    labels = list(task.option_labels)
    current = dict(zip(labels, action.prior_probabilities, strict=True))
    return [
        {
            "role": "system",
            "content": (
                "You are a calibrated hypothetical-evidence clinical judge. Update a "
                "distribution over the finite exam-answer labels after one specified "
                "patient reply. Return strict JSON only."
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
                + f"\n\nClinical question:\n{task.question}\n\n"
                f"Options:\n{_options_text(task)}\n\n"
                f"Current answer distribution: {json.dumps(current)}\n\n"
                f"Hypothetical next interaction:\nDoctor: {action.query}\n"
                f"Patient response category: {hypothetical_outcome}\n\n"
                "Update the probability that each option is the exam's correct answer. "
                "Interpret Yes or No as direct evidence about the doctor's predicate and "
                "do not invent any additional hidden facts. Options may be diagnoses, "
                "mechanisms, next steps, or treatment priorities; findings associated with "
                "different options can coexist, so reason about which option is correct "
                "rather than treating option text as mutually exclusive patient states. "
                "Return "
                + json.dumps(
                    {"probabilities": {label: 0.0 for label in labels}}
                )
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
                f"Select at most {max_facts} fact indices that explicitly establish either "
                "Yes or No for the question. "
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
                "qualifier, frequency, severity, or timing. Direct arithmetic comparison of "
                "an explicit recorded value with an explicit query threshold is allowed. Return "
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
                "category. Do not infer an unstated qualifier, frequency, severity, or timing. "
                "Directly comparing an explicit value with an explicit threshold is allowed. "
                "If neither Yes nor No is explicitly entailed, set clean=false and "
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
