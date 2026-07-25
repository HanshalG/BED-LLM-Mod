import contextlib
import io
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from core import BeliefState
from helpers import Config, format_categorical_belief_summary, \
    is_uniform_belief_state, print_and_log, reverse_history, _binary_entropy, convert_string_to_array
if TYPE_CHECKING:
    from model import Model
else:
    Model = Any
from environments.animals.prompts import candidate_generation_system_message, conditional_question_generation_prompt, \
    unconditional_question_generation_prompt, weighted_conditional_question_generation_prompt, \
    weighted_unconditional_question_generation_prompt, \
    candidate_generation_system_message_naive, \
    question_generation_prompt_naive, weighted_question_generation_prompt_naive, answer_likelihood_messages
from environments.animals.beliefs import _update_beliefs_many, ensure_animals_belief_state, update_beliefs_batched

from helpers import write_to_log


@dataclass(frozen=True)
class CandidateCoverageDynamics:
    """One-step support-survival diagnostic for a candidate question.

    ``truth`` is deliberately not stored here.  It is used only after the
    production branch update returns to measure whether the target survived
    hypothesis regeneration and filtering.
    """

    question: str
    immediate_eig: float
    p_yes: float
    p_no: float
    current_support_retention_if_yes: float
    current_support_retention_if_no: float
    expected_current_support_retention: float
    expected_surviving_map_mass: float
    expected_dynamic_brier_score: float
    dynamic_brier_gain: float
    truth_covered_if_yes: bool
    truth_covered_if_no: bool
    support_size_if_yes: int
    support_size_if_no: int
    support_if_yes: tuple[str, ...]
    support_if_no: tuple[str, ...]

    @property
    def expected_truth_coverage(self) -> float:
        return (
            self.p_yes * float(self.truth_covered_if_yes)
            + self.p_no * float(self.truth_covered_if_no)
        )


def _format_question_preview(questions: list[str]) -> str:
    if len(questions) == 0:
        return "[]"
    return repr(questions)


def generate_candidate_questions(beliefs: BeliefState | list[str], history_questioner: list[dict[str, str]],
                                 questioner: Model, generation_temperature: float, num_questions: int,
                                 verbose: bool = True) -> list[str]:
    belief_state = ensure_animals_belief_state(beliefs)
    # if there are less than 3 beliefs left, best question is always to check one of them
    if len(belief_state.hypotheses) in [1, 2]:
        top_belief = belief_state.hypotheses[int(np.argmax(belief_state.probabilities))]
        if verbose:
            print(f"[candidate-gen] Only {len(belief_state.hypotheses)} belief(s) left, switching to direct guess")
        return [f"Is it {top_belief}?"]

    if verbose:
        print(f"[candidate-gen] Building candidates from {len(belief_state.hypotheses)} belief(s) and {len(history_questioner) // 2} prior round(s)")
    if is_uniform_belief_state(belief_state):
        question_prompt = conditional_question_generation_prompt(belief_state.hypotheses, num_questions)
    else:
        if verbose:
            print(f"[categorical] Candidate generation weights: {format_categorical_belief_summary(belief_state)}")
        weighted_beliefs = sorted(
            zip(belief_state.hypotheses, belief_state.probabilities),
            key=lambda entry: entry[1],
            reverse=True,
        )
        question_prompt = weighted_conditional_question_generation_prompt(weighted_beliefs, num_questions)

    messages = ([candidate_generation_system_message()] + reverse_history(history_questioner) + [question_prompt])
    candidate_questions = questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
    candidate_questions = convert_string_to_array(candidate_questions)
    if verbose:
        print(f"[candidate-gen] Received {len(candidate_questions)} candidate(s) from conditional generation")
    if verbose and not is_uniform_belief_state(belief_state):
        print(
            f"[categorical] Candidate questions after conditional pass ({len(candidate_questions)}): "
            f"{_format_question_preview(candidate_questions)}"
        )

    if len(candidate_questions) < num_questions:
        if verbose:
            print(f"[candidate-gen] Backfilling {num_questions - len(candidate_questions)} more candidate(s)")
        if is_uniform_belief_state(belief_state):
            backfill_prompt = unconditional_question_generation_prompt(
                candidate_questions,
                num_questions - len(candidate_questions),
            )
        else:
            weighted_beliefs = sorted(
                zip(belief_state.hypotheses, belief_state.probabilities),
                key=lambda entry: entry[1],
                reverse=True,
            )
            backfill_prompt = weighted_unconditional_question_generation_prompt(
                weighted_beliefs,
                candidate_questions,
                num_questions - len(candidate_questions),
            )
        messages = ([candidate_generation_system_message()] + reverse_history(history_questioner) + [backfill_prompt])
        new_candidate_questions = questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
        candidate_questions = candidate_questions + convert_string_to_array(new_candidate_questions)
        if verbose:
            print(f"[candidate-gen] Candidate pool now has {len(candidate_questions)} question(s)")
        if verbose and not is_uniform_belief_state(belief_state):
            print(
                f"[categorical] Candidate questions after backfill ({len(candidate_questions)}): "
                f"{_format_question_preview(candidate_questions)}"
            )

    return candidate_questions


def _draw_belief_samples(beliefs: BeliefState | list[str], deterministic: bool,
                         num_mc_samples: int) -> tuple[list[str] | np.ndarray, list[float] | None]:
    belief_state = ensure_animals_belief_state(beliefs)
    if len(belief_state.hypotheses) == 0:
        return [], None

    if deterministic:
        return belief_state.hypotheses, belief_state.probabilities

    if len(belief_state.hypotheses) <= num_mc_samples:
        return belief_state.hypotheses, belief_state.probabilities

    sampled_indices = np.random.choice(
        len(belief_state.hypotheses),
        size=num_mc_samples,
        replace=True,
        p=belief_state.probabilities,
    )
    samples = np.array([belief_state.hypotheses[index] for index in sampled_indices])
    return samples, None


def _score_questions_from_samples(samples: list[str] | np.ndarray, sample_probabilities: list[float] | None,
                                  cand_questions: list[str], eig: bool, questioner: Model,
                                  answer_temperature: float, block_size: int) -> tuple[list[float], list[float], list[float]]:
    if len(cand_questions) == 0 or len(samples) == 0:
        return [0.0] * len(cand_questions), [0.0] * len(cand_questions), [0.0] * len(cand_questions)

    probabilities = _answer_probability_rows(
        samples,
        cand_questions,
        questioner,
        answer_temperature,
        block_size,
    )
    return _score_questions_from_probability_rows(
        probabilities,
        len(samples),
        sample_probabilities,
        len(cand_questions),
        eig,
    )


def _answer_probability_rows(
    samples: list[str] | np.ndarray,
    cand_questions: list[str],
    questioner: Model,
    answer_temperature: float,
    block_size: int,
) -> list[dict[str, float]]:
    conversations = []
    for question in cand_questions:
        for sample in samples:
            conversations.append(answer_likelihood_messages(sample, question, ["Yes", "No"]))

    return questioner.chat_probabilities_messages_batched(
        conversations,
        ["Yes", "No"],
        temperature=answer_temperature,
        block_size=block_size,
    )


def _score_questions_from_probability_rows(probabilities: list[dict[str, float]], num_samples: int,
                                           sample_probabilities: list[float] | None, num_questions: int,
                                           eig: bool) -> tuple[list[float], list[float], list[float]]:
    if num_questions == 0 or num_samples == 0:
        return [0.0] * num_questions, [0.0] * num_questions, [0.0] * num_questions

    question_values = [0.0] * num_questions
    p_yes_values = [0.0] * num_questions
    p_no_values = [0.0] * num_questions

    for i in range(num_questions):
        answers = probabilities[i * num_samples:(i + 1) * num_samples]
        p_yes = []
        p_no = []
        entropy_sum = []

        for answer in answers:
            p_yes.append(answer["Yes"])
            p_no.append(answer["No"])
            entropy_sum.append(_binary_entropy(answer["Yes"], answer["No"]))

        if sample_probabilities is None:
            p_hat_yes = float(np.mean(p_yes))
            p_hat_no = float(np.mean(p_no))
            expected_entropy = float(np.mean(entropy_sum))
        else:
            p_hat_yes = float(np.dot(sample_probabilities, p_yes))
            p_hat_no = float(np.dot(sample_probabilities, p_no))
            expected_entropy = float(np.dot(sample_probabilities, entropy_sum))
        entropy = _binary_entropy(p_hat_yes, p_hat_no)

        p_yes_values[i] = p_hat_yes
        p_no_values[i] = p_hat_no
        if eig:
            question_values[i] = entropy - expected_entropy
        else:
            question_values[i] = entropy

    return question_values, p_yes_values, p_no_values


def _future_beliefs_for_answer(beliefs: BeliefState | list[str], history_questioner: list[dict[str, str]], question: str,
                               answer: str, questioner: Model, deterministic: bool, config: Config) -> BeliefState:
    belief_state = ensure_animals_belief_state(beliefs)
    hypothetical_history = history_questioner + [
        {"role": "assistant", "content": question},
        {"role": "user", "content": answer},
    ]
    # Forward search must mirror the live belief-update pipeline so hypothetical
    # branch scoring matches the beliefs we would actually carry into the next turn.
    if not config.forward_search_verbose:
        quiet_config = replace(config, log_path=None)
        with contextlib.redirect_stdout(io.StringIO()):
            return update_beliefs_batched(hypothetical_history, belief_state, questioner, deterministic, quiet_config)
    return update_beliefs_batched(hypothetical_history, belief_state, questioner, deterministic, config)


def _future_beliefs_for_answers_batched(
    beliefs: BeliefState | list[str],
    history_questioner: list[dict[str, str]],
    question_answers: list[tuple[str, str]],
    questioner: Model,
    deterministic: bool,
    config: Config,
) -> list[BeliefState]:
    """Run independent counterfactual branches through the production batch path."""
    belief_state = ensure_animals_belief_state(beliefs)
    histories = [
        _history_with_answer(history_questioner, question, answer)
        for question, answer in question_answers
    ]
    if not config.forward_search_verbose:
        quiet_config = replace(config, log_path=None)
        with contextlib.redirect_stdout(io.StringIO()):
            return _update_beliefs_many(histories, belief_state, questioner, deterministic, quiet_config)
    return _update_beliefs_many(histories, belief_state, questioner, deterministic, config)


def evaluate_candidate_coverage_dynamics(
    beliefs: BeliefState | list[str],
    history_questioner: list[dict[str, str]],
    cand_questions: list[str],
    truth: str,
    deterministic: bool,
    questioner: Model,
    config: Config,
    exact_current_support: bool | None = None,
) -> list[CandidateCoverageDynamics]:
    """Measure branch-dependent truth coverage using the live update pipeline.

    This is an observational diagnostic, not a policy score. Candidate EIG
    and branch probabilities use exactly the existing likelihood scorer.
    ``exact_current_support`` can request deterministic scoring over every
    current hypothesis without changing the production branch-update path
    selected by ``deterministic``. The target is compared with returned
    supports only; it is never included in prompt construction, likelihood
    scoring, or belief regeneration.
    """
    belief_state = ensure_animals_belief_state(beliefs)
    if not cand_questions or not belief_state.hypotheses:
        return []

    score_deterministically = (
        deterministic
        if exact_current_support is None
        else exact_current_support
    )
    samples, sample_probabilities = _draw_belief_samples(
        belief_state,
        score_deterministically,
        config.num_mc_samples,
    )
    probability_rows = _answer_probability_rows(
        samples,
        cand_questions,
        questioner,
        config.answer_temperature,
        config.batched_block_size,
    )
    immediate_eigs, p_yes_values, p_no_values = _score_questions_from_probability_rows(
        probability_rows,
        len(samples),
        sample_probabilities,
        len(cand_questions),
        eig=True,
    )
    truth_key = truth.strip().casefold()

    branch_questions_answers = [
        (question, answer)
        for question in cand_questions
        for answer in ("Yes", "No")
    ]
    future_beliefs = _future_beliefs_for_answers_batched(
        belief_state,
        history_questioner,
        branch_questions_answers,
        questioner,
        deterministic,
        config,
    )
    if len(future_beliefs) != len(branch_questions_answers):
        raise ValueError(
            "Counterfactual batch update returned the wrong number of belief states"
        )

    dynamics: list[CandidateCoverageDynamics] = []
    for question_index, (question, immediate_eig, p_yes, p_no) in enumerate(zip(
        cand_questions,
        immediate_eigs,
        p_yes_values,
        p_no_values,
    )):
        future_yes = ensure_animals_belief_state(future_beliefs[2 * question_index])
        future_no = ensure_animals_belief_state(future_beliefs[2 * question_index + 1])
        yes_support = {hypothesis.strip().casefold() for hypothesis in future_yes.hypotheses}
        no_support = {hypothesis.strip().casefold() for hypothesis in future_no.hypotheses}
        yes_covered = any(hypothesis.strip().casefold() == truth_key for hypothesis in future_yes.hypotheses)
        no_covered = any(hypothesis.strip().casefold() == truth_key for hypothesis in future_no.hypotheses)
        if sample_probabilities is None:
            sample_weights = [1.0 / len(samples)] * len(samples)
        else:
            sample_weights = sample_probabilities
        question_rows = probability_rows[
            question_index * len(samples):(question_index + 1) * len(samples)
        ]
        current_support_retention_if_yes = 0.0
        current_support_retention_if_no = 0.0
        expected_pseudo_truth_probability = 0.0
        yes_surviving_masses: list[float] = []
        no_surviving_masses: list[float] = []
        yes_probabilities = {
            hypothesis.strip().casefold(): float(probability)
            for hypothesis, probability in future_yes
        }
        no_probabilities = {
            hypothesis.strip().casefold(): float(probability)
            for hypothesis, probability in future_no
        }
        for hypothesis, weight, answer_probabilities in zip(samples, sample_weights, question_rows):
            hypothesis_key = str(hypothesis).strip().casefold()
            yes_mass = float(weight) * answer_probabilities["Yes"]
            no_mass = float(weight) * answer_probabilities["No"]
            if hypothesis_key in yes_support:
                current_support_retention_if_yes += yes_mass
                yes_surviving_masses.append(yes_mass)
                expected_pseudo_truth_probability += (
                    yes_mass * yes_probabilities[hypothesis_key]
                )
            if hypothesis_key in no_support:
                current_support_retention_if_no += no_mass
                no_surviving_masses.append(no_mass)
                expected_pseudo_truth_probability += (
                    no_mass * no_probabilities[hypothesis_key]
                )
        expected_current_support_retention = (
            current_support_retention_if_yes
            + current_support_retention_if_no
        )
        expected_surviving_map_mass = (
            max(yes_surviving_masses, default=0.0)
            + max(no_surviving_masses, default=0.0)
        )
        expected_dynamic_brier_score = (
            2.0 * expected_pseudo_truth_probability
            - float(p_yes) * sum(probability**2 for probability in future_yes.probabilities)
            - float(p_no) * sum(probability**2 for probability in future_no.probabilities)
        )
        current_brier_score = sum(
            probability**2 for probability in belief_state.probabilities
        )
        dynamics.append(
            CandidateCoverageDynamics(
                question=question,
                immediate_eig=float(immediate_eig),
                p_yes=float(p_yes),
                p_no=float(p_no),
                current_support_retention_if_yes=current_support_retention_if_yes,
                current_support_retention_if_no=current_support_retention_if_no,
                expected_current_support_retention=expected_current_support_retention,
                expected_surviving_map_mass=expected_surviving_map_mass,
                expected_dynamic_brier_score=expected_dynamic_brier_score,
                dynamic_brier_gain=(
                    expected_dynamic_brier_score - current_brier_score
                ),
                truth_covered_if_yes=yes_covered,
                truth_covered_if_no=no_covered,
                support_size_if_yes=future_yes.support_size,
                support_size_if_no=future_no.support_size,
                support_if_yes=tuple(str(item) for item in future_yes.hypotheses),
                support_if_no=tuple(str(item) for item in future_no.hypotheses),
            )
        )
    return dynamics


def _history_with_answer(history_questioner: list[dict[str, str]], question: str, answer: str) -> list[dict[str, str]]:
    return history_questioner + [
        {"role": "assistant", "content": question},
        {"role": "user", "content": answer},
    ]


def _evaluate_questions_forward_search_recursive(
    beliefs: BeliefState | list[str],
    history_questioner: list[dict[str, str]],
    cand_questions: list[str],
    eig: bool,
    deterministic: bool,
    questioner: Model,
    config: Config,
    depth: int,
) -> tuple[list[float], list[list[float]]]:
    belief_state = ensure_animals_belief_state(beliefs)
    if depth <= 0:
        raise ValueError("search depth must be a positive integer")
    if len(cand_questions) == 0:
        return [], []
    if depth == 1:
        values = evaluate_questions_batched(
            belief_state,
            cand_questions,
            eig,
            deterministic,
            questioner,
            config.answer_temperature,
            config.num_mc_samples,
            config.batched_block_size,
            verbose=config.forward_search_verbose,
        )
        return values, [[value] for value in values]

    samples, sample_probabilities = _draw_belief_samples(belief_state, deterministic, config.num_mc_samples)
    if config.forward_search_verbose:
        print(
            f"[question-score] Depth-{depth} evaluation for {len(cand_questions)} question(s) "
            f"using {len(samples)} sample(s)"
        )
    immediate_values, p_yes_values, p_no_values = _score_questions_from_samples(
        samples,
        sample_probabilities,
        cand_questions,
        eig,
        questioner,
        config.answer_temperature,
        config.batched_block_size,
    )

    horizon_values = [[value] for value in immediate_values]
    for i, question in enumerate(cand_questions):
        if config.forward_search_verbose:
            print(
                f"[question-score] Exploring future branches at depth {depth} "
                f"for question {i + 1}/{len(cand_questions)}: {question}"
            )
        expected_future_values = [0.0] * (depth - 1)
        future_value_yes = 0.0
        future_value_no = 0.0

        for answer, branch_probability in (("Yes", p_yes_values[i]), ("No", p_no_values[i])):
            if branch_probability == 0.0:
                continue

            future_beliefs = ensure_animals_belief_state(
                _future_beliefs_for_answer(
                    belief_state,
                    history_questioner,
                    question,
                    answer,
                    questioner,
                    deterministic,
                    config,
                )
            )
            if config.forward_search_verbose and config.belief_state_mode == "categorical":
                print_and_log(
                    f"[categorical] Branch '{question}' -> {answer} (p={branch_probability:.3f}): "
                    f"{format_categorical_belief_summary(future_beliefs)}",
                    config,
                )
            if len(future_beliefs.hypotheses) == 0:
                continue

            hypothetical_history = _history_with_answer(history_questioner, question, answer)
            future_questions = generate_candidate_questions(
                future_beliefs,
                hypothetical_history,
                questioner,
                config.generation_temperature_diverse,
                config.target_num_questions,
                verbose=config.forward_search_verbose,
            )
            if len(future_questions) == 0:
                continue

            future_values, future_horizon_values = _evaluate_questions_forward_search_recursive(
                future_beliefs,
                hypothetical_history,
                future_questions,
                eig,
                deterministic,
                questioner,
                config,
                depth - 1,
            )
            if len(future_values) == 0:
                continue

            branch_future_values = []
            for horizon_index in range(depth - 1):
                available_values = [
                    row[horizon_index]
                    for row in future_horizon_values
                    if len(row) > horizon_index
                ]
                branch_future_values.append(max(available_values) if available_values else 0.0)
            branch_future_value = branch_future_values[-1]
            for horizon_index, branch_horizon_value in enumerate(branch_future_values):
                expected_future_values[horizon_index] += branch_probability * branch_horizon_value
            if answer == "Yes":
                future_value_yes = branch_future_value
            else:
                future_value_no = branch_future_value

        horizon_values[i].extend(
            immediate_values[i] + expected_future_value
            for expected_future_value in expected_future_values
        )
        if config.forward_search_verbose:
            print(
                f"[question-score] Question summary: immediate={immediate_values[i]:.4f}, "
                f"future_yes={future_value_yes:.4f}, future_no={future_value_no:.4f}, total={horizon_values[i][-1]:.4f}"
            )

    return [row[-1] for row in horizon_values], horizon_values


def _write_forward_search_summary_line(message: str, config: Config) -> None:
    print(message)
    if config.log_path is not None:
        write_to_log(f"{message}\n", config)


def _format_forward_search_horizon_row(horizon_values: list[float], max_horizon: int) -> str:
    return ", ".join(
        f"{horizon_index + 1}-step={value:.6f}"
        for horizon_index, value in enumerate(horizon_values[:max_horizon])
    )


def _log_compact_forward_search_summary(cand_questions: list[str], horizon_values: list[list[float]],
                                        config: Config) -> None:
    if len(cand_questions) == 0 or len(horizon_values) == 0:
        return

    max_horizon = min(3, max((len(row) for row in horizon_values), default=0))
    if max_horizon == 0:
        return

    _write_forward_search_summary_line("[forward-search] Compact top-3 summary", config)
    for horizon_index in range(max_horizon):
        if horizon_index == 0:
            title = "Top 3 1-step / immediate EIG"
        else:
            title = f"Top 3 {horizon_index + 1}-step EIG"
        _write_forward_search_summary_line(f"[forward-search] {title}", config)
        ranked_indices = sorted(
            (
                index
                for index, row in enumerate(horizon_values)
                if len(row) > horizon_index
            ),
            key=lambda index: horizon_values[index][horizon_index],
            reverse=True,
        )[:3]
        for rank, index in enumerate(ranked_indices, start=1):
            _write_forward_search_summary_line(
                f"[forward-search] {rank}. {cand_questions[index]} "
                f"({_format_forward_search_horizon_row(horizon_values[index], max_horizon)})",
                config,
            )


def _log_forward_search_summary(cand_questions: list[str], horizon_values: list[list[float]],
                                config: Config) -> None:
    if len(cand_questions) == 0 or len(horizon_values) == 0:
        return

    immediate_values = [row[0] for row in horizon_values]
    total_values = [row[-1] for row in horizon_values]
    if len(immediate_values) == 0 or len(total_values) == 0:
        return
    if not config.forward_search_verbose:
        _log_compact_forward_search_summary(cand_questions, horizon_values, config)
        return

    optimal_immediate_index = int(np.argmax(immediate_values))
    optimal_immediate_question = cand_questions[optimal_immediate_index]
    write_to_log(f"Optimal immediate question: {optimal_immediate_question}\n", config)
    write_to_log(f"Immediate EIG: {immediate_values[optimal_immediate_index]}\n", config)
    write_to_log(
        f"Future value: {total_values[optimal_immediate_index] - immediate_values[optimal_immediate_index]}\n",
        config,
    )
    write_to_log(f"Total value: {total_values[optimal_immediate_index]}\n", config)

    optimal_total_index = int(np.argmax(total_values))
    optimal_total_question = cand_questions[optimal_total_index]
    write_to_log(f"Optimal total question: {optimal_total_question}\n", config)
    write_to_log(f"Immediate EIG: {immediate_values[optimal_total_index]}\n", config)
    write_to_log(
        f"Future value: {total_values[optimal_total_index] - immediate_values[optimal_total_index]}\n",
        config,
    )
    write_to_log(f"Total value: {total_values[optimal_total_index]}\n", config)

    if config.forward_search_verbose:
        for i, question in enumerate(cand_questions):
            print(f"Question: {question}")
            print(f"Immediate EIG: {immediate_values[i]}")
            print(f"Future value: {total_values[i] - immediate_values[i]}")
            print(f"Total value: {total_values[i]}")


def evaluate_questions_forward_search(beliefs: BeliefState | list[str], history_questioner: list[dict[str, str]],
                                      cand_questions: list[str],
                                      eig: bool, deterministic: bool, questioner: Model, config: Config,
                                      depth: int = 2) -> list[float]:
    belief_state = ensure_animals_belief_state(beliefs)
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise ValueError("search depth must be a positive integer")
    if depth == 1:
        if config.forward_search_verbose:
            print(f"[question-score] Depth-1 evaluation for {len(cand_questions)} question(s)")
        values = evaluate_questions_batched(
            belief_state,
            cand_questions,
            eig,
            deterministic,
            questioner,
            config.answer_temperature,
            config.num_mc_samples,
            config.batched_block_size,
            verbose=config.forward_search_verbose,
        )
        if not config.forward_search_verbose:
            _log_forward_search_summary(cand_questions, [[value] for value in values], config)
        return values
    total_values, horizon_values = _evaluate_questions_forward_search_recursive(
        belief_state,
        history_questioner,
        cand_questions,
        eig,
        deterministic,
        questioner,
        config,
        depth,
    )
    _log_forward_search_summary(cand_questions, horizon_values, config)
    return total_values


def evaluate_questions_batched(beliefs: BeliefState | list[str], cand_questions: list[str], eig: bool,
                               deterministic: bool, questioner: Model, answer_temperature: float,
                               num_mc_samples: int, block_size: int, verbose: bool = True) -> list[float]:
    samples, sample_probabilities = _draw_belief_samples(beliefs, deterministic, num_mc_samples)
    if verbose:
        print(f"[question-score] Batched scoring for {len(cand_questions)} question(s) using {len(samples)} sample(s)")
    question_values, _p_yes_values, _p_no_values = _score_questions_from_samples(
        samples,
        sample_probabilities,
        cand_questions,
        eig,
        questioner,
        answer_temperature,
        block_size,
    )
    return question_values


def generate_candidate_question_naive(history_questioner: list[dict[str, str]], questioner: Model,
                                      generation_temperature: float,
                                      prior_beliefs: BeliefState | None = None,
                                      belief_context_label: str = "prior distribution") -> str:
    if prior_beliefs is None or len(prior_beliefs.hypotheses) == 0:
        question_prompt = question_generation_prompt_naive()
    else:
        weighted_beliefs = sorted(
            zip(prior_beliefs.hypotheses, prior_beliefs.probabilities),
            key=lambda entry: entry[1],
            reverse=True,
        )
        question_prompt = weighted_question_generation_prompt_naive(weighted_beliefs, belief_context_label)
    messages = ([candidate_generation_system_message_naive()] + reverse_history(history_questioner) +
                [question_prompt])
    return questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
