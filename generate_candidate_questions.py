from dataclasses import dataclass, field

import numpy as np

from helpers import BeliefState, Config, ensure_belief_state, format_categorical_belief_summary, \
    is_uniform_belief_state, print_and_log, reverse_history, _binary_entropy, convert_string_to_array
from model import Model
from prompts import candidate_generation_system_message, conditional_question_generation_prompt, \
    unconditional_question_generation_prompt, weighted_conditional_question_generation_prompt, \
    weighted_unconditional_question_generation_prompt, \
    candidate_generation_system_message_naive, \
    question_generation_prompt_naive, weighted_question_generation_prompt_naive, answer_likelihood_messages
from update_beliefs import _update_beliefs_many, update_beliefs_batched

from helpers import write_to_log


@dataclass
class _DepthTwoBranch:
    question_index: int
    question: str
    answer: str
    branch_probability: float
    history: list[dict[str, str]]
    future_beliefs: BeliefState = field(default_factory=lambda: BeliefState([], []))
    future_questions: list[str] = field(default_factory=list)
    future_values: list[float] = field(default_factory=list)


def _format_question_preview(questions: list[str]) -> str:
    if len(questions) == 0:
        return "[]"
    return repr(questions)


def generate_candidate_questions(beliefs: BeliefState | list[str], history_questioner: list[dict[str, str]],
                                 questioner: Model, generation_temperature: float, num_questions: int) -> list[str]:
    belief_state = ensure_belief_state(beliefs)
    # if there are less than 3 beliefs left, best question is always to check one of them
    if len(belief_state.beliefs) in [1, 2]:
        top_belief = belief_state.beliefs[int(np.argmax(belief_state.probabilities))]
        print(f"[candidate-gen] Only {len(belief_state.beliefs)} belief(s) left, switching to direct guess")
        return [f"Is it {top_belief}?"]

    print(f"[candidate-gen] Building candidates from {len(belief_state.beliefs)} belief(s) and {len(history_questioner) // 2} prior round(s)")
    if is_uniform_belief_state(belief_state):
        question_prompt = conditional_question_generation_prompt(belief_state.beliefs, num_questions)
    else:
        print(f"[categorical] Candidate generation weights: {format_categorical_belief_summary(belief_state)}")
        weighted_beliefs = sorted(
            zip(belief_state.beliefs, belief_state.probabilities),
            key=lambda entry: entry[1],
            reverse=True,
        )
        question_prompt = weighted_conditional_question_generation_prompt(weighted_beliefs, num_questions)

    messages = ([candidate_generation_system_message()] + reverse_history(history_questioner) + [question_prompt])
    candidate_questions = questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
    candidate_questions = convert_string_to_array(candidate_questions)
    print(f"[candidate-gen] Received {len(candidate_questions)} candidate(s) from conditional generation")
    if not is_uniform_belief_state(belief_state):
        print(
            f"[categorical] Candidate questions after conditional pass ({len(candidate_questions)}): "
            f"{_format_question_preview(candidate_questions)}"
        )

    if len(candidate_questions) < num_questions:
        print(f"[candidate-gen] Backfilling {num_questions - len(candidate_questions)} more candidate(s)")
        if is_uniform_belief_state(belief_state):
            backfill_prompt = unconditional_question_generation_prompt(
                candidate_questions,
                num_questions - len(candidate_questions),
            )
        else:
            weighted_beliefs = sorted(
                zip(belief_state.beliefs, belief_state.probabilities),
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
        print(f"[candidate-gen] Candidate pool now has {len(candidate_questions)} question(s)")
        if not is_uniform_belief_state(belief_state):
            print(
                f"[categorical] Candidate questions after backfill ({len(candidate_questions)}): "
                f"{_format_question_preview(candidate_questions)}"
            )

    return candidate_questions


def _conditional_question_prompt_for_belief_state(belief_state: BeliefState, num_questions: int) -> dict[str, str]:
    if is_uniform_belief_state(belief_state):
        return conditional_question_generation_prompt(belief_state.beliefs, num_questions)

    weighted_beliefs = sorted(
        zip(belief_state.beliefs, belief_state.probabilities),
        key=lambda entry: entry[1],
        reverse=True,
    )
    return weighted_conditional_question_generation_prompt(weighted_beliefs, num_questions)


def _backfill_question_prompt_for_belief_state(belief_state: BeliefState, candidate_questions: list[str],
                                               num_questions: int) -> dict[str, str]:
    missing_questions = num_questions - len(candidate_questions)
    if is_uniform_belief_state(belief_state):
        return unconditional_question_generation_prompt(candidate_questions, missing_questions)

    weighted_beliefs = sorted(
        zip(belief_state.beliefs, belief_state.probabilities),
        key=lambda entry: entry[1],
        reverse=True,
    )
    return weighted_unconditional_question_generation_prompt(
        weighted_beliefs,
        candidate_questions,
        missing_questions,
    )


def _generate_future_candidate_questions_batched(branch_beliefs: list[BeliefState], branch_histories: list[list[dict[str, str]]],
                                                 questioner: Model, generation_temperature: float,
                                                 num_questions: int, block_size: int) -> list[list[str]]:
    if len(branch_beliefs) != len(branch_histories):
        raise ValueError("branch_beliefs and branch_histories must have the same length")
    if not branch_beliefs:
        return []

    future_questions = [[] for _ in branch_beliefs]
    conditional_indices: list[int] = []
    conditional_messages: list[list[dict[str, str]]] = []

    for idx, (belief_state, history_questioner) in enumerate(zip(branch_beliefs, branch_histories)):
        if len(belief_state.beliefs) == 0:
            continue
        if len(belief_state.beliefs) in [1, 2]:
            top_belief = belief_state.beliefs[int(np.argmax(belief_state.probabilities))]
            print(f"[candidate-gen] Only {len(belief_state.beliefs)} belief(s) left, switching to direct guess")
            future_questions[idx] = [f"Is it {top_belief}?"]
            continue

        print(
            f"[candidate-gen] Building candidates from {len(belief_state.beliefs)} belief(s) and "
            f"{len(history_questioner) // 2} prior round(s)"
        )
        if not is_uniform_belief_state(belief_state):
            print(f"[categorical] Candidate generation weights: {format_categorical_belief_summary(belief_state)}")

        prompt = _conditional_question_prompt_for_belief_state(belief_state, num_questions)
        conditional_indices.append(idx)
        conditional_messages.append(
            [candidate_generation_system_message()] + reverse_history(history_questioner) + [prompt]
        )

    if conditional_messages:
        conditional_completions = questioner.chat_complete_messages_batched(
            batch_messages=conditional_messages,
            temperature=generation_temperature,
            block_size=block_size,
        )
        if len(conditional_completions) != len(conditional_messages):
            raise ValueError(
                f"Expected {len(conditional_messages)} conditional completions, received {len(conditional_completions)}"
            )

        for idx, completion in zip(conditional_indices, conditional_completions):
            candidate_questions = convert_string_to_array(completion)
            future_questions[idx] = candidate_questions
            print(f"[candidate-gen] Received {len(candidate_questions)} candidate(s) from conditional generation")
            if not is_uniform_belief_state(branch_beliefs[idx]):
                print(
                    f"[categorical] Candidate questions after conditional pass ({len(candidate_questions)}): "
                    f"{_format_question_preview(candidate_questions)}"
                )

    backfill_indices: list[int] = []
    backfill_messages: list[list[dict[str, str]]] = []
    for idx, (belief_state, history_questioner, candidate_questions) in enumerate(
        zip(branch_beliefs, branch_histories, future_questions)
    ):
        if len(belief_state.beliefs) == 0 or len(belief_state.beliefs) in [1, 2]:
            continue
        if len(candidate_questions) >= num_questions:
            continue

        print(f"[candidate-gen] Backfilling {num_questions - len(candidate_questions)} more candidate(s)")
        prompt = _backfill_question_prompt_for_belief_state(belief_state, candidate_questions, num_questions)
        backfill_indices.append(idx)
        backfill_messages.append(
            [candidate_generation_system_message()] + reverse_history(history_questioner) + [prompt]
        )

    if backfill_messages:
        backfill_completions = questioner.chat_complete_messages_batched(
            batch_messages=backfill_messages,
            temperature=generation_temperature,
            block_size=block_size,
        )
        if len(backfill_completions) != len(backfill_messages):
            raise ValueError(
                f"Expected {len(backfill_messages)} backfill completions, received {len(backfill_completions)}"
            )

        for idx, completion in zip(backfill_indices, backfill_completions):
            future_questions[idx] = future_questions[idx] + convert_string_to_array(completion)
            print(f"[candidate-gen] Candidate pool now has {len(future_questions[idx])} question(s)")
            if not is_uniform_belief_state(branch_beliefs[idx]):
                print(
                    f"[categorical] Candidate questions after backfill ({len(future_questions[idx])}): "
                    f"{_format_question_preview(future_questions[idx])}"
                )

    return future_questions


def _draw_belief_samples(beliefs: BeliefState | list[str], deterministic: bool,
                         num_mc_samples: int) -> tuple[list[str] | np.ndarray, list[float] | None]:
    belief_state = ensure_belief_state(beliefs)
    if len(belief_state.beliefs) == 0:
        return [], None

    if deterministic:
        return belief_state.beliefs, belief_state.probabilities

    if len(belief_state.beliefs) <= num_mc_samples:
        return belief_state.beliefs, belief_state.probabilities

    sampled_indices = np.random.choice(
        len(belief_state.beliefs),
        size=num_mc_samples,
        replace=True,
        p=belief_state.probabilities,
    )
    samples = np.array([belief_state.beliefs[index] for index in sampled_indices])
    return samples, None


def _score_questions_from_samples(samples: list[str] | np.ndarray, sample_probabilities: list[float] | None,
                                  cand_questions: list[str], eig: bool, questioner: Model,
                                  answer_temperature: float, block_size: int) -> tuple[list[float], list[float], list[float]]:
    if len(cand_questions) == 0 or len(samples) == 0:
        return [0.0] * len(cand_questions), [0.0] * len(cand_questions), [0.0] * len(cand_questions)

    conversations = []
    for question in cand_questions:
        for sample in samples:
            conversations.append(answer_likelihood_messages(sample, question, ["Yes", "No"]))

    probabilities = questioner.chat_probabilities_messages_batched(
        conversations,
        ["Yes", "No"],
        temperature=answer_temperature,
        block_size=block_size,
    )

    return _score_questions_from_probability_rows(
        probabilities,
        len(samples),
        sample_probabilities,
        len(cand_questions),
        eig,
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


def _score_future_questions_batched(branch_beliefs: list[BeliefState], branch_future_questions: list[list[str]], eig: bool,
                                    deterministic: bool, questioner: Model, answer_temperature: float,
                                    num_mc_samples: int, block_size: int) -> list[list[float]]:
    if len(branch_beliefs) != len(branch_future_questions):
        raise ValueError("branch_beliefs and branch_future_questions must have the same length")
    if not branch_beliefs:
        return []

    branch_samples = []
    conversations = []
    active_branch_indices = []

    for branch_idx, (belief_state, future_questions) in enumerate(zip(branch_beliefs, branch_future_questions)):
        samples, sample_probabilities = _draw_belief_samples(belief_state, deterministic, num_mc_samples)
        branch_samples.append((samples, sample_probabilities))
        if len(samples) == 0 or len(future_questions) == 0:
            continue

        active_branch_indices.append(branch_idx)
        for question in future_questions:
            for sample in samples:
                conversations.append(answer_likelihood_messages(sample, question, ["Yes", "No"]))

    probabilities = []
    if conversations:
        probabilities = questioner.chat_probabilities_messages_batched(
            conversations,
            ["Yes", "No"],
            temperature=answer_temperature,
            block_size=block_size,
        )

    branch_future_values = [[] for _ in branch_beliefs]
    probability_offset = 0
    for branch_idx in range(len(branch_beliefs)):
        samples, sample_probabilities = branch_samples[branch_idx]
        future_questions = branch_future_questions[branch_idx]
        if len(samples) == 0 or len(future_questions) == 0:
            continue

        branch_probability_count = len(samples) * len(future_questions)
        branch_probabilities = probabilities[probability_offset:probability_offset + branch_probability_count]
        probability_offset += branch_probability_count
        branch_values, _p_yes_values, _p_no_values = _score_questions_from_probability_rows(
            branch_probabilities,
            len(samples),
            sample_probabilities,
            len(future_questions),
            eig,
        )
        branch_future_values[branch_idx] = branch_values

    return branch_future_values


def _future_beliefs_for_answer(beliefs: BeliefState | list[str], history_questioner: list[dict[str, str]], question: str,
                               answer: str, questioner: Model, deterministic: bool, config: Config) -> BeliefState:
    belief_state = ensure_belief_state(beliefs)
    hypothetical_history = history_questioner + [
        {"role": "assistant", "content": question},
        {"role": "user", "content": answer},
    ]
    # Depth-2 search must mirror the live belief-update pipeline so hypothetical
    # branch scoring matches the beliefs we would actually carry into the next turn.
    return update_beliefs_batched(hypothetical_history, belief_state, questioner, deterministic, config)


def _build_depth_two_branches(cand_questions: list[str], history_questioner: list[dict[str, str]],
                              p_yes_values: list[float], p_no_values: list[float]) -> list[_DepthTwoBranch]:
    branches = []
    for question_index, question in enumerate(cand_questions):
        for answer, branch_probability in (("Yes", p_yes_values[question_index]), ("No", p_no_values[question_index])):
            branches.append(
                _DepthTwoBranch(
                    question_index=question_index,
                    question=question,
                    answer=answer,
                    branch_probability=branch_probability,
                    history=history_questioner + [
                        {"role": "assistant", "content": question},
                        {"role": "user", "content": answer},
                    ],
                )
            )
    return branches


def _supports_batched_depth_two(questioner: Model) -> bool:
    return callable(getattr(questioner, "chat_complete_messages_batched", None))


def evaluate_questions_forward_search(beliefs: BeliefState | list[str], history_questioner: list[dict[str, str]],
                                      cand_questions: list[str],
                                      eig: bool, deterministic: bool, questioner: Model, config: Config,
                                      depth: int = 2) -> list[float]:
    belief_state = ensure_belief_state(beliefs)
    if depth == 1:
        print(f"[question-score] Depth-1 evaluation for {len(cand_questions)} question(s)")
        return evaluate_questions_batched(
            belief_state,
            cand_questions,
            eig,
            deterministic,
            questioner,
            config.answer_temperature,
            config.num_mc_samples,
            config.batched_block_size,
        )
    if depth != 2:
        raise ValueError("evaluate_questions_forward_search only supports depth=1 or depth=2")

    #samples number of beliefs to sample from the current beliefs
    samples, sample_probabilities = _draw_belief_samples(belief_state, deterministic, config.num_mc_samples)
    print(f"[question-score] Depth-2 evaluation for {len(cand_questions)} question(s) using {len(samples)} sample(s)")

    #immediate values is the expected 1 step info gain of asking the candidate questions
    immediate_values, p_yes_values, p_no_values = _score_questions_from_samples(
        samples,
        sample_probabilities,
        cand_questions,
        eig,
        questioner,
        config.answer_temperature,
        config.batched_block_size,
    )

    if not _supports_batched_depth_two(questioner):
        total_values = immediate_values.copy()
        for i, question in enumerate(cand_questions):
            print(f"[question-score] Exploring future branches for question {i + 1}/{len(cand_questions)}: {question}")
            expected_future_value = 0.0
            future_value_yes = 0.0
            future_value_no = 0.0

            for answer, branch_probability in (("Yes", p_yes_values[i]), ("No", p_no_values[i])):
                future_beliefs = _future_beliefs_for_answer(
                    belief_state,
                    history_questioner,
                    question,
                    answer,
                    questioner,
                    deterministic,
                    config,
                )
                future_beliefs = ensure_belief_state(future_beliefs)
                if config.belief_state_mode == "categorical":
                    print_and_log(
                        f"[categorical] Branch '{question}' -> {answer} (p={branch_probability:.3f}): "
                        f"{format_categorical_belief_summary(future_beliefs)}",
                        config,
                    )
                if len(future_beliefs.beliefs) == 0:
                    continue

                hypothetical_history = history_questioner + [
                    {"role": "assistant", "content": question},
                    {"role": "user", "content": answer},
                ]
                future_questions = generate_candidate_questions(
                    future_beliefs,
                    hypothetical_history,
                    questioner,
                    config.generation_temperature_diverse,
                    config.target_num_questions,
                )
                if len(future_questions) == 0:
                    continue

                future_values = evaluate_questions_batched(
                    future_beliefs,
                    future_questions,
                    eig,
                    deterministic,
                    questioner,
                    config.answer_temperature,
                    config.num_mc_samples,
                    config.batched_block_size,
                )
                if len(future_values) == 0:
                    continue

                branch_future_value = max(future_values)
                expected_future_value += branch_probability * branch_future_value
                if answer == "Yes":
                    future_value_yes = branch_future_value
                else:
                    future_value_no = branch_future_value

            total_values[i] += expected_future_value
            print(
                f"[question-score] Question summary: immediate={immediate_values[i]:.4f}, "
                f"future_yes={future_value_yes:.4f}, future_no={future_value_no:.4f}, total={total_values[i]:.4f}"
            )

        optimal_immediate_question = cand_questions[np.argmax(immediate_values)]
        write_to_log(f"Optimal immediate question: {optimal_immediate_question}\n", config)
        write_to_log(f"Immediate EIG: {immediate_values[np.argmax(immediate_values)]}\n", config)
        write_to_log(
            f"Future value: {total_values[np.argmax(immediate_values)] - immediate_values[np.argmax(immediate_values)]}\n",
            config,
        )
        write_to_log(f"Total value: {total_values[np.argmax(immediate_values)]}\n", config)

        optimal_total_question = cand_questions[np.argmax(total_values)]
        write_to_log(f"Optimal total question: {optimal_total_question}\n", config)
        write_to_log(f"Immediate EIG: {immediate_values[np.argmax(total_values)]}\n", config)
        write_to_log(
            f"Future value: {total_values[np.argmax(total_values)] - immediate_values[np.argmax(total_values)]}\n",
            config,
        )
        write_to_log(f"Total value: {total_values[np.argmax(total_values)]}\n", config)

        for i, question in enumerate(cand_questions):
            print(f"Question: {question}")
            print(f"Immediate EIG: {immediate_values[i]}")
            print(f"Future value: {total_values[i] - immediate_values[i]}")
            print(f"Total value: {total_values[i]}")

        return total_values

    total_values = immediate_values.copy()
    branches = _build_depth_two_branches(cand_questions, history_questioner, p_yes_values, p_no_values)
    active_branch_indices = [
        idx
        for idx, branch in enumerate(branches)
        if branch.branch_probability != 0.0
    ]

    if active_branch_indices:
        future_beliefs = _update_beliefs_many(
            [branches[idx].history for idx in active_branch_indices],
            belief_state,
            questioner,
            deterministic,
            config,
        )
        for branch_idx, future_belief_state in zip(active_branch_indices, future_beliefs):
            branches[branch_idx].future_beliefs = ensure_belief_state(future_belief_state)

        future_questions = _generate_future_candidate_questions_batched(
            [branches[idx].future_beliefs for idx in active_branch_indices],
            [branches[idx].history for idx in active_branch_indices],
            questioner,
            config.generation_temperature_diverse,
            config.target_num_questions,
            config.batched_block_size,
        )
        for branch_idx, branch_questions in zip(active_branch_indices, future_questions):
            branches[branch_idx].future_questions = branch_questions

        future_values = _score_future_questions_batched(
            [branches[idx].future_beliefs for idx in active_branch_indices],
            [branches[idx].future_questions for idx in active_branch_indices],
            eig,
            deterministic,
            questioner,
            config.answer_temperature,
            config.num_mc_samples,
            config.batched_block_size,
        )
        for branch_idx, branch_values in zip(active_branch_indices, future_values):
            branches[branch_idx].future_values = branch_values

    branch_lookup = {(branch.question_index, branch.answer): branch for branch in branches}
    for i, question in enumerate(cand_questions):
        print(f"[question-score] Exploring future branches for question {i + 1}/{len(cand_questions)}: {question}")
        expected_future_value = 0.0
        future_value_yes = 0.0
        future_value_no = 0.0

        for answer, branch_probability in (("Yes", p_yes_values[i]), ("No", p_no_values[i])):
            branch = branch_lookup[(i, answer)]
            future_beliefs = branch.future_beliefs
            if config.belief_state_mode == "categorical":
                print_and_log(
                    f"[categorical] Branch '{question}' -> {answer} (p={branch_probability:.3f}): "
                    f"{format_categorical_belief_summary(future_beliefs)}",
                    config,
                )
            if len(future_beliefs.beliefs) == 0 or len(branch.future_questions) == 0 or len(branch.future_values) == 0:
                continue

            branch_future_value = max(branch.future_values)
            expected_future_value += branch_probability * branch_future_value
            if answer == "Yes":
                future_value_yes = branch_future_value
            else:
                future_value_no = branch_future_value

        total_values[i] += expected_future_value
        print(
            f"[question-score] Question summary: immediate={immediate_values[i]:.4f}, "
            f"future_yes={future_value_yes:.4f}, future_no={future_value_no:.4f}, total={total_values[i]:.4f}"
        )

    optimal_immediate_question = cand_questions[np.argmax(immediate_values)]
    write_to_log(f"Optimal immediate question: {optimal_immediate_question}\n", config)
    write_to_log(f"Immediate EIG: {immediate_values[np.argmax(immediate_values)]}\n", config)
    write_to_log(
        f"Future value: {total_values[np.argmax(immediate_values)] - immediate_values[np.argmax(immediate_values)]}\n",
        config,
    )
    write_to_log(f"Total value: {total_values[np.argmax(immediate_values)]}\n", config)

    optimal_total_question = cand_questions[np.argmax(total_values)]
    write_to_log(f"Optimal total question: {optimal_total_question}\n", config)
    write_to_log(f"Immediate EIG: {immediate_values[np.argmax(total_values)]}\n", config)
    write_to_log(
        f"Future value: {total_values[np.argmax(total_values)] - immediate_values[np.argmax(total_values)]}\n",
        config,
    )
    write_to_log(f"Total value: {total_values[np.argmax(total_values)]}\n", config)

    #goes to wandb logger
    for i, question in enumerate(cand_questions):
        print(f"Question: {question}")
        print(f"Immediate EIG: {immediate_values[i]}")
        print(f"Future value: {total_values[i] - immediate_values[i]}")
        print(f"Total value: {total_values[i]}")

    return total_values

def evaluate_questions_batched(beliefs: BeliefState | list[str], cand_questions: list[str], eig: bool,
                               deterministic: bool, questioner: Model, answer_temperature: float,
                               num_mc_samples: int, block_size: int) -> list[float]:
    samples, sample_probabilities = _draw_belief_samples(beliefs, deterministic, num_mc_samples)
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
                                      prior_beliefs: BeliefState | None = None) -> str:
    if prior_beliefs is None or len(prior_beliefs.beliefs) == 0:
        question_prompt = question_generation_prompt_naive()
    else:
        weighted_beliefs = sorted(
            zip(prior_beliefs.beliefs, prior_beliefs.probabilities),
            key=lambda entry: entry[1],
            reverse=True,
        )
        question_prompt = weighted_question_generation_prompt_naive(weighted_beliefs)
    messages = ([candidate_generation_system_message_naive()] + reverse_history(history_questioner) +
                [question_prompt])
    return questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
