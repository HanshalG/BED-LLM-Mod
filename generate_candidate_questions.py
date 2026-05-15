import numpy as np

from helpers import BeliefState, Config, ensure_belief_state, format_categorical_belief_summary, \
    is_uniform_belief_state, print_and_log, reverse_history, _binary_entropy, convert_string_to_array
from model import Model
from prompts import candidate_generation_system_message, conditional_question_generation_prompt, \
    unconditional_question_generation_prompt, weighted_conditional_question_generation_prompt, \
    weighted_unconditional_question_generation_prompt, \
    candidate_generation_system_message_naive, \
    question_generation_prompt_naive, weighted_question_generation_prompt_naive, answer_likelihood_messages
from update_beliefs import update_beliefs_batched

from helpers import write_to_log


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


def _future_beliefs_for_answer(beliefs: BeliefState | list[str], history_questioner: list[dict[str, str]], question: str,
                               answer: str, questioner: Model, deterministic: bool, config: Config) -> BeliefState:
    belief_state = ensure_belief_state(beliefs)
    hypothetical_history = history_questioner + [
        {"role": "assistant", "content": question},
        {"role": "user", "content": answer},
    ]
    # Forward search must mirror the live belief-update pipeline so hypothetical
    # branch scoring matches the beliefs we would actually carry into the next turn.
    return update_beliefs_batched(hypothetical_history, belief_state, questioner, deterministic, config)


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
) -> tuple[list[float], list[float]]:
    belief_state = ensure_belief_state(beliefs)
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
        )
        return values, values

    samples, sample_probabilities = _draw_belief_samples(belief_state, deterministic, config.num_mc_samples)
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

    total_values = immediate_values.copy()
    for i, question in enumerate(cand_questions):
        print(
            f"[question-score] Exploring future branches at depth {depth} "
            f"for question {i + 1}/{len(cand_questions)}: {question}"
        )
        expected_future_value = 0.0
        future_value_yes = 0.0
        future_value_no = 0.0

        for answer, branch_probability in (("Yes", p_yes_values[i]), ("No", p_no_values[i])):
            if branch_probability == 0.0:
                continue

            future_beliefs = ensure_belief_state(
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
            if config.belief_state_mode == "categorical":
                print_and_log(
                    f"[categorical] Branch '{question}' -> {answer} (p={branch_probability:.3f}): "
                    f"{format_categorical_belief_summary(future_beliefs)}",
                    config,
                )
            if len(future_beliefs.beliefs) == 0:
                continue

            hypothetical_history = _history_with_answer(history_questioner, question, answer)
            future_questions = generate_candidate_questions(
                future_beliefs,
                hypothetical_history,
                questioner,
                config.generation_temperature_diverse,
                config.target_num_questions,
            )
            if len(future_questions) == 0:
                continue

            future_values, _future_immediate_values = _evaluate_questions_forward_search_recursive(
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

    return total_values, immediate_values


def _log_forward_search_summary(cand_questions: list[str], immediate_values: list[float],
                                total_values: list[float], config: Config) -> None:
    if len(cand_questions) == 0 or len(immediate_values) == 0 or len(total_values) == 0:
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

    for i, question in enumerate(cand_questions):
        print(f"Question: {question}")
        print(f"Immediate EIG: {immediate_values[i]}")
        print(f"Future value: {total_values[i] - immediate_values[i]}")
        print(f"Total value: {total_values[i]}")


def evaluate_questions_forward_search(beliefs: BeliefState | list[str], history_questioner: list[dict[str, str]],
                                      cand_questions: list[str],
                                      eig: bool, deterministic: bool, questioner: Model, config: Config,
                                      depth: int = 2) -> list[float]:
    belief_state = ensure_belief_state(beliefs)
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise ValueError("search depth must be a positive integer")
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
    total_values, immediate_values = _evaluate_questions_forward_search_recursive(
        belief_state,
        history_questioner,
        cand_questions,
        eig,
        deterministic,
        questioner,
        config,
        depth,
    )
    _log_forward_search_summary(cand_questions, immediate_values, total_values, config)
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
                                      prior_beliefs: BeliefState | None = None,
                                      belief_context_label: str = "prior distribution") -> str:
    if prior_beliefs is None or len(prior_beliefs.beliefs) == 0:
        question_prompt = question_generation_prompt_naive()
    else:
        weighted_beliefs = sorted(
            zip(prior_beliefs.beliefs, prior_beliefs.probabilities),
            key=lambda entry: entry[1],
            reverse=True,
        )
        question_prompt = weighted_question_generation_prompt_naive(weighted_beliefs, belief_context_label)
    messages = ([candidate_generation_system_message_naive()] + reverse_history(history_questioner) +
                [question_prompt])
    return questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
