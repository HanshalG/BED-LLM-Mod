import numpy as np
import wandb

from helpers import Config, reverse_history, _binary_entropy, convert_string_to_array
from model import Model
from prompts import candidate_generation_system_message, conditional_question_generation_prompt, \
    unconditional_question_generation_prompt, candidate_generation_system_message_naive, \
    question_generation_prompt_naive, answer_question_yesno_system_prompt
from update_beliefs import check_beliefs_batched, update_beliefs_batched

from helpers import write_to_log


def generate_candidate_questions(beliefs: list[str], history_questioner: list[dict[str, str]],
                                 questioner: Model, generation_temperature: float, num_questions: int) -> list[str]:
    # if there are less than 3 beliefs left, best question is always to check one of them
    if len(beliefs) in [1, 2]:
        print(f"[candidate-gen] Only {len(beliefs)} belief(s) left, switching to direct guess")
        return [f"Is it {beliefs[0]}?"]

    print(f"[candidate-gen] Building candidates from {len(beliefs)} belief(s) and {len(history_questioner) // 2} prior round(s)")
    messages = ([candidate_generation_system_message()] + reverse_history(history_questioner) +
                [conditional_question_generation_prompt(beliefs, num_questions)])
    candidate_questions = questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
    candidate_questions = convert_string_to_array(candidate_questions)
    print(f"[candidate-gen] Received {len(candidate_questions)} candidate(s) from conditional generation")

    if len(candidate_questions) < num_questions:
        print(f"[candidate-gen] Backfilling {num_questions - len(candidate_questions)} more candidate(s)")
        messages = ([candidate_generation_system_message()] + reverse_history(history_questioner) +
                    [unconditional_question_generation_prompt(candidate_questions, num_questions - len(candidate_questions))])
        new_candidate_questions = questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
        candidate_questions = candidate_questions + convert_string_to_array(new_candidate_questions)
        print(f"[candidate-gen] Candidate pool now has {len(candidate_questions)} question(s)")

    return candidate_questions


def _draw_belief_samples(beliefs: list[str], deterministic: bool, num_mc_samples: int) -> tuple[int, list[str] | np.ndarray]:
    if len(beliefs) == 0:
        return 0, []

    if deterministic:
        return len(beliefs), beliefs

    if len(beliefs) < num_mc_samples:
        return len(beliefs), beliefs

    return num_mc_samples, np.random.choice(beliefs, size=num_mc_samples, replace=True)


def _score_questions_from_samples(samples: list[str] | np.ndarray, cand_questions: list[str], eig: bool, questioner: Model,
                                  answer_temperature: float, block_size: int) -> tuple[list[float], list[float], list[float]]:
    if len(cand_questions) == 0 or len(samples) == 0:
        return [0.0] * len(cand_questions), [0.0] * len(cand_questions), [0.0] * len(cand_questions)

    conversations = []
    for question in cand_questions:
        for sample in samples:
            user_question = {"role": "user", "content": f"{question}"}
            messages = [answer_question_yesno_system_prompt(entity=sample), user_question]
            conversations.append(messages)

    probabilities = questioner.chat_probabilities_messages_batched(
        conversations,
        ["Yes", "No"],
        temperature=answer_temperature,
        block_size=block_size,
    )

    num_samples = len(samples)
    question_values = [0.0] * len(cand_questions)
    p_yes_values = [0.0] * len(cand_questions)
    p_no_values = [0.0] * len(cand_questions)

    for i, _question in enumerate(cand_questions):
        answers = probabilities[i * num_samples:(i + 1) * num_samples]
        p_yes = []
        p_no = []
        entropy_sum = []

        for answer in answers:
            p_yes.append(answer["Yes"])
            p_no.append(answer["No"])
            entropy_sum.append(_binary_entropy(answer["Yes"], answer["No"]))

        p_hat_yes = float(np.mean(p_yes))
        p_hat_no = float(np.mean(p_no))
        entropy = _binary_entropy(p_hat_yes, p_hat_no)

        p_yes_values[i] = p_hat_yes
        p_no_values[i] = p_hat_no
        if eig:
            question_values[i] = entropy - float(np.mean(entropy_sum))
        else:
            question_values[i] = entropy

    return question_values, p_yes_values, p_no_values


def _future_beliefs_for_answer(beliefs: list[str], history_questioner: list[dict[str, str]], question: str, answer: str,
                               questioner: Model, deterministic: bool, config: Config) -> list[str]:
    hypothetical_history = history_questioner + [
        {"role": "assistant", "content": question},
        {"role": "user", "content": answer},
    ]

    if deterministic:
        return update_beliefs_batched(hypothetical_history, beliefs, questioner, deterministic, config)

    return check_beliefs_batched(
        beliefs,
        hypothetical_history[-2:],
        questioner,
        config.answer_temperature,
        config.batched_block_size,
        config.threshold_rejection_probability,
    )


def evaluate_questions_forward_search(beliefs: list[str], history_questioner: list[dict[str, str]], cand_questions: list[str],
                                      eig: bool, deterministic: bool, questioner: Model, config: Config,
                                      depth: int = 2) -> list[float]:
    if depth == 1:
        print(f"[question-score] Depth-1 evaluation for {len(cand_questions)} question(s)")
        return evaluate_questions_batched(
            beliefs,
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
    _num_samples, samples = _draw_belief_samples(beliefs, deterministic, config.num_mc_samples)
    print(f"[question-score] Depth-2 evaluation for {len(cand_questions)} question(s) using {len(samples)} sample(s)")

    #immediate values is the expected 1 step info gain of asking the candidate questions
    immediate_values, p_yes_values, p_no_values = _score_questions_from_samples(
        samples,
        cand_questions,
        eig,
        questioner,
        config.answer_temperature,
        config.batched_block_size,
    )

    total_values = immediate_values.copy()
    for i, question in enumerate(cand_questions):
        print(f"[question-score] Exploring future branches for question {i + 1}/{len(cand_questions)}: {question}")
        expected_future_value = 0.0
        future_value_yes = 0.0
        future_value_no = 0.0

        for answer, branch_probability in (("Yes", p_yes_values[i]), ("No", p_no_values[i])):
            #future beliefs can either use full update or just filtered beliefs
            future_beliefs = _future_beliefs_for_answer(
                beliefs,
                history_questioner,
                question,
                answer,
                questioner,
                deterministic,
                config,
            )
            if len(future_beliefs) == 0:
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

    #goes to wandb logger
    for i, question in enumerate(cand_questions):
        print(f"Question: {question}")
        print(f"Immediate EIG: {immediate_values[i]}")
        print(f"Future value: {total_values[i] - immediate_values[i]}")
        print(f"Total value: {total_values[i]}")

    return total_values

def evaluate_questions_batched(beliefs: list[str], cand_questions: list[str], eig: bool, deterministic: bool,
                                   questioner: Model, answer_temperature: float, num_mc_samples: int, block_size: int) -> list[float]:
    _num_samples, samples = _draw_belief_samples(beliefs, deterministic, num_mc_samples)
    print(f"[question-score] Batched scoring for {len(cand_questions)} question(s) using {len(samples)} sample(s)")
    question_values, _p_yes_values, _p_no_values = _score_questions_from_samples(
        samples,
        cand_questions,
        eig,
        questioner,
        answer_temperature,
        block_size,
    )
    return question_values


def generate_candidate_question_naive(history_questioner: list[dict[str, str]], questioner: Model,
                                      generation_temperature: float) -> str:
    messages = ([candidate_generation_system_message_naive()] + reverse_history(history_questioner) +
                [question_generation_prompt_naive()])
    return questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
