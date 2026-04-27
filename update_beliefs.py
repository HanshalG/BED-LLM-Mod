from helpers import (
    BeliefState,
    Config,
    _average_labeled_distributions_from_completions,
    clean_generated_belief_labels,
    ensure_belief_state,
    convert_string_to_array,
    _distribution_with_valid_count_from_batched_messages,
    _distribution_with_valid_count_from_messages,
    format_categorical_belief_summary,
    format_belief_state,
    make_belief_state,
    make_uniform_belief_state,
    print_and_log,
    reverse_history,
    sample_permuted_history_messages,
    sort_belief_state_descending,
)
from model import Model
from prompts import generate_animals_system_prompt, generate_more_animals_system_prompt, \
    answer_question_yesno_system_prompt, belief_distribution_system_prompt, belief_distribution_user_prompt, \
    generate_animals_user_prompt, validate_animal_name_system_prompt, validate_animal_name_user_prompt


def generate_new_beliefs(system_prompt: dict[str, str], history_questioner: list[dict[str, str]],
                         questioner: Model, generation_temperature: float, config: Config) -> list[str]:
    print(f"[beliefs] Generating beliefs from {len(history_questioner) // 2} answered round(s)")
    messages = ([system_prompt] + reverse_history(history_questioner) + [generate_animals_user_prompt()])
    new_beliefs = questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
    raw_beliefs = convert_string_to_array(new_beliefs)
    cleaned_beliefs = clean_generated_belief_labels(raw_beliefs)
    print(f"[beliefs] Generated {len(raw_beliefs)} raw belief(s)")
    print(f"[beliefs] {len(cleaned_beliefs)} belief(s) remain after structural cleanup")
    if config.belief_state_mode == "categorical":
        print_and_log(
            f"[categorical] Structural cleanup retained {len(cleaned_beliefs)}/{len(raw_beliefs)} generated belief(s)",
            config,
        )
    return cleaned_beliefs


def _generate_new_beliefs_many(system_prompts: list[dict[str, str]], histories_questioner: list[list[dict[str, str]]],
                               questioner: Model, generation_temperature: float, config: Config) -> list[list[str]]:
    if len(system_prompts) != len(histories_questioner):
        raise ValueError("system_prompts and histories_questioner must have the same length")
    if not system_prompts:
        return []

    batch_messages = [
        [system_prompt] + reverse_history(history_questioner) + [generate_animals_user_prompt()]
        for system_prompt, history_questioner in zip(system_prompts, histories_questioner)
    ]
    completions = questioner.chat_complete_messages_batched(
        batch_messages=batch_messages,
        temperature=generation_temperature,
        block_size=config.batched_block_size,
    )
    if len(completions) != len(batch_messages):
        raise ValueError(
            f"Expected {len(batch_messages)} belief generations, received {len(completions)}"
        )

    branch_beliefs: list[list[str]] = []
    for history_questioner, completion in zip(histories_questioner, completions):
        print(f"[beliefs] Generating beliefs from {len(history_questioner) // 2} answered round(s)")
        raw_beliefs = convert_string_to_array(completion)
        cleaned_beliefs = clean_generated_belief_labels(raw_beliefs)
        print(f"[beliefs] Generated {len(raw_beliefs)} raw belief(s)")
        print(f"[beliefs] {len(cleaned_beliefs)} belief(s) remain after structural cleanup")
        if config.belief_state_mode == "categorical":
            print_and_log(
                f"[categorical] Structural cleanup retained {len(cleaned_beliefs)}/{len(raw_beliefs)} generated belief(s)",
                config,
            )
        branch_beliefs.append(cleaned_beliefs)

    return branch_beliefs


def filter_valid_animal_names_batched(beliefs: list[str], checker: Model, block_size: int) -> list[str]:
    if len(beliefs) == 0:
        return beliefs

    print(f"[beliefs] Validating {len(beliefs)} cleaned belief name(s)")
    conversations = [
        [validate_animal_name_system_prompt(), validate_animal_name_user_prompt(belief)]
        for belief in beliefs
    ]
    completions = checker.chat_complete_messages_batched(
        conversations,
        temperature=0.0,
        block_size=block_size,
        max_new_tokens=8,
    )
    if len(completions) != len(beliefs):
        raise ValueError(
            f"Expected {len(beliefs)} validity completions, received {len(completions)}"
        )

    filtered_beliefs = [
        belief
        for belief, completion in zip(beliefs, completions)
        if completion.strip() == "Yes"
    ]
    print(f"[beliefs] {len(filtered_beliefs)} belief(s) remain after animal-name validation")
    return filtered_beliefs


def _filter_valid_animal_names_many(branch_beliefs: list[list[str]], checker: Model, block_size: int) -> list[list[str]]:
    if not branch_beliefs:
        return []

    flattened_beliefs = [belief for beliefs in branch_beliefs for belief in beliefs]
    if len(flattened_beliefs) == 0:
        return [[] for _ in branch_beliefs]

    print(f"[beliefs] Validating {len(flattened_beliefs)} cleaned belief name(s)")
    conversations = [
        [validate_animal_name_system_prompt(), validate_animal_name_user_prompt(belief)]
        for belief in flattened_beliefs
    ]
    completions = checker.chat_complete_messages_batched(
        conversations,
        temperature=0.0,
        block_size=block_size,
        max_new_tokens=8,
    )
    if len(completions) != len(flattened_beliefs):
        raise ValueError(
            f"Expected {len(flattened_beliefs)} validity completions, received {len(completions)}"
        )

    filtered_branch_beliefs: list[list[str]] = []
    offset = 0
    for beliefs in branch_beliefs:
        branch_completions = completions[offset:offset + len(beliefs)]
        offset += len(beliefs)
        filtered_branch_beliefs.append(
            [
                belief
                for belief, completion in zip(beliefs, branch_completions)
                if completion.strip() == "Yes"
            ]
        )

    print(
        f"[beliefs] {sum(len(beliefs) for beliefs in filtered_branch_beliefs)} belief(s) remain after animal-name validation"
    )
    return filtered_branch_beliefs


def score_beliefs_batched(beliefs: list[str], history_questioner: list[dict[str, str]], questioner: Model,
                          config: Config) -> BeliefState:
    belief_state = make_belief_state(beliefs, fallback_to_uniform=True)
    if len(belief_state.beliefs) == 0:
        return belief_state

    messages = (
        [belief_distribution_system_prompt()]
        + reverse_history(history_questioner)
        + [belief_distribution_user_prompt(belief_state.beliefs)]
    )
    if config.belief_distribution_permute_history:
        batch_messages = [
            [belief_distribution_system_prompt()] + permuted_history + [belief_distribution_user_prompt(belief_state.beliefs)]
            for permuted_history in sample_permuted_history_messages(
                history_questioner,
                config.belief_distribution_num_calls,
            )
        ]
        distribution, valid_distribution_count = _distribution_with_valid_count_from_batched_messages(
            batch_messages,
            belief_state.beliefs,
            temperature=config.belief_probability_temperature,
            complete_messages_batched=questioner.chat_complete_messages_batched,
            block_size=config.batched_block_size,
            fallback_to_uniform=True,
        )
    else:
        distribution, valid_distribution_count = _distribution_with_valid_count_from_messages(
            messages,
            belief_state.beliefs,
            temperature=config.belief_probability_temperature,
            complete_message=questioner.chat_complete,
            num_calls=config.belief_distribution_num_calls,
            fallback_to_uniform=True,
        )
    scored_state = make_belief_state(
        belief_state.beliefs,
        [distribution[belief] for belief in belief_state.beliefs],
        fallback_to_uniform=False,
    )
    scored_state = sort_belief_state_descending(scored_state)

    print(f"[beliefs] Scored belief state: {format_belief_state(scored_state)}")
    if config.belief_state_mode == "categorical":
        if config.belief_distribution_permute_history:
            detail = (
                f"(permuted-history, {valid_distribution_count}/{config.belief_distribution_num_calls} valid)"
            )
        else:
            detail = f"({valid_distribution_count}/{config.belief_distribution_num_calls} valid)"
        print_and_log(
            "[categorical] Scored belief distribution "
            f"{detail}: "
            f"{format_categorical_belief_summary(scored_state)}",
            config,
        )
    return scored_state


def _score_beliefs_many(branch_beliefs: list[list[str]], histories_questioner: list[list[dict[str, str]]], questioner: Model,
                        config: Config) -> list[BeliefState]:
    if len(branch_beliefs) != len(histories_questioner):
        raise ValueError("branch_beliefs and histories_questioner must have the same length")
    if not branch_beliefs:
        return []

    deduped_states = [make_uniform_belief_state(beliefs) for beliefs in branch_beliefs]
    if config.belief_state_mode != "categorical":
        return deduped_states

    branch_prompt_counts: list[int] = []
    branch_labels: list[list[str]] = []
    batch_messages: list[list[dict[str, str]]] = []
    active_branch_indices: list[int] = []

    for branch_idx, (belief_state, history_questioner) in enumerate(zip(deduped_states, histories_questioner)):
        if len(belief_state.beliefs) == 0:
            branch_prompt_counts.append(0)
            branch_labels.append([])
            continue

        labels = belief_state.beliefs
        branch_labels.append(labels)
        active_branch_indices.append(branch_idx)
        if config.belief_distribution_permute_history:
            prompts = [
                [belief_distribution_system_prompt()] + permuted_history + [belief_distribution_user_prompt(labels)]
                for permuted_history in sample_permuted_history_messages(
                    history_questioner,
                    config.belief_distribution_num_calls,
                )
            ]
        else:
            messages = (
                [belief_distribution_system_prompt()]
                + reverse_history(history_questioner)
                + [belief_distribution_user_prompt(labels)]
            )
            prompts = [messages for _ in range(config.belief_distribution_num_calls)]

        branch_prompt_counts.append(len(prompts))
        batch_messages.extend(prompts)

    completions: list[str] = []
    if batch_messages:
        completions = questioner.chat_complete_messages_batched(
            batch_messages=batch_messages,
            temperature=config.belief_probability_temperature,
            block_size=config.batched_block_size,
        )
        if len(completions) != len(batch_messages):
            raise ValueError(
                f"Expected {len(batch_messages)} batched distribution completions, received {len(completions)}"
            )

    scored_states: list[BeliefState] = []
    completion_offset = 0
    for branch_idx, belief_state in enumerate(deduped_states):
        labels = branch_labels[branch_idx]
        if len(labels) == 0:
            scored_states.append(belief_state)
            continue

        prompt_count = branch_prompt_counts[branch_idx]
        branch_completions = completions[completion_offset:completion_offset + prompt_count]
        completion_offset += prompt_count
        distribution, valid_distribution_count = _average_labeled_distributions_from_completions(
            branch_completions,
            labels,
            fallback_to_uniform=True,
        )
        scored_state = make_belief_state(
            labels,
            [distribution[belief] for belief in labels],
            fallback_to_uniform=False,
        )
        scored_state = sort_belief_state_descending(scored_state)

        print(f"[beliefs] Scored belief state: {format_belief_state(scored_state)}")
        detail = f"({valid_distribution_count}/{config.belief_distribution_num_calls} valid)"
        if config.belief_distribution_permute_history:
            detail = (
                f"(permuted-history, {valid_distribution_count}/{config.belief_distribution_num_calls} valid)"
            )
        print_and_log(
            "[categorical] Scored belief distribution "
            f"{detail}: "
            f"{format_categorical_belief_summary(scored_state)}",
            config,
        )
        scored_states.append(scored_state)

    return scored_states


def build_belief_state(beliefs: list[str], history_questioner: list[dict[str, str]], questioner: Model,
                       config: Config) -> BeliefState:
    deduped_state = make_uniform_belief_state(beliefs)
    if config.belief_state_mode == "categorical":
        return score_beliefs_batched(deduped_state.beliefs, history_questioner, questioner, config)
    return deduped_state


def _build_belief_states_many(branch_beliefs: list[list[str]], histories_questioner: list[list[dict[str, str]]],
                              questioner: Model, config: Config) -> list[BeliefState]:
    return _score_beliefs_many(branch_beliefs, histories_questioner, questioner, config)


def initialize_belief_state(beliefs: list[str], history_questioner: list[dict[str, str]], questioner: Model,
                            config: Config) -> BeliefState:
    print(f"[beliefs] Initializing {config.belief_state_mode} belief state")
    cleaned_beliefs = clean_generated_belief_labels(beliefs)
    if len(cleaned_beliefs) != len(beliefs):
        print(
            f"[beliefs] Structural cleanup retained {len(cleaned_beliefs)}/{len(beliefs)} opening belief(s)"
        )

    validated_beliefs = filter_valid_animal_names_batched(
        cleaned_beliefs,
        questioner,
        config.batched_block_size,
    )
    if len(validated_beliefs) == 0 and len(cleaned_beliefs) > 0:
        print("[beliefs] Opening animal-name validation rejected every candidate, falling back to cleaned beliefs")
        validated_beliefs = cleaned_beliefs

    belief_state = build_belief_state(validated_beliefs, history_questioner, questioner, config)
    if config.belief_state_mode == "categorical":
        print_and_log(
            f"[categorical] Initial weighted beliefs: {format_categorical_belief_summary(belief_state)}",
            config,
        )
    return belief_state


def check_beliefs_batched(beliefs: list[str], history_questioner: list[dict[str, str]], checker: Model,
                          answer_temperature: float, block_size: int, threshold_rejection_probability: float) -> list[str]:
    if len(beliefs) == 0 or len(history_questioner) == 0:
        print(f"[beliefs] Skipping belief filtering: beliefs={len(beliefs)}, history_pairs={len(history_questioner) // 2}")
        return beliefs

    filtered_beliefs = []
    print(f"[beliefs] Filtering {len(beliefs)} belief(s) against {len(history_questioner) // 2} answered round(s)")

    conversations = []
    answers = []
    # generate one conversation for each belief and question to check the belief for
    for new_belief in beliefs:
        for (i, history_message) in enumerate(history_questioner):
            if history_message["role"] != "assistant":
                continue
            question = history_message["content"]
            user_question = {"role": "user", "content": question}
            conversations.append([answer_question_yesno_system_prompt(entity=new_belief), user_question])
            answers.append(history_questioner[i+1]["content"].lower())

    # get answers for all conversations in parallel
    probabilities = checker.chat_probabilities_messages_batched(conversations, ["Yes", "No"],
                                                                temperature=answer_temperature, block_size=block_size)

    # for each belief, check if all question-answer-pairs fit and filter out otherwise
    for i, new_belief in enumerate(beliefs):
        valid_belief = True
        probs_belief = probabilities[i*int(len(history_questioner)/2):(i+1)*int(len(history_questioner)/2)]
        for j, answer in enumerate(probs_belief):
            if ((answer["Yes"] > (1 - threshold_rejection_probability) and answers[j] == "no")
                    or (answer["Yes"] < threshold_rejection_probability and answers[j] == "yes")):
                valid_belief = False
                break
        if valid_belief:
            filtered_beliefs.append(new_belief)

    print(f"[beliefs] {len(filtered_beliefs)} belief(s) remain after filtering")
    return filtered_beliefs


def _check_beliefs_many(branch_beliefs: list[list[str]], histories_questioner: list[list[dict[str, str]]], checker: Model,
                        answer_temperature: float, block_size: int, threshold_rejection_probability: float) -> list[list[str]]:
    if len(branch_beliefs) != len(histories_questioner):
        raise ValueError("branch_beliefs and histories_questioner must have the same length")
    if not branch_beliefs:
        return []

    filtered_branch_beliefs: list[list[str]] = [[] for _ in branch_beliefs]
    conversations = []
    answers_by_branch: list[list[str]] = []
    num_pairs_by_branch: list[int] = []

    for branch_idx, (beliefs, history_questioner) in enumerate(zip(branch_beliefs, histories_questioner)):
        if len(beliefs) == 0 or len(history_questioner) == 0:
            print(
                f"[beliefs] Skipping belief filtering: beliefs={len(beliefs)}, history_pairs={len(history_questioner) // 2}"
            )
            filtered_branch_beliefs[branch_idx] = list(beliefs)
            answers_by_branch.append([])
            num_pairs_by_branch.append(0)
            continue

        print(f"[beliefs] Filtering {len(beliefs)} belief(s) against {len(history_questioner) // 2} answered round(s)")
        branch_answers = []
        for i, history_message in enumerate(history_questioner):
            if history_message["role"] != "assistant":
                continue
            branch_answers.append(history_questioner[i + 1]["content"].lower())
        answers_by_branch.append(branch_answers)
        num_pairs_by_branch.append(len(branch_answers))

        for new_belief in beliefs:
            for history_message in history_questioner:
                if history_message["role"] != "assistant":
                    continue
                user_question = {"role": "user", "content": history_message["content"]}
                conversations.append([answer_question_yesno_system_prompt(entity=new_belief), user_question])

    if not conversations:
        return filtered_branch_beliefs

    probabilities = checker.chat_probabilities_messages_batched(
        conversations,
        ["Yes", "No"],
        temperature=answer_temperature,
        block_size=block_size,
    )

    probability_offset = 0
    for branch_idx, beliefs in enumerate(branch_beliefs):
        num_pairs = num_pairs_by_branch[branch_idx]
        if num_pairs == 0:
            continue
        branch_answers = answers_by_branch[branch_idx]
        filtered_beliefs = []
        for new_belief in beliefs:
            valid_belief = True
            probs_belief = probabilities[probability_offset:probability_offset + num_pairs]
            probability_offset += num_pairs
            for answer_idx, answer in enumerate(probs_belief):
                if (
                    (answer["Yes"] > (1 - threshold_rejection_probability) and branch_answers[answer_idx] == "no")
                    or (answer["Yes"] < threshold_rejection_probability and branch_answers[answer_idx] == "yes")
                ):
                    valid_belief = False
                    break
            if valid_belief:
                filtered_beliefs.append(new_belief)
        filtered_branch_beliefs[branch_idx] = filtered_beliefs
        print(f"[beliefs] {len(filtered_beliefs)} belief(s) remain after filtering")

    return filtered_branch_beliefs


def _update_beliefs_many(histories_questioner: list[list[dict[str, str]]], beliefs: BeliefState | list[str], questioner: Model,
                         deterministic: bool, config: Config) -> list[BeliefState]:
    belief_state = ensure_belief_state(beliefs)
    if not histories_questioner:
        return []

    prior_beliefs = belief_state.beliefs
    generation_temperature = config.generation_temperature_diverse
    max_num_samples = config.max_num_samples
    min_num_samples = config.min_num_samples
    answer_temperature = config.answer_temperature
    block_size = config.batched_block_size
    threshold_rejection_probability = config.threshold_rejection_probability

    current_system_prompts = [
        generate_animals_system_prompt(max_num_samples, min_num_samples)
        for _ in histories_questioner
    ]
    beliefs_new_many = _generate_new_beliefs_many(
        current_system_prompts,
        histories_questioner,
        questioner,
        generation_temperature,
        config,
    )

    if deterministic:
        return _build_belief_states_many(beliefs_new_many, histories_questioner, questioner, config)

    beliefs_new_many = _filter_valid_animal_names_many(
        beliefs_new_many,
        questioner,
        block_size,
    )
    beliefs_new_many = _check_beliefs_many(
        beliefs_new_many,
        histories_questioner,
        questioner,
        answer_temperature,
        block_size,
        threshold_rejection_probability,
    )

    filtered_beliefs_old_many = _check_beliefs_many(
        [list(prior_beliefs) for _ in histories_questioner],
        [history_questioner[-2:] for history_questioner in histories_questioner],
        questioner,
        answer_temperature,
        block_size,
        threshold_rejection_probability,
    )

    beliefs_updated_many = [
        make_belief_state(beliefs_new + filtered_beliefs_old, fallback_to_uniform=True).beliefs
        for beliefs_new, filtered_beliefs_old in zip(beliefs_new_many, filtered_beliefs_old_many)
    ]

    for retry_idx in range(2):
        retry_indices = [
            idx
            for idx, beliefs_updated in enumerate(beliefs_updated_many)
            if len(beliefs_updated) < min_num_samples
        ]
        if not retry_indices:
            break

        retry_prompts = []
        retry_histories = []
        for idx in retry_indices:
            current_system_prompts[idx] = generate_more_animals_system_prompt(
                beliefs_updated_many[idx],
                min_num_samples - len(beliefs_updated_many[idx]),
            )
            retry_prompts.append(current_system_prompts[idx])
            retry_histories.append(histories_questioner[idx])
            print(f"[beliefs] Retry {retry_idx + 1}/2 to reach minimum of {min_num_samples} belief(s)")

        beliefs_retry_many = _generate_new_beliefs_many(
            retry_prompts,
            retry_histories,
            questioner,
            generation_temperature,
            config,
        )
        beliefs_retry_many = _filter_valid_animal_names_many(
            beliefs_retry_many,
            questioner,
            block_size,
        )
        beliefs_retry_many = _check_beliefs_many(
            beliefs_retry_many,
            retry_histories,
            questioner,
            answer_temperature,
            block_size,
            threshold_rejection_probability,
        )

        for idx, beliefs_retry in zip(retry_indices, beliefs_retry_many):
            beliefs_updated_many[idx] = make_belief_state(
                beliefs_retry + beliefs_updated_many[idx],
                fallback_to_uniform=True,
            ).beliefs
            print(f"[beliefs] After retry {retry_idx + 1}, belief pool has {len(beliefs_updated_many[idx])} candidate(s)")

    fallback_indices = [
        idx
        for idx, beliefs_updated in enumerate(beliefs_updated_many)
        if len(beliefs_updated) == 0
    ]
    if fallback_indices:
        for _idx in fallback_indices:
            print("[beliefs] No valid beliefs survived filtering, falling back to unfiltered generation")

        fallback_beliefs_many = _generate_new_beliefs_many(
            [current_system_prompts[idx] for idx in fallback_indices],
            [histories_questioner[idx] for idx in fallback_indices],
            questioner,
            generation_temperature,
            config,
        )
        fallback_beliefs_many = _filter_valid_animal_names_many(
            fallback_beliefs_many,
            questioner,
            block_size,
        )
        for idx, fallback_beliefs in zip(fallback_indices, fallback_beliefs_many):
            beliefs_updated_many[idx] = fallback_beliefs

    return _build_belief_states_many(beliefs_updated_many, histories_questioner, questioner, config)


def update_beliefs_batched(history: list[(str, str)], beliefs: BeliefState | list[str], questioner: Model,
                           deterministic: bool, config: Config) -> BeliefState:
    belief_state = ensure_belief_state(beliefs)
    prior_beliefs = belief_state.beliefs
    prior_summary = format_categorical_belief_summary(belief_state)
    generation_temperature, max_num_samples, min_num_samples = config.generation_temperature_diverse, config.max_num_samples, config.min_num_samples
    answer_temperature, block_size, threshold_rejection_probability = config.answer_temperature, config.batched_block_size, config.threshold_rejection_probability

    # generate new beliefs
    print(f"[beliefs] Updating beliefs with {len(prior_beliefs)} prior belief(s); deterministic={deterministic}")
    if config.belief_state_mode == "categorical":
        if len(history) >= 2:
            latest_question = history[-2]["content"]
            latest_answer = history[-1]["content"]
            print_and_log(
                f"[categorical] Updating weighted beliefs after answer '{latest_answer}' to question: {latest_question}",
                config,
            )
        print_and_log(f"[categorical] Prior weighted beliefs: {prior_summary}", config)
    system_prompt = generate_animals_system_prompt(max_num_samples, min_num_samples)
    beliefs_new = generate_new_beliefs(system_prompt, history, questioner, generation_temperature, config)
    if config.belief_state_mode == "categorical":
        print_and_log(
            f"[categorical] Generated {len(beliefs_new)} new categorical candidate(s)",
            config,
        )

    # in split baseline, return the beliefs sampled using the current history
    if deterministic:
        deterministic_state = build_belief_state(beliefs_new, history, questioner, config)
        print(f"[beliefs] Deterministic mode returning {len(deterministic_state.beliefs)} belief(s)")
        return deterministic_state

    # filter new beliefs according to previous questions+answers
    beliefs_new = filter_valid_animal_names_batched(
        beliefs_new,
        questioner,
        block_size,
    )
    if config.belief_state_mode == "categorical":
        print_and_log(
            f"[categorical] Retained {len(beliefs_new)} generated belief(s) after animal-name validation",
            config,
        )
    beliefs_new = check_beliefs_batched(
        beliefs_new,
        history,
        questioner,
        answer_temperature,
        block_size,
        threshold_rejection_probability,
    )

    # filter previous beliefs with new question+answer
    filtered_beliefs_old = check_beliefs_batched(
        prior_beliefs,
        history[-2:],
        questioner,
        answer_temperature,
        block_size,
        threshold_rejection_probability,
    )
    if config.belief_state_mode == "categorical":
        print_and_log(
            f"[categorical] Retained {len(filtered_beliefs_old)} prior categorical belief(s) after the latest answer",
            config,
        )
    # throw out duplicates
    beliefs_updated = make_belief_state(beliefs_new + filtered_beliefs_old, fallback_to_uniform=True).beliefs
    print(f"[beliefs] {len(beliefs_updated)} unique belief(s) remain after merge")
    if config.belief_state_mode == "categorical":
        print_and_log(
            f"[categorical] Merged categorical pool now has {len(beliefs_updated)} candidate(s)",
            config,
        )

    # try to generate new beliefs twice more if no sufficient number could be generated
    for retry_idx in range(2):
        if len(beliefs_updated) >= min_num_samples:
            break
        print(f"[beliefs] Retry {retry_idx + 1}/2 to reach minimum of {min_num_samples} belief(s)")
        system_prompt = generate_more_animals_system_prompt(beliefs_updated, min_num_samples - len(beliefs_updated))
        beliefs_new = generate_new_beliefs(system_prompt, history, questioner, generation_temperature, config)
        beliefs_new = filter_valid_animal_names_batched(
            beliefs_new,
            questioner,
            block_size,
        )
        if config.belief_state_mode == "categorical":
            print_and_log(
                f"[categorical] Retained {len(beliefs_new)} retry-generated belief(s) after animal-name validation",
                config,
            )
        beliefs_new = check_beliefs_batched(
            beliefs_new,
            history,
            questioner,
            answer_temperature,
            block_size,
            threshold_rejection_probability,
        )
        beliefs_updated = make_belief_state(beliefs_new + beliefs_updated, fallback_to_uniform=True).beliefs
        print(f"[beliefs] After retry {retry_idx + 1}, belief pool has {len(beliefs_updated)} candidate(s)")
        if config.belief_state_mode == "categorical":
            print_and_log(
                f"[categorical] Retry {retry_idx + 1} produced a categorical pool of {len(beliefs_updated)} candidate(s)",
                config,
            )

    # If no valid beliefs at all can be generated, generate unfiltered to continue the game
    if len(beliefs_updated) == 0:
        print("[beliefs] No valid beliefs survived filtering, falling back to unfiltered generation")
        if config.belief_state_mode == "categorical":
            print_and_log(
                "[categorical] No valid weighted beliefs survived filtering; falling back to unfiltered generation",
                config,
            )
        beliefs_updated = generate_new_beliefs(system_prompt, history, questioner, generation_temperature, config)
        beliefs_updated = filter_valid_animal_names_batched(
            beliefs_updated,
            questioner,
            block_size,
        )
        if config.belief_state_mode == "categorical":
            print_and_log(
                f"[categorical] Retained {len(beliefs_updated)} fallback-generated belief(s) after animal-name validation",
                config,
            )

    updated_state = build_belief_state(beliefs_updated, history, questioner, config)
    print(f"[beliefs] Belief update complete with {len(updated_state.beliefs)} candidate(s)")
    if config.belief_state_mode == "categorical":
        print_and_log(
            f"[categorical] Weighted belief transition: before={prior_summary} -> after={format_categorical_belief_summary(updated_state)}",
            config,
        )
    return updated_state
