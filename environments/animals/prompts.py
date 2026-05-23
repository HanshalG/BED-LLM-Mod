"""Public surface for animals (20 Questions) prompts.

This module re-exports every prompt function defined in the top-level
:mod:`prompts` module so that new code can import them via the same
``environments.<env_name>.prompts`` pattern used by the location-finding
environment.

The bodies still live in :mod:`prompts`; this is a stable public name only.
"""

from __future__ import annotations

from prompts import (
    answer_likelihood_messages,
    answer_likelihood_system_prompt,
    answer_likelihood_user_prompt,
    answer_question_yesno_system_prompt,
    answer_question_yesnocorrect_system_prompt,
    belief_distribution_system_prompt,
    belief_distribution_user_prompt,
    candidate_generation_system_message,
    candidate_generation_system_message_naive,
    conditional_question_generation_prompt,
    convert_to_prompt_message,
    generate_animals_system_prompt,
    generate_animals_user_prompt,
    generate_more_animals_system_prompt,
    generate_original_animals_system_prompt,
    greedy_sample_animal_system_prompt,
    greedy_sample_animal_system_prompt_naive,
    greedy_sample_animal_user_prompt,
    greedy_sample_animal_user_prompt_naive,
    is_answer_likelihood_messages,
    probability_answer_scores_prompt,
    question_generation_prompt_naive,
    unconditional_question_generation_prompt,
    validate_animal_name_system_prompt,
    validate_animal_name_user_prompt,
    weighted_conditional_question_generation_prompt,
    weighted_greedy_sample_animal_user_prompt_naive,
    weighted_question_generation_prompt_naive,
    weighted_unconditional_question_generation_prompt,
)


__all__ = [
    "answer_likelihood_messages",
    "answer_likelihood_system_prompt",
    "answer_likelihood_user_prompt",
    "answer_question_yesno_system_prompt",
    "answer_question_yesnocorrect_system_prompt",
    "belief_distribution_system_prompt",
    "belief_distribution_user_prompt",
    "candidate_generation_system_message",
    "candidate_generation_system_message_naive",
    "conditional_question_generation_prompt",
    "convert_to_prompt_message",
    "generate_animals_system_prompt",
    "generate_animals_user_prompt",
    "generate_more_animals_system_prompt",
    "generate_original_animals_system_prompt",
    "greedy_sample_animal_system_prompt",
    "greedy_sample_animal_system_prompt_naive",
    "greedy_sample_animal_user_prompt",
    "greedy_sample_animal_user_prompt_naive",
    "is_answer_likelihood_messages",
    "probability_answer_scores_prompt",
    "question_generation_prompt_naive",
    "unconditional_question_generation_prompt",
    "validate_animal_name_system_prompt",
    "validate_animal_name_user_prompt",
    "weighted_conditional_question_generation_prompt",
    "weighted_greedy_sample_animal_user_prompt_naive",
    "weighted_question_generation_prompt_naive",
    "weighted_unconditional_question_generation_prompt",
]
