from core import BeliefState
from helpers import reverse_history
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from model import Model
else:
    Model = Any
from environments.animals.prompts import greedy_sample_animal_system_prompt, greedy_sample_animal_user_prompt, \
    greedy_sample_animal_system_prompt_naive, greedy_sample_animal_user_prompt_naive, \
    weighted_greedy_sample_animal_user_prompt_naive


def sample_beliefs(beliefs: list[str], history_questioner: list[dict[str, str]], questioner: Model,
                   generation_temperature: float) -> str:
    messages = ([greedy_sample_animal_system_prompt()] + reverse_history(history_questioner) +
                [greedy_sample_animal_user_prompt(beliefs)])
    return questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]


def sample_beliefs_naive(history_questioner: list[dict[str, str]], questioner: Model, generation_temperature: float,
                         prior_beliefs: BeliefState | None = None) -> str:
    if prior_beliefs is None or len(prior_beliefs.hypotheses) == 0:
        user_prompt = greedy_sample_animal_user_prompt_naive()
    else:
        weighted_beliefs = sorted(
            zip(prior_beliefs.hypotheses, prior_beliefs.probabilities),
            key=lambda entry: entry[1],
            reverse=True,
        )
        user_prompt = weighted_greedy_sample_animal_user_prompt_naive(weighted_beliefs)
    messages = ([greedy_sample_animal_system_prompt_naive()] + reverse_history(history_questioner) +
                [user_prompt])
    return questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
