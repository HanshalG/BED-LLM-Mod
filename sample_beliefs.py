from helpers import reverse_history
from model import Model
from prompts import greedy_sample_animal_system_prompt, greedy_sample_animal_user_prompt, \
    greedy_sample_animal_system_prompt_naive, greedy_sample_animal_user_prompt_naive


def sample_beliefs(beliefs: list[str], history_questioner: list[dict[str, str]], questioner: Model,
                   generation_temperature: float) -> str:
    messages = ([greedy_sample_animal_system_prompt()] + reverse_history(history_questioner) +
                [greedy_sample_animal_user_prompt(beliefs)])
    return questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]


def sample_beliefs_naive(history_questioner: list[dict[str, str]], questioner: Model, generation_temperature: float) -> str:
    messages = ([greedy_sample_animal_system_prompt_naive()] + reverse_history(history_questioner) +
                [greedy_sample_animal_user_prompt_naive()])
    return questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
