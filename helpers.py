from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from prompts import answer_question_yesnocorrect_system_prompt, generate_original_animals_system_prompt, \
    probability_answer_scores_prompt

if TYPE_CHECKING:
    from model import Model


@dataclass
class Config:
    version: int = 0
    model_names: list[tuple[str, str]] = field(default_factory=list)
    method_names: list[str] = field(default_factory=list)
    animals: list[list[str]] = field(default_factory=list)
    batched_block_size: int = 50
    generation_temperature_diverse: float = 1.3
    generation_temperature_simple: float = 1.0
    answer_temperature: float = 0.7
    target_num_questions: int = 15
    num_mc_samples: int = 15
    max_num_samples: int = 50
    min_num_samples: int = 15
    threshold_rejection_probability: float = 0.2
    run_id: str = ""
    log_path: Path | None = None


def load_config(path: str) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config(
        version = raw.get("version", 0),
        model_names = [(p[0], p[1]) for p in raw.get("model_names", [])],
        method_names = raw.get("method_names", raw.get("extraction_methods", [])),
        animals = raw.get("animals", []),
        batched_block_size = raw.get("batched_block_size", 50),
        generation_temperature_diverse = raw.get("generation_temperature_diverse", 1.3),
        generation_temperature_simple = raw.get("generation_temperature_simple", 1.0),
        answer_temperature = raw.get("answer_temperature", 0.7),
        target_num_questions = raw.get("target_num_questions", 15),
        num_mc_samples = raw.get("num_mc_samples", 15),
        max_num_samples = raw.get("max_num_samples", 50),
        min_num_samples = raw.get("min_num_samples", 15),
        threshold_rejection_probability = raw.get("threshold_rejection_probability", 0.2),
    )


def build_output_stem(run_id: str, method_name: str, questioner: str, answerer: str, version: int) -> str:
    return (
        f"{run_id}_{method_name}_Q:{questioner.replace('/', '_')},"
        f"A:{answerer.replace('/', '_')}_{version}_animals"
    )


def write_to_log(message: str, config: Config) -> None:
    if config.log_path is None:
        raise ValueError("config.log_path must be set before logging")

    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    with config.log_path.open("a", encoding="utf-8") as file:
        file.write(message)


def _build_probability_messages(messages: list[dict[str, str]], responses: list[str]) -> list[dict[str, str]]:
    probability_messages = [dict(message) for message in messages]
    instruction = probability_answer_scores_prompt(responses)["content"]

    if probability_messages and probability_messages[-1]["role"] == "user":
        original_content = probability_messages[-1]["content"].rstrip()
        if original_content:
            probability_messages[-1]["content"] = f"{original_content}\n\n{instruction}"
        else:
            probability_messages[-1]["content"] = instruction
        return probability_messages

    probability_messages.append({"role": "user", "content": instruction})
    return probability_messages


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_probability_response(response_text: str, responses: list[str]) -> dict[str, float]:
    if not responses:
        return {}

    try:
        payload = json.loads(_strip_code_fences(response_text))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid probability JSON: {response_text!r}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Probability response must be a JSON object: {response_text!r}")

    scores: list[float] = []
    for response in responses:
        raw_value = payload.get(response, 0.0)
        try:
            score = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Probability for {response!r} must be numeric: {raw_value!r}") from exc
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"Probability for {response!r} must be finite and non-negative: {raw_value!r}")
        scores.append(score)

    total = sum(scores)
    if total <= 0.0:
        raise ValueError(f"Probability response must contain a positive total weight: {response_text!r}")

    return {
        response: score / total
        for response, score in zip(responses, scores)
    }


def _probability_results_from_messages(batch_messages: list[list[dict[str, str]]], responses: list[str], block_size: int,
                                       complete_messages_batched: Callable[..., list[str]]) -> list[dict[str, float]]:
    probability_messages = [
        _build_probability_messages(messages, responses)
        for messages in batch_messages
    ]
    completions = complete_messages_batched(
        probability_messages,
        temperature=0.0,
        block_size=block_size,
        max_new_tokens=4096,
    )
    return [
        _normalize_probability_response(completion, responses)
        for completion in completions
    ]


# prompts ask to generate collection of entities, one on each line --> convert the returned string to an array
def convert_string_to_array(response):
    return [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]


def _binary_entropy(p_yes: float, p_no: float) -> float:
    p_yes_clipped = max(p_yes, 1e-12)
    p_no_clipped = max(p_no, 1e-12)
    return - (p_yes_clipped * np.log(p_yes_clipped) + p_no_clipped * np.log(p_no_clipped))


# reverses a messages array so that the final question comes first
def reverse_history(history_questioner: list[dict[str,str]]) -> list[dict[str,str]]:
    blocks = [history_questioner[i:i+2] for i in range(0, len(history_questioner), 2)]
    return [x for b in blocks[::-1] for x in b]


def get_question_answered(question: str, goal_object: str, answerer: Model, answer_temperature: float) -> str:
    user_question = {"role": "user", "content": f"{question}"}
    messages = [answer_question_yesnocorrect_system_prompt(entity=goal_object), user_question]
    return answerer.chat_complete(messages=messages, temperature=answer_temperature)[0]


def generate_original_beliefs(questioner: Model, config: Config) -> list[str]:
    generation_temperature, max_num_samples, min_num_samples = config.generation_temperature_diverse, config.max_num_samples, config.min_num_samples
    user_question = {"role": "user", "content": f"Let\'s start the game of 20 questions. Generate a diverse "
                                                f"set of animals, at least {min_num_samples}."}
    messages = [generate_original_animals_system_prompt(max_num_samples), user_question]
    new_beliefs = questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
    return convert_string_to_array(new_beliefs)
