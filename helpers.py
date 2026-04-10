from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import yaml

from prompts import answer_question_yesnocorrect_system_prompt, generate_original_animals_system_prompt, \
    probability_answer_scores_prompt

if TYPE_CHECKING:
    from model import Model


ReasoningEffort = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ModelSpec:
    model: str
    thinking: bool | None = None
    reasoning_effort: ReasoningEffort | None = None


@dataclass(frozen=True)
class ModelPair:
    questioner: ModelSpec
    answerer: ModelSpec


@dataclass
class Config:
    version: int = 0
    model_pairs: list[ModelPair] = field(default_factory=list)
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


def _normalize_model_spec(raw_spec: object, side_name: str) -> ModelSpec:
    if not isinstance(raw_spec, dict):
        raise ValueError(f"{side_name} must be a mapping with at least a 'model' field")

    model_name = raw_spec.get("model")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"{side_name}.model must be a non-empty string")

    thinking = raw_spec.get("thinking")
    if thinking is not None and not isinstance(thinking, bool):
        raise ValueError(f"{side_name}.thinking must be a boolean when provided")

    reasoning_effort = raw_spec.get("reasoning_effort")
    if reasoning_effort is not None and reasoning_effort not in {"low", "medium", "high"}:
        raise ValueError(f"{side_name}.reasoning_effort must be one of: low, medium, high")

    is_qwen = model_name.startswith("Qwen/")
    is_gemma = model_name.startswith("google/gemma-4")
    is_harmony = model_name.startswith("openai/gpt-oss")
    if is_harmony:
        if thinking is not None:
            raise ValueError(f"{side_name}.thinking is not supported for {model_name}")
        return ModelSpec(
            model=model_name,
            reasoning_effort=reasoning_effort or "low",
        )

    if is_qwen or is_gemma:
        if reasoning_effort is not None:
            raise ValueError(f"{side_name}.reasoning_effort is not supported for {model_name}")
        return ModelSpec(
            model=model_name,
            thinking=False if thinking is None else thinking,
        )

    if thinking is not None:
        raise ValueError(f"{side_name}.thinking is only supported for Qwen and Gemma 4 models")
    if reasoning_effort is not None:
        raise ValueError(f"{side_name}.reasoning_effort is only supported for gpt-oss models")

    return ModelSpec(model=model_name)


def _normalize_model_pair(raw_pair: object, index: int) -> ModelPair:
    if not isinstance(raw_pair, dict):
        raise ValueError(f"Each model_pairs entry must be a mapping, got {type(raw_pair).__name__}")

    if "questioner" not in raw_pair or "answerer" not in raw_pair:
        raise ValueError(f"model_pairs[{index}] must contain both 'questioner' and 'answerer'")

    return ModelPair(
        questioner=_normalize_model_spec(raw_pair["questioner"], f"model_pairs[{index}].questioner"),
        answerer=_normalize_model_spec(raw_pair["answerer"], f"model_pairs[{index}].answerer"),
    )


def load_config(path: str) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    model_pairs = [
        _normalize_model_pair(pair, index)
        for index, pair in enumerate(raw.get("model_pairs", []))
    ]
    return Config(
        version = raw.get("version", 0),
        model_pairs = model_pairs,
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


def build_models(model_pairs: list[ModelPair], build_model_adapter: Callable[[ModelSpec], "Model"]) -> dict[ModelSpec, "Model"]:
    model_specs = {
        pair.questioner
        for pair in model_pairs
    } | {
        pair.answerer
        for pair in model_pairs
    }
    return {
        spec: build_model_adapter(spec)
        for spec in model_specs
    }


def _model_spec_stem(spec: ModelSpec) -> str:
    base = spec.model.replace("/", "_")
    if spec.reasoning_effort is not None:
        return f"{base}__reasoning-{spec.reasoning_effort}"
    if spec.thinking is not None:
        return f"{base}__thinking-{'on' if spec.thinking else 'off'}"
    return base


def build_output_stem(run_id: str, method_name: str, questioner: ModelSpec, answerer: ModelSpec, version: int) -> str:
    return (
        f"{run_id}_{method_name}_Q:{_model_spec_stem(questioner)},"
        f"A:{_model_spec_stem(answerer)}_{version}_animals"
    )


def resolve_run_id() -> str:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        return slurm_job_id

    return datetime.now().strftime("%Y%m%dT%H%M%S")


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
