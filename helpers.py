from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
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
BeliefStateMode = Literal["uniform", "categorical"]
GameMode = Literal["animals", "wordle"]


@dataclass(frozen=True)
class ModelSpec:
    model: str
    thinking: bool | None = None
    reasoning_effort: ReasoningEffort | None = None
    use_logprobs: bool = False


@dataclass(frozen=True)
class ModelPair:
    questioner: ModelSpec
    answerer: ModelSpec


@dataclass(frozen=True)
class BeliefState:
    beliefs: list[str] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.beliefs) != len(self.probabilities):
            raise ValueError("BeliefState beliefs and probabilities must have the same length")


@dataclass
class Config:
    version: int = 0
    model_pairs: list[ModelPair] = field(default_factory=list)
    method_names: list[str] = field(default_factory=list)
    animals: list[list[str]] = field(default_factory=list)
    game: GameMode = "animals"
    wordle_solution_words_path: str | None = None
    wordle_valid_words_path: str | None = None
    max_wordle_guesses: int = 6
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float = 0.88
    max_model_len: int = 4096
    batched_block_size: int = 50
    generation_temperature_diverse: float = 1.3
    generation_temperature_simple: float = 1.0
    answer_temperature: float = 0.7
    search_depth: int = 1
    target_num_questions: int = 15
    num_mc_samples: int = 15
    max_num_samples: int = 50
    min_num_samples: int = 15
    threshold_rejection_probability: float = 0.2
    belief_state_mode: BeliefStateMode = "uniform"
    belief_probability_temperature: float = 0.0
    belief_distribution_num_calls: int = 1
    belief_distribution_permute_history: bool = False
    probability_parse_fallback_to_uniform: bool = True
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

    use_logprobs = raw_spec.get("use_logprobs", False)
    if not isinstance(use_logprobs, bool):
        raise ValueError(f"{side_name}.use_logprobs must be a boolean when provided")

    is_qwen = model_name.startswith("Qwen/")
    is_qwen25 = model_name.startswith("Qwen/Qwen2.5")
    is_gemma = model_name.startswith("google/gemma-4")
    is_harmony = model_name.startswith("openai/gpt-oss")
    if is_harmony:
        if thinking is not None:
            raise ValueError(f"{side_name}.thinking is not supported for {model_name}")
        if use_logprobs:
            raise ValueError(f"{side_name}.use_logprobs is only supported for Qwen2.5 models")
        return ModelSpec(
            model=model_name,
            reasoning_effort=reasoning_effort or "low",
        )

    if is_qwen or is_gemma:
        if reasoning_effort is not None:
            raise ValueError(f"{side_name}.reasoning_effort is not supported for {model_name}")
        if use_logprobs and not is_qwen25:
            raise ValueError(f"{side_name}.use_logprobs is only supported for Qwen2.5 models")
        return ModelSpec(
            model=model_name,
            thinking=False if thinking is None else thinking,
            use_logprobs=use_logprobs,
        )

    if thinking is not None:
        raise ValueError(f"{side_name}.thinking is only supported for Qwen and Gemma 4 models")
    if reasoning_effort is not None:
        raise ValueError(f"{side_name}.reasoning_effort is only supported for gpt-oss models")
    if use_logprobs:
        raise ValueError(f"{side_name}.use_logprobs is only supported for Qwen2.5 models")

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
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    game = raw.get("game", "animals")
    if game not in {"animals", "wordle"}:
        raise ValueError("game must be one of: animals, wordle")
    model_pairs = [
        _normalize_model_pair(pair, index)
        for index, pair in enumerate(raw.get("model_pairs", []))
    ]
    wordle_solution_words_path = raw.get("wordle_solution_words_path")
    if isinstance(wordle_solution_words_path, str):
        solution_path = Path(wordle_solution_words_path)
        if not solution_path.is_absolute():
            solution_path = config_path.parent / solution_path
        wordle_solution_words_path = str(solution_path)
    if game == "wordle":
        if not isinstance(wordle_solution_words_path, str) or not wordle_solution_words_path:
            raise ValueError("wordle_solution_words_path is required when game is wordle")
        if not Path(wordle_solution_words_path).is_file():
            raise ValueError(f"wordle_solution_words_path does not exist: {wordle_solution_words_path}")
    wordle_valid_words_path = raw.get("wordle_valid_words_path")
    if isinstance(wordle_valid_words_path, str):
        valid_words_path = Path(wordle_valid_words_path)
        if not valid_words_path.is_absolute():
            valid_words_path = config_path.parent / valid_words_path
        wordle_valid_words_path = str(valid_words_path)
    elif game == "wordle":
        default_valid_words_path = config_path.parent / "valid-wordle-words.txt"
        if default_valid_words_path.is_file():
            wordle_valid_words_path = str(default_valid_words_path)
    if wordle_valid_words_path is not None:
        if not isinstance(wordle_valid_words_path, str) or not wordle_valid_words_path:
            raise ValueError("wordle_valid_words_path must be a non-empty string when provided")
        if not Path(wordle_valid_words_path).is_file():
            raise ValueError(f"wordle_valid_words_path does not exist: {wordle_valid_words_path}")

    max_wordle_guesses = raw.get("max_wordle_guesses", 6)
    if not isinstance(max_wordle_guesses, int) or isinstance(max_wordle_guesses, bool):
        raise ValueError("max_wordle_guesses must be an integer")
    if max_wordle_guesses < 1:
        raise ValueError("max_wordle_guesses must be at least 1")
    tensor_parallel_size = raw.get("tensor_parallel_size")
    if tensor_parallel_size is not None:
        if not isinstance(tensor_parallel_size, int) or isinstance(tensor_parallel_size, bool):
            raise ValueError("tensor_parallel_size must be an integer when provided")
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")
    gpu_memory_utilization = raw.get("gpu_memory_utilization", 0.88)
    if not isinstance(gpu_memory_utilization, (int, float)) or isinstance(gpu_memory_utilization, bool):
        raise ValueError("gpu_memory_utilization must be numeric")
    gpu_memory_utilization = float(gpu_memory_utilization)
    if gpu_memory_utilization <= 0.0 or gpu_memory_utilization > 1.0:
        raise ValueError("gpu_memory_utilization must be in the interval (0, 1]")
    max_model_len = raw.get("max_model_len", 4096)
    if not isinstance(max_model_len, int) or isinstance(max_model_len, bool):
        raise ValueError("max_model_len must be an integer")
    if max_model_len < 1:
        raise ValueError("max_model_len must be at least 1")

    belief_state_mode = raw.get("belief_state_mode", "uniform")
    if belief_state_mode not in {"uniform", "categorical"}:
        raise ValueError("belief_state_mode must be one of: uniform, categorical")
    belief_distribution_num_calls = raw.get("belief_distribution_num_calls", 1)
    if not isinstance(belief_distribution_num_calls, int) or isinstance(belief_distribution_num_calls, bool):
        raise ValueError("belief_distribution_num_calls must be an integer")
    if belief_distribution_num_calls < 1:
        raise ValueError("belief_distribution_num_calls must be at least 1")
    belief_distribution_permute_history = raw.get("belief_distribution_permute_history", False)
    if not isinstance(belief_distribution_permute_history, bool):
        raise ValueError("belief_distribution_permute_history must be a boolean")
    probability_parse_fallback_to_uniform = raw.get("probability_parse_fallback_to_uniform", True)
    if not isinstance(probability_parse_fallback_to_uniform, bool):
        raise ValueError("probability_parse_fallback_to_uniform must be a boolean")
    search_depth = raw.get("search_depth", 1)
    if not isinstance(search_depth, int) or isinstance(search_depth, bool):
        raise ValueError("search_depth must be an integer")
    if search_depth < 1:
        raise ValueError("search_depth must be at least 1")
    if game != "wordle" and search_depth not in {1, 2}:
        raise ValueError("search_depth must be one of: 1, 2")
    return Config(
        game = game,
        version = raw.get("version", 0),
        model_pairs = model_pairs,
        method_names = raw.get("method_names", raw.get("extraction_methods", [])),
        animals = raw.get("animals", []),
        wordle_solution_words_path = wordle_solution_words_path,
        wordle_valid_words_path = wordle_valid_words_path,
        max_wordle_guesses = max_wordle_guesses,
        tensor_parallel_size = tensor_parallel_size,
        gpu_memory_utilization = gpu_memory_utilization,
        max_model_len = max_model_len,
        batched_block_size = raw.get("batched_block_size", 50),
        generation_temperature_diverse = raw.get("generation_temperature_diverse", 1.3),
        generation_temperature_simple = raw.get("generation_temperature_simple", 1.0),
        answer_temperature = raw.get("answer_temperature", 0.7),
        search_depth = search_depth,
        target_num_questions = raw.get("target_num_questions", 15),
        num_mc_samples = raw.get("num_mc_samples", 15),
        max_num_samples = raw.get("max_num_samples", 50),
        min_num_samples = raw.get("min_num_samples", 15),
        threshold_rejection_probability = raw.get("threshold_rejection_probability", 0.2),
        belief_state_mode = belief_state_mode,
        belief_probability_temperature = raw.get("belief_probability_temperature", 0.0),
        belief_distribution_num_calls = belief_distribution_num_calls,
        belief_distribution_permute_history = belief_distribution_permute_history,
        probability_parse_fallback_to_uniform = probability_parse_fallback_to_uniform,
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
    parts = [spec.model.replace("/", "_")]
    if spec.reasoning_effort is not None:
        parts.append(f"reasoning-{spec.reasoning_effort}")
    if spec.thinking is not None:
        parts.append(f"thinking-{'on' if spec.thinking else 'off'}")
    if spec.use_logprobs:
        parts.append("logprobs-on")
    return "__".join(parts)


def build_output_stem(
    run_id: str,
    method_name: str,
    questioner: ModelSpec,
    answerer: ModelSpec,
    version: int,
    belief_state_mode: BeliefStateMode = "uniform",
    search_depth: int = 1,
    game: GameMode = "animals",
) -> str:
    return (
        f"{run_id}_{method_name}_Q:{_model_spec_stem(questioner)},"
        f"A:{_model_spec_stem(answerer)}_{belief_state_mode}_depth-{search_depth}_{version}_{game}"
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


def print_and_log(message: str, config: Config) -> None:
    print(message)
    if config.log_path is not None:
        write_to_log(f"{message}\n", config)


def _json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_ready(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def format_config_for_log(config: Config) -> str:
    return json.dumps(_json_ready(asdict(config)), indent=2, sort_keys=True)


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


def _extract_first_balanced_json_object(text: str) -> str | None:
    stripped = _strip_code_fences(text)
    start_idx: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for idx, char in enumerate(stripped):
        if start_idx is None:
            if char == "{":
                start_idx = idx
                depth = 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start_idx:idx + 1]

    return None


def _normalize_labeled_distribution_response(response_text: str, labels: list[str]) -> dict[str, float]:
    if not labels:
        return {}

    normalized_text = _strip_code_fences(response_text)
    try:
        payload = json.loads(normalized_text)
    except (json.JSONDecodeError, TypeError) as exc:
        balanced_payload = _extract_first_balanced_json_object(response_text)
        if balanced_payload is None:
            raise ValueError(f"Invalid probability JSON: {response_text!r}") from exc
        try:
            payload = json.loads(balanced_payload)
        except (json.JSONDecodeError, TypeError) as balanced_exc:
            raise ValueError(f"Invalid probability JSON: {response_text!r}") from balanced_exc

    if not isinstance(payload, dict):
        raise ValueError(f"Probability response must be a JSON object: {response_text!r}")

    scores: list[float] = []
    for label in labels:
        raw_value = payload.get(label, 0.0)
        try:
            score = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Probability for {label!r} must be numeric: {raw_value!r}") from exc
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"Probability for {label!r} must be finite and non-negative: {raw_value!r}")
        scores.append(score)

    total = sum(scores)
    if total <= 0.0:
        raise ValueError(f"Probability response must contain a positive total weight: {response_text!r}")

    return {
        label: score / total
        for label, score in zip(labels, scores)
    }


def _normalize_probability_response(response_text: str, responses: list[str]) -> dict[str, float]:
    return _normalize_labeled_distribution_response(response_text, responses)


def _uniform_probability_response(responses: list[str]) -> dict[str, float]:
    if not responses:
        return {}

    probability = 1.0 / len(responses)
    return {
        response: probability
        for response in responses
    }


def _probability_results_from_messages(batch_messages: list[list[dict[str, str]]], responses: list[str], block_size: int,
                                       temperature: float,
                                       complete_messages_batched: Callable[..., list[str]],
                                       fallback_to_uniform: bool = False) -> list[dict[str, float]]:
    probability_messages = [
        _build_probability_messages(messages, responses)
        for messages in batch_messages
    ]
    results: list[dict[str, float] | None] = [None] * len(probability_messages)
    pending_indices = list(range(len(probability_messages)))
    raw_completions: dict[int, str] = {}

    for _attempt in range(3):
        if not pending_indices:
            break

        completions = complete_messages_batched(
            [probability_messages[index] for index in pending_indices],
            temperature=temperature,
            block_size=block_size,
            max_new_tokens=64,
        )

        if len(completions) != len(pending_indices):
            raise ValueError(
                f"Expected {len(pending_indices)} probability completions, received {len(completions)}"
            )

        failed_indices: list[int] = []
        for index, completion in zip(pending_indices, completions):
            raw_completions[index] = completion
            try:
                results[index] = _normalize_probability_response(completion, responses)
            except ValueError:
                failed_indices.append(index)

        pending_indices = failed_indices

    if pending_indices:
        if fallback_to_uniform:
            for index in pending_indices:
                failed_completion = raw_completions.get(index, "")
                print(
                    f"Failed to parse probability JSON for index {index}, "
                    f"assigning uniform probabilities {failed_completion!r}"
                )
                results[index] = _uniform_probability_response(responses)
        else:
            failure_details = ", ".join(
                f"{index}: {raw_completions.get(index, '')!r}"
                for index in pending_indices
            )
            raise ValueError(
                "Failed to parse probability JSON after 3 attempts for "
                f"{len(pending_indices)} item(s): {failure_details}"
            )

    return [result for result in results if result is not None]


def _distribution_from_messages(messages: list[dict[str, str]], labels: list[str], temperature: float,
                                complete_message: Callable[..., list[str]],
                                num_calls: int = 1,
                                fallback_to_uniform: bool = False) -> dict[str, float]:
    distribution, _valid_count = _distribution_with_valid_count_from_messages(
        messages,
        labels,
        temperature,
        complete_message,
        num_calls=num_calls,
        fallback_to_uniform=fallback_to_uniform,
    )
    return distribution


def _distribution_with_valid_count_from_messages(messages: list[dict[str, str]], labels: list[str], temperature: float,
                                                 complete_message: Callable[..., list[str]],
                                                 num_calls: int = 1,
                                                 fallback_to_uniform: bool = False) -> tuple[dict[str, float], int]:
    if not labels:
        return {}, 0
    if num_calls < 1:
        raise ValueError("num_calls must be at least 1")

    completions = complete_message(messages=messages, temperature=temperature, num_responses=num_calls)
    if len(completions) != num_calls:
        raise ValueError(f"Expected {num_calls} distribution completions, received {len(completions)}")

    return _average_labeled_distributions_from_completions(
        completions,
        labels,
        fallback_to_uniform=fallback_to_uniform,
    )


def _distribution_with_valid_count_from_batched_messages(batch_messages: list[list[dict[str, str]]], labels: list[str],
                                                         temperature: float,
                                                         complete_messages_batched: Callable[..., list[str]],
                                                         block_size: int,
                                                         max_new_tokens: int = 8192,
                                                         fallback_to_uniform: bool = False) -> tuple[dict[str, float], int]:
    if not labels:
        return {}, 0
    if not batch_messages:
        raise ValueError("batch_messages must contain at least one prompt")

    completions = complete_messages_batched(
        batch_messages=batch_messages,
        temperature=temperature,
        block_size=block_size,
        max_new_tokens=max_new_tokens,
    )
    if len(completions) != len(batch_messages):
        raise ValueError(
            f"Expected {len(batch_messages)} batched distribution completions, received {len(completions)}"
        )

    return _average_labeled_distributions_from_completions(
        completions,
        labels,
        fallback_to_uniform=fallback_to_uniform,
    )


def _average_labeled_distributions_from_completions(completions: list[str], labels: list[str],
                                                    fallback_to_uniform: bool = False) -> tuple[dict[str, float], int]:
    completion_count = len(completions)
    valid_distributions: list[dict[str, float]] = []
    failed_completions: list[str] = []

    for completion in completions:
        try:
            valid_distributions.append(_normalize_labeled_distribution_response(completion, labels))
        except ValueError:
            failed_completions.append(completion)

    if valid_distributions:
        averaged_distribution = {
            label: sum(distribution[label] for distribution in valid_distributions) / len(valid_distributions)
            for label in labels
        }
        return averaged_distribution, len(valid_distributions)

    if fallback_to_uniform:
        print(
            f"Failed to parse belief distribution JSON for all {completion_count} completion(s), "
            f"assigning uniform distribution from completions {failed_completions!r}"
        )
        return _uniform_probability_response(labels), 0

    raise ValueError(
        f"Failed to parse belief distribution JSON for all {completion_count} completion(s): {failed_completions!r}"
    )

# prompts ask to generate collection of entities, one on each line --> convert the returned string to an array
def convert_string_to_array(response):
    return [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]


_BELIEF_MAX_LENGTH = 80
_BELIEF_EXPLANATION_PATTERN = re.compile(
    r"\b("
    r"because|however|therefore|based on|provided clues|previous answers|"
    r"contradiction|fit these geographic exclusions|logic of the previous answers|"
    r"there are no naturally occurring"
    r")\b",
    flags=re.IGNORECASE,
)
_BELIEF_REASONING_PAREN_PATTERN = re.compile(
    r"\((?:yes|no|incorrect|has|because|but|not)\b",
    flags=re.IGNORECASE,
)
_BELIEF_QUESTION_LIKE_PATTERN = re.compile(
    r"^(?:is|are|was|were|does|do|did|can|could|should|would|will)\b",
    flags=re.IGNORECASE,
)


def _is_plausible_animal_label(text: str) -> bool:
    has_alpha = False
    for char in text:
        if char.isalpha():
            has_alpha = True
            continue
        if char in {" ", "-", "'"}:
            continue
        return False
    return has_alpha


def normalize_belief_label(raw_belief: str) -> str | None:
    cleaned_belief = re.sub(r"\s+", " ", raw_belief.strip())
    if not cleaned_belief:
        return None

    cleaned_belief = re.sub(r"\s*\([^)]*\)", "", cleaned_belief)
    cleaned_belief = re.sub(r"\s+", " ", cleaned_belief).strip()
    cleaned_belief = cleaned_belief.strip(" -:;,.")
    if not cleaned_belief:
        return None

    if len(cleaned_belief) > _BELIEF_MAX_LENGTH:
        return None
    if "->" in cleaned_belief or "?" in cleaned_belief:
        return None
    if _BELIEF_QUESTION_LIKE_PATTERN.match(cleaned_belief):
        return None
    if _BELIEF_EXPLANATION_PATTERN.search(cleaned_belief):
        return None
    if _BELIEF_REASONING_PAREN_PATTERN.search(cleaned_belief):
        return None
    if cleaned_belief.count(",") >= 3:
        return None
    if not _is_plausible_animal_label(cleaned_belief):
        return None

    return cleaned_belief


def clean_generated_belief_labels(raw_beliefs: list[str]) -> list[str]:
    cleaned_beliefs: list[str] = []
    for raw_belief in raw_beliefs:
        cleaned_belief = normalize_belief_label(raw_belief)
        if cleaned_belief is not None:
            cleaned_beliefs.append(cleaned_belief)
    return cleaned_beliefs


def _binary_entropy(p_yes: float, p_no: float) -> float:
    p_yes_clipped = max(p_yes, 1e-12)
    p_no_clipped = max(p_no, 1e-12)
    return - (p_yes_clipped * np.log(p_yes_clipped) + p_no_clipped * np.log(p_no_clipped))


def make_belief_state(beliefs: list[str], probabilities: list[float] | None = None,
                      fallback_to_uniform: bool = False) -> BeliefState:
    if probabilities is None:
        probabilities = [1.0] * len(beliefs)

    if len(beliefs) != len(probabilities):
        raise ValueError("beliefs and probabilities must have the same length")

    merged_beliefs: list[str] = []
    merged_probabilities: list[float] = []
    belief_indices: dict[str, int] = {}

    for belief, probability in zip(beliefs, probabilities):
        cleaned_belief = belief.strip()
        if not cleaned_belief:
            continue

        try:
            numeric_probability = float(probability)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid belief probability: {probability!r}") from exc
        if not math.isfinite(numeric_probability) or numeric_probability < 0.0:
            raise ValueError(f"Belief probabilities must be finite and non-negative: {probability!r}")

        belief_key = cleaned_belief.lower()
        existing_index = belief_indices.get(belief_key)
        if existing_index is None:
            belief_indices[belief_key] = len(merged_beliefs)
            merged_beliefs.append(cleaned_belief)
            merged_probabilities.append(numeric_probability)
        else:
            merged_probabilities[existing_index] += numeric_probability

    if not merged_beliefs:
        return BeliefState([], [])

    total_probability = sum(merged_probabilities)
    if total_probability <= 0.0:
        if not fallback_to_uniform:
            raise ValueError("Belief probabilities must sum to a positive value")
        uniform_probability = 1.0 / len(merged_beliefs)
        return BeliefState(merged_beliefs, [uniform_probability] * len(merged_beliefs))

    return BeliefState(
        merged_beliefs,
        [probability / total_probability for probability in merged_probabilities],
    )


def ensure_belief_state(beliefs: BeliefState | list[str]) -> BeliefState:
    if isinstance(beliefs, BeliefState):
        return beliefs
    return make_uniform_belief_state(beliefs)


def make_uniform_belief_state(beliefs: list[str]) -> BeliefState:
    deduped_beliefs = make_belief_state(beliefs, fallback_to_uniform=True).beliefs
    if not deduped_beliefs:
        return BeliefState([], [])

    uniform_probability = 1.0 / len(deduped_beliefs)
    return BeliefState(deduped_beliefs, [uniform_probability] * len(deduped_beliefs))


def sort_belief_state_descending(belief_state: BeliefState) -> BeliefState:
    ordered_entries = sorted(
        zip(belief_state.beliefs, belief_state.probabilities),
        key=lambda entry: entry[1],
        reverse=True,
    )
    return BeliefState(
        [belief for belief, _probability in ordered_entries],
        [probability for _belief, probability in ordered_entries],
    )


def format_belief_state(belief_state: BeliefState, top_n: int | None = None) -> str:
    if len(belief_state.beliefs) == 0:
        return "[]"

    entries = list(zip(belief_state.beliefs, belief_state.probabilities))
    if top_n is not None:
        entries = sorted(entries, key=lambda entry: entry[1], reverse=True)[:top_n]

    formatted_entries = [
        f"{belief} ({probability:.3f})"
        for belief, probability in entries
    ]
    return "[" + ", ".join(formatted_entries) + "]"


def format_categorical_belief_summary(belief_state: BeliefState, top_n: int | None = None) -> str:
    if top_n is None:
        return f"{len(belief_state.beliefs)} belief(s): {format_belief_state(belief_state)}"

    top_count = min(top_n, len(belief_state.beliefs))
    return f"{len(belief_state.beliefs)} belief(s): {format_belief_state(belief_state, top_n=top_count)}"


def is_uniform_belief_state(belief_state: BeliefState, tolerance: float = 1e-9) -> bool:
    if len(belief_state.beliefs) <= 1:
        return True

    uniform_probability = 1.0 / len(belief_state.beliefs)
    return all(
        math.isclose(probability, uniform_probability, rel_tol=tolerance, abs_tol=tolerance)
        for probability in belief_state.probabilities
    )


# reverses a messages array so that the final question comes first
def reverse_history(history_questioner: list[dict[str,str]]) -> list[dict[str,str]]:
    blocks = split_history_into_qa_blocks(history_questioner)
    return [x for b in blocks[::-1] for x in b]


def split_history_into_qa_blocks(history_questioner: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    return [history_questioner[i:i + 2] for i in range(0, len(history_questioner), 2)]


def sample_permuted_history_messages(history_questioner: list[dict[str, str]], num_samples: int) -> list[list[dict[str, str]]]:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    blocks = split_history_into_qa_blocks(history_questioner)
    if len(blocks) <= 1:
        return [[dict(message) for message in history_questioner] for _ in range(num_samples)]

    permuted_histories: list[list[dict[str, str]]] = []
    for _ in range(num_samples):
        permutation = np.random.permutation(len(blocks))
        permuted_histories.append([
            dict(message)
            for block_index in permutation
            for message in blocks[block_index]
        ])

    return permuted_histories


def get_question_answered(question: str, goal_object: str, answerer: Model, answer_temperature: float) -> str:
    user_question = {"role": "user", "content": f"{question}"}
    messages = [answer_question_yesnocorrect_system_prompt(entity=goal_object), user_question]
    return answerer.chat_complete(messages=messages, temperature=answer_temperature)[0]


def is_guess_correct_via_answerer(guess: str, goal_object: str, answerer: Model, answer_temperature: float) -> bool:
    return get_question_answered(
        f"Is it {guess}?",
        goal_object,
        answerer,
        answer_temperature,
    ) == "Correct!"


def generate_original_beliefs(questioner: Model, config: Config) -> list[str]:
    generation_temperature, max_num_samples, min_num_samples = config.generation_temperature_diverse, config.max_num_samples, config.min_num_samples
    user_question = {"role": "user", "content": f"Let\'s start the game of 20 questions. Generate a diverse "
                                                f"set of animals, at least {min_num_samples}."}
    messages = [generate_original_animals_system_prompt(max_num_samples), user_question]
    new_beliefs = questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
    raw_beliefs = convert_string_to_array(new_beliefs)
    cleaned_beliefs = clean_generated_belief_labels(raw_beliefs)
    print(f"[beliefs] Generated {len(raw_beliefs)} raw opening belief(s)")
    print(f"[beliefs] {len(cleaned_beliefs)} opening belief(s) remain after structural cleanup")
    return cleaned_beliefs
