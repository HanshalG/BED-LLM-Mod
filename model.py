from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import date
import math
import os
import time

import torch
import wandb
from helpers import ModelSpec, _probability_results_from_messages, write_to_log, Config
from openai_harmony import (
    Conversation as HarmonyConversation,
    HarmonyEncodingName,
    Message as HarmonyMessage,
    Role as HarmonyRole,
    load_harmony_encoding,
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


@contextmanager
def _temporary_cuda_visible_devices(cuda_visible_devices: str | None):
    if cuda_visible_devices is None:
        yield
        return

    previous_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_value


def _count_cuda_visible_devices(cuda_visible_devices: str) -> int:
    devices = [
        device.strip()
        for device in cuda_visible_devices.split(",")
        if device.strip()
    ]
    if not devices:
        raise ValueError("cuda_visible_devices must list at least one device")
    return len(devices)


class Model(ABC):
    @abstractmethod
    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def chat_complete_messages_batched(self, batch_messages: list[list[dict[str, str]]], temperature: float,
                                       block_size: int, max_new_tokens: int = 8192) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int) -> list[dict[str, float]]:
        raise NotImplementedError


class BaseVLLMAdapter(Model):
    def __init__(self, spec: ModelSpec, config: Config, tensor_parallel_size: int | None = None, dtype: str = "bfloat16"):
        self.spec = spec
        self.config = config
        self.model_name = spec.model
        self.use_logprobs = spec.use_logprobs
        self.tokenizer = self._build_tokenizer()

        if tensor_parallel_size is None:
            tensor_parallel_size = spec.tensor_parallel_size
        if tensor_parallel_size is None:
            tensor_parallel_size = config.tensor_parallel_size
        if tensor_parallel_size is None and spec.cuda_visible_devices is not None:
            tensor_parallel_size = _count_cuda_visible_devices(spec.cuda_visible_devices)
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()

        max_model_len = spec.max_model_len or config.max_model_len
        gpu_memory_utilization = spec.gpu_memory_utilization or config.gpu_memory_utilization

        with _temporary_cuda_visible_devices(spec.cuda_visible_devices):
            self.llm = LLM(
                model=self.model_name,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
            )

    def _tokenizer_kwargs(self) -> dict[str, object]:
        return {}

    def _build_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name, **self._tokenizer_kwargs())

    def _chat_template_kwargs(self) -> dict[str, object]:
        return {}

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self._chat_template_kwargs(),
        )

    def _build_sampling_params(self, temperature: float, max_tokens: int, n: int) -> SamplingParams:
        return SamplingParams(
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            max_tokens=max_tokens,
            n=n,
        )

    def _normalize_completion_output(self, output) -> str:
        return output.text.lstrip()

    def _chat_probabilities_messages_batched_via_logprobs(
        self,
        messages: list[list[dict[str, str]]],
        responses: list[str],
        temperature: float,
        block_size: int,
    ) -> list[dict[str, float]]:
        prompts = [self._messages_to_prompt(prompt_messages) for prompt_messages in messages]
        tokenized_prompts = [
            self.tokenizer(prompt, add_special_tokens=False).input_ids
            for prompt in prompts
        ]
        base_prompt_lengths = [len(token_ids) for token_ids in tokenized_prompts]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
        )

        results: list[dict[str, float]] = []

        for block_start in range(0, len(prompts), block_size):
            block_prompts = prompts[block_start:block_start + block_size]
            block_base_lengths = base_prompt_lengths[block_start:block_start + block_size]
            block_response_log_scores: list[list[float]] = []

            for response in responses:
                full_prompts = [prompt + response for prompt in block_prompts]
                outputs = self.llm.generate(full_prompts, sampling_params=sampling_params)

                block_scores: list[float] = []
                for output, base_length in zip(outputs, block_base_lengths):
                    prompt_logprobs = output.prompt_logprobs
                    score = 0.0
                    for position in range(base_length, len(prompt_logprobs)):
                        token_logprobs = prompt_logprobs[position]
                        score += next(iter(token_logprobs.values())).logprob

                    block_scores.append(score)

                block_response_log_scores.append(block_scores)

            for convo_idx in range(len(block_prompts)):
                log_scores = [
                    block_response_log_scores[response_idx][convo_idx]
                    for response_idx in range(len(responses))
                ]
                scaled_scores = [score / temperature for score in log_scores]
                max_score = max(scaled_scores)
                exp_scores = [math.exp(score - max_score) for score in scaled_scores]
                normalization = sum(exp_scores)
                probabilities = [score / normalization for score in exp_scores]
                results.append(dict(zip(responses, probabilities)))

        return results

    def _chat_complete_messages_batched(self, batch_messages: list[list[dict[str, str]]], temperature: float,
                                        block_size: int, max_new_tokens: int) -> list[str]:
        prompts = [self._messages_to_prompt(messages) for messages in batch_messages]
        completions: list[str] = []

        for start_idx in range(0, len(prompts), block_size):
            block_prompts = prompts[start_idx:start_idx + block_size]
            sampling_params = self._build_sampling_params(
                temperature=temperature,
                max_tokens=max_new_tokens,
                n=1,
            )
            outputs = self.llm.generate(block_prompts, sampling_params)
            completions.extend(
                self._normalize_completion_output(output.outputs[0]) if output.outputs else ""
                for output in outputs
            )

        return completions

    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1) -> list[str]:
        start_time = time.perf_counter()
        prompt = self._messages_to_prompt(messages)
        sampling_params = self._build_sampling_params(
            temperature=temperature,
            max_tokens=8192,
            n=num_responses,
        )
        outputs = self.llm.generate([prompt], sampling_params)
        completions = [
            self._normalize_completion_output(output)
            for output in outputs[0].outputs
        ]

        elapsed_time = time.perf_counter() - start_time
        wandb.log({
            "event": "Chat completion",
            "number_input_tokens": len(outputs[0].prompt_token_ids),
            "elapsed_time": elapsed_time,
        })

        return completions

    def chat_complete_messages_batched(self, batch_messages: list[list[dict[str, str]]], temperature: float,
                                       block_size: int, max_new_tokens: int = 8192) -> list[str]:
        start_time = time.perf_counter()
        completions = self._chat_complete_messages_batched(
            batch_messages=batch_messages,
            temperature=temperature,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
        )

        elapsed_time = time.perf_counter() - start_time
        wandb.log({
            "event": "Batched chat completion",
            "number_conversations": len(batch_messages),
            "elapsed_time_batched": elapsed_time,
        })

        return completions

    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int) -> list[dict[str, float]]:
        start_time = time.perf_counter()

        if self.use_logprobs:
            results = self._chat_probabilities_messages_batched_via_logprobs(
                messages,
                responses,
                temperature,
                block_size,
            )
        else:
            results = _probability_results_from_messages(
                messages,
                responses,
                block_size,
                temperature,
                self._chat_complete_messages_batched,
                fallback_to_uniform=self.config.probability_parse_fallback_to_uniform,
            )

        elapsed_time = time.perf_counter() - start_time
        wandb.log({
            "event": "Batched probability determination",
            "number_conversations": len(messages) * len(responses),
            "elapsed_time_batched": elapsed_time,
        })

        return results


class QwenVLLMAdapter(BaseVLLMAdapter):
    def __init__(self, spec: ModelSpec, config: Config, tensor_parallel_size: int | None = None, dtype: str = "bfloat16"):
        super().__init__(spec=spec, config=config, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
        self.thinking = bool(spec.thinking)

    def _chat_template_kwargs(self) -> dict[str, object]:
        return {"enable_thinking": self.thinking}

    def _normalize_completion_output(self, output) -> str:
        text = output.text.lstrip()
        if not self.thinking:
            return text
        thought, separator, final_text = text.rpartition("</think>")
        if not separator or not final_text.strip():
            raise ValueError(f"{self.model_name} returned reasoning output without a final answer")
        #write_to_log(f"Reasoning trace: {thought}\n", self.config)
        return final_text.strip()


class GemmaVLLMAdapter(BaseVLLMAdapter):
    def __init__(self, spec: ModelSpec, config: Config, tensor_parallel_size: int | None = None, dtype: str = "bfloat16"):
        super().__init__(spec=spec, config=config, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
        self.thinking = bool(spec.thinking)

    def _tokenizer_kwargs(self) -> dict[str, object]:
        return {
            "trust_remote_code": True,
        }

    def _chat_template_kwargs(self) -> dict[str, object]:
        return {"enable_thinking": self.thinking}

    def _normalize_completion_output(self, output) -> str:
        parse_response = getattr(self.tokenizer, "parse_response", None)
        if parse_response is None:
            raise ValueError(f"{self.model_name} tokenizer does not expose parse_response()")

        token_ids = getattr(output, "token_ids", None)
        if not token_ids:
            raise ValueError(f"{self.model_name} completion is missing token_ids required for parse_response()")

        #write reasoning trace to log
        #write_to_log(f"Reasoning trace: {self.tokenizer.decode(token_ids)}\n", self.config)
        
        parsed = parse_response(token_ids)
        if isinstance(parsed, dict):
            content = parsed.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

        raise ValueError(f"{self.model_name} returned reasoning output without a final answer")


class HarmonyVLLMAdapter(BaseVLLMAdapter):
    _HARMONY_KNOWLEDGE_CUTOFF = "2024-06"

    def __init__(self, spec: ModelSpec, config: Config, tensor_parallel_size: int | None = None):
        self._harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.reasoning_effort = spec.reasoning_effort or "low"
        super().__init__(spec=spec, config=config, tensor_parallel_size=tensor_parallel_size, dtype="bfloat16")
        self.tokenizer = None

    def _build_tokenizer(self):
        return None

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> dict[str, list[int]]:
        return {"prompt_token_ids": self._messages_to_harmony_prompt_token_ids(messages)}

    def _messages_to_harmony_prompt_token_ids(self, messages: list[dict[str, str]]) -> list[int]:
        harmony_messages = [
            HarmonyMessage.from_role_and_content(HarmonyRole.SYSTEM, self._harmony_system_message())
        ]
        harmony_messages.extend(
            self._local_message_to_harmony_message(message)
            for message in messages
        )
        conversation = HarmonyConversation.from_messages(harmony_messages)
        return self._harmony_encoding.render_conversation_for_completion(conversation, HarmonyRole.ASSISTANT)

    def _harmony_system_message(self) -> str:
        return (
            "You are ChatGPT, a large language model trained by OpenAI.\n"
            f"Knowledge cutoff: {self._HARMONY_KNOWLEDGE_CUTOFF}\n"
            f"Current date: {date.today().isoformat()}\n\n"
            f"Reasoning: {self.reasoning_effort}\n\n"
            "# Valid channels: analysis, commentary, final. Channel must be included for every message."
        )

    @staticmethod
    def _local_message_to_harmony_message(message: dict[str, str]):
        role = message["role"]
        content = message["content"]

        if role == "system":
            return HarmonyMessage.from_role_and_content(
                HarmonyRole.DEVELOPER,
                f"# Instructions\n\n{content}",
            )
        if role == "assistant":
            return HarmonyMessage.from_role_and_content(
                HarmonyRole.ASSISTANT,
                content,
            ).with_channel("final")
        if role == "developer":
            return HarmonyMessage.from_role_and_content(HarmonyRole.DEVELOPER, content)

        return HarmonyMessage.from_role_and_content(HarmonyRole(role), content)

    @staticmethod
    def _extract_harmony_message_text(message) -> str:
        return "".join(
            content.text
            for content in message.content
            if getattr(content, "text", "")
        ).strip()

    def _build_sampling_params(self, temperature: float, max_tokens: int, n: int) -> SamplingParams:
        return SamplingParams(
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            max_tokens=max_tokens,
            n=n,
            stop_token_ids=self._harmony_encoding.stop_tokens_for_assistant_actions(),
            skip_special_tokens=False,
        )

    def _normalize_completion_output(self, output) -> str:
        token_ids = getattr(output, "token_ids", None)
        if not token_ids:
            raise ValueError(f"{self.model_name} completion is missing token_ids required for harmony parsing")

        parsed_messages = self._harmony_encoding.parse_messages_from_completion_tokens(
            token_ids,
            HarmonyRole.ASSISTANT,
        )
        final_messages = [
            self._extract_harmony_message_text(message)
            for message in parsed_messages
            if (message.channel or "").startswith("final")
            and self._extract_harmony_message_text(message)
        ]
        if not final_messages:
            raise ValueError(f"{self.model_name} completion did not include a final harmony message")
        return final_messages[-1]


def build_model_adapter(spec: ModelSpec, config: Config, tensor_parallel_size: int | None = None, dtype: str = "bfloat16") -> Model:
    if spec.model.startswith("openai/gpt-oss"):
        return HarmonyVLLMAdapter(spec=spec, config=config, tensor_parallel_size=tensor_parallel_size)
    if spec.model.startswith("google/gemma-4"):
        return GemmaVLLMAdapter(spec=spec, config=config, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
    if spec.model.startswith("Qwen/Qwen3.5"):
        return QwenVLLMAdapter(spec=spec, config=config, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
    return BaseVLLMAdapter(spec=spec, config=config, tensor_parallel_size=tensor_parallel_size, dtype=dtype)


VLLMAdapter = BaseVLLMAdapter
