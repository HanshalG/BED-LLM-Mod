from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
import time

import torch
import wandb
from helpers import ModelSpec, _probability_results_from_messages
from openai_harmony import (
    Conversation as HarmonyConversation,
    HarmonyEncodingName,
    Message as HarmonyMessage,
    Role as HarmonyRole,
    load_harmony_encoding,
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class Model(ABC):
    @abstractmethod
    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int) -> list[dict[str, float]]:
        raise NotImplementedError


class BaseVLLMAdapter(Model):
    def __init__(self, spec: ModelSpec, tensor_parallel_size: int | None = None, dtype: str = "bfloat16"):
        self.spec = spec
        self.model_name = spec.model
        self.tokenizer = self._build_tokenizer()

        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()

        self.llm = LLM(
            model=self.model_name,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
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
            max_tokens=256,
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

    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int) -> list[dict[str, float]]:
        start_time = time.perf_counter()

        results = _probability_results_from_messages(
            messages,
            responses,
            block_size,
            self._chat_complete_messages_batched,
        )

        elapsed_time = time.perf_counter() - start_time
        wandb.log({
            "event": "Batched probability determination",
            "number_conversations": len(messages) * len(responses),
            "elapsed_time_batched": elapsed_time,
        })

        return results


class QwenVLLMAdapter(BaseVLLMAdapter):
    def __init__(self, spec: ModelSpec, tensor_parallel_size: int | None = None, dtype: str = "bfloat16"):
        super().__init__(spec=spec, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
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
        return final_text.strip()


class GemmaVLLMAdapter(BaseVLLMAdapter):
    def __init__(self, spec: ModelSpec, tensor_parallel_size: int | None = None, dtype: str = "bfloat16"):
        super().__init__(spec=spec, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
        self.thinking = bool(spec.thinking)

    def _tokenizer_kwargs(self) -> dict[str, object]:
        return {
            "trust_remote_code": True,
            "limit_mm_per_prompt": {"image": 0, "audio": 0},
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

        parsed = parse_response(token_ids)
        if isinstance(parsed, dict):
            content = parsed.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

        raise ValueError(f"{self.model_name} returned reasoning output without a final answer")


class HarmonyVLLMAdapter(BaseVLLMAdapter):
    _HARMONY_KNOWLEDGE_CUTOFF = "2024-06"

    def __init__(self, spec: ModelSpec, tensor_parallel_size: int | None = None):
        self._harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.reasoning_effort = spec.reasoning_effort or "low"
        super().__init__(spec=spec, tensor_parallel_size=tensor_parallel_size, dtype="bfloat16")
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


def build_model_adapter(spec: ModelSpec, tensor_parallel_size: int | None = None, dtype: str = "bfloat16") -> Model:
    if spec.model.startswith("openai/gpt-oss"):
        return HarmonyVLLMAdapter(spec=spec, tensor_parallel_size=tensor_parallel_size)
    if spec.model.startswith("google/gemma-4"):
        return GemmaVLLMAdapter(spec=spec, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
    if spec.model.startswith("Qwen/Qwen3.5"):
        return QwenVLLMAdapter(spec=spec, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
    return BaseVLLMAdapter(spec=spec, tensor_parallel_size=tensor_parallel_size, dtype=dtype)


VLLMAdapter = BaseVLLMAdapter
