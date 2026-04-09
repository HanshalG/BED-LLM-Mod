from abc import ABC, abstractmethod
from datetime import date
import time

import torch
import wandb
from helpers import _probability_results_from_messages
from openai_harmony import (
    Conversation as HarmonyConversation,
    HarmonyEncodingName,
    Message as HarmonyMessage,
    Role as HarmonyRole,
    load_harmony_encoding,
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# Abstract interface for all LLM provider adapters
class Model(ABC):
    # Generate n completions (strings) for the provided messages at the given temperature
    @abstractmethod
    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1) -> list[str]:
        raise NotImplementedError


    # for each conversation, get the probability of each of the responses occurring (relative to each other)
    @abstractmethod
    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int) -> list[dict[str, float]]:
        raise NotImplementedError


class VLLMAdapter(Model):
    _HARMONY_KNOWLEDGE_CUTOFF = "2024-06"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B", tensor_parallel_size: int | None = None,
                 dtype: str = "float16"):
        self.model_name = model_name
        self._uses_harmony = "openai/gpt-oss" in model_name
        self.tokenizer = None
        self._harmony_encoding = None

        if self._uses_harmony:
            dtype = "bfloat16"
            self._harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

        if "Qwen" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 0, "audio": 0}
            )
        elif "google" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 0, "audio": 0}
            )
        elif not self._uses_harmony:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()

        self.llm = LLM(
            model=model_name,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
        )

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str | dict[str, list[int]]:
        if self._uses_harmony:
            return {"prompt_token_ids": self._messages_to_harmony_prompt_token_ids(messages)}

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


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
            "Reasoning: low\n\n"
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
        kwargs = {
            "temperature": temperature,
            "top_k": 50,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "n": n,
        }
        if self._uses_harmony:
            kwargs["stop_token_ids"] = self._harmony_encoding.stop_tokens_for_assistant_actions()
            kwargs["skip_special_tokens"] = False
        return SamplingParams(**kwargs)


    def _normalize_completion_output(self, output) -> str:
        if not self._uses_harmony:
            return output.text.lstrip()

        token_ids = getattr(output, "token_ids", None)
        if token_ids:
            parsed_messages = self._harmony_encoding.parse_messages_from_completion_tokens(
                token_ids,
                HarmonyRole.ASSISTANT,
                strict=False,
            )
            for preferred_channel in ("final", "commentary", "analysis"):
                channel_texts = [
                    self._extract_harmony_message_text(message)
                    for message in parsed_messages
                    if (message.channel or "").startswith(preferred_channel)
                    and self._extract_harmony_message_text(message)
                ]
                if channel_texts:
                    return channel_texts[-1]

            parsed_texts = [
                self._extract_harmony_message_text(message)
                for message in parsed_messages
                if self._extract_harmony_message_text(message)
            ]
            if parsed_texts:
                return parsed_texts[-1]

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
            max_tokens=4096,
            n=num_responses,
        )

        outputs = self.llm.generate([prompt], sampling_params)

        completions = [
            self._normalize_completion_output(o)
            for o in outputs[0].outputs
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
