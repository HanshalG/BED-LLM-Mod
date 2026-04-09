from abc import ABC, abstractmethod
import time

import torch
import wandb
from helpers import _probability_results_from_messages
from transformers import AutoTokenizer, AutoModelForCausalLM
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
    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B", tensor_parallel_size: int | None = None,
                 dtype: str = "float16"):
        self.model_name = model_name

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
        elif "openai" in model_name:
            dtype="bfloat16"
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name
        )

        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()

        self.llm = LLM(
            model=model_name,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
        )


    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


    def _chat_complete_messages_batched(self, batch_messages: list[list[dict[str, str]]], temperature: float,
                                        block_size: int, max_new_tokens: int) -> list[str]:
        prompts = [self._messages_to_prompt(messages) for messages in batch_messages]
        completions: list[str] = []

        for start_idx in range(0, len(prompts), block_size):
            block_prompts = prompts[start_idx:start_idx + block_size]
            sampling_params = SamplingParams(
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                max_tokens=max_new_tokens,
                n=1,
            )
            outputs = self.llm.generate(block_prompts, sampling_params)
            completions.extend(
                output.outputs[0].text.lstrip() if output.outputs else ""
                for output in outputs
            )

        return completions


    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1) -> list[str]:
        start_time = time.perf_counter()

        prompt = self._messages_to_prompt(messages)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            max_tokens=256,
            n=num_responses,
        )

        outputs = self.llm.generate([prompt], sampling_params)

        completions = [
            o.text.lstrip()
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
