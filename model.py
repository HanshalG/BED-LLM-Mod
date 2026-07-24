from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import date
import json
import math
import os
import threading
import time
from typing import Any

import torch
try:
    import wandb
except ModuleNotFoundError:
    class _DisabledWandb:
        run = None

        @staticmethod
        def log(_payload: dict[str, Any]) -> None:
            return None

    wandb = _DisabledWandb()
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


def _wandb_log(payload: dict[str, object]) -> None:
    if getattr(wandb, "run", None) is not None:
        wandb.log(payload)


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
                                       block_size: int, max_new_tokens: int | None = None) -> list[str]:
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
        self._initialize_usage_counters()

        if tensor_parallel_size is None:
            tensor_parallel_size = spec.tensor_parallel_size
        if tensor_parallel_size is None:
            tensor_parallel_size = config.tensor_parallel_size
        if tensor_parallel_size is None and spec.cuda_visible_devices is not None:
            tensor_parallel_size = _count_cuda_visible_devices(spec.cuda_visible_devices)
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()

        max_model_len = spec.max_model_len or config.max_model_len
        self.max_model_len = max_model_len
        gpu_memory_utilization = spec.gpu_memory_utilization or config.gpu_memory_utilization

        extra_kwargs_raw = os.environ.get("BED_LLM_VLLM_KWARGS")
        extra_kwargs = json.loads(extra_kwargs_raw) if extra_kwargs_raw else {}
        if not isinstance(extra_kwargs, dict):
            raise ValueError("BED_LLM_VLLM_KWARGS must decode to a JSON object")

        with _temporary_cuda_visible_devices(spec.cuda_visible_devices):
            self.llm = LLM(
                model=self.model_name,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                **extra_kwargs,
            )

    def _initialize_usage_counters(self) -> None:
        self._usage_lock = threading.Lock()
        self._usage_requests = 0
        self._usage_prompt_tokens = 0
        self._usage_completion_tokens = 0

    def _ensure_usage_counters(self) -> None:
        if not hasattr(self, "_usage_lock"):
            self._initialize_usage_counters()

    def _record_request_outputs(self, request_outputs) -> None:
        self._ensure_usage_counters()
        requests = list(request_outputs)
        prompt_tokens = sum(
            len(getattr(request, "prompt_token_ids", None) or []) for request in requests
        )
        completion_tokens = sum(
            len(getattr(output, "token_ids", None) or [])
            for request in requests
            for output in (getattr(request, "outputs", None) or [])
        )
        with self._usage_lock:
            self._usage_requests += len(requests)
            self._usage_prompt_tokens += prompt_tokens
            self._usage_completion_tokens += completion_tokens

    def usage_snapshot(self) -> dict[str, object]:
        self._ensure_usage_counters()
        with self._usage_lock:
            requests = self._usage_requests
            prompt_tokens = self._usage_prompt_tokens
            completion_tokens = self._usage_completion_tokens
        model_usage = {
            "requests": requests,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
        }
        return {
            "backend": "vllm",
            "model": self.model_name,
            "run_cost_usd": 0.0,
            **model_usage,
            "model_usage": {self.model_name: model_usage},
            "forced_exits": 0,
        }

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

    def _prompt_token_count(self, prompt) -> int:
        prompt_token_ids = prompt.get("prompt_token_ids") if isinstance(prompt, dict) else None
        if prompt_token_ids is not None:
            return len(prompt_token_ids)
        return len(self.tokenizer(prompt, add_special_tokens=False).input_ids)

    def _max_new_tokens_for_prompts(self, prompts: list, requested_max_tokens: int) -> int:
        if not prompts:
            return requested_max_tokens
        remaining_context = [
            self.max_model_len - self._prompt_token_count(prompt)
            for prompt in prompts
        ]
        max_new_tokens = min(requested_max_tokens, min(remaining_context))
        if max_new_tokens < 1:
            longest_prompt = max(self._prompt_token_count(prompt) for prompt in prompts)
            raise ValueError(
                f"Prompt length {longest_prompt} leaves no room for generation "
                f"within max_model_len={self.max_model_len}"
            )
        return max_new_tokens

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

    def _completion_raw_text(self, output) -> str:
        return (getattr(output, "text", "") or "").lstrip()

    def _completion_has_final_answer(self, output) -> bool:
        return True

    def _forced_thinking_exit_enabled(self) -> bool:
        return False

    def _forced_thinking_closer(self) -> str:
        return ""

    def _forced_final_prompt(self, prompt, output) -> str:
        raw_completion = self._completion_raw_text(output)
        closer = self._forced_thinking_closer()
        stripped_raw = raw_completion.rstrip()
        stripped_closer = closer.lstrip()
        for boundary in ("</think>", "<channel|>"):
            if stripped_raw.endswith(boundary) and stripped_closer.startswith(boundary):
                closer = stripped_closer[len(boundary):]
                break
        return f"{prompt}{raw_completion}{closer}"

    def _strip_forced_final_output(self, text: str) -> str:
        stripped = text.strip()
        prefix = "Final Answer:"
        for boundary in ("</think>", "<channel|>"):
            if boundary in stripped:
                stripped = stripped.rpartition(boundary)[2].strip()
        if prefix in stripped:
            stripped = stripped.rpartition(prefix)[2].strip()
        return stripped

    def _normalize_forced_final_output(self, output) -> str:
        return self._strip_forced_final_output(self._completion_raw_text(output))

    def _completion_hit_generation_budget(self, output, max_tokens: int) -> bool:
        finish_reason = getattr(output, "finish_reason", None)
        if isinstance(finish_reason, str) and finish_reason.lower() == "length":
            return True

        token_ids = getattr(output, "token_ids", None)
        return token_ids is not None and len(token_ids) >= max_tokens

    def _completion_token_count(self, output) -> int | None:
        token_ids = getattr(output, "token_ids", None)
        if token_ids is None:
            return None
        return len(token_ids)

    def _completion_finish_reason(self, output) -> str:
        finish_reason = getattr(output, "finish_reason", None)
        return finish_reason if isinstance(finish_reason, str) else ""

    @staticmethod
    def _request_prompt_token_count(request_output) -> int | None:
        prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
        if prompt_token_ids is None:
            return None
        return len(prompt_token_ids)

    def _log_llm_token_usage(
        self,
        *,
        call_type: str,
        prompt,
        output=None,
        request_output=None,
        temperature: float,
        max_new_tokens: int,
        block_index: int | None = None,
        response_index: int | None = None,
    ) -> None:
        if self.config.log_path is None:
            return

        prompt_tokens = (
            self._request_prompt_token_count(request_output)
            if request_output is not None
            else None
        )
        if prompt_tokens is None:
            prompt_tokens = self._prompt_token_count(prompt)
        completion_tokens = (
            self._completion_token_count(output)
            if output is not None
            else None
        )
        payload = {
            "event": "llm_token_usage",
            "model": self.model_name,
            "call_type": call_type,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens + completion_tokens
                if completion_tokens is not None
                else None
            ),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if block_index is not None:
            payload["block_index"] = block_index
        if response_index is not None:
            payload["response_index"] = response_index
        finish_reason = self._completion_finish_reason(output) if output is not None else ""
        if finish_reason:
            payload["finish_reason"] = finish_reason

        try:
            write_to_log(json.dumps(payload, sort_keys=True) + "\n", self.config)
        except Exception:
            return

    def _forced_final_max_new_tokens(self) -> int:
        return 0

    def _forced_final_reserve_tokens(self) -> int:
        closer = self._forced_thinking_closer()
        if not closer:
            return self._forced_final_max_new_tokens()
        return self._forced_final_max_new_tokens() + len(
            self.tokenizer(closer, add_special_tokens=False).input_ids
        )

    def _first_pass_max_new_tokens_for_prompts(self, prompts: list, requested_max_tokens: int) -> int:
        return self._max_new_tokens_for_prompts(prompts, requested_max_tokens)

    def _maybe_force_final_answers(
        self,
        prompts: list,
        outputs: list,
        initial_max_tokens: int,
        temperature: float,
    ) -> list[str]:
        completions = [
            self._normalize_completion_output(output) if output is not None else ""
            for output in outputs
        ]
        if not self._forced_thinking_exit_enabled():
            return completions

        forced_indices = [
            idx
            for idx, output in enumerate(outputs)
            if output is not None
            and not self._completion_has_final_answer(output)
        ]
        if not forced_indices:
            return completions

        budget_limited_indices = [
            idx
            for idx in forced_indices
            if self._completion_hit_generation_budget(outputs[idx], initial_max_tokens)
        ]
        continuation_prompts = [
            self._forced_final_prompt(prompts[idx], outputs[idx])
            for idx in forced_indices
        ]
        try:
            final_max_new_tokens = self._max_new_tokens_for_prompts(
                continuation_prompts,
                self._forced_final_max_new_tokens(),
            )
        except ValueError:
            return completions

        sampling_params = self._build_sampling_params(
            temperature=temperature,
            max_tokens=final_max_new_tokens,
            n=1,
        )
        continuation_outputs = self.llm.generate(continuation_prompts, sampling_params)
        self._record_request_outputs(continuation_outputs)
        forced_count = 0
        empty_count = 0
        for continuation_idx, (original_idx, request_output) in enumerate(
            zip(forced_indices, continuation_outputs)
        ):
            forced_output = request_output.outputs[0] if request_output.outputs else None
            self._log_llm_token_usage(
                call_type="forced_final",
                prompt=continuation_prompts[continuation_idx],
                output=forced_output,
                request_output=request_output,
                temperature=temperature,
                max_new_tokens=final_max_new_tokens,
                response_index=original_idx,
            )
            if not request_output.outputs:
                empty_count += 1
                continue
            forced_text = self._normalize_forced_final_output(request_output.outputs[0])
            if forced_text:
                completions[original_idx] = forced_text
                forced_count += 1
            else:
                empty_count += 1

        _wandb_log({
            "event": "Forced thinking exit",
            "forced_thinking_exit_count": forced_count,
            "forced_thinking_exit_length_count": len(budget_limited_indices),
            "forced_thinking_exit_early_stop_count": len(forced_indices) - len(budget_limited_indices),
            "forced_thinking_exit_empty_count": empty_count,
        })
        self._log_forced_thinking_exits(
            prompts,
            outputs,
            continuation_outputs,
            forced_indices,
            initial_max_tokens,
            final_max_new_tokens,
            set(budget_limited_indices),
        )

        return completions

    @staticmethod
    def _completion_excerpt(text: str, limit: int = 500) -> str:
        trace_chars = os.environ.get("BED_LLM_REASONING_TRACE_CHARS")
        if trace_chars:
            if trace_chars.lower() == "full":
                limit = 0
            else:
                try:
                    limit = max(0, int(trace_chars))
                except ValueError:
                    limit = 500
        text = text.strip()
        if limit <= 0:
            return text
        if len(text) <= limit:
            return text
        half = max(1, limit // 2)
        return f"{text[:half]}\n...\n{text[-half:]}"

    def _log_forced_thinking_exits(
        self,
        prompts: list,
        raw_outputs: list,
        forced_outputs: list,
        forced_indices: list[int],
        initial_max_tokens: int,
        final_max_new_tokens: int,
        budget_limited_indices: set[int],
    ) -> None:
        if os.environ.get("BED_LLM_LOG_REASONING_TRACES") != "1" or self.config.log_path is None:
            return
        for original_idx, forced_output in zip(forced_indices, forced_outputs):
            forced_text = ""
            if forced_output.outputs:
                forced_text = self._completion_raw_text(forced_output.outputs[0])
            raw_output = raw_outputs[original_idx]
            prompt_tokens = self._prompt_token_count(prompts[original_idx])
            raw_tokens = self._completion_token_count(raw_output)
            write_to_log(
                "Forced thinking exit:\n"
                f"model={self.model_name}\n"
                f"prompt_tokens={prompt_tokens}\n"
                f"first_pass_max_tokens={initial_max_tokens}\n"
                f"first_pass_output_tokens={raw_tokens if raw_tokens is not None else 'unknown'}\n"
                f"first_pass_finish_reason={self._completion_finish_reason(raw_output) or 'unknown'}\n"
                f"first_pass_budget_limited={original_idx in budget_limited_indices}\n"
                f"forced_final_max_tokens={final_max_new_tokens}\n"
                f"raw_completion_excerpt:\n{self._completion_excerpt(self._completion_raw_text(raw_output))}\n\n"
                f"forced_final_excerpt:\n{self._completion_excerpt(forced_text)}\n\n",
                self.config,
            )

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
                self._record_request_outputs(outputs)

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
                probability_temperature = (
                    float(temperature)
                    if float(temperature) > 0.0
                    else 1.0
                )
                scaled_scores = [
                    score / probability_temperature
                    for score in log_scores
                ]
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
            block_max_new_tokens = self._first_pass_max_new_tokens_for_prompts(block_prompts, max_new_tokens)
            sampling_params = self._build_sampling_params(
                temperature=temperature,
                max_tokens=block_max_new_tokens,
                n=1,
            )
            outputs = self.llm.generate(block_prompts, sampling_params)
            self._record_request_outputs(outputs)
            completion_outputs = [
                output.outputs[0] if output.outputs else None
                for output in outputs
            ]
            for block_offset, (prompt, request_output, output) in enumerate(
                zip(block_prompts, outputs, completion_outputs)
            ):
                self._log_llm_token_usage(
                    call_type="batched_chat",
                    prompt=prompt,
                    output=output,
                    request_output=request_output,
                    temperature=temperature,
                    max_new_tokens=block_max_new_tokens,
                    block_index=start_idx + block_offset,
                )
            completions.extend(
                self._maybe_force_final_answers(
                    block_prompts,
                    completion_outputs,
                    block_max_new_tokens,
                    temperature,
                )
            )

        return completions

    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1) -> list[str]:
        start_time = time.perf_counter()
        prompt = self._messages_to_prompt(messages)
        max_new_tokens = self._first_pass_max_new_tokens_for_prompts([prompt], self.config.location_max_new_tokens)
        sampling_params = self._build_sampling_params(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=num_responses,
        )
        outputs = self.llm.generate([prompt], sampling_params)
        self._record_request_outputs(outputs)
        request_output = outputs[0]
        for response_idx, output in enumerate(request_output.outputs):
            self._log_llm_token_usage(
                call_type="chat",
                prompt=prompt,
                output=output,
                request_output=request_output,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                response_index=response_idx,
            )
        completions = self._maybe_force_final_answers(
            [prompt for _output in outputs[0].outputs],
            list(outputs[0].outputs),
            max_new_tokens,
            temperature,
        )

        elapsed_time = time.perf_counter() - start_time
        _wandb_log({
            "event": "Chat completion",
            "number_input_tokens": len(outputs[0].prompt_token_ids),
            "elapsed_time": elapsed_time,
        })

        return completions

    def chat_complete_messages_batched(self, batch_messages: list[list[dict[str, str]]], temperature: float,
                                       block_size: int, max_new_tokens: int | None = None) -> list[str]:
        start_time = time.perf_counter()
        completions = self._chat_complete_messages_batched(
            batch_messages=batch_messages,
            temperature=temperature,
            block_size=block_size,
            max_new_tokens=max_new_tokens or self.config.location_max_new_tokens,
        )

        elapsed_time = time.perf_counter() - start_time
        _wandb_log({
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
                max_new_tokens=self.config.location_max_new_tokens,
            )

        elapsed_time = time.perf_counter() - start_time
        _wandb_log({
            "event": "Batched probability determination",
            "number_conversations": len(messages) * len(responses),
            "elapsed_time_batched": elapsed_time,
        })

        return results


class QwenVLLMAdapter(BaseVLLMAdapter):
    def __init__(self, spec: ModelSpec, config: Config, tensor_parallel_size: int | None = None, dtype: str = "bfloat16"):
        super().__init__(spec=spec, config=config, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
        self.thinking = bool(spec.thinking)
        self.thinking_max_new_tokens = spec.thinking_max_new_tokens or 4096
        self.thinking_final_max_new_tokens = spec.thinking_final_max_new_tokens or 512

    def _chat_template_kwargs(self) -> dict[str, object]:
        return {"enable_thinking": self.thinking}

    def _forced_thinking_exit_enabled(self) -> bool:
        return self.thinking

    def _forced_thinking_closer(self) -> str:
        return "\n</think>\nFinal Answer:"

    def _forced_final_max_new_tokens(self) -> int:
        return self.thinking_final_max_new_tokens

    def _first_pass_max_new_tokens_for_prompts(self, prompts: list, requested_max_tokens: int) -> int:
        if not self.thinking:
            return super()._first_pass_max_new_tokens_for_prompts(prompts, requested_max_tokens)
        requested = min(requested_max_tokens, self.thinking_max_new_tokens)
        reserve = self._forced_final_reserve_tokens()
        remaining_with_reserve = [
            self.max_model_len - self._prompt_token_count(prompt) - reserve
            for prompt in prompts
        ]
        if remaining_with_reserve and min(remaining_with_reserve) >= 1:
            requested = min(requested, min(remaining_with_reserve))
        return self._max_new_tokens_for_prompts(prompts, requested)

    def _completion_has_final_answer(self, output) -> bool:
        if not self.thinking:
            return True
        _thought, separator, final_text = self._completion_raw_text(output).rpartition("</think>")
        return bool(separator and final_text.strip())

    def _normalize_completion_output(self, output) -> str:
        text = self._completion_raw_text(output)
        if not self.thinking:
            return text
        thought, separator, final_text = text.rpartition("</think>")
        if not separator or not final_text.strip():
            if os.environ.get("BED_LLM_LOG_REASONING_TRACES") == "1" and self.config.log_path is not None:
                write_to_log(f"Qwen raw completion without final answer:\n{text}\n\n", self.config)
            return text.strip()
        if os.environ.get("BED_LLM_LOG_REASONING_TRACES") == "1" and self.config.log_path is not None:
            write_to_log(
                f"Qwen reasoning trace:\n{thought}\n\nQwen final answer:\n{final_text.strip()}\n\n",
                self.config,
            )
        return final_text.strip()


class GemmaVLLMAdapter(BaseVLLMAdapter):
    def __init__(self, spec: ModelSpec, config: Config, tensor_parallel_size: int | None = None, dtype: str = "bfloat16"):
        super().__init__(spec=spec, config=config, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
        self.thinking = bool(spec.thinking)
        self.thinking_max_new_tokens = spec.thinking_max_new_tokens or 4096
        self.thinking_final_max_new_tokens = spec.thinking_final_max_new_tokens or 512

    def _tokenizer_kwargs(self) -> dict[str, object]:
        return {
            "trust_remote_code": True,
        }

    def _chat_template_kwargs(self) -> dict[str, object]:
        return {"enable_thinking": self.thinking}

    def _forced_thinking_exit_enabled(self) -> bool:
        return self.thinking

    def _forced_thinking_closer(self) -> str:
        return "\n<channel|>\nFinal Answer:"

    def _forced_final_max_new_tokens(self) -> int:
        return self.thinking_final_max_new_tokens

    def _first_pass_max_new_tokens_for_prompts(self, prompts: list, requested_max_tokens: int) -> int:
        if not self.thinking:
            return super()._first_pass_max_new_tokens_for_prompts(prompts, requested_max_tokens)
        requested = min(requested_max_tokens, self.thinking_max_new_tokens)
        reserve = self._forced_final_reserve_tokens()
        remaining_with_reserve = [
            self.max_model_len - self._prompt_token_count(prompt) - reserve
            for prompt in prompts
        ]
        if remaining_with_reserve and min(remaining_with_reserve) >= 1:
            requested = min(requested, min(remaining_with_reserve))
        return self._max_new_tokens_for_prompts(prompts, requested)

    @staticmethod
    def _strip_thinking_channels(text: str) -> str:
        remaining = text
        kept_parts: list[str] = []
        while "<|channel>" in remaining:
            before, _marker, after_marker = remaining.partition("<|channel>")
            kept_parts.append(before)
            _channel_name, closer, after_channel = after_marker.partition("<channel|>")
            if not closer:
                remaining = ""
                break
            remaining = after_channel
        kept_parts.append(remaining)
        return "".join(kept_parts).strip()

    def _parse_response_content(self, output) -> str | None:
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
                stripped_content = self._strip_thinking_channels(content)
                if stripped_content:
                    return stripped_content
        return None

    def _completion_has_final_answer(self, output) -> bool:
        if not self.thinking:
            return True
        return self._parse_response_content(output) is not None

    def _normalize_completion_output(self, output) -> str:
        #write reasoning trace to log
        #write_to_log(f"Reasoning trace: {self.tokenizer.decode(token_ids)}\n", self.config)

        content = self._parse_response_content(output)
        if content is not None:
            return content

        # Gemma occasionally emits only a reasoning channel for structured JSON prompts.
        # Keep the run alive and let downstream JSON parsers/fallbacks handle the text.
        token_ids = getattr(output, "token_ids", None)
        fallback_text = getattr(output, "text", "") or self.tokenizer.decode(token_ids)
        return fallback_text.strip()


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
    if spec.model.startswith("Qwen/Qwen3."):
        return QwenVLLMAdapter(spec=spec, config=config, tensor_parallel_size=tensor_parallel_size, dtype=dtype)
    return BaseVLLMAdapter(spec=spec, config=config, tensor_parallel_size=tensor_parallel_size, dtype=dtype)


VLLMAdapter = BaseVLLMAdapter
