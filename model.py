import math
from abc import ABC, abstractmethod
import time

import torch
import wandb
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


class HuggingFaceAdapter(Model):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B", device: str | None = None, dtype=torch.float16,
                 use_flash_attn: bool = True):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        attn_impl = None
        if self.device == "cuda" and use_flash_attn and self.dtype in (
                torch.float16,
                torch.bfloat16,
        ):
            attn_impl = "flash_attention_2"

        model_kwargs = {
            "torch_dtype": self.dtype,
        }
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        if self.device == "cpu":
            self.model.to("cpu")

        self.model.eval()

        self.model.config.use_cache = True


    def _messages_to_prompt(self, messages):
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )


    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1,
                      max_new_tokens: int = 256) -> list[str]:
        start_time = time.perf_counter()

        prompt = self._messages_to_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            do_sample=temperature > 0,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            num_return_sequences=num_responses,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with torch.inference_mode():
            outputs = self.model.generate(**gen_kwargs)

        gen_ids = outputs[:, input_len:]
        completions = self.tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True,
        )

        completions = [c.lstrip() for c in completions]

        elapsed_time = time.perf_counter() - start_time
        wandb.log({
            "event": "Chat completion",
            "number_input_tokens": input_len,
            "elapsed_time": elapsed_time,
        })

        return completions


    def chat_probabilities_messages_batched(self, batch_messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int) -> list[dict[str, float]]:
        start_time = time.perf_counter()

        device = self.model.device

        prompts = [self._messages_to_prompt(msgs) for msgs in batch_messages]
        batch_size = len(prompts)
        num_responses = len(responses)

        scores = torch.zeros(batch_size, num_responses, device=device)

        for start_idx in range(0, batch_size, block_size):
            end_idx = min(start_idx + block_size, batch_size)
            block_prompts = prompts[start_idx:end_idx]

            block_prompt_tok = self.tokenizer(
                block_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            block_prompt_attn = block_prompt_tok["attention_mask"].to(device)
            block_prompt_lens = block_prompt_attn.sum(dim=1)  # shape: [block_B]

            for r_idx, resp in enumerate(responses):
                block_full_texts = [p + resp for p in block_prompts]

                inputs = self.tokenizer(
                    block_full_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                ).to(device)

                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                with torch.inference_mode():
                    logits = self.model(**inputs).logits

                logits = logits[:, :-1, :] / temperature
                target_ids = input_ids[:, 1:]

                log_probs = torch.log_softmax(logits, dim=-1)

                token_log_probs = log_probs.gather(
                    dim=-1,
                    index=target_ids.unsqueeze(-1)
                ).squeeze(-1)

                seq_lens = attention_mask.sum(dim=1)
                pos = torch.arange(logits.size(1), device=device).unsqueeze(0)
                valid_mask = pos < (seq_lens - 1).unsqueeze(1)
                start = (block_prompt_lens - 2).clamp(min=0)
                resp_mask = pos >= start.unsqueeze(1)
                mask = valid_mask & resp_mask

                token_log_probs = token_log_probs.masked_fill(~mask, 0.0)
                block_scores_for_resp = token_log_probs.sum(dim=1)

                scores[start_idx:end_idx, r_idx] = block_scores_for_resp

        probs = torch.softmax(scores, dim=1)  # [B, R]

        results: list[dict[str, float]] = []
        for b in range(batch_size):
            conv_probs = {responses[r]: probs[b, r].item() for r in range(num_responses)}
            results.append(conv_probs)

        elapsed_time = time.perf_counter() - start_time
        wandb.log({
            "event": "Batched probability determination",
            "number_conversations": len(batch_messages) * len(responses),
            "elapsed_time_batched": elapsed_time,
        })

        return results


class VLLMAdapter(Model):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B", tensor_parallel_size: int | None = None,
                 dtype: str = "float16"):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()

        self.llm = LLM(
            model=model_name,
            max_model_len=1024,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            language_model_only=True,
            
        )


    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )


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

        prompts = [self._messages_to_prompt(m) for m in messages]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
        )

        tokenized_prompts = [
            self.tokenizer(p, add_special_tokens=False).input_ids
            for p in prompts
        ]

        base_lens = [len(toks) for toks in tokenized_prompts]

        results: list[dict[str, float]] = []

        for block_start in range(0, len(prompts), block_size):
            block_prompts = prompts[block_start:block_start + block_size]
            base_token_lens = base_lens[block_start:block_start + block_size]

            # Collect log-probs per response, per conversation
            # shape: [num_responses][block_size]
            response_logprobs = []

            for response in responses:
                full_prompts = [
                    p + response for p in block_prompts
                ]

                outputs = self.llm.generate(
                    full_prompts,
                    sampling_params=sampling_params,
                )

                block_logprobs = [0.0] * len(outputs)

                for i, output in enumerate(outputs):
                    prompt_logprobs = output.prompt_logprobs
                    start = base_token_lens[i]

                    total_lp = 0.0
                    # exactly one token for 'Yes'/'No'
                    for j in range(start, len(prompt_logprobs)):
                        token_lp_dict = prompt_logprobs[j]
                        # first item is always the 'fixed' response
                        total_lp += next(iter(token_lp_dict.values())).logprob

                    block_logprobs[i] = total_lp

                response_logprobs.append(block_logprobs)

            # Convert log-probs to probabilities PER CONVERSATION
            # transpose to [block_size][num_responses]
            for convo_idx in range(len(block_prompts)):
                lps = [
                    response_logprobs[r][convo_idx]
                    for r in range(len(responses))
                ]

                # calculate softmax
                scaled = [lp / temperature for lp in lps]
                max_lp = max(scaled)
                exps = [math.exp(lp - max_lp) for lp in scaled]
                norm = sum(exps)

                probs = [e / norm for e in exps]

                results.append(dict(zip(responses, probs)))

        elapsed_time = time.perf_counter() - start_time
        wandb.log({
            "event": "Batched probability determination",
            "number_conversations": len(messages) * len(responses),
            "elapsed_time_batched": elapsed_time,
        })

        return results
