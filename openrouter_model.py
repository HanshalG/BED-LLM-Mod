"""OpenRouter chat adapter with cost accounting and a hard project budget."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import fcntl
import http.client

from helpers import Config, ModelSpec, _probability_results_from_messages, write_to_log


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
WAFER_REASONING_TRUNCATION_PREFIX = (
    "[wafer: response was truncated before the model finished its internal reasoning."
)
OPENROUTER_GEMMA_4_26B_A4B = "google/gemma-4-26b-a4b-it"
_SPEND_LOCK = threading.Lock()


class OpenRouterBudgetError(RuntimeError):
    pass


class OpenRouterBudgetTracker:
    def __init__(self, config: Config, model: str) -> None:
        self.path = Path(config.openrouter_spend_path)
        self.budget = float(config.openrouter_budget_usd)
        self.projected = float(config.openrouter_projected_cost_usd)
        self.run_budget = (
            float(config.openrouter_run_budget_usd)
            if config.openrouter_run_budget_usd is not None
            else None
        )
        self.run_id = config.run_id or "unassigned"
        self.model = model
        with self._locked_transaction():
            payload = self._read()
            spent = float(payload.get("total_spent_usd", 0.0))
            budget = self._effective_budget(payload)
            run_cost = float(((payload.get("runs") or {}).get(self.run_id) or {}).get("cost_usd", 0.0))
            if spent + self.projected > budget + 1e-12:
                raise OpenRouterBudgetError(
                    f"Projected OpenRouter spend ${self.projected:.4f} plus existing "
                    f"${spent:.4f} exceeds ${budget:.2f} budget"
                )
            if self.run_budget is not None and run_cost + self.projected > self.run_budget + 1e-12:
                raise OpenRouterBudgetError(
                    f"Projected OpenRouter run spend ${self.projected:.4f} plus existing "
                    f"${run_cost:.4f} exceeds ${self.run_budget:.2f} run budget"
                )

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"budget_usd": self.budget, "total_spent_usd": 0.0, "runs": {}}
        payload = json.loads(self.path.read_text())
        if not isinstance(payload, dict):
            raise OpenRouterBudgetError(f"Invalid spend ledger at {self.path}")
        return payload

    @contextmanager
    def _locked_transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with _SPEND_LOCK, lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _atomic_write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _effective_budget(self, payload: dict[str, Any]) -> float:
        """Keep an authorized top-up from being reverted by older live workers."""
        return max(self.budget, float(payload.get("budget_usd", self.budget)))

    def add(self, cost: float, usage: dict[str, Any]) -> float:
        if cost < 0.0:
            raise OpenRouterBudgetError("OpenRouter reported negative cost")
        with self._locked_transaction():
            payload = self._read()
            budget = self._effective_budget(payload)
            total = float(payload.get("total_spent_usd", 0.0)) + cost
            if total > budget + 1e-9:
                raise OpenRouterBudgetError(
                    f"OpenRouter charge would exceed ${budget:.2f} budget: ${total:.6f}"
                )
            runs = payload.setdefault("runs", {})
            run = runs.setdefault(
                self.run_id,
                {"backend": "openrouter", "model": self.model, "cost_usd": 0.0, "requests": 0},
            )
            run_cost = float(run.get("cost_usd", 0.0)) + cost
            if self.run_budget is not None and run_cost > self.run_budget + 1e-9:
                raise OpenRouterBudgetError(
                    f"OpenRouter charge would exceed ${self.run_budget:.2f} run budget: ${run_cost:.6f}"
                )
            run["cost_usd"] = run_cost
            run["requests"] = int(run.get("requests", 0)) + 1
            run["prompt_tokens"] = int(run.get("prompt_tokens", 0)) + int(usage.get("prompt_tokens", 0) or 0)
            run["completion_tokens"] = int(run.get("completion_tokens", 0)) + int(usage.get("completion_tokens", 0) or 0)
            details = usage.get("completion_tokens_details") or {}
            run["reasoning_tokens"] = int(run.get("reasoning_tokens", 0)) + int(details.get("reasoning_tokens", 0) or 0)
            model_usage = run.setdefault("model_usage", {})
            model_run = model_usage.setdefault(
                self.model,
                {
                    "cost_usd": 0.0,
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                },
            )
            model_run["cost_usd"] = float(model_run["cost_usd"]) + cost
            model_run["requests"] = int(model_run["requests"]) + 1
            model_run["prompt_tokens"] = int(model_run["prompt_tokens"]) + int(
                usage.get("prompt_tokens", 0) or 0
            )
            model_run["completion_tokens"] = int(model_run["completion_tokens"]) + int(
                usage.get("completion_tokens", 0) or 0
            )
            model_run["reasoning_tokens"] = int(model_run["reasoning_tokens"]) + int(
                details.get("reasoning_tokens", 0) or 0
            )
            payload["budget_usd"] = budget
            payload["total_spent_usd"] = total
            self._atomic_write(payload)
            return total

    def snapshot(self) -> dict[str, Any]:
        with self._locked_transaction():
            payload = self._read()
            budget = self._effective_budget(payload)
            run = (payload.get("runs") or {}).get(self.run_id, {})
            return {
                "backend": "openrouter",
                "model": self.model,
                "run_cost_usd": float(run.get("cost_usd", 0.0)),
                "requests": int(run.get("requests", 0)),
                "prompt_tokens": int(run.get("prompt_tokens", 0)),
                "completion_tokens": int(run.get("completion_tokens", 0)),
                "reasoning_tokens": int(run.get("reasoning_tokens", 0)),
                "model_usage": run.get("model_usage", {}),
                "total_spent_usd": float(payload.get("total_spent_usd", 0.0)),
                "budget_usd": budget,
                "remaining_usd": budget - float(payload.get("total_spent_usd", 0.0)),
                "run_budget_usd": self.run_budget,
                "run_remaining_usd": (
                    self.run_budget - float(run.get("cost_usd", 0.0))
                    if self.run_budget is not None
                    else None
                ),
            }


class OpenRouterAdapter:
    def __init__(self, spec: ModelSpec, config: Config) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for backend=openrouter")
        self.api_key = api_key
        self.spec = spec
        self.config = config
        self.model_name = spec.model
        self.thinking = bool(spec.thinking)
        self.reasoning_enabled = bool(
            spec.thinking
            or (
                spec.reasoning_effort is not None
                and spec.reasoning_effort != "none"
            )
            or spec.reasoning_max_tokens
        )
        requested_output = (
            int((spec.thinking_max_new_tokens or 4096) + (spec.thinking_final_max_new_tokens or 512))
            if self.thinking
            else int(config.openrouter_max_output_tokens)
        )
        self.max_tokens = min(int(spec.max_model_len or config.max_model_len), requested_output)
        self.concurrency = int(config.openrouter_concurrency)
        self.max_retries = int(config.openrouter_max_retries)
        self.backoff_seconds = float(config.openrouter_backoff_seconds)
        self.request_timeout_seconds = float(config.openrouter_request_timeout_seconds)
        task_seeds = {
            "paprika_customer_service": config.paprika_seed,
            "mediq": config.mediq_seed,
        }
        self.seed = task_seeds.get(config.task)
        self.tracker = OpenRouterBudgetTracker(config, self.model_name)
        self.forced_exits = 0
        self.local_cost_usd = 0.0
        self.local_requests = 0
        self.local_prompt_tokens = 0
        self.local_completion_tokens = 0
        self.local_reasoning_tokens = 0
        self.forced_final_requests = 0
        self.forced_final_successes = 0
        self.http_attempts = 0
        self.retry_count = 0
        self._usage_lock = threading.Lock()
        self._budget_warning_emitted = False
        self._budget_warning_lock = threading.Lock()

    def _warn_near_budget_once(self, cumulative: float) -> None:
        budget = float(self.config.openrouter_budget_usd)
        if cumulative < 0.9 * budget:
            return
        with self._budget_warning_lock:
            if self._budget_warning_emitted:
                return
            self._budget_warning_emitted = True
        print(
            f"WARNING: cumulative OpenRouter spend is ${cumulative:.2f} "
            f"of ${budget:.2f}; ${budget - cumulative:.2f} remains"
        )

    def _payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        n: int,
        max_tokens: int | None = None,
        *,
        disable_reasoning: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": int(max_tokens or self.max_tokens),
            "n": int(n),
        }
        if disable_reasoning:
            payload["reasoning"] = {"enabled": False, "exclude": True}
        elif self.spec.reasoning_max_tokens is not None:
            payload["reasoning"] = {
                "max_tokens": self.spec.reasoning_max_tokens,
                "exclude": False,
            }
        elif self.spec.reasoning_effort is not None:
            payload["reasoning"] = {
                "effort": self.spec.reasoning_effort,
                "exclude": False,
            }
        elif self.thinking:
            payload["reasoning"] = {"enabled": True, "exclude": False}
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        if response_format is not None:
            payload["response_format"] = response_format
            payload["provider"] = {"require_parameters": True}
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_endpoint("chat/completions", payload)

    def _post_endpoint(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{OPENROUTER_BASE_URL}/{endpoint}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/HanshalG/BED-LLM-Mod",
                "X-Title": "BED-LLM Path E",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            with self._usage_lock:
                self.http_attempts += 1
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                    raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self.backoff_seconds * (2**attempt)
            except (
                urllib.error.URLError,
                TimeoutError,
                http.client.HTTPException,
                ConnectionError,
                json.JSONDecodeError,
            ) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"OpenRouter request failed after retries: {exc}") from exc
                delay = self.backoff_seconds * (2**attempt)
            with self._usage_lock:
                self.retry_count += 1
            time.sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _responses_input(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "message",
                "role": message["role"],
                "content": [
                    {
                        "type": "input_text",
                        "text": str(message.get("content", "")),
                    }
                ],
            }
            for message in messages
        ]

    def _responses_payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        if (
            response_format.get("type") != "json_schema"
            or not isinstance(response_format.get("json_schema"), dict)
        ):
            raise ValueError(
                "Responses structured output requires a JSON Schema format"
            )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": self._responses_input(messages),
            "temperature": float(temperature),
            "top_p": 0.95,
            "max_output_tokens": int(max_tokens or self.max_tokens),
            "text": {"format": response_format["json_schema"] | {
                "type": "json_schema"
            }},
            "provider": {"require_parameters": True},
            "store": False,
        }
        if self.spec.reasoning_effort is not None:
            payload["reasoning"] = {
                "effort": self.spec.reasoning_effort,
            }
        elif self.spec.reasoning_max_tokens is not None:
            payload["reasoning"] = {
                "max_tokens": self.spec.reasoning_max_tokens,
            }
        elif self.thinking:
            payload["reasoning"] = {"effort": "medium"}
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        return payload

    @staticmethod
    def _responses_output_text(data: dict[str, Any]) -> str:
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        parts = []
        for item in data.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if (
                    isinstance(content, dict)
                    and content.get("type") == "output_text"
                ):
                    parts.append(str(content.get("text") or ""))
        return "".join(parts).strip()

    def _complete_responses_structured(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any],
    ) -> str:
        data = self._post_endpoint(
            "responses",
            self._responses_payload(
                messages,
                temperature,
                max_tokens,
                response_format,
            ),
        )
        usage = data.get("usage")
        if not isinstance(usage, dict) or usage.get("cost") is None:
            raise RuntimeError(
                "OpenRouter Responses result is missing required usage.cost"
            )
        input_tokens = int(
            usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        )
        output_tokens = int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        )
        output_details = usage.get("output_tokens_details") or {}
        reasoning_tokens = int(
            output_details.get(
                "reasoning_tokens",
                (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens", 0
                ),
            )
            or 0
        )
        normalized_usage = dict(usage)
        normalized_usage["prompt_tokens"] = input_tokens
        normalized_usage["completion_tokens"] = output_tokens
        normalized_usage["completion_tokens_details"] = {
            "reasoning_tokens": reasoning_tokens
        }
        cost = float(usage["cost"])
        cumulative = self.tracker.add(cost, normalized_usage)
        self.local_cost_usd += cost
        self.local_requests += 1
        self.local_prompt_tokens += input_tokens
        self.local_completion_tokens += output_tokens
        self.local_reasoning_tokens += reasoning_tokens
        if data.get("status") != "completed":
            self.forced_exits += 1
            raise RuntimeError(
                f"OpenRouter Responses status is {data.get('status')!r}"
            )
        content = self._responses_output_text(data)
        if not content:
            raise RuntimeError("OpenRouter Responses result has no output text")
        if self.config.log_path is not None:
            write_to_log(
                json.dumps(
                    {
                        "event": "llm_token_usage",
                        "backend": "openrouter_responses",
                        "model": self.model_name,
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "total_tokens": usage.get("total_tokens"),
                        "cost_usd": cost,
                        "cumulative_cost_usd": cumulative,
                        "temperature": temperature,
                        "status": data.get("status"),
                    },
                    sort_keys=True,
                )
                + "\n",
                self.config,
            )
        self._warn_near_budget_once(cumulative)
        return content

    @staticmethod
    def _content(choice: dict[str, Any]) -> str:
        content = (choice.get("message") or {}).get("content", "")
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
        return str(content).strip()

    @staticmethod
    def _is_reasoning_truncation_notice(content: str) -> bool:
        normalized = " ".join(content.strip().lower().split())
        return (
            normalized.startswith(WAFER_REASONING_TRUNCATION_PREFIX)
            and "increase max_tokens" in normalized
            and "disable thinking" in normalized
        )

    def _complete_request(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        n: int = 1,
        max_tokens: int | None = None,
        *,
        allow_forced_final: bool = True,
        disable_reasoning: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> list[str]:
        data = self._post(
            self._payload(
                messages,
                temperature,
                n,
                max_tokens,
                disable_reasoning=disable_reasoning,
                response_format=response_format,
            )
        )
        choices = data.get("choices")
        usage = data.get("usage")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenRouter response has no choices")
        if not isinstance(usage, dict) or usage.get("cost") is None:
            raise RuntimeError("OpenRouter response is missing required usage.cost")
        cost = float(usage["cost"])
        cumulative = self.tracker.add(cost, usage)
        self.local_cost_usd += cost
        self.local_requests += 1
        self.local_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.local_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        self.local_reasoning_tokens += int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        )
        for choice in choices:
            if choice.get("finish_reason") == "length":
                self.forced_exits += 1
        payload = {
            "event": "llm_token_usage",
            "backend": "openrouter",
            "model": self.model_name,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            "total_tokens": usage.get("total_tokens"),
            "cost_usd": cost,
            "cumulative_cost_usd": cumulative,
            "temperature": temperature,
            "finish_reasons": [choice.get("finish_reason") for choice in choices],
            "reasoning_enabled": self.reasoning_enabled,
        }
        if self.config.log_path is not None:
            write_to_log(json.dumps(payload, sort_keys=True) + "\n", self.config)
            if any(choice.get("finish_reason") == "length" for choice in choices):
                write_to_log("Forced thinking exit (OpenRouter finish_reason=length)\n", self.config)
        self._warn_near_budget_once(cumulative)
        contents = [self._content(choice) for choice in choices]
        provider_truncation_notice = self._is_reasoning_truncation_notice(
            contents[0]
        )
        if (
            allow_forced_final
            and self.reasoning_enabled
            and n == 1
            and choices[0].get("finish_reason") == "length"
            and (not contents[0] or provider_truncation_notice)
        ):
            message = choices[0].get("message") or {}
            preserved: dict[str, Any] = {
                "role": "assistant",
                "content": (
                    ""
                    if provider_truncation_notice
                    else message.get("content") or ""
                ),
            }
            if message.get("reasoning_details") is not None:
                preserved["reasoning_details"] = message["reasoning_details"]
            elif message.get("reasoning") is not None:
                preserved["reasoning"] = message["reasoning"]
            elif message.get("reasoning_content") is not None:
                preserved["reasoning_content"] = message["reasoning_content"]
            forced_messages = [
                *messages,
                preserved,
                {
                    "role": "user",
                    "content": (
                        "Continue from the preserved reasoning and return only the "
                        "final answer requested by the original user. Do not include "
                        "reasoning, explanation, or extra prose."
                    ),
                },
            ]
            self.forced_final_requests += 1
            if self.config.log_path is not None:
                if provider_truncation_notice:
                    write_to_log(
                        "OpenRouter provider truncation notice normalized\n",
                        self.config,
                    )
                write_to_log(
                    "OpenRouter forced-final continuation request\n", self.config
                )
            forced = self._complete_request(
                forced_messages,
                0.0,
                n=1,
                max_tokens=int(self.spec.thinking_final_max_new_tokens or 512),
                allow_forced_final=False,
                disable_reasoning=True,
                response_format=response_format,
            )
            if forced[0]:
                self.forced_final_successes += 1
            return forced
        return contents

    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1) -> list[str]:
        return self._complete_request(messages, temperature, n=num_responses)

    def chat_complete_messages_batched(self, batch_messages: list[list[dict[str, str]]], temperature: float, block_size: int, max_new_tokens: int | None = None) -> list[str]:
        del block_size
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [
                executor.submit(self._complete_request, messages, temperature, 1, max_new_tokens)
                for messages in batch_messages
            ]
            return [future.result()[0] for future in futures]

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del block_size
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [
                executor.submit(
                    self._complete_request,
                    messages,
                    temperature,
                    1,
                    max_new_tokens,
                    response_format=response_format,
                )
                for messages in batch_messages
            ]
            return [future.result()[0] for future in futures]

    def responses_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del block_size
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [
                executor.submit(
                    self._complete_responses_structured,
                    messages,
                    temperature,
                    max_new_tokens,
                    response_format,
                )
                for messages in batch_messages
            ]
            return [future.result() for future in futures]

    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str], temperature: float, block_size: int) -> list[dict[str, float]]:
        return _probability_results_from_messages(
            messages,
            responses,
            block_size,
            temperature,
            self.chat_complete_messages_batched,
            fallback_to_uniform=self.config.probability_parse_fallback_to_uniform,
            max_new_tokens=self.max_tokens,
        )

    def usage_snapshot(self) -> dict[str, Any]:
        snapshot = self.tracker.snapshot()
        snapshot.update(
            {
                "adapter_cost_usd": self.local_cost_usd,
                "adapter_requests": self.local_requests,
                "adapter_prompt_tokens": self.local_prompt_tokens,
                "adapter_completion_tokens": self.local_completion_tokens,
                "adapter_reasoning_tokens": self.local_reasoning_tokens,
            }
        )
        snapshot["forced_exits"] = self.forced_exits
        snapshot["forced_final_requests"] = self.forced_final_requests
        snapshot["forced_final_successes"] = self.forced_final_successes
        snapshot["http_attempts"] = self.http_attempts
        snapshot["retry_count"] = self.retry_count
        snapshot["reasoning_enabled"] = self.reasoning_enabled
        return snapshot
