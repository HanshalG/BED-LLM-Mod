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
OPENROUTER_GEMMA_4_26B_A4B = "google/gemma-4-26b-a4b-it"
_SPEND_LOCK = threading.Lock()


class OpenRouterBudgetError(RuntimeError):
    pass


class OpenRouterBudgetTracker:
    def __init__(self, config: Config, model: str) -> None:
        self.path = Path(config.openrouter_spend_path)
        self.budget = float(config.openrouter_budget_usd)
        self.projected = float(config.openrouter_projected_cost_usd)
        self.run_id = config.run_id or "unassigned"
        self.model = model
        with self._locked_transaction():
            payload = self._read()
            spent = float(payload.get("total_spent_usd", 0.0))
            budget = self._effective_budget(payload)
            if spent + self.projected > budget + 1e-12:
                raise OpenRouterBudgetError(
                    f"Projected OpenRouter spend ${self.projected:.4f} plus existing "
                    f"${spent:.4f} exceeds ${budget:.2f} budget"
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
            run["cost_usd"] = float(run.get("cost_usd", 0.0)) + cost
            run["requests"] = int(run.get("requests", 0)) + 1
            run["prompt_tokens"] = int(run.get("prompt_tokens", 0)) + int(usage.get("prompt_tokens", 0) or 0)
            run["completion_tokens"] = int(run.get("completion_tokens", 0)) + int(usage.get("completion_tokens", 0) or 0)
            details = usage.get("completion_tokens_details") or {}
            run["reasoning_tokens"] = int(run.get("reasoning_tokens", 0)) + int(details.get("reasoning_tokens", 0) or 0)
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
                "total_spent_usd": float(payload.get("total_spent_usd", 0.0)),
                "budget_usd": budget,
                "remaining_usd": budget - float(payload.get("total_spent_usd", 0.0)),
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
        requested_output = (
            int((spec.thinking_max_new_tokens or 4096) + (spec.thinking_final_max_new_tokens or 512))
            if self.thinking
            else int(config.openrouter_max_output_tokens)
        )
        self.max_tokens = min(int(spec.max_model_len or config.max_model_len), requested_output)
        self.concurrency = int(config.openrouter_concurrency)
        self.max_retries = int(config.openrouter_max_retries)
        self.backoff_seconds = float(config.openrouter_backoff_seconds)
        self.seed = config.paprika_seed if config.task == "paprika_customer_service" else None
        self.tracker = OpenRouterBudgetTracker(config, self.model_name)
        self.forced_exits = 0
        self.local_cost_usd = 0.0
        self.local_requests = 0
        self.local_prompt_tokens = 0
        self.local_completion_tokens = 0
        self.local_reasoning_tokens = 0

    def _payload(self, messages: list[dict[str, str]], temperature: float, n: int, max_tokens: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": int(max_tokens or self.max_tokens),
            "n": int(n),
        }
        if self.thinking:
            payload["reasoning"] = {"enabled": True, "exclude": False}
        if self.seed is not None:
            payload["seed"] = int(self.seed)
        return payload

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{OPENROUTER_BASE_URL}/chat/completions",
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
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
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
            time.sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _content(choice: dict[str, Any]) -> str:
        content = (choice.get("message") or {}).get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict)).strip()
        return str(content).strip()

    def _complete_request(self, messages: list[dict[str, str]], temperature: float, n: int = 1, max_tokens: int | None = None) -> list[str]:
        data = self._post(self._payload(messages, temperature, n, max_tokens))
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
            "reasoning_enabled": self.thinking,
        }
        if self.config.log_path is not None:
            write_to_log(json.dumps(payload, sort_keys=True) + "\n", self.config)
            if any(choice.get("finish_reason") == "length" for choice in choices):
                write_to_log("Forced thinking exit (OpenRouter finish_reason=length)\n", self.config)
        if cumulative >= 18.0:
            print(f"WARNING: cumulative OpenRouter spend is ${cumulative:.2f}; request top-up before more runs")
        return [self._content(choice) for choice in choices]

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
        return snapshot
