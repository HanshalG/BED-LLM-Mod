#!/usr/bin/env python3
"""Run fresh Tau2 MMS semantics with response-level durable checkpoints."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import tau2_mms_array_semantic_calibration as prior
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint

SCHEMA_VERSION = 1
INTERFACE_VERSION = "tau2-mms-checkpointed-semantic-calibration-1"
MODEL_ID = prior.MODEL_ID
MODEL_SEEDS = tuple(range(202608130300, 202608130306))
MAX_TOKENS = 6_000
EXPECTED_REQUESTS = 6
CONCURRENCY = 2
RUN_CAP_USD = 0.10
PROJECTED_COST_USD = 0.06
MAX_REQUEST_COST_USD = 0.01
PROTOCOL = REPO_ROOT / "results/nonmyopic/TAU2_MMS_CHECKPOINTED_SEMANTIC_CALIBRATION_PROTOCOL_20260813.md"
MANIFEST = REPO_ROOT / "results/nonmyopic/tau2_mms_checkpointed_semantic_calibration/CALIBRATION_MANIFEST.json"

canonical_json = prior.canonical_json
sha256_file = prior.sha256_file
episode_hash = prior.episode_hash
public_episode = prior.public_episode
response_format = prior.response_format
messages = prior.messages
parse = prior.parse
calibration_gates = prior.calibration_gates
usage = prior.usage


def checkpoint_partial(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def selected_episodes() -> list[dict[str, Any]]:
    tasks = json.loads(prior.source.TASKS_PATH.read_text())
    splits = json.loads(prior.source.SPLITS_PATH.read_text())
    selected, _ = prior.source.select_episodes(
        [str(row["id"]) for row in tasks], list(splits["base"])
    )
    by_hash = {episode_hash(row): row for row in selected["reserve"]}
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest.get("episodes", [])
    if len(rows) != 6 or len({row.get("episode_sha256") for row in rows}) != 6:
        raise ValueError("checkpointed MMS manifest changed")
    episodes = []
    for row in rows:
        episode = by_hash.get(row.get("episode_sha256"))
        if episode is None or row.get("family") != episode["family"] or len(episode["worlds"]) != 4:
            raise ValueError("checkpointed MMS episode binding changed")
        episodes.append(episode)
    old_hashes = {episode_hash(row) for row in prior.selected_episodes()}
    if old_hashes & {episode_hash(row) for row in episodes}:
        raise ValueError("checkpointed MMS cohort overlaps predecessor")
    return episodes


class CheckpointingAdapter(prior.base.NonReasoningSeededAdapter):
    def __init__(self, *args: Any, partial_path: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.partial_path = partial_path

    def chat_complete_seeded_messages_batched_structured(
        self, batch_messages: Sequence[list[dict[str, str]]], seeds: Sequence[int],
        *, temperature: float, response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        if len(batch_messages) != len(seeds):
            raise ValueError("message and seed counts differ")

        def request(index: int) -> str:
            self._per_request_seed.value = int(seeds[index])
            try:
                return self._complete_request(
                    batch_messages[index], temperature, 1, max_new_tokens,
                    response_format=response_format,
                )[0]
            finally:
                del self._per_request_seed.value

        completed: dict[int, str] = {}
        first_error: BaseException | None = None
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = {executor.submit(request, index): index for index in range(len(seeds))}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    completed[index] = future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                checkpoint_partial(
                    self.partial_path,
                    {
                        "interface_version": INTERFACE_VERSION,
                        "model_id": MODEL_ID,
                        "expected_seeds": list(seeds),
                        "completed": [
                            {"index": item, "seed": int(seeds[item]), "response": completed[item]}
                            for item in sorted(completed)
                        ],
                        "complete": len(completed) == len(seeds) and first_error is None,
                    },
                )
        if first_error is not None:
            raise first_error
        if len(completed) != len(seeds):
            raise RuntimeError("checkpointed MMS response set incomplete")
        return [completed[index] for index in range(len(seeds))]


def build_adapter(*, run_id: str, output_dir: Path) -> CheckpointingAdapter:
    config = Config(
        task="animals", run_id=run_id, log_path=output_dir / "run.log",
        openrouter_budget_usd=245.0, openrouter_run_budget_usd=RUN_CAP_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=CONCURRENCY, openrouter_max_retries=0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=MAX_REQUEST_COST_USD,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return CheckpointingAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65_536),
        config, partial_path=output_dir / "private/PARTIAL_RAW_RESPONSES.json",
    )


def run(*, output_dir: Path, adapter: Any, daily_budget_status: Mapping[str, Any] | None = None) -> dict[str, Any]:
    episodes = selected_episodes()
    public = [public_episode(row, index) for index, row in enumerate(episodes)]
    prompts = [messages(row) for row in public]
    privacy = {
        "prompt_sha256": [hashlib.sha256(canonical_json(row).encode()).hexdigest() for row in prompts],
        "selected_task_ids_in_prompts": False, "source_fault_ids_in_prompts": False,
        "raw_tool_responses_in_prompts": False, "repair_or_endpoint_outcomes_in_prompts": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"; private.mkdir(parents=True, exist_ok=True)
    checkpoint(private / "PROMPT_PRIVACY.json", privacy)
    raw = adapter.chat_complete_seeded_messages_batched_structured(
        prompts, MODEL_SEEDS, temperature=0.0,
        response_format=response_format(), max_new_tokens=MAX_TOKENS,
    )
    if len(raw) != EXPECTED_REQUESTS:
        raise ValueError("checkpointed MMS response count changed")
    checkpoint(private / "RAW_RESPONSES.json", {"model_id": MODEL_ID, "seeds": list(MODEL_SEEDS), "responses": list(raw)})
    ordering = {"all_partial_responses_banked": True, "complete_raw_bank_created": True, "official_calibration_loaded_after_complete_bank": False}
    checkpoint(private / "ORDERING.json", ordering)
    parsed = [parse(text, row) for text, row in zip(raw, public, strict=True)]
    observed = prior.base.official_observations(episodes)
    ordering["official_calibration_loaded_after_complete_bank"] = True
    checkpoint(private / "ORDERING.json", ordering)
    scores = prior.base.score_responses(episodes, parsed, observed)
    gates = calibration_gates(scores)
    usage_value, serving = usage(adapter)
    passed = gates["all_calibration_gates_pass"] and all(serving.values())
    result = {
        "schema_version": SCHEMA_VERSION, "interface_version": INTERFACE_VERSION,
        "status": "mms_checkpointed_semantic_pass" if passed else "mms_checkpointed_semantic_null",
        "authorizes": "prospective_paired_development_protocol_only" if passed else "nothing",
        "protocol_sha256": sha256_file(PROTOCOL), "manifest_sha256": sha256_file(MANIFEST),
        "model": MODEL_ID, "model_seeds": list(MODEL_SEEDS), "privacy": privacy,
        "ordering": ordering, "usage": usage_value, "serving_gates": serving,
        "metrics": scores["metrics"], "episode_metrics": scores["episode_metrics"],
        "calibration_gates": gates, "daily_budget_status": dict(daily_budget_status or {}),
        "development_confirmation_reserve_opened": False,
        "repair_or_task_success_endpoints_opened": False,
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result
