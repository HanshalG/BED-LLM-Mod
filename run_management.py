from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._+-]+")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_path_name(name: str) -> str:
    sanitized = _SAFE_NAME_PATTERN.sub("-", name.strip())
    sanitized = sanitized.strip(".-_")
    return sanitized or "run"


def default_run_name(config_path: Path) -> str:
    return sanitize_path_name(config_path.stem)


def run_directory(output_root: Path, run_id: str, run_name: str) -> Path:
    return output_root / f"{sanitize_path_name(run_id)}_{sanitize_path_name(run_name)}"


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _json_ready(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_commit(cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


@dataclass
class RunItem:
    item_id: str
    item_dir: Path
    metadata: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def metadata_path(self) -> Path:
        return self.item_dir / "metadata.json"

    def write_metadata(self) -> None:
        write_json(self.metadata_path, self.metadata)


@dataclass
class RunContext:
    run_id: str
    run_name: str
    run_dir: Path
    config_path: Path
    task: str | None
    started_at: str
    git_commit: str | None
    slurm_job_id: str | None
    items: list[RunItem] = field(default_factory=list)
    status: str = "running"
    completed_at: str | None = None
    error: str | None = None

    @property
    def log_path(self) -> Path:
        return self.run_dir / "run.log"

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def config_snapshot_path(self) -> Path:
        return self.run_dir / "config.resolved.json"

    def relative_to_run(self, path: Path) -> str:
        return path.relative_to(self.run_dir).as_posix()

    def new_item(self, index: int, method_name: str, metadata: dict[str, Any]) -> RunItem:
        item_id = f"{index:03d}_{sanitize_path_name(method_name)}"
        item_dir = self.run_dir / "items" / item_id
        item_dir.mkdir(parents=True, exist_ok=True)
        item = RunItem(
            item_id=item_id,
            item_dir=item_dir,
            metadata={
                "item_id": item_id,
                "run_id": self.run_id,
                "run_name": self.run_name,
                **metadata,
            },
        )
        self.items.append(item)
        item.write_metadata()
        self.write_metadata()
        return item

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "run_dir": str(self.run_dir.resolve()),
            "config_path": str(self.config_path.resolve()),
            "task": self.task,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "git_commit": self.git_commit,
            "slurm_job_id": self.slurm_job_id,
            "error": self.error,
            "items": [
                {
                    **item.metadata,
                    "item_dir": self.relative_to_run(item.item_dir),
                    "metadata_path": self.relative_to_run(item.metadata_path),
                }
                for item in self.items
            ],
        }

    def metrics_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "items": [
                {
                    **item.metadata,
                    "item_dir": self.relative_to_run(item.item_dir),
                    "metrics": item.metrics,
                }
                for item in self.items
            ],
        }

    def write_metadata(self) -> None:
        write_json(self.metadata_path, self.metadata_payload())

    def write_metrics(self) -> None:
        write_json(self.metrics_path, self.metrics_payload())

    def write_config_snapshot(self, config: Any) -> None:
        write_json(self.config_snapshot_path, config)

    def finish(self, status: str = "completed", error: str | None = None) -> None:
        self.status = status
        self.completed_at = utc_now_iso()
        self.error = error
        self.write_metadata()
        self.write_metrics()


def create_run_context(
    output_root: Path,
    run_id: str,
    run_name: str | None,
    config_path: Path,
    task: str | None = None,
    cwd: Path | None = None,
) -> RunContext:
    resolved_run_name = sanitize_path_name(run_name) if run_name else default_run_name(config_path)
    run_dir = run_directory(output_root, run_id, resolved_run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "items").mkdir(exist_ok=True)
    context = RunContext(
        run_id=run_id,
        run_name=resolved_run_name,
        run_dir=run_dir,
        config_path=config_path,
        task=task,
        started_at=utc_now_iso(),
        git_commit=git_commit(cwd),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    context.write_metadata()
    return context


def add_item_artifact(item: RunItem, artifact_name: str, path: Path, run_context: RunContext) -> None:
    artifacts = item.metadata.setdefault("artifacts", {})
    artifacts[artifact_name] = run_context.relative_to_run(path)
    item.write_metadata()


def set_item_metrics(item: RunItem, metrics: dict[str, Any]) -> None:
    item.metrics = metrics
    item.write_metadata()


def model_spec_metadata(spec: Any) -> dict[str, Any]:
    return _json_ready(spec)


def item_search_depth(config: Any) -> int:
    if getattr(config, "task", "animals") == "location_finding":
        return config.location_search_depth
    return config.search_depth


def item_base_metadata(config: Any, method_name: str, pair: Any) -> dict[str, Any]:
    return {
        "method": method_name,
        "task": config.task,
        "version": config.version,
        "belief_state_mode": config.belief_state_mode,
        "search_depth": item_search_depth(config),
        "questioner": model_spec_metadata(pair.questioner),
        "answerer": model_spec_metadata(pair.answerer),
        "artifacts": {},
    }
