"""Load released Paprika customer-service tasks without changing their semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .types import PaprikaTask


PAPRIKA_REPOSITORY = "https://github.com/tajwarfahim/paprika.git"
PAPRIKA_COMMIT = "8470461554f12f9f9215ee5e78ea19ba1d576b5c"
PAPRIKA_CUSTOMER_SERVICE_RELATIVE_PATH = Path(
    "llm_exploration/game/game_configs/customer_service.json"
)
PAPRIKA_CUSTOMER_SERVICE_SHA256 = (
    "0ebadf112db200501d90cda51aa9df618e889d408a6f4ecb7a8a6dcfac3d3c5e"
)


def _require_text(item: dict[str, Any], key: str, *, split: str, index: int) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Paprika {split}[{index}].{key} must be non-empty text")
    return value.strip()


def load_paprika_tasks(
    path: str | Path,
    *,
    split: str = "eval",
    verify_official_hash: bool = False,
) -> list[PaprikaTask]:
    """Load one released split and expose ``env`` as the hidden solution.

    ``verify_official_hash`` is intended for the complete pinned upstream file. Tests
    and local fixtures deliberately leave it disabled.
    """
    source = Path(path)
    raw_bytes = source.read_bytes()
    if verify_official_hash:
        digest = hashlib.sha256(raw_bytes).hexdigest()
        if digest != PAPRIKA_CUSTOMER_SERVICE_SHA256:
            raise ValueError(
                "Paprika customer-service file hash mismatch: "
                f"expected {PAPRIKA_CUSTOMER_SERVICE_SHA256}, got {digest}"
            )
    payload = json.loads(raw_bytes)
    if split not in {"train", "eval"}:
        raise ValueError("Paprika split must be 'train' or 'eval'")
    items = payload.get(split)
    if not isinstance(items, list) or not items:
        raise ValueError(f"Paprika payload must contain a non-empty {split!r} list")

    tasks: list[PaprikaTask] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Paprika {split}[{index}] must be an object")
        scenario = _require_text(item, "agent", split=split, index=index)
        solution = _require_text(item, "env", split=split, index=index)
        identity = (scenario.casefold(), solution.casefold())
        if identity in seen:
            raise ValueError(f"Paprika {split}[{index}] duplicates an earlier task")
        seen.add(identity)
        tasks.append(
            PaprikaTask(
                task_id=f"customer_service:{split}:{index:04d}",
                scenario=scenario,
                solution=solution,
                split=split,
            )
        )
    return tasks

