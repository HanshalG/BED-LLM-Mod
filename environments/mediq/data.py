"""Load the pinned official MediQ data release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .types import MediQTask


MEDIQ_REPOSITORY = "https://github.com/stellali7/MediQ.git"
MEDIQ_COMMIT = "faa2ce62fef0423e35af4c31d7537aad973173eb"
MEDIQ_IMEDQA_DEV_RELATIVE_PATH = Path("data/all_dev_good.jsonl")
MEDIQ_IMEDQA_DEV_SHA256 = (
    "3bfc7090d060dd8d11e4237344ed78846707faab433a84d078191627ad3c9526"
)
MEDIQ_ICRAFT_MD_RELATIVE_PATH = Path("data/all_craft_md.jsonl")
MEDIQ_ICRAFT_MD_SHA256 = (
    "658441e6c6692c84d78fdf1c5ddb406ee6705f4a43330f0f759a1b941d8bfcdc"
)

_OFFICIAL_HASHES = {
    "imedqa": MEDIQ_IMEDQA_DEV_SHA256,
    "icraft_md": MEDIQ_ICRAFT_MD_SHA256,
}


def _require_text(row: dict[str, Any], key: str, index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MediQ row {index} field {key!r} must be non-empty text")
    return value.strip()


def _clean_fact(value: Any, *, row_index: int, fact_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"MediQ row {row_index} fact {fact_index} must be non-empty text"
        )
    return re.sub(r"^\s*\d+\s*[.)]\s*", "", value).strip()


def load_mediq_tasks(
    path: str | Path,
    *,
    dataset: str = "imedqa",
    verify_official_hash: bool = False,
    skip_unusable_tasks: bool = True,
) -> list[MediQTask]:
    """Load one official JSONL file without exposing hidden context initially."""
    tasks, _excluded_source_ids, _raw_row_count = load_mediq_tasks_with_report(
        path,
        dataset=dataset,
        verify_official_hash=verify_official_hash,
        skip_unusable_tasks=skip_unusable_tasks,
    )
    return tasks


def load_mediq_tasks_with_report(
    path: str | Path,
    *,
    dataset: str = "imedqa",
    verify_official_hash: bool = False,
    skip_unusable_tasks: bool = True,
) -> tuple[list[MediQTask], tuple[str, ...], int]:
    """Load tasks and report official rows lacking an interactive patient record."""
    if dataset not in _OFFICIAL_HASHES:
        raise ValueError("MediQ dataset must be one of: imedqa, icraft_md")
    source = Path(path)
    raw_bytes = source.read_bytes()
    if verify_official_hash:
        digest = hashlib.sha256(raw_bytes).hexdigest()
        expected = _OFFICIAL_HASHES[dataset]
        if digest != expected:
            raise ValueError(
                f"MediQ {dataset} file hash mismatch: expected {expected}, got {digest}"
            )

    tasks: list[MediQTask] = []
    excluded_source_ids: list[str] = []
    seen_ids: set[str] = set()
    rows = [line for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    for index, line in enumerate(rows):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"MediQ row {index} must be an object")
        source_id = str(row.get("id", index))
        task_id = f"mediq:{dataset}:{source_id}"
        if task_id in seen_ids:
            raise ValueError(f"MediQ task ID {task_id!r} is duplicated")
        seen_ids.add(task_id)

        raw_options = row.get("options")
        if not isinstance(raw_options, dict) or len(raw_options) < 2:
            raise ValueError(f"MediQ row {index} requires at least two options")
        options: list[tuple[str, str]] = []
        for raw_label, raw_text in raw_options.items():
            label = str(raw_label).strip().upper()
            if len(label) != 1 or not label.isalpha():
                raise ValueError(f"MediQ row {index} has invalid option label {raw_label!r}")
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError(f"MediQ row {index} option {label} must be non-empty")
            options.append((label, raw_text.strip()))
        options.sort()
        if len({label for label, _text in options}) != len(options):
            raise ValueError(f"MediQ row {index} has duplicate option labels")

        answer_idx = _require_text(row, "answer_idx", index).upper()
        answer = _require_text(row, "answer", index)
        option_map = dict(options)
        if answer_idx not in option_map:
            raise ValueError(f"MediQ row {index} answer_idx is not an option")
        if option_map[answer_idx].casefold() != answer.casefold():
            raise ValueError(f"MediQ row {index} answer text does not match answer_idx")

        raw_context = row.get("context")
        if isinstance(raw_context, str):
            context = (raw_context.strip(),) if raw_context.strip() else ()
        elif isinstance(raw_context, list):
            context = tuple(
                str(item).strip() for item in raw_context if str(item).strip()
            )
        else:
            context = ()
        raw_facts = row.get("facts", row.get("atomic_facts"))
        usable_facts = (
            isinstance(raw_facts, list)
            and bool(raw_facts)
            and all(isinstance(value, str) and value.strip() for value in raw_facts)
        )
        if not context or not usable_facts:
            if skip_unusable_tasks:
                excluded_source_ids.append(source_id)
                continue
            missing = "context" if not context else "atomic facts"
            raise ValueError(f"MediQ row {index} requires non-empty {missing}")
        initial_value = row.get("initial_info", context[0])
        if not isinstance(initial_value, str) or not initial_value.strip():
            raise ValueError(f"MediQ row {index} initial information is empty")

        facts = tuple(
            _clean_fact(value, row_index=index, fact_index=fact_index)
            for fact_index, value in enumerate(raw_facts)
        )
        tasks.append(
            MediQTask(
                task_id=task_id,
                source_id=source_id,
                dataset=dataset,
                question=_require_text(row, "question", index),
                options=tuple(options),
                answer_idx=answer_idx,
                answer=answer,
                initial_info=initial_value.strip(),
                context=context,
                facts=facts,
            )
        )
    if not tasks:
        raise ValueError("MediQ data file contains no tasks")
    return tasks, tuple(excluded_source_ids), len(rows)
