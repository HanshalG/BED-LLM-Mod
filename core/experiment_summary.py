"""Task-level summaries produced from :class:`core.bed_runner.RunResult`."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentSummary:
    """Per-round metric series and optional log lines for main / W&B."""

    metrics: dict[str, list[float]] = field(default_factory=dict)
    logs: dict[str, str] | None = None
