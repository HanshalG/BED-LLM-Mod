from __future__ import annotations

import pytest

from scripts.chembench_costed_repeat_legacy_medium_cutoff import (
    FROZEN_PID,
    parse_cpu_time,
    validate_legacy_process,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("00:00.00", 0.0),
        ("327:03.07", 327 * 60 + 3.07),
        ("05:27:03.07", 5 * 3600 + 27 * 60 + 3.07),
        ("2-05:27:03.07", 2 * 86400 + 5 * 3600 + 27 * 60 + 3.07),
    ),
)
def test_parse_cpu_time(value: str, expected: float) -> None:
    assert parse_cpu_time(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ("", "bad", "1:00:60.0", "1:60:00.0"))
def test_parse_cpu_time_rejects_malformed_values(value: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        parse_cpu_time(value)


def test_validate_legacy_process_requires_exact_frozen_identity() -> None:
    snapshot = {
        "pid": FROZEN_PID,
        "ps": (
            "python scripts/chembench_costed_repeat_corridor.py "
            "--required-commit d9559a2f --difficulty medium"
        ),
    }
    validate_legacy_process(snapshot)
    with pytest.raises(RuntimeError, match="no longer identifies"):
        validate_legacy_process({**snapshot, "ps": "python unrelated.py"})
