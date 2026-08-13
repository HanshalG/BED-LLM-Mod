from __future__ import annotations

from scripts import number_game_fresh_control_mechanics_localization as localize


def _draw(valid: int, **rejected: int) -> dict[str, object]:
    counts = {key: 0 for key in localize.EXPECTED_REJECTION_KEYS}
    counts.update(rejected)
    return {
        "codec_mode": "strict_json",
        "valid_unique_count": valid,
        "rejected": counts,
    }


def test_failure_attribution_precedence() -> None:
    malformed = _draw(15, missing_or_incomplete=1)
    assert localize.failure_attribution(
        draw_diagnostics=[malformed, _draw(24)], pool_count=30
    ) == "codec_or_incomplete"
    assert localize.failure_attribution(
        draw_diagnostics=[_draw(15, duplicate_extension=9), _draw(24)],
        pool_count=30,
    ) == "duplicate_collapse"
    assert localize.failure_attribution(
        draw_diagnostics=[_draw(15, invalid_expression=9), _draw(24)],
        pool_count=30,
    ) == "invalid_rule"
    assert localize.failure_attribution(
        draw_diagnostics=[_draw(16), _draw(16)], pool_count=23
    ) == "cross_draw_overlap"


def test_quantile_uses_linear_interpolation() -> None:
    assert localize.quantile([1, 2, 3, 4], 0.5) == 2.5
    assert localize.quantile([1, 2, 3, 4], 0.0) == 1.0
    assert localize.quantile([1, 2, 3, 4], 1.0) == 4.0
