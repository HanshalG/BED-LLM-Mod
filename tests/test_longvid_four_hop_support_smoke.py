from __future__ import annotations

from pathlib import Path

from helpers import load_config
from scripts.longvid_four_hop_support_smoke import (
    DeterministicFixtureModel,
    FIXTURE,
    fixture_sha256,
    parse_support,
    run_smoke,
    valid_anchor_count,
)


def _support(anchor: str = "QUESTION", suffix: int = 0) -> str:
    return "\n".join(
        (
            f"H{index}|{10 + index}|{anchor}|"
            f"{anchor} search route {suffix} {index}|"
            f"Distinct evidence chain hypothesis {suffix} number {index}"
        )
        for index in range(1, 7)
    )


def test_fixture_hash_is_stable() -> None:
    assert fixture_sha256() == (
        "6bf2ccc7952689533b9bc609f53b75d1aa4a793ee233f63520e7cf2292cde7ad"
    )


def test_parse_support_rejects_extra_text_and_duplicate_queries() -> None:
    parsed = parse_support(_support())
    assert len(parsed) == 6
    try:
        parse_support("preface\n" + _support())
    except ValueError as exc:
        assert "exactly 6 lines" in str(exc)
    else:
        raise AssertionError("extra text should fail")
    duplicate = "\n".join(
        f"H{i}|{10+i}|QUESTION|same query|Distinct hypothesis number {i}"
        for i in range(1, 7)
    )
    try:
        parse_support(duplicate)
    except ValueError as exc:
        assert "search queries must be distinct" in str(exc)
    else:
        raise AssertionError("duplicate queries should fail")


def test_anchor_validation_requires_visible_novel_query_token() -> None:
    branch = FIXTURE["branches"][0]
    support = parse_support(_support(anchor="astrolabe"))
    assert valid_anchor_count(support, branch) == 6
    invalid = parse_support(_support(anchor="Mira", suffix=1))
    assert valid_anchor_count(invalid, branch) == 0


def test_dry_smoke_passes_exact_ten_request_gates(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        str(
            root
            / "configs"
            / "config_longvid_four_hop_support_smoke_openrouter.yaml"
        )
    )
    model = DeterministicFixtureModel()
    payload = run_smoke(
        config,
        raw_path=tmp_path / "raw.json",
        model=model,
    )
    assert payload["status"] == "passed"
    assert payload["usage"]["physical_requests"] == 10
    assert payload["diagnostics"]["unique_branch_support_count"] == 8
    assert payload["gates"]["all_pass"]
