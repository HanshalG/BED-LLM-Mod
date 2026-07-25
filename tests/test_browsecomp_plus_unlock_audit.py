from __future__ import annotations

from scripts.browsecomp_plus_unlock_audit import (
    decrypt_string,
    derive_key,
    root_queries,
    summarize_records,
)


def test_decryption_matches_xor_reference() -> None:
    import base64

    plaintext = "private query text"
    password = "test password"
    key = derive_key(password, len(plaintext.encode("utf-8")))
    ciphertext = bytes(
        value ^ key[index]
        for index, value in enumerate(plaintext.encode("utf-8"))
    )
    encoded = base64.b64encode(ciphertext).decode("ascii")
    assert decrypt_string(encoded, password) == plaintext


def test_root_queries_are_unique_deterministic_clauses_with_fallbacks() -> None:
    query = (
        "Find the inventor, who worked in optics, and identify the year; "
        "then compare that date with the later patent, which lists a company."
    )
    roots = root_queries(query)
    assert roots[0] == query
    assert 3 <= len(roots) <= 5
    assert len(set(roots)) == len(roots)
    assert all(len(root.split()) >= 2 for root in roots)


def _record(
    *,
    root_count: int = 5,
    diversity: int = 3,
    pair_gain: int = 1,
    changed: bool = True,
    gap: int = 1,
    evidence_count: int = 5,
    direct: int = 0,
    gold: int = 1,
) -> dict[str, int]:
    return {
        "root_count": root_count,
        "distinct_first_results": diversity,
        "pair_gain": pair_gain,
        "oracle_root_index": 1 if changed else 0,
        "immediate_root_index": 0,
        "nonmyopic_gap": gap,
        "evidence_count": evidence_count,
        "direct_query_evidence_utility": direct,
        "oracle_pair_gold_utility": gold,
    }


def test_summary_passes_frozen_thresholds_exactly() -> None:
    records = [
        _record(
            root_count=3 if index < 16 else 2,
            diversity=3 if index < 10 else 2,
            pair_gain=1 if index < 8 else 0,
            changed=index < 4,
            gap=1 if index < 3 else 0,
        )
        for index in range(20)
    ]
    summary = summarize_records(records)
    assert summary["analyzable_count"] == 16
    assert summary["mean_distinct_first_results"] == 2.5
    assert summary["pair_gain_count"] == 8
    assert summary["mean_pair_gain"] == 0.4
    assert summary["root_change_count"] == 4
    assert summary["nonmyopic_gap_count"] == 3
    assert summary["mean_nonmyopic_gap"] == 0.15
    assert summary["passed"] is True


def test_summary_fails_saturated_direct_coverage() -> None:
    records = [_record(direct=5) for _ in range(20)]
    summary = summarize_records(records)
    assert summary["direct_query_evidence_coverage"] == 1.0
    assert summary["gates"]["direct_coverage_below_0_60"] is False
    assert summary["passed"] is False
