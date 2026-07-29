from __future__ import annotations

import hashlib

from scripts import number_game_classical_grammar_irreducibility_audit as audit
from scripts.number_game_generator_aware_bed import compile_expression


def _item(expression: str) -> dict[str, object]:
    extension = compile_expression(expression)
    return {
        "name": expression,
        "expression": expression,
        "positive_count": sum(extension),
        "extension_sha256": hashlib.sha256(bytes(extension)).hexdigest(),
    }


def test_classical_grammar_bank_is_large_closed_and_deterministic() -> None:
    bank, diagnostics = audit.build_classical_grammar_bank()

    assert len(bank) >= audit.MIN_BANK_EXTENSIONS
    assert 0 not in bank
    assert audit.ALL_MASK not in bank
    assert all((audit.ALL_MASK ^ mask) in bank for mask in bank)
    assert diagnostics["sha256"] == audit.grammar_sha256(bank)
    assert diagnostics["core_atom_count"] > 300
    assert diagnostics["extended_atom_count"] > 1_000


def test_stage_summary_uses_extension_membership_not_syntax() -> None:
    even = _item("n % 2 == 0")
    even_alias = _item("divisible(n, 2)")
    prime = _item("is_prime(n)")
    bank = {audit.item_mask(even)}

    result = audit.stage_summary(
        [even, even_alias, prime],
        bank=bank,
    )

    assert result["occurrences"] == 3
    assert result["unique_extensions"] == 2
    assert result["grammar_novel_occurrences"] == 1
    assert result["grammar_novel_unique_extensions"] == 1


def test_nearest_bank_distance_counts_extension_bits() -> None:
    bank = {
        audit.extension_mask(compile_expression("n % 2 == 0")),
        audit.extension_mask(compile_expression("n % 2 != 0")),
    }
    target = bank.copy().pop() ^ (1 << 3) ^ (1 << 7)

    result = audit.nearest_bank_distances([target], bank=bank)

    assert result["minimum_hamming_count"] == 2
    assert result["median_hamming_count"] == 2
    assert result["fraction_at_least_two"] == 1.0


def test_status_distinguishes_power_from_efficacy() -> None:
    assert (
        audit.audit_status(
            bank_gate=True,
            support_gate=True,
            endpoint_gate=True,
            efficacy_gate=True,
        )
        == "positive_irreducibility_audit"
    )
    assert (
        audit.audit_status(
            bank_gate=True,
            support_gate=True,
            endpoint_gate=True,
            efficacy_gate=False,
        )
        == "mechanism_only"
    )
    assert (
        audit.audit_status(
            bank_gate=True,
            support_gate=True,
            endpoint_gate=False,
            efficacy_gate=False,
        )
        == "inconclusive"
    )
