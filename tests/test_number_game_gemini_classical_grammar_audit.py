from __future__ import annotations

from scripts import number_game_gemini_classical_grammar_audit as audit


def test_frozen_gemini_evidence_has_two_disjoint_32_tree_studies() -> None:
    sources, endpoints, hashes = audit.load_frozen_evidence()

    assert set(sources) == {
        "fixed_policy_fresh_endpoints",
        "fresh_tree_replication",
    }
    assert len(endpoints["trees"]) == 64
    assert {
        row["source_study"] for row in endpoints["trees"]
    } == set(sources)
    assert all(len(row["supports"]) == 16 for row in endpoints["trees"])
    assert len(hashes) == 7


def test_prior_audit_binds_expected_bank_and_support_novelty() -> None:
    prior = audit.verify_prior_audit()

    assert prior["bank"]["sha256"] == audit.EXPECTED_BANK_SHA256
    assert prior["support_novelty"]["all_gates_pass"]


def test_audit_status_is_fail_closed() -> None:
    assert (
        audit.audit_status(
            bank_gate=False,
            endpoint_gate=True,
            efficacy_gate=True,
        )
        == "mechanics_failure"
    )
    assert (
        audit.audit_status(
            bank_gate=True,
            endpoint_gate=False,
            efficacy_gate=True,
        )
        == "inconclusive"
    )
    assert (
        audit.audit_status(
            bank_gate=True,
            endpoint_gate=True,
            efficacy_gate=False,
        )
        == "negative"
    )
    assert (
        audit.audit_status(
            bank_gate=True,
            endpoint_gate=True,
            efficacy_gate=True,
        )
        == "positive_irreducibility_audit"
    )
