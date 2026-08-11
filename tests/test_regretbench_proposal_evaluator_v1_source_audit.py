from __future__ import annotations

import copy

from scripts import regretbench_proposal_evaluator_v1_source_audit as audit


def test_source_audit_passes_and_freezes_only_untouched_pair(tmp_path):
    result = audit.write_audit(tmp_path)
    assert result["status"] == "source_protocol_pass"
    assert result["authorizes"] == "proposal_evaluator_v1_exact20_smoke_only"
    assert result["classification"]["paid_root_positions"] == [0, 1]
    assert result["classification"]["untouched_positions"] == [2, 3]
    assert result["model_calls_made"] == 0
    assert result["endpoint_outcomes_opened"] is False


def test_prompt_classification_rejects_reuse_of_successor_hash():
    roots = [
        {"root_payload_sha256": f"root-{index}"}
        for index in range(4)
    ]
    paid = [{"payload_sha256": "root-0"}, {"payload_sha256": "root-1"}]
    paid.extend({"payload_sha256": f"transition-{index}"} for index in range(4))
    passed = audit.classify_root_payloads(roots, paid)
    assert passed["paid_root_positions"] == [0, 1]
    assert passed["untouched_positions"] == [2, 3]

    leaked = copy.deepcopy(paid)
    leaked[-1]["payload_sha256"] = "root-2"
    failed = audit.classify_root_payloads(roots, leaked)
    assert failed["positions_two_three_absent_from_all_paid_payloads"] is False
    assert failed["untouched_positions"] == [3]
