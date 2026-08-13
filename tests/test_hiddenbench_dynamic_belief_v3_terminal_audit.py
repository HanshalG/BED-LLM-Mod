from __future__ import annotations

import inspect
import json
from pathlib import Path

from scripts import hiddenbench_dynamic_belief_v3_terminal_audit as audit


def test_terminal_audit_passes_and_keeps_endpoints_closed() -> None:
    result = audit.audit()
    assert result["status"] == "terminal_audit_pass"
    assert result["authorizes"] == "nothing"
    assert result["strict_root_envelopes"] == 2
    assert result["schema_echoes"] == 2
    assert result["charged_responses"] == 2
    assert result["zero_cost_error_responses"] == 2
    assert result["registered_answers_opened"] is False
    assert result["endpoint_scores_opened"] is False
    assert all(result["gates"].values())


def test_terminal_audit_is_independent_of_v3_producer_and_endpoint_modules() -> None:
    source = inspect.getsource(audit)
    assert "hiddenbench_dynamic_belief_v3_serving" not in source
    assert "hiddenbench_dynamic_belief_v3_endpoint" not in source


def test_terminal_audit_rejects_tampered_failure(
    tmp_path: Path,
) -> None:
    failure = tmp_path / "failure.json"
    failure.write_bytes(audit.FAILURE.read_bytes())
    value = json.loads(failure.read_text())
    value["authorizes"] = "endpoint"
    failure.write_text(json.dumps(value))
    result = audit.audit(failure_path=failure)
    assert result["status"] == "terminal_audit_failed"
    assert result["gates"]["bound_terminal_artifacts"] is False
    assert result["gates"]["failure_is_non_authorizing"] is False
