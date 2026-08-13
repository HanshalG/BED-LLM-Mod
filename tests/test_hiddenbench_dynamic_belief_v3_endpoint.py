from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.hiddenbench_dynamic_belief_v3_endpoint_custodian import endpoint_rows
from scripts.hiddenbench_dynamic_belief_v3_pass_token import create_token, verify_token


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


def test_endpoint_mapping_uses_only_synthetic_rows() -> None:
    rows = [{"possible_answers": ["red", "blue", "green"], "correct_answer": "green"} for _ in range(4)]
    assert endpoint_rows(rows) == {"endpoints": [{"slot": f"T{i + 1}", "correct_option_id": "O3"} for i in range(4)]}


def test_pass_token_binds_every_label_free_artifact(tmp_path: Path) -> None:
    paths = {name: tmp_path / name for name in ("binding", "manifest", "audit", "raw", "result", "verification")}
    write(paths["binding"], {"binding": 1}); write(paths["manifest"], {"manifest": 1}); write(paths["audit"], {"audit": 1}); write(paths["raw"], {"raw": 1})
    import hashlib
    raw_hash = hashlib.sha256(paths["raw"].read_bytes()).hexdigest()
    write(paths["result"], {"status": "serving_pass", "authorizes": "endpoint_only", "gates": {"all": True}, "raw_response_sha256": raw_hash, "registered_answers_opened": False, "endpoint_scores_opened": False})
    write(paths["verification"], {"status": "verification_pass", "gates": {"all": True}, "registered_answers_loaded": False})
    token = create_token(execution_binding=paths["binding"], source_manifest=paths["manifest"], source_audit=paths["audit"], raw_responses=paths["raw"], label_free_result=paths["result"], verification=paths["verification"])
    token_path = tmp_path / "token"; write(token_path, token)
    assert verify_token(token_path, execution_binding=paths["binding"], source_manifest=paths["manifest"], source_audit=paths["audit"], raw_responses=paths["raw"], label_free_result=paths["result"], verification=paths["verification"])["status"] == "label_free_pass"
    write(paths["raw"], {"raw": 2})
    with pytest.raises(RuntimeError):
        verify_token(token_path, execution_binding=paths["binding"], source_manifest=paths["manifest"], source_audit=paths["audit"], raw_responses=paths["raw"], label_free_result=paths["result"], verification=paths["verification"])


def test_real_source_endpoint_invocation_is_absent_outside_dated_wrapper() -> None:
    offenders = []
    for path in Path("scripts").glob("hiddenbench_dynamic_belief_v3*.py"):
        if path.name in {"hiddenbench_dynamic_belief_v3_endpoint_custodian.py", "hiddenbench_dynamic_belief_v3_aug13_execute.py"}:
            continue
        text = path.read_text()
        if "endpoint_custodian.py" in text or "endpoint_rows(select" in text:
            offenders.append(path.name)
    assert offenders == []
