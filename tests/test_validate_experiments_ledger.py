from scripts.validate_experiments_ledger import summary_payload, validate_experiments_ledger


HEADER = """# Experiments Ledger

| Date | Run / job | Config | Model | Status | Key metric / purpose | Artifacts | Commit / tag |
|---|---|---|---|---|---|---|---|
"""


def test_validate_experiments_ledger_passes_current_repo():
    payload = summary_payload(validate_experiments_ledger("EXPERIMENTS.md"))

    assert payload["ok"] is True
    names = {check["name"] for check in payload["checks"]}
    assert "required_ranking_fidelity" in names
    assert "required_cost_vs_depth" in names
    assert "complete_artifacts_exist" in names


def test_validate_experiments_ledger_rejects_active_status_rows(tmp_path):
    (tmp_path / "artifact.md").write_text("ok", encoding="utf-8")
    ledger = tmp_path / "EXPERIMENTS.md"
    ledger.write_text(
        HEADER
        + "| 2026-07-09 | `job` | cfg | model | Running on `gh200` | purpose | `artifact.md` | abc123 |\n",
        encoding="utf-8",
    )

    payload = summary_payload(validate_experiments_ledger(ledger, root=tmp_path))

    assert payload["ok"] is False
    active = next(check for check in payload["checks"] if check["name"] == "no_active_status_rows")
    assert active["ok"] is False
    assert "`job`" in active["detail"]


def test_validate_experiments_ledger_checks_complete_artifact_paths(tmp_path):
    ledger = tmp_path / "EXPERIMENTS.md"
    ledger.write_text(
        HEADER
        + "| 2026-07-09 | `done` | cfg | model | Complete | purpose | `missing.md` | abc123 |\n",
        encoding="utf-8",
    )

    payload = summary_payload(validate_experiments_ledger(ledger, root=tmp_path))

    assert payload["ok"] is False
    artifacts = next(check for check in payload["checks"] if check["name"] == "complete_artifacts_exist")
    assert artifacts["ok"] is False
    assert "missing.md" in artifacts["detail"]
