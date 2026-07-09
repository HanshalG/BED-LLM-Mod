from pathlib import Path

from scripts.path_a_preflight import _check_configs, _check_split_tools, run_preflight


ROOT = Path(__file__).resolve().parents[1]


def test_path_a_preflight_passes_local_launch_readiness_checks():
    payload = run_preflight(ROOT)

    assert payload["ok"] is True
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["configs"]["ok"] is True
    assert "26B-A4B" in checks["configs"]["detail"]
    assert checks["gh200_launcher"]["ok"] is True
    assert checks["split_tools"]["ok"] is True
    assert checks["launch_commands"]["ok"] is True
    assert "split-MPP30" in checks["launch_commands"]["detail"]
    assert "oat12 excluded" in checks["launch_commands"]["detail"]
    assert "package artifacts listed" in checks["launch_commands"]["detail"]
    assert payload["paper_validation"]["ok"] is True
    assert any(
        check["name"] == "paper_page_count" and check["detail"] == "6 pages"
        for check in payload["paper_validation"]["checks"]
    )
    assert payload["ledger_validation"]["ok"] is True
    assert any(
        check["name"] == "required_cost_vs_depth" and check["ok"] is True
        for check in payload["ledger_validation"]["checks"]
    )
    assert payload["package_validation"]["ok"] is False


def test_path_a_preflight_reports_missing_configs(tmp_path):
    check = _check_configs(tmp_path)

    assert check.ok is False
    assert "missing" in check.detail


def test_path_a_preflight_reports_missing_split_tools(tmp_path):
    check = _check_split_tools(tmp_path)

    assert check.ok is False
    assert "combine_location_fixed_root_depth_sweeps.py" in check.detail
