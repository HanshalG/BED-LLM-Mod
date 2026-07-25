from pathlib import Path

from helpers import load_config
from scripts.clariq_dynamic_support_smoke import DeterministicFixtureModel
from scripts.clariq_dynamic_support_v2_serving import (
    MANIFEST_SHA256,
    load_manifest,
    run_serving,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "results/nonmyopic/clariq_dynamic_support_v2_manifest/MANIFEST.json"
)
CONFIG = ROOT / "configs/config_clariq_dynamic_support_openrouter.yaml"


def test_v2_manifest_binding_reproduces() -> None:
    assert MANIFEST_SHA256 == (
        "fa5a34e55ab455359a4a64bd2aba00ea5789f27fb2f2d932ea9e2330cf90ca03"
    )
    task = load_manifest(MANIFEST)
    assert task["topic_id"] == "60"
    assert task["root_count"] == 15
    assert task["branch_count"] == 90


def test_v2_serving_fixture_uses_one_call_and_no_endpoint(
    tmp_path: Path,
) -> None:
    config = load_config(str(CONFIG))
    payload = run_serving(
        config,
        manifest_path=MANIFEST,
        raw_path=tmp_path / "RAW.json",
        model=DeterministicFixtureModel(),
    )
    assert payload["usage"]["physical_requests"] == 1
    assert payload["protocol"]["branch_requests_sent"] == 0
    assert payload["protocol"]["endpoint_loaded"] is False
    assert payload["gates"]["support_parses"] is True
