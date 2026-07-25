from __future__ import annotations

import json

from scripts.ambigdocs_binary_serving_smoke import (
    DEV_SHA256,
    EXPECTED_QID,
    ROW_INDEX,
    verify_source,
)


def test_verify_frozen_ambigdocs_source(tmp_path) -> None:
    rows = [{"qid": 0, "documents": []} for _ in range(3610)]
    rows[ROW_INDEX] = {
        "qid": EXPECTED_QID,
        "documents": [{"title": str(index), "text": "x"} for index in range(6)],
    }
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(rows))
    import scripts.ambigdocs_binary_serving_smoke as module

    original = module.DEV_SHA256
    try:
        module.DEV_SHA256 = module.hashlib.sha256(path.read_bytes()).hexdigest()
        assert verify_source(path)["qid"] == EXPECTED_QID
    finally:
        module.DEV_SHA256 = original
    assert DEV_SHA256.startswith("43ab72")
