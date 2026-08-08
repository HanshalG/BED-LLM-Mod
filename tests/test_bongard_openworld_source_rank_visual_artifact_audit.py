from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import bongard_openworld_source_rank_visual_artifact_audit as audit


def test_image_features_are_finite_and_exact_width() -> None:
    output = BytesIO()
    Image.new("RGB", (37, 29), color=(30, 90, 170)).save(output, format="PNG")
    features = audit.image_features(output.getvalue())
    assert features.shape == (len(audit.FEATURE_NAMES),)
    assert np.isfinite(features).all()


def test_full_frozen_source_rank_visual_artifact_audit_passes(
    tmp_path: Path,
) -> None:
    result = audit.run_audit(output_path=tmp_path / "MANIFEST.json")
    assert result["status"] == "source_rank_visual_artifact_pass"
    assert result["all_gates_pass"] is True
    assert result["authorizes_paid_calls"] is False
    assert result["model_calls"] == 0
    assert result["cost_usd"] == 0.0
    assert result["feature_protocol"]["total_features"] == 42
    assert result["feature_protocol"]["selected_l2"] in audit.L2_GRID
    confirmation = result["confirmation"]
    assert confirmation["rows"] == 960
    assert confirmation["endpoint_rows"] == 192
    assert confirmation["auc"] < audit.MAX_CONFIRMATION_AUC
    assert confirmation["maximum_probability"] < audit.MAX_CONFIRMATION_PROBABILITY
