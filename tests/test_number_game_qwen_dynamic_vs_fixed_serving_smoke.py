from __future__ import annotations

from scripts import number_game_qwen_dynamic_vs_fixed_serving_smoke as smoke


def test_configured_smoke_uses_fresh_interface_and_seeds(monkeypatch, tmp_path):
    observed = {}

    def fake_run_smoke(*, output_dir, run_id, adapter=None):
        observed.update(
            {
                "output_dir": output_dir,
                "run_id": run_id,
                "adapter": adapter,
                "interface_version": smoke.base.INTERFACE_VERSION,
                "model_seeds": smoke.base.MODEL_SEEDS,
            }
        )
        return {"status": "passed"}

    original_interface = smoke.base.INTERFACE_VERSION
    original_seeds = smoke.base.MODEL_SEEDS
    monkeypatch.setattr(smoke.base, "run_smoke", fake_run_smoke)

    result = smoke.run_smoke(
        output_dir=tmp_path,
        run_id="fresh-smoke",
        adapter="fake",
    )

    assert result == {"status": "passed"}
    assert observed["interface_version"] == smoke.INTERFACE_VERSION
    assert observed["model_seeds"] == smoke.MODEL_SEEDS
    assert observed["adapter"] == "fake"
    assert smoke.base.INTERFACE_VERSION == original_interface
    assert smoke.base.MODEL_SEEDS == original_seeds
