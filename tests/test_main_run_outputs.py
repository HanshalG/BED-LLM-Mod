import importlib
import json
import sys
import types

import numpy as np


def test_main_writes_one_run_directory_without_legacy_outputs(monkeypatch, tmp_path):
    fake_wandb = types.ModuleType("wandb")
    wandb_init_calls = []
    wandb_log_calls = []
    fake_wandb.init = lambda **kwargs: wandb_init_calls.append(kwargs)
    fake_wandb.log = lambda payload: wandb_log_calls.append(payload)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    fake_model = types.ModuleType("model")
    fake_model.build_model_adapter = lambda spec, config: f"adapter:{spec.model}"
    monkeypatch.setitem(sys.modules, "model", fake_model)

    from core.experiment_summary import ExperimentSummary

    def _fake_run_from_config(*args, **kwargs):
        return (
            types.SimpleNamespace(trials=()),
            ExperimentSummary(
                metrics={
                    "accuracy": [1, 0, 1],
                    "correct_belief_mass": [0.8, 0.6, 1.0],
                }
            ),
        )

    monkeypatch.setattr("core.experiment.run_from_config", _fake_run_from_config)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: 0
animals:
  - ["cat"]
model_pairs:
  - questioner:
      model: "Qwen/Qwen3.5-4B"
      thinking: false
    answerer:
      model: "Qwen/Qwen3.5-4B"
      thinking: false
method_names:
  - "EIG"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("SLURM_JOB_ID", "999")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "-c",
            str(config_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-name",
            "Smoke Run",
        ],
    )

    main_module = importlib.import_module("main")
    importlib.reload(main_module)
    main_module.main()

    run_dir = tmp_path / "runs" / "999_Smoke-Run"
    item_dir = run_dir / "items" / "000_EIG"
    assert run_dir.is_dir()
    assert (run_dir / "run.log").exists()
    assert (run_dir / "config.resolved.json").exists()
    assert np.load(item_dir / "accuracy.npy").tolist() == [1, 0, 1]
    assert np.load(item_dir / "correct_belief_mass.npy").tolist() == [0.8, 0.6, 1.0]
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "results").exists()

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    item_metadata = json.loads((item_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["run_id"] == "999"
    assert metadata["run_name"] == "Smoke-Run"
    assert metadata["items"][0]["artifacts"]["accuracy"] == "items/000_EIG/accuracy.npy"
    assert item_metadata["questioner"]["model"] == "Qwen/Qwen3.5-4B"
    assert metrics["items"][0]["metrics"]["accuracy"] == [1, 0, 1]
    assert wandb_init_calls[0]["name"] == "999_Smoke-Run"
    assert wandb_init_calls[0]["config"]["run_dir"] == str(run_dir.resolve())
    assert wandb_log_calls
