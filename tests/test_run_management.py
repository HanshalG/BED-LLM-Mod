import json
from pathlib import Path

import numpy as np

from helpers import Config, ModelPair, ModelSpec
from run_management import (
    create_run_context,
    default_run_name,
    item_base_metadata,
    sanitize_path_name,
    set_item_metrics,
    write_json,
)


def test_sanitize_path_name_and_default_run_name():
    assert sanitize_path_name(" Gemma 4 / EIG: animals ") == "Gemma-4-EIG-animals"
    assert sanitize_path_name("...") == "run"
    assert default_run_name(Path("configs/config_gemma4.yaml")) == "config_gemma4"


def test_create_run_context_writes_initial_metadata(tmp_path):
    config_path = tmp_path / "configs" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text("model_pairs: []\n", encoding="utf-8")

    context = create_run_context(
        output_root=tmp_path / "runs",
        run_id="123456",
        run_name="Smoke Test",
        config_path=config_path,
        task="animals",
        cwd=tmp_path,
    )

    assert context.run_dir == tmp_path / "runs" / "123456_Smoke-Test"
    assert context.log_path == context.run_dir / "run.log"
    assert context.config_snapshot_path == context.run_dir / "config.resolved.json"
    metadata = json.loads(context.metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_id"] == "123456"
    assert metadata["run_name"] == "Smoke-Test"
    assert metadata["task"] == "animals"
    assert metadata["status"] == "running"
    assert metadata["items"] == []


def test_run_context_creates_deterministic_item_directories_and_metrics(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_pairs: []\n", encoding="utf-8")
    context = create_run_context(
        output_root=tmp_path / "runs",
        run_id="run123",
        run_name=None,
        config_path=config_path,
        task="animals",
        cwd=tmp_path,
    )
    config = Config(
        version=2,
        task="animals",
        belief_state_mode="categorical",
        search_depth=2,
    )
    pair = ModelPair(
        questioner=ModelSpec(model="Qwen/Qwen3.5-4B", thinking=False),
        answerer=ModelSpec(model="google/gemma-4-E4B-it", thinking=True),
    )

    item = context.new_item(0, "EIG", item_base_metadata(config, "EIG", pair))
    set_item_metrics(item, {"accuracy": np.array([1.0, 0.5])})
    context.write_metrics()

    assert item.item_id == "000_EIG"
    assert item.item_dir == context.run_dir / "items" / "000_EIG"
    item_metadata = json.loads(item.metadata_path.read_text(encoding="utf-8"))
    assert item_metadata["method"] == "EIG"
    assert item_metadata["questioner"]["model"] == "Qwen/Qwen3.5-4B"
    metrics = json.loads(context.metrics_path.read_text(encoding="utf-8"))
    assert metrics["items"][0]["metrics"]["accuracy"] == [1.0, 0.5]


def test_write_json_serializes_paths_dataclasses_and_numpy(tmp_path):
    output = tmp_path / "payload.json"
    payload = {
        "path": tmp_path / "artifact.npy",
        "spec": ModelSpec(model="Qwen/Qwen3.5-4B", thinking=False),
        "values": np.array([1, 2, 3]),
    }

    write_json(output, payload)

    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["path"] == str(tmp_path / "artifact.npy")
    assert parsed["spec"]["thinking"] is False
    assert parsed["values"] == [1, 2, 3]
