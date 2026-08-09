from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import bongard_openworld_siglip_embedding_extract as extract
from scripts import bongard_openworld_vlm_bed as bed


def _image_bytes(value: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (20, 18), color=(value, value, value)).save(
        output, format="PNG"
    )
    return output.getvalue()


def _task(task_id: str) -> bed.VisualTask:
    image_ids = tuple(f"image-{index:02d}" for index in range(14))
    return bed.VisualTask(
        task_id=task_id,
        image_ids=image_ids,
        initial_history=((image_ids[0], True), (image_ids[7], False)),
        candidate_ids=image_ids[1:7] + image_ids[8:10],
        endpoint_ids=(image_ids[10], image_ids[11]),
        image_bytes={
            image_id: _image_bytes(index)
            for index, image_id in enumerate(image_ids)
        },
        actual_labels={},
    )


def test_extract_embeddings_is_label_free_and_checks_normalization() -> None:
    tasks = {
        partition: [_task(f"{partition}-task-{index}") for index in range(size)]
        for partition, size in extract.PARTITION_SIZES.items()
    }

    def embed(images: list[bytes]) -> np.ndarray:
        values = np.zeros((len(images), extract.EMBEDDING_DIMENSION), dtype=np.float32)
        values[:, 0] = 1.0
        return values

    rows, embeddings = extract.extract_embeddings(tasks, embed, batch_size=17)
    assert len(rows) == sum(extract.PARTITION_SIZES.values()) * bed.NUM_IMAGES
    assert embeddings.shape == (len(rows), extract.EMBEDDING_DIMENSION)
    assert all(
        set(row) == {"partition", "task_id", "image_id", "image_sha256"}
        for row in rows
    )
    assert all("label" not in row and "role" not in row for row in rows)


def test_resumable_extraction_reuses_completed_batches(tmp_path: Path) -> None:
    tasks = {
        partition: [_task(f"{partition}-task")]
        for partition in extract.PARTITION_SIZES
    }
    calls = []

    def interrupted(images: list[bytes]) -> np.ndarray:
        calls.append(len(images))
        if len(calls) > 1:
            raise RuntimeError("synthetic interruption")
        values = np.zeros((len(images), extract.EMBEDDING_DIMENSION), dtype=np.float32)
        values[:, 0] = 1.0
        return values

    with np.testing.assert_raises_regex(RuntimeError, "synthetic interruption"):
        extract.extract_embeddings_resumable(
            tasks,
            interrupted,
            work_dir=tmp_path,
            batch_size=10,
        )
    progress = json.loads(
        (tmp_path / "PROGRESS.json").read_text(encoding="utf-8")
    )
    assert progress["next_start"] == 10
    assert progress["model_id"] == extract.MODEL_ID
    assert progress["model_revision"] == extract.MODEL_REVISION
    assert progress["model_weights_sha256"] == extract.MODEL_WEIGHTS_SHA256

    resumed_calls = []

    def resumed(images: list[bytes]) -> np.ndarray:
        resumed_calls.append(len(images))
        values = np.zeros((len(images), extract.EMBEDDING_DIMENSION), dtype=np.float32)
        values[:, 0] = 1.0
        return values

    rows, embeddings = extract.extract_embeddings_resumable(
        tasks,
        resumed,
        work_dir=tmp_path,
        batch_size=10,
    )
    assert len(rows) == 42
    assert embeddings.shape == (42, extract.EMBEDDING_DIMENSION)
    assert sum(resumed_calls) == 32
