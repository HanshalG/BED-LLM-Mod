from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from scripts import bongard_openworld_dinov2_embedding_extract as extract
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
    assert all(set(row) == {"partition", "task_id", "image_id", "image_sha256"} for row in rows)
    assert all("label" not in row and "role" not in row for row in rows)
