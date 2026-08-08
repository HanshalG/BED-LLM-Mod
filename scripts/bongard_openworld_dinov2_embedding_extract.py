#!/usr/bin/env python3
"""Extract label-free DINOv2 embeddings for frozen Bongard BED tasks."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-dinov2-embedding-extract-1"
MODEL_ID = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MODEL_WEIGHTS_SHA256 = (
    "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
)
MODEL_CONFIG_SHA256 = (
    "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
)
PREPROCESSOR_CONFIG_SHA256 = (
    "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"
)
PARTITION_SIZES = {"mechanics": 4, "development": 64, "confirmation": 96}
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_tasks() -> dict[str, list[bed.VisualTask]]:
    return {
        partition: bed.load_validation_partition_tasks(
            partition, include_endpoint_labels=False
        )
        for partition in PARTITION_SIZES
    }


def image_rows(
    tasks_by_partition: Mapping[str, Sequence[bed.VisualTask]],
) -> list[dict[str, Any]]:
    rows = []
    for partition in PARTITION_SIZES:
        for task in tasks_by_partition[partition]:
            for image_id in task.image_ids:
                rows.append(
                    {
                        "partition": partition,
                        "task_id": task.task_id,
                        "image_id": image_id,
                        "image_sha256": hashlib.sha256(
                            task.image_bytes[image_id]
                        ).hexdigest(),
                    }
                )
    return rows


def load_local_model() -> tuple[Callable[[Sequence[bytes]], np.ndarray], dict[str, Any]]:
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    ).eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    def embed(batch: Sequence[bytes]) -> np.ndarray:
        images = []
        for data in batch:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Palette images with Transparency expressed in bytes.*",
                )
                with Image.open(BytesIO(data)) as source:
                    images.append(source.convert("RGB"))
        inputs = {
            key: value.to(device)
            for key, value in processor(images=images, return_tensors="pt").items()
        }
        with torch.inference_mode():
            values = model(**inputs).last_hidden_state[:, 0]
            values = torch.nn.functional.normalize(values, dim=1)
        return values.cpu().numpy().astype(np.float32)

    metadata = {
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    return embed, metadata


def extract_embeddings(
    tasks_by_partition: Mapping[str, Sequence[bed.VisualTask]],
    embed: Callable[[Sequence[bytes]], np.ndarray],
    *,
    batch_size: int = BATCH_SIZE,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = image_rows(tasks_by_partition)
    task_by_id = {
        task.task_id: task
        for tasks in tasks_by_partition.values()
        for task in tasks
    }
    batches = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        values = embed(
            [
                task_by_id[row["task_id"]].image_bytes[row["image_id"]]
                for row in batch_rows
            ]
        )
        if values.shape != (len(batch_rows), EMBEDDING_DIMENSION):
            raise ValueError(
                f"embedding batch has shape {values.shape}, expected "
                f"{(len(batch_rows), EMBEDDING_DIMENSION)}"
            )
        if not np.isfinite(values).all():
            raise ValueError("embedding batch contains non-finite values")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
            raise ValueError("DINO embeddings are not unit normalized")
        batches.append(values.astype(np.float32, copy=False))
    return rows, np.concatenate(batches, axis=0)


def write_artifacts(
    *,
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "INDEX.json"
    embeddings_path = output_dir / "EMBEDDINGS.npy"
    manifest_path = output_dir / "MANIFEST.json"
    for path in (index_path, embeddings_path, manifest_path):
        if path.exists():
            raise FileExistsError(path)

    index_path.write_text(
        bed.canonical_json({"rows": list(rows)}) + "\n", encoding="utf-8"
    )
    with embeddings_path.open("wb") as handle:
        np.save(handle, embeddings, allow_pickle=False)

    partition_counts = {
        partition: sum(row["partition"] == partition for row in rows)
        for partition in PARTITION_SIZES
    }
    gates = {
        "exact_partition_image_counts": partition_counts
        == {partition: size * bed.NUM_IMAGES for partition, size in PARTITION_SIZES.items()},
        "opaque_image_index_has_no_labels_or_roles": all(
            set(row) == {"partition", "task_id", "image_id", "image_sha256"}
            for row in rows
        ),
        "embedding_array_is_exact_finite_unit_normalized_float32": (
            embeddings.shape == (sum(partition_counts.values()), EMBEDDING_DIMENSION)
            and embeddings.dtype == np.float32
            and np.isfinite(embeddings).all()
            and np.allclose(
                np.linalg.norm(embeddings, axis=1), 1.0, rtol=0.0, atol=1e-5
            )
        ),
    }
    all_gates_pass = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "embedding_extract_pass" if all_gates_pass else "fail",
        "model_calls": 0,
        "cost_usd": 0.0,
        "authorizes_paid_calls": False,
        "endpoint_labels_accessed": False,
        "candidate_labels_accessed": False,
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weights_sha256": MODEL_WEIGHTS_SHA256,
            "config_sha256": MODEL_CONFIG_SHA256,
            "preprocessor_config_sha256": PREPROCESSOR_CONFIG_SHA256,
            **dict(runtime),
        },
        "artifacts": {
            "index": str(index_path.relative_to(REPO_ROOT)),
            "index_sha256": sha256_file(index_path),
            "embeddings": str(embeddings_path.relative_to(REPO_ROOT)),
            "embeddings_sha256": sha256_file(embeddings_path),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
        },
        "partition_image_counts": partition_counts,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "interpretation": (
            "This artifact contains frozen-image DINOv2 representations only. "
            "It reads no candidate or endpoint outcomes and provides no efficacy result."
        ),
    }
    checkpoint(manifest_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results/nonmyopic/bongard_openworld_dinov2_classical_baseline/"
            "embedding-extract-20260808"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = default_tasks()
    embed, runtime = load_local_model()
    rows, embeddings = extract_embeddings(tasks, embed)
    result = write_artifacts(
        output_dir=args.output_dir,
        rows=rows,
        embeddings=embeddings,
        runtime=runtime,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
