#!/usr/bin/env python3
"""Extract label-free SigLIP embeddings for frozen Bongard BED tasks."""

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
INTERFACE_VERSION = "bongard-openworld-siglip-embedding-extract-1"
MODEL_ID = "google/siglip2-so400m-patch16-512"
MODEL_REVISION = "ceea1cba8130d8271436da4828633198c176a775"
MODEL_WEIGHTS_SHA256 = (
    "a621bd212e1b3329b428595f9693217e19587afe826adf3e5c241a16392e8973"
)
MODEL_CONFIG_SHA256 = (
    "048fd125a0ce9fdf98919bb3651c8596510488282cf76402eb838418a8029c79"
)
PREPROCESSOR_CONFIG_SHA256 = (
    "63ff380d3e424f93e6fbca5cc8e74eeed882e96fd6be2b7728aed308f3ad1513"
)
PARTITION_SIZES = {"mechanics": 4, "development": 64, "confirmation": 96}
EMBEDDING_DIMENSION = 1152
BATCH_SIZE = 14


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


def verify_local_model_files() -> dict[str, str]:
    from transformers.utils.hub import cached_file

    expected = {
        "model.safetensors": MODEL_WEIGHTS_SHA256,
        "config.json": MODEL_CONFIG_SHA256,
        "preprocessor_config.json": PREPROCESSOR_CONFIG_SHA256,
    }
    observed = {
        filename: sha256_file(
            Path(
                cached_file(
                    MODEL_ID,
                    filename,
                    revision=MODEL_REVISION,
                    local_files_only=True,
                )
            )
        )
        for filename in expected
    }
    if observed != expected:
        raise ValueError(
            f"local SigLIP model files changed: expected {expected}, observed {observed}"
        )
    return observed


def load_local_model() -> tuple[Callable[[Sequence[bytes]], np.ndarray], dict[str, Any]]:
    import torch
    from transformers import AutoImageProcessor, SiglipVisionModel

    verified_files = verify_local_model_files()
    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        use_fast=False,
    )
    model = SiglipVisionModel.from_pretrained(
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
            values = model(**inputs).pooler_output
            values = torch.nn.functional.normalize(values, dim=1)
        return values.cpu().numpy().astype(np.float32)

    metadata = {
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "vision_tower_only": True,
        "verified_local_files": verified_files,
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
            raise ValueError("SigLIP embeddings are not unit normalized")
        batches.append(values.astype(np.float32, copy=False))
    return rows, np.concatenate(batches, axis=0)


def _validate_embedding_batch(values: np.ndarray, expected_rows: int) -> None:
    if values.shape != (expected_rows, EMBEDDING_DIMENSION):
        raise ValueError(
            f"embedding batch has shape {values.shape}, expected "
            f"{(expected_rows, EMBEDDING_DIMENSION)}"
        )
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("embedding batch must be finite float32")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError("SigLIP embeddings are not unit normalized")


def extract_embeddings_resumable(
    tasks_by_partition: Mapping[str, Sequence[bed.VisualTask]],
    embed: Callable[[Sequence[bytes]], np.ndarray],
    *,
    work_dir: Path,
    batch_size: int = BATCH_SIZE,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = image_rows(tasks_by_partition)
    rows_sha256 = hashlib.sha256(
        bed.canonical_json({"rows": rows}).encode("utf-8")
    ).hexdigest()
    task_by_id = {
        task.task_id: task
        for tasks in tasks_by_partition.values()
        for task in tasks
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    partial_path = work_dir / "EMBEDDINGS.partial.npy"
    progress_path = work_dir / "PROGRESS.json"
    expected_progress = {
        "interface_version": INTERFACE_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "preprocessor_config_sha256": PREPROCESSOR_CONFIG_SHA256,
        "rows_sha256": rows_sha256,
        "rows": len(rows),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "batch_size": batch_size,
    }
    if progress_path.exists() or partial_path.exists():
        if not progress_path.exists() or not partial_path.exists():
            raise ValueError("incomplete SigLIP resume checkpoint")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if any(progress.get(key) != value for key, value in expected_progress.items()):
            raise ValueError("SigLIP resume checkpoint does not match frozen rows")
        next_start = progress.get("next_start")
        if (
            not isinstance(next_start, int)
            or next_start < 0
            or next_start > len(rows)
            or (next_start != len(rows) and next_start % batch_size != 0)
        ):
            raise ValueError("SigLIP resume checkpoint has invalid progress")
        memmap = np.lib.format.open_memmap(partial_path, mode="r+")
        if memmap.shape != (len(rows), EMBEDDING_DIMENSION) or memmap.dtype != np.float32:
            raise ValueError("SigLIP resume array has invalid shape or dtype")
    else:
        next_start = 0
        memmap = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), EMBEDDING_DIMENSION),
        )
        progress = {**expected_progress, "next_start": 0}
        progress_path.write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for start in range(next_start, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        values = embed(
            [
                task_by_id[row["task_id"]].image_bytes[row["image_id"]]
                for row in batch_rows
            ]
        )
        _validate_embedding_batch(values, len(batch_rows))
        end = start + len(batch_rows)
        memmap[start:end] = values
        memmap.flush()
        progress = {**expected_progress, "next_start": end}
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(progress_path)
        print(f"siglip_embedding_progress {end}/{len(rows)}", flush=True)

    embeddings = np.asarray(memmap).copy()
    del memmap
    _validate_embedding_batch(embeddings, len(rows))
    return rows, embeddings


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
            "This artifact contains frozen-image SigLIP representations only. "
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
            / "results/nonmyopic/bongard_openworld_siglip_classical_baseline/"
            "embedding-extract-20260809"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = default_tasks()
    embed, runtime = load_local_model()
    rows, embeddings = extract_embeddings_resumable(
        tasks,
        embed,
        work_dir=args.output_dir,
    )
    result = write_artifacts(
        output_dir=args.output_dir,
        rows=rows,
        embeddings=embeddings,
        runtime=runtime,
    )
    (args.output_dir / "PROGRESS.json").unlink()
    (args.output_dir / "EMBEDDINGS.partial.npy").unlink()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
