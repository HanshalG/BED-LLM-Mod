from __future__ import annotations

import argparse
import json
import os
import time

import wandb

from helpers import Config, ModelPair, ModelSpec
from model import build_model_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    args = parser.parse_args()

    batch_size = args.batch_size or args.block_size
    os.environ.setdefault(
        "BED_LLM_VLLM_KWARGS",
        json.dumps({"max_num_seqs": 100, "enforce_eager": False}),
    )
    wandb.init(mode="disabled")

    spec = ModelSpec(
        model=args.model,
        thinking=True,
        thinking_max_new_tokens=4096,
        thinking_final_max_new_tokens=512,
        cuda_visible_devices="0",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    config = Config(
        model_pairs=[ModelPair(questioner=spec, answerer=spec)],
        location_max_new_tokens=args.max_new_tokens,
    )
    model = build_model_adapter(spec, config)

    batch_messages = [
        [
            {
                "role": "user",
                "content": (
                    "Return exactly this JSON object and no extra commentary: "
                    f'{{"index": {idx}, "ok": true}}'
                ),
            }
        ]
        for idx in range(batch_size)
    ]

    print(
        "START block_size="
        f"{args.block_size} batch_size={batch_size} max_new_tokens={args.max_new_tokens}"
    )
    start = time.perf_counter()
    completions = model.chat_complete_messages_batched(
        batch_messages,
        temperature=0.0,
        block_size=args.block_size,
        max_new_tokens=args.max_new_tokens,
    )
    elapsed = time.perf_counter() - start
    nonempty = sum(1 for completion in completions if completion.strip())
    print(
        "RESULT "
        f"block_size={args.block_size} batch_size={batch_size} "
        f"completions={len(completions)} nonempty={nonempty} elapsed={elapsed:.2f}s"
    )
    if len(completions) != batch_size:
        raise RuntimeError(f"Expected {batch_size} completions, got {len(completions)}")


if __name__ == "__main__":
    main()
