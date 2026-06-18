from __future__ import annotations

import argparse
import json
import os
import time

import wandb

from helpers import Config, ModelPair, ModelSpec
from model import build_model_adapter


def _messages(batch_size: int) -> list[list[dict[str, str]]]:
    return [
        [
            {
                "role": "user",
                "content": (
                    "Return exactly this JSON object and no extra commentary: "
                    f'{{"index": {idx}, "answer": "ok"}}'
                ),
            }
        ]
        for idx in range(batch_size)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-sizes", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    args = parser.parse_args()

    block_sizes = [int(value) for value in args.block_sizes.split(",") if value.strip()]
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
    batch_messages = _messages(args.batch_size)

    print(
        "BENCHMARK_START "
        f"batch_size={args.batch_size} max_new_tokens={args.max_new_tokens} "
        f"repeats={args.repeats} block_sizes={block_sizes}",
        flush=True,
    )
    for block_size in block_sizes:
        for repeat_idx in range(args.repeats):
            start = time.perf_counter()
            completions = model.chat_complete_messages_batched(
                batch_messages,
                temperature=0.0,
                block_size=block_size,
                max_new_tokens=args.max_new_tokens,
            )
            elapsed = time.perf_counter() - start
            nonempty = sum(1 for completion in completions if completion.strip())
            tokens_per_second = (
                args.batch_size * args.max_new_tokens / elapsed
                if elapsed > 0.0
                else 0.0
            )
            print(
                "BENCHMARK_RESULT "
                f"block_size={block_size} repeat={repeat_idx} "
                f"batch_size={args.batch_size} completions={len(completions)} "
                f"nonempty={nonempty} elapsed={elapsed:.2f}s "
                f"approx_output_tokens_per_s={tokens_per_second:.2f}",
                flush=True,
            )
            if len(completions) != args.batch_size:
                raise RuntimeError(
                    f"Expected {args.batch_size} completions, got {len(completions)}"
                )
    print("BENCHMARK_DONE", flush=True)


if __name__ == "__main__":
    main()
