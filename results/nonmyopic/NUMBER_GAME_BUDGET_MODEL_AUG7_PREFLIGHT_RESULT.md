# Number Game August 7 Budget-Model Preflight Result

Latest run: 2026-08-06 22:29 Europe/London, before any August 7 formal seed.

Status: **ready without paid calls**.

The read-only orchestrator preflight verifies:

- the mechanics-clean August 6 Qwen source and its science-blind control
  authorization;
- exact source artifact hashes and the prior source/control predecessor hashes;
- 128 frozen reliability cases per budget model;
- the 3,584-case stress manifest `017ef555...bb71cd`, preregistration hash,
  and deferred-authorization amendment hash `259465d0...9fb7`;
- precharge reservation amendment hash `4b731619...30510` and exact
  per-model worst-case attempt reservations;
- pristine control, reliability, stress, and wrapper paths;
- the untouched August 7 account-wide ledger SHA-256
  `d889a0dd...7a47f`;
- expected full-sequence cost `$4.96` under the `$5.00` hard cap, with `$0.04`
  nominal slack; stress is guaranteed when measured control cost is at most
  `$3.25`, or may be authorized after both reliability reconciliations when
  total recorded spend is at most `$3.45`;
- authenticated OpenRouter credits `$245.00`, usage `$217.297890263`, and
  balance `$27.702109737`;
- zero usage after the frozen opening baseline.

The live OpenRouter catalog exposes every exact endpoint with text input,
structured output, and at least the required 4,200 completion tokens:

| Model | Input / output per 1M | Request reservation | Prompt tokens covered after max output |
|---|---:|---:|---:|
| `qwen/qwen3.7-plus` | `$0.32 / $1.28` | `$0.0100` | `14,450` |
| `openai/gpt-5.6-luna` | `$0.10 / $0.60` | `$0.0040` | `14,800` |
| `deepseek/deepseek-v4-flash-0731` | `$0.09 / $0.18` | `$0.0015` | `8,267` |

Each attempt reservation covers the frozen 4,200-token output maximum and at least
8,000 prompt tokens at the live catalog price. Reservations are acquired under
the shared spend-ledger lock before HTTP dispatch; concurrent workers and
retries wait for settled headroom instead of sharing the same apparent
remainder. Ambiguous failed attempts retain their own reservation.

The focused adapter, config, source, control, reliability, stress, wrapper, and
budget suite passes 163 tests after the reservation amendment. The preflight
made zero model calls and wrote zero files. The authoritative paid command remains
the single orchestrator command
in `NUMBER_GAME_BUDGET_MODEL_AUG7_EXECUTION.md`; it reruns this preflight
automatically before any write or adapter. Component CLIs must not be launched
separately.
