# Number Game August 7 Budget-Model Preflight Result

Run: 2026-08-06 07:30 Europe/London, before any August 7 formal seed.

Status: **ready without paid calls**.

The read-only orchestrator preflight verifies:

- the mechanics-clean August 6 Qwen source and its science-blind control
  authorization;
- exact source artifact hashes and the prior source/control predecessor hashes;
- 128 frozen reliability cases per budget model;
- the 3,584-case stress manifest `017ef555...bb71cd` and preregistration hash;
- pristine control, reliability, stress, and wrapper paths;
- the untouched August 7 account-wide ledger SHA-256
  `d889a0dd...7a47f`;
- expected full-sequence cost `$4.96` under the `$5.00` hard cap, with `$0.04`
  nominal slack and stress conditional on measured control cost at most `$3.25`;
- authenticated OpenRouter credits `$245.00`, usage `$217.297890263`, and
  balance `$27.702109737`;
- zero usage after the frozen opening baseline.

The live OpenRouter catalog exposes every exact endpoint with text input,
structured output, and at least the required 4,200 completion tokens:

| Model | Context | Max completion | Input / output per 1M |
|---|---:|---:|---:|
| `qwen/qwen3.7-plus` | 1,000,000 | 131,072 | `$0.32 / $1.28` |
| `openai/gpt-5.6-luna` | 1,050,000 | 128,000 | `$0.10 / $0.60` |
| `deepseek/deepseek-v4-flash-0731` | 1,048,576 | 65,536 | `$0.09 / $0.18` |

The focused control, verifier, reliability, stress, daily-stage, and budget
suite passes 69 tests. The preflight made zero model calls and wrote zero
files. The authoritative paid command remains the single orchestrator command
in `NUMBER_GAME_BUDGET_MODEL_AUG7_EXECUTION.md`; it reruns this preflight
automatically before any write or adapter. Component CLIs must not be launched
separately.
