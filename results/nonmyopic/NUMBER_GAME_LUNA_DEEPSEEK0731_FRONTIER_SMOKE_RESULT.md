# Number Game Luna + DeepSeek 0731 Frontier Smoke Result

Date completed: 2026-08-05

## Decision

Advance `openai/gpt-5.6-luna` to the separately frozen paired efficacy32 gate.
Close `deepseek/deepseek-v4-flash-0731` under this interface without prompt,
parser, seed, or threshold repair.

## Results

| Model | Status | Initial valid mean | Conditioned valid mean | Conditioned minimum | Cost |
|---|---|---:|---:|---:|---:|
| GPT-5.6 Luna | passed | 22.0 | 15.375 | 10 | `$0.003577600` |
| DeepSeek V4 Flash 0731 | gated null | 21.5 | 11.5 | 0 | `$0.003258592` |

Luna completed exactly ten accepted/HTTP requests with zero retries,
provider-error retries, reasoning tokens, or forced exits. Every response had
24 schema-valid items before semantic filtering. Its merged first supports
contained 21--25 hypotheses and merged second supports contained 15--22.

DeepSeek also completed clean transport and strict top-level parsing, but one
two-observation response produced 24 invalid executable expressions and zero
valid hypotheses. The linked retained support remained nonempty through parent
retention, but the frozen conditioned-generation gate failed. Lower price does
not override that failure.

The observed combined cost was `$0.006836192`, or 0.137% of the `$5.00` daily
allowance. Generic benchmarks remain screening evidence only: the direct task
gate selects Luna and rejects 0731 for this planner interface.

## Artifacts

- Luna RESULT SHA256:
  `9affa0b48ebc7ae2adfc6867e4b5bf56a47e0dc28c92c8894f94ae1896211496`;
- Luna private raw SHA256:
  `a8122a2f14106be59a1a242f78597c4b6980abbd19df6a40744aaad5e4095e73`;
- DeepSeek 0731 RESULT SHA256:
  `029d9458bed196f788e421601c280de96fe69f3140bad083c7d779ee6496eb91`;
- DeepSeek 0731 private raw SHA256:
  `82d4c5b12c2e561e045ce3ee78941b377ed1f3f04b6916ae214247a6d1891da5`.
