# Number Game Qwen History-Blind Matched-32 V2 Failure Result

Date: 2026-07-30

## Status

**Evaluator failed closed before mechanics or scientific scoring.** All 3,072
fresh responses completed and parsed, but the V2 wrapper recursively called
its patched mechanics function until Python raised `RecursionError`.

No mechanics aggregate, canonical endpoint, or scientific result was
produced. V2 responses are permanently banked and may not be scored or reused.

## Transport Audit

The hash-bound log and controls establish:

- accepted requests: `3,072`;
- finish reason: `3,072/3,072` `stop`;
- reasoning tokens: `0`;
- accepted cost: `$3.20377344`;
- parsed trees / branch slots / draws: `32 / 1,536 / 3,072`;
- parser: `3,072/3,072` strict JSON;
- valid unique rules per raw draw: `18..24`;
- pooled unique rules: `24..38`;
- second-draw novelty minimum: `0`, with `1/1,536` below two.

These are failure diagnostics only. They do not authorize endpoint scoring.

## Root Cause

Inside the V2 context, `base.mechanics_gates` was replaced by
`v2.mechanics_gates`. The replacement then called `base.mechanics_gates`,
which now referred to itself. The unit test exercised the replacement outside
the patched context and therefore missed the recursion.

## Decision

Bank V2. A fresh-seed V3 may capture the immutable original base-gate function
before patching and must test mechanics dispatch **inside** the configured
context, plus the full synthetic 3,072-response path, before launch.

## Provenance

- public `V2_RUNNER_FAILURE.json` SHA256:
  `4b3abc38059f11dceffb82ba1996d11ec7ecc1eae51616f4a2d63c895da2abd6`
- uncommitted controls SHA256:
  `f2c8c345eb0b1c3d7658372b5f9cb78d8b78ccd5b1b23b61ff96425d4a252e15`
- uncommitted run-log SHA256:
  `0c61b3c3a0d638b6c0d4a52c85a5db6b67fe4bf023ab798f8ac392117baa207b`
- private raw-response SHA256:
  `3057d850408338efdad1fe344d9210ebfd4fdbaefe174e65b3832113b4086994`
- endpoint accessed: `false`
