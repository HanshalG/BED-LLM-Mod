# Number Game Fresh-Control Mechanics Localization Protocol

Date frozen: 2026-08-13, before aggregating the already banked control diagnostics.

## Objective

Localize the single mechanics failure in the August 7 fully fresh Qwen
history-blind control without evaluating its sealed policy endpoint. The result will
decide what, if any, genuinely new support-generation interface is worth developing;
it cannot rescue, score, rerun, or reclassify the failed control.

## Immutable Inputs

- `FAILURE.json` SHA-256:
  `bf2a119f23e8241a7c2f694a3d3009104b0f3151073d37407cb6ddd7a917c244`;
- `CONTROLS.json` SHA-256:
  `513def73b8e75f31eca123e593c2e9b0deab715fbd52ed6d3833020564237672`;
- private `RAW_RESPONSES.json` SHA-256:
  `f8201be3cd199f895a6d971d4fa713beaee01660d4419a341c167a986cded241`.

The audit reads `FAILURE.json` and only the `diagnostic` objects in
`CONTROLS.json`. It must not deserialize support expressions, source trees, targets,
conditional supports, per-root endpoint values, or raw response content. The private
raw hash is verified from bytes only.

## Frozen Outputs

Report only aggregate or anonymous mechanics information:

- slot and draw counts;
- valid-count and pooled-count histograms/minima/quantiles;
- counts below the frozen draw floor 16 and pool floor 24;
- rejection totals by reason;
- second-draw novel-contribution distribution;
- for every failing slot, only anonymous local tree index, history index, stage,
  draw valid counts, pool count, novelty counts, and rejection counts.

Do not report query indices, answer labels, rule names, expressions, extensions,
target identities, selected roots, Brier values, or policy outcomes.

## Attribution

Classify each failure using fixed precedence:

1. `codec_or_incomplete` if codec is not strict or any item is missing/incomplete;
2. `invalid_rule` if invalid expression/name/fields alone explains the shortfall;
3. `duplicate_collapse` if duplicate extensions alone explains the shortfall;
4. `mixed_rule_rejections` if multiple rejection types are needed;
5. `cross_draw_overlap` if both draws meet 16 but the pool misses 24;
6. `unexplained` otherwise.

## Decision

- If failures are only isolated duplicate/invalid-rule collapse with otherwise strong
  support margins, freeze a new value-free support-generation mechanics gate on fresh
  synthetic histories before selecting any scientific cohort. It must improve valid
  unique support without changing endpoint thresholds.
- If failures are broad or dominated by cross-draw overlap, do not spend on another
  same-family full control; return to a new semantic environment/interface.

No outcome of this audit authorizes a paid call. Any successor requires its own
prospective protocol, fresh seeds, exact schema rehearsal, and account-wide budget
gate.
