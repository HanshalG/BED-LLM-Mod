# tau-Knowledge Shared-Support Finalist Preregistration

## Purpose

Remove candidate-position preference from the tau finalist decision. The
balanced A/B duel produced a frozen `+7` document, zero-loss directional result
but failed because only 6/13 comparisons were position-consistent.

This successor evaluates each finalist independently against the same pooled
semantic information-need support. It tests a shared belief/value scale rather
than side-by-side preference.

## Frozen Inputs

- Primary proposal artifact SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`
- Secondary proposal artifact SHA-256:
  `627dfe7641dca9fade90890dc90f818da1c6967c08b7878208224cd961fd7f20`
- Failed pooled rank artifact SHA-256:
  `9eef6301598149066fe552bb2ccfcf734bfeb11ae01f6fc727f05d8cea4f99eb`
- Same 13 frozen changed-finalist tasks.
- Model: `openai/gpt-5.4`, temperature zero, reasoning disabled.

Required-document labels and endpoint values are hidden from every call.

## Policy

For each changed task and each finalist independently:

1. Show the same deduplicated union of the two initial eight-need hypothesis
   sets, with stable `H1...Hn` IDs.
2. Show exactly one finalist's first evidence and four displayed continuation
   branches.
3. Ask the model to choose its best branch and label every shared need:
   `C` covered, `P` partial, or `U` unsupported.
4. Repeat with independently permuted need, follow-up, and document
   presentation.
5. Score a map with frozen weights `C=2`, `P=1`, `U=0`.
6. Override myopic only when the lower of the two non-myopic totals is strictly
   greater than the higher of the two myopic totals. Otherwise use myopic.

There is no A/B candidate order in any prompt. The endpoint remains the same
exact oracle continuation under the selected root, isolating the first link.

## Serving Gate

Run the first two changed tasks:

```text
2 tasks x 2 finalists x 2 map replicates = 8 logical calls
```

Require complete strict maps, exact request accounting, zero reasoning tokens,
and zero forced exits. Bounded transport retries are logged separately; no
semantic repair or reissue is allowed.

## Development Gate

If serving passes, run exact 52 logical calls over all 13 changed tasks.
Every criterion must pass:

- mean within-candidate categorical replicate agreement at least `.70`;
- at least `3` robust non-myopic overrides;
- selected-root oracle-tail gain at least `4` documents over myopic;
- task wins exceed losses by at least `3`;
- losses at most `1`;
- selected-root-versus-all accuracy at least `.70`; and
- selected-root-versus-all accuracy gain at least `+.08`.

Pass authorizes a separately frozen fresh-task execution. Failure closes
shared-support finalist scoring on tau.

## Budget

- Serving projected/cap: `$0.12 / $0.50`.
- Development projected/cap: `$0.80 / $1.75`.
- Authenticated pre-gate balance: `$29.668149594`.
- No fixed reserve.
- OpenRouter only; no OatML or Slurm.
