# tau-Knowledge Balanced Finalist Duel Preregistration

## Purpose

Test a top-focused successor to the failed complete ten-root ranking policy.
The frozen shared-comparative run improved selected-root oracle-tail coverage
by six documents and improved selected-root-versus-all accuracy, but its global
ranking of all lower roots was worse than myopic.

This method does not change that failed result. It uses the already-frozen
myopic and non-myopic Borda winners as a two-root shortlist, then asks a
position-balanced semantic comparator whether the non-myopic finalist should
override the stable myopic finalist.

## Frozen Inputs

- Primary proposal artifact SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`
- Secondary proposal artifact SHA-256:
  `627dfe7641dca9fade90890dc90f818da1c6967c08b7878208224cd961fd7f20`
- Failed pooled rank artifact SHA-256:
  `9eef6301598149066fe552bb2ccfcf734bfeb11ae01f6fc727f05d8cea4f99eb`
- Model: `openai/gpt-5.4`, temperature zero, reasoning disabled.
- Tasks: the same 20 open development tasks.
- Changed finalists: `13/20`, fixed by the rank artifact.

Required-document labels and endpoint values are never sent to the model.

## Policy

For each changed task:

1. Compare the myopic finalist and non-myopic finalist with both complete
   two-step future trees visible.
2. Call once with myopic as A and non-myopic as B.
3. Call once with non-myopic as A and myopic as B.
4. Each response independently summarizes the documented coverage of A and B,
   then selects A, B, or a genuine tie.
5. Override the myopic finalist only if both calls select the same canonical
   non-myopic root.
6. If the calls disagree, either calls a tie, or both finalists were already
   identical, select the myopic finalist.

Thus position instability cannot hurt relative to the myopic control. The
endpoint is the same exact oracle continuation under the selected root, so the
test remains strictly about the first link.

## Serving Gate

The first two tasks have different finalists. Run exact four logical requests:

```text
2 changed tasks x 2 A/B orders
```

Require complete strict JSON, exact request accounting, zero reasoning tokens,
and zero forced exits. Bounded transport retries are logged separately; no
semantic repair or reissue is allowed.

## Development Gate

If serving passes, run exact 26 logical requests over the 13 changed tasks.
Every condition must pass:

- at least `8/13` duels have a unanimous canonical winner;
- at most `5/13` use disagreement/tie fallback;
- selected-root oracle-tail total gains at least `4` documents over myopic;
- task wins exceed losses by at least `3`;
- losses are at most `1`;
- selected-root-versus-all accuracy is at least `.70`; and
- selected-root-versus-all accuracy gains at least `+.08` over myopic.

Pass authorizes a separately frozen fresh-task execution that generates two
new proposal pools, obtains matched pooled rankings, and applies the same
balanced finalist duel before opening endpoints. Failure closes tau balanced
dueling.

## Budget

- Serving projected/cap: `$0.10 / $0.40`.
- Development projected/cap: `$0.75 / $1.50`.
- Conservative balance before this gate: `$29.997867594`.
- No fixed reserve.
- OpenRouter only; no OatML or Slurm.
