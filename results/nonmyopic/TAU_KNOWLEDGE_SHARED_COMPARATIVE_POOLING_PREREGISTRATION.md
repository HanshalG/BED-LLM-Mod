# tau-Knowledge Shared Comparative Pooling Preregistration

## Purpose

Test whether tau-Knowledge's unstable first-link result is primarily a
cross-generation calibration problem. Two independently generated five-root
trees for each of the same 20 open tasks contain a pooled root oracle of 51
required documents, but prior selection rules could not compare their
independent numeric score scales.

This method puts all ten roots on one shared semantic comparison scale. It is
development on already-open tasks, not new-task confirmation.

## Frozen Inputs

- Primary V3.1 artifact SHA-256:
  `f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae`
- Independent execution artifact SHA-256:
  `627dfe7641dca9fade90890dc90f818da1c6967c08b7878208224cd961fd7f20`
- The primary artifact passed its efficacy gates. The independent execution is
  a complete scored `gate_failed` artifact; its reversed efficacy is the
  instability this development test is designed to address.
- Tasks: the same 20 IDs in primary-artifact order.
- Roots: five from each artifact, preserving source order.
- Model: `openai/gpt-5.4`, temperature zero, reasoning disabled.

Required-document labels are never included in a model prompt. Minor
source-opening punctuation/text drift between the two historical executions is
recorded; the primary opening is the shared scoring context.

## Policies

For each task:

1. **Myopic pooled ranker** sees all ten root queries and first-result evidence,
   but no refreshed beliefs, follow-up queries, or follow-up results.
2. **Non-myopic pooled ranker** sees the same ten roots and each complete
   two-step semantic future tree.
3. Each policy receives exactly three independent root-order permutations.
4. Each call returns one complete ten-root ordering, with no numeric ratings.
5. The policy order is the Borda sum of its three rankings; canonical root
   index breaks exact ties.
6. A seeded random root is reported as a control.

Both learned policies are evaluated at the first link using the exact best
available continuation under their selected root. This oracle continuation is
used only after rankings freeze and is identical for both policies. It removes
the noisy second-link selector from the estimand.

The matched comparison is therefore:

```text
same 10 generated roots + same 3 ranking calls + same continuation endpoint
```

with only semantic future visibility changed.

## Position And Scale Controls

- No independently calibrated 0--100 scores are requested.
- Root IDs are anonymous and all ten appear exactly once.
- Three deterministic, independently seeded presentation orders are frozen
  with seed `24419`.
- Myopic and non-myopic presentation orders differ.
- Pairwise agreement among the three rankings is reported.

This follows the known lesson that comparative LLM judgments have position
bias: permutation aggregation is part of the frozen policy, not a post-hoc
repair.

## Serving Gate

Run the first two open tasks with exact 12 logical requests:

```text
2 tasks x 2 policies x 3 permutations
```

Require:

- all 12 complete strict permutations;
- exact logical request accounting;
- zero reasoning tokens and forced exits; and
- input length at most 100,000 characters.

Transport retries may occur only through the configured bounded adapter and are
logged separately. There is no semantic response repair or reissue.

## Development Gate

If serving passes, run all 20 tasks with exact 120 logical requests. The
development method passes only if every condition holds:

- non-myopic root pairwise accuracy at least `.62`;
- non-myopic minus myopic pairwise accuracy at least `+.05`;
- selected-root oracle-tail total at least `+4` documents over myopic;
- task wins exceed losses by at least `2`;
- myopic and non-myopic roots differ on at least `4/20` tasks;
- mean non-myopic replicate rank agreement at least `.55`; and
- non-myopic selected-root total at least `+5` over seeded random.

Pass authorizes a separately frozen execution on untouched tau-Knowledge tasks
with two fresh proposal generations per task. Failure closes shared
comparative pooling on tau.

## Budget

- Serving projected/cap: `$0.20 / $0.75`.
- Development projected/cap: `$2.50 / $5.00`.
- Live OpenRouter balance before implementation: `$32.496637594`.
- No fixed reserve.
- OpenRouter only; no OatML or Slurm.
