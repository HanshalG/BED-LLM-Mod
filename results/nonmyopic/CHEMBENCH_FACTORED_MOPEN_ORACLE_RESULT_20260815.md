# ChemBench Factored M-open Oracle Result

Date: 2026-08-15 (Europe/London)

## Decision

**Passed and independently replayed.** Selective, source-calibrated support
expansion plus a bounded typed-edit pool preserves a strong, exactly monotonic
non-myopic opportunity on all three opened v4 ChemBench slices.

This result used no LLM/API call and cost $0. It authorizes a small prospective
LLM proposal-semantics and transition-fidelity gate. It does not authorize v5
efficacy and is not yet an LLM result.

## Primary Result

| Policy level | Aggregate terminal MSE | Successive reduction |
| --- | ---: | ---: |
| d1 | 0.05439110 | - |
| d2 | 0.03781695 | 30.47% |
| d3 | 0.03176550 | 16.00% |

- d2 versus d1: 56 practical wins, 64 ties, and 24 losses.
- d3 versus d2: 27 practical wins, 94 ties, and 23 losses.
- d2 changes the d1 root on all three slices.
- d3 changes the d2 root on two of three slices.
- Planned terminal risk equals uniform truth-conditional replay within the
  frozen numerical tolerance at every slice and level.
- All 16 conjunctive conditions pass in both production and independent replay.

## Slice Results

| Slice | d1 MSE | d2 MSE | d3 MSE | d2 reduction | d3 reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy/v4 | 0.043997 | 0.025624 | 0.024154 | 41.76% | 5.74% |
| medium/v4 | 0.056148 | 0.042825 | 0.028816 | 23.73% | 32.71% |
| hard/v4 | 0.063029 | 0.045002 | 0.042326 | 28.60% | 5.95% |

Every slice is non-increasing from d1 to d2 to d3. The easy and hard d3 links
clear the frozen 5% threshold narrowly but prospectively; medium supplies a
large second-link gain.

## Trigger And Support Behavior

The 90th-percentile surprise threshold was calibrated only on primitive v2
source worlds. Its actual weighted false-trigger mass is 9.72% on easy, 3.55%
on medium, and 9.26% on hard, satisfying the at-most-10% gate.

On outside-support v4 planning transitions, expansion fires on 30.75% of easy,
35.36% of medium, and 27.68% of hard unique transitions. Thus the transition is
neither always-expand nor inert.

| Slice | Unique transitions | Triggered | Accepted edits | Contractions |
| --- | ---: | ---: | ---: | ---: |
| easy/v4 | 25,110 | 7,722 | 30,888 | 7,722 |
| medium/v4 | 27,486 | 9,720 | 38,880 | 9,720 |
| hard/v4 | 19,461 | 5,387 | 21,548 | 5,387 |

Every accepted candidate round-trips through a typed edit, every proposal is
novel along its history, support never exceeds 12, and every full support has at
least four core mechanism families. Non-triggering branches issue no proposer
request.

## Controls

The fixed primitive-support d1 terminal MSE is 0.2213/0.4409/0.4563 on
easy/medium/hard, far above the dynamic-support policy. Structural expansion is
therefore necessary in this outside-support cohort.

The union-only diagnostic exactly matches the bounded-pool d1/d2/d3 values on
all slices. Under this oracle, the evidence-discarded models have negligible
effect on selected policies and forecasts, so contraction imposes no observed
performance cost. This does not establish that contraction helps; its purpose
here is to prevent duplicate/support proliferation before an imperfect LLM
proposer is introduced.

The call-matched d1 replay adds zero proposal misses. An immutable bank replay
reconstructs every transition audit, root, planned value, per-truth loss,
control, comparison, and gate condition.

## Comparison With Always-Expand Oracle

The original always-expand v4 oracle produced d1/d2/d3
0.03919/0.03204/0.02925. Selective expansion makes d1 harder, as expected, but
leaves deeper policies able to anticipate which assays are likely to trigger
useful model creation. The resulting successive relative gains increase from
18.26%/8.71% to 30.47%/16.00%.

This is the desired structural mechanism: an early experiment is valuable not
only for its immediate likelihood update, but because a surprising outcome can
unlock executable hypotheses that improve later experiment choice.

## Scientific Boundary

The proposal source is still a registry oracle ranked by complete-history
likelihood. Consequently this result proves the numerical architecture and the
existence of a selective M-open horizon gap, not that an LLM can identify the
right edits.

The next gate must test the exact LLM first link:

1. pool-wide residual report to typed executable edit;
2. strict validity and no duplicate/dead-end reproposals;
3. held-out edit-family recall and proposal-induced action regret;
4. frozen-atlas versus fresh-LLM transition fidelity;
5. matched history-blind and random-edit controls.

Only a full semantic and transition-fidelity pass may open v4 LLM policy
development. Untouched v5 remains sealed.

## Artifacts

- Required implementation commit: `b9f5192a`.
- Raw result content SHA256:
  `2f1cdb224a5628d9b9e342c8d29dccf064fd9165db4a0fe40358be4192c7eb81`.
- Raw transition-bank content SHA256:
  `b9af378701f813c3a61cd5725e8f54b4b461826196c39b3386726fe7cc404db1`.
- Result archive SHA256:
  `247f0b3996ce3ea58a71e37af8cc314dad1399d5f8987850b590b615ae6f3d2e`.
- Transition-bank archive SHA256:
  `4a48da4968407059806bdc70bee6f7dce987d36f63bbc5e7ab8f91b60ce280b3`.
- Independent verification SHA256:
  `d97daa7dacbbeb7ab0a08907fd4e3785f92a319e0aa224f5ebdfc5f9059594f3`.
- The gzip archives are deterministic (`gzip -n -9`). Decompressing them
  reproduces the raw content hashes above.
