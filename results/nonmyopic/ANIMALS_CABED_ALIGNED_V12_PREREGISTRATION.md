# Animals CA-BED Aligned V12 Preregistration

Date frozen: 2026-07-24

## Question

Can depth-two planning outperform one-step EIG inside a coherent LLM-native
semantic environment?

V11 established that independently generated hidden-target answers are not a
stable realization of its target-blind semantic likelihood table. V12 freezes
one response function: for each generated question, Gemma labels every animal
in the 64-element support Yes or No in one target-blind batch. The label at the
hidden target is then the realized observation, while the full row supplies
the likelihood model.

No target identity is included in the semantic prompt. The table is generated
before its hidden-target label is read. No label is regenerated, repaired, or
selected after observing an endpoint.

## Frozen Design

- Gemma 4 26B A4B, non-thinking, temperature zero.
- Same fixed 64-animal support, selection seed 24310, target split, frozen
  prehistories, width 4/3 shared trees, and controls as V10/V11.
- The LLM generates every root and branch-specific follow-up question.
- One strict 64-row Yes/No semantic classification defines each question's
  response function.
- Raw labels 1/0 are smoothed with frozen confidence 0.7 to Bayesian
  likelihoods 0.85/0.15.
- The realized observation is the unsmoothed hidden-target label from that
  same precomputed table.
- Depth one scores immediate EIG.
- Depth two scores immediate EIG plus predictive-answer-weighted best
  follow-up EIG.
- Seeded random selects from the same shared root tree.
- Primary endpoint is truth log-posterior gain; entropy is secondary.
- No reasoning, numerical-probability generation, fallback, parser repair,
  response resampling, replacement state, or threshold change is allowed.

This tests planning under an LLM-defined semantic observation model. It does
not claim agreement with an external zoological oracle.

## Stages

### Serving Smoke

- Wombat and Aardvark only.
- Hard cap $0.75.
- Requires two complete trees, complete binary semantic tables, duplicate
  deterministic table lookups, finite complementary likelihoods, and zero
  reasoning.
- No two-state efficacy threshold.

### Formal Ranking

- The same untouched 24 formal targets.
- Hard cap $5.00.
- Authorized only if smoke passes unchanged and the live balance preserves the
  $25 Monday reserve.

The V10/V11 formal conjunction remains unchanged: at least 6 distinct depth-two
roots; mean depth-two score/truth-gain Spearman at least 0.20 and at least 0.10
above depth one; mean truth-NLL gains over depth one and random at least 0.02
with positive paired 90% bootstrap lower bounds; at least 14 depth-two wins;
and final entropy no more than 0.02 nats worse than depth one.
