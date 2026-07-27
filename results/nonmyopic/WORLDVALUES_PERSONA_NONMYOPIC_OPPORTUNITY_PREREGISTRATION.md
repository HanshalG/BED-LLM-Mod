# WorldValues Persona Non-Myopic Opportunity Preregistration

## Status

Frozen before computing any entropy, root, continuation, or gate outcome from
the official persona-response matrix.

This is a zero-call structural audit. It cannot establish an LLM policy result.
Its only purpose is to determine whether the replayable World Values persona
model contains enough exact two-step structure to justify a small
path-dependent LLM-support smoke.

## Source

- Official repository:
  `https://github.com/yw3453/adaptive-query-ai-persona-priors`
- Commit:
  `fbd8e19eed6af960b64e3afae13e3bcf4020b73f`
- Direct-distribution response matrix:
  `data/WorldValuesBench/worldvalues_simulated.csv`
- Matrix SHA-256:
  `24d5d7b3a9bdf94894952dc7bfff39d409f8f78c8131ccd7de87fb292068638e`
- Ordered 91-question hash:
  `1309815471ee599723f58af7a394676b42027178bd715787eef78a2d11f11971`

The release contains 2,058 text personas and GPT-5-mini-elicited categorical
response distributions over 91 four-option WorldValuesBench questions. The
license-gated real WVS response matrix is not used in this audit.

## Frozen Tasks

NumPy seed `24711` independently permutes the ordered 91 question IDs for each
of 20 tasks:

- first 8 questions: held-out prediction targets;
- next 24 questions: feasible query actions;
- remaining 59 questions: unused.

The complete canonical task-spec hash is:

`da1ae2fdc0d1f0495e8e6c01c78a1c656dab0416076419c65f2e3d5e52d5602d`.

Question text, persona text, and response probabilities do not affect the
split.

## Exact Model

- Latent state: one of all 2,058 released personas.
- Prior: uniform over personas.
- Response likelihood: the released four-category distribution for each
  persona-question pair, normalized only for floating-point drift.
- Target objective: mean posterior-predictive entropy, in nats, over the eight
  target questions.
- Myopic root: candidate with minimum expected target entropy after one answer.
- Adaptive d2 root: candidate with minimum expected target entropy after its
  answer and the answer-conditioned best remaining followup.
- Ties: first candidate in the frozen order.

The myopic control receives the same 24 roots and, after its selected root, its
own exact answer-conditioned best followup. Thus the comparison isolates the
first action rather than giving d2 extra actions or candidates.

A strict tradeoff requires all three:

1. adaptive d2 and myopic choose different roots;
2. adaptive d2 has strictly worse one-step expected target entropy; and
3. adaptive d2 has strictly better two-step expected target entropy.

Tolerance is `1e-10`.

## Frozen Gates

All conditions must pass:

- exactly 20 tasks complete;
- mean initial target entropy is at least `.5` nats;
- immediate and final root scores vary on all 20 tasks;
- adaptive d2 changes the root on at least 5/20 tasks;
- at least 4/20 tasks are strict tradeoffs;
- mean d2 final advantage over the myopic root is at least `.001` nats;
- strict total final advantage is at least `.01` nats; and
- mean strict immediate sacrifice is at least `.002` nats.

No question resampling, persona subset, prior fitting, temperature adjustment,
objective substitution, threshold repair, or task pooling follows failure.

## Authorization

A full pass authorizes only a separately frozen, exact 10-call, nonreasoning
OpenRouter serving smoke capped at `$0.20`. That smoke must make the LLM
generate an initial weighted persona support and regenerate it after complete
hypothetical question-answer histories. A fixed-support d2 ablation is
mandatory. No real-user or synthetic-user efficacy run is authorized by this
structural result alone.

Failure closes this WorldValues construction before any model call.

## Budget

OpenRouter calls: `0`; cost: `$0`; OatML/cluster use: none. The protected
`$25` reserve and stricter `$8.3292075` pre-Monday allowance are unchanged.
