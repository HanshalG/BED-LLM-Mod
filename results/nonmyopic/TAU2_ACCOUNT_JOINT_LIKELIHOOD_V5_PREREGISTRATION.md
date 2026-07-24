# Tau2 Account Joint-Likelihood V5 Preregistration

Date frozen: 2026-07-24

## Status and Question

V5 is transparent posthoc development after observing the V4 failure. V4 showed
that full GPT-5.4 often understood the zero-information customer lookup and its
informative line-record successor, but independently generated branch policies
changed under ticket paraphrases.

V5 tests a distinct architecture: can one jointly elicited LLM semantic
likelihood table support reliable exact non-myopic planning? This removes LLM
branch-action generation but keeps the LLM load-bearing as the forward model.
It is not a claim about path-dependent support regeneration.

## Environment

The official Tau2 four-world account-only factorial remains:

- allowance available/exhausted;
- account roaming enabled/disabled;
- device roaming fixed off; and
- airplane mode fixed on.

All six initial diagnostic actions have zero immediate information under the
official simulator. Only `customer_lookup` unlocks account reads;
`line_details` then partitions all four worlds. The exact two-step setup value
is `ln(4) = 1.386294` nats; direct roots have zero two-step value.

## V5 Interface

- Model: full `openai/gpt-5.4` through OpenRouter.
- Reasoning disabled; temperature zero.
- One request per ticket variant.
- The model sees the four semantic hypotheses and documented action meanings.
- It predicts one short observable category for every one of nine actions under
  every hypothesis in a single table.
- It receives no official outputs, scores, entropy values, information gains,
  preferred actions, or policy choices.
- Identical predicted categories define deterministic likelihood partitions.
- Exact one-step and two-step planning is computed from that table.
- Official Tau2 outputs remain hidden until evaluation.
- Missing, reordered, duplicate, or empty table cells fail closed.

This architecture is LLM-Modulo: semantic likelihood construction is delegated
to the LLM and planning arithmetic is exact and auditable.

## Fresh Splits

V5 does not reuse any V4 smoke or formal ticket string.

- Serving smoke: two new variants, seed `24321`.
- Formal ranking: twelve new variants.
- Conditional confirmation: twenty-four additional new variants and a balanced
  hidden-world schedule shuffled once with seed `24321`.

No failed or incomplete variant may be replaced. Raw responses remain private;
parsed likelihood tables, actions, scores, endpoints, usage, and hashes are
public.

## Serving Smoke

Exactly two requests, projected `$0.05`, hard cap `$0.50`. Both variants must:

1. preserve one predicted outcome across all worlds for every initial action;
2. predict four distinct `line_details` outcomes;
3. make exact depth two select `customer_lookup`;
4. make depth one avoid `customer_lookup`;
5. make the lookup successor `line_details`; and
6. complete with zero reasoning tokens and finite scores.

Failure closes V5 without formal use.

## Formal Ranking Gate

Exactly twelve requests, projected `$0.20`, hard cap `$1.00`. All gates are
conjunctive:

1. all variants parse; request count is exactly twelve; reasoning is zero;
2. at least 10/12 preserve observational equivalence for every initial action;
3. at least 10/12 predict four distinct line-detail outcomes;
4. depth one selects lookup 0/12;
5. depth two selects lookup at least 10/12;
6. lookup chooses line details at least 10/12;
7. predicted versus official depth-two action-value Spearman is at least `.50`;
8. mean top-one regret improves by at least `1.00` nat; and
9. depth two strictly beats depth one on at least 10/12 variants.

Failure closes V5. Passage authorizes only the frozen confirmation.

## Conditional Paired Confirmation

Exactly twenty-four fresh requests, projected `$0.40`, hard cap `$1.50`.
The same jointly predicted table is shared by:

- exact depth-two planning;
- sequential greedy depth-one planning; and
- a seeded random root/follow-up control.

Each predicted initial action must have one observable category, so its
successor is executable without truth leakage. Chosen two-action sequences are
evaluated against the official four-world simulator with common hidden worlds.
Endpoints are final posterior entropy, trapezoidal entropy AUC, and truth log
posterior.

Passage requires:

1. all 24 variants and exact requests with zero reasoning;
2. depth two selects lookup and beats greedy on at least 20/24;
3. depth two beats random on at least 16/24;
4. mean final-entropy gain over greedy at least `1.00` nat with positive paired
   90% bootstrap lower bound;
5. mean final-entropy gain over random at least `.50` nat with positive lower
   bound; and
6. paired truth-log-posterior lower bounds above zero versus both controls.

Even passage is evidence for LLM-generated semantic likelihoods plus exact
non-myopic planning, not unaided LLM planning or path-dependent support.
