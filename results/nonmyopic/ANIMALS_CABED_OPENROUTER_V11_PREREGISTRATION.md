# Animals CA-BED OpenRouter V11 Preregistration

Date frozen: 2026-07-24

## Question

Can exact depth-two lookahead over LLM-generated animal questions outperform
one-step EIG when both methods use the same fixed support, shared semantic
likelihood table, and realized answers?

V11 is a distinct OpenRouter-native likelihood interface for the untouched V10
targets. It is not a rerun of any observed endpoint. OatML execution remains
paused.

## Load-Bearing LLM Work

Gemma 4 26B A4B runs non-thinking at temperature zero and:

1. generates four root questions and three branch-specific follow-ups;
2. labels all 64 support animals Yes or No for each generated question in one
   strict target-blind semantic batch; and
3. independently answers each realized hidden-animal question twice.

The semantic labels define raw probabilities 1/0. Frozen confidence smoothing
of 0.7 produces likelihoods 0.85/0.15. Belief updates, immediate EIG, and
depth-two tree evaluation are exact after those LLM outputs. The model never
sees which support animal is the realized target during semantic batching.

This replaces V10's local prompt-logprob primitive, which is unavailable
through the current OpenRouter adapter. It avoids generated numerical
probabilities and keeps the semantic forward model auditable. No reasoning
tokens, probability fallback, parser repair, response resampling, replacement
state, or post-result threshold change is allowed.

## Frozen Design

- Selection seed and target split: unchanged from untouched V10 seed 24310.
- Fixed support: the same committed 64 animals.
- Root/follow-up widths: 4/3.
- Uniform initial prior and two frozen prehistory questions per state.
- Depth one: immediate EIG.
- Depth two: immediate EIG plus expected best branch follow-up EIG.
- Random: seeded random root from the same shared tree.
- Every root is realized for retrospective within-state ranking.
- Primary endpoint: truth log-posterior gain, or truth-NLL reduction.
- Secondary endpoint: posterior entropy.

## Stages

### Serving Smoke

- Targets: Wombat and Aardvark.
- Hard run cap: $0.75.
- Requires two complete shared trees, valid binary semantic classifications,
  duplicate-answer agreement, finite complementary likelihoods, and zero
  reasoning tokens.
- No efficacy threshold is read from two states.

### Formal Ranking Gate

- The 24 V10 formal targets remain untouched.
- Hard run cap: $5.00.
- Run only after the smoke passes unchanged and after a live balance check.

The formal conjunction is unchanged from V10:

1. all 24 states and shared trees complete;
2. all duplicated answers agree and are binary;
3. depth two selects a distinct root on at least 6/24 states;
4. mean depth-two score versus realized truth-gain Spearman is at least 0.20;
5. that Spearman exceeds depth one's by at least 0.10;
6. mean truth-NLL improvement over depth one is at least 0.02 nats, its paired
   90% bootstrap lower bound is positive, and depth two wins at least 14/24;
7. mean truth-NLL improvement over seeded random is at least 0.02 nats with a
   positive paired 90% bootstrap lower bound; and
8. mean final entropy is no more than 0.02 nats worse than depth one.

Passing authorizes a fresh sequential confirmation. Failure closes this exact
batched-label, epsilon-0.7, width-4/3 design.
