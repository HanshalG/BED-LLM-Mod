# Animals CA-BED Shared-Tree V10 Preregistration

Date frozen: 2026-07-24

## Question

Can exact depth-two lookahead over LLM-generated binary questions outperform
one-step EIG when both methods use the same fixed hypothesis support, the same
LLM semantic likelihood table, and the same realized answers?

This is a distinct route from the failed path-dependent Animals and MovieLens
experiments. The LLM remains necessary for candidate-question generation and
for `p(Yes | animal, question)`, but simulated and live belief updates are exact
Bayesian updates on a fixed support. This follows the CA-BED architecture and
tests whether semantic likelihoods are coherent enough for two-step planning
before reintroducing generative support dynamics.

## Frozen Design

- Selection seed: `24310`.
- Model: Gemma 4 26B A4B, non-thinking, for question generation, semantic
  likelihoods, and hidden-animal answers. Semantic likelihoods use the model's
  prompt log-probabilities for the complete `Yes` and `No` labels; they do not
  use generated probability JSON or parser retries.
- Fixed support: 64 animals listed in the committed V10 config.
- Likelihood confidence smoothing:
  `p' = 0.7 * p_LLM + 0.3 * 0.5`.
- Root width: 4.
- Follow-up width: 3 per Yes/No branch.
- Horizon: exactly two questions.
- Initial prior: uniform.
- Each state receives two target-independent prehistory questions from the
  frozen cyclic schedule in the implementation. Their answers are observed
  before root candidates are generated.
- Root candidates are generated once per state. Depth one, depth two, and
  random controls share these candidates and their cached likelihood rows.
- For every root, branch-specific follow-up candidates are generated once and
  shared by every scorer.
- Depth-one score is immediate EIG.
- Depth-two score is immediate EIG plus the predictive-answer-weighted maximum
  follow-up EIG.
- Realized utility for every root uses the hidden animal's duplicated,
  independently requested root answer, then the duplicated answer to that
  root's branch-optimal follow-up.
- Primary realized endpoint is truth log-posterior gain, equivalently reduction
  in truth NLL. Entropy drop is secondary.
- No direct animal guesses, repeated history questions, incomplete candidate
  sets, answer disagreements, invalid probability rows, retries, replacement
  states, or post-result threshold changes are allowed.

Smoke targets, disjoint from formal:

1. Wombat
2. Aardvark

Formal targets, in frozen order:

1. Red fox
2. Reindeer
3. Bald eagle
4. Saltwater crocodile
5. Manatee
6. Horseshoe crab
7. Cheetah
8. Tasmanian devil
9. Coyote
10. African grey parrot
11. Alpaca
12. Giant panda
13. African elephant
14. Green sea turtle
15. Kangaroo
16. Praying mantis
17. Meerkat
18. Yak
19. Tiger shark
20. Bottlenose dolphin
21. Honey badger
22. Pangolin
23. Giant squid
24. Walrus

## Frozen Formal Gates

All gates are conjunctive.

1. All 24 states complete with four roots and both three-question branch menus.
2. Every duplicated hidden-animal answer agrees and is `Yes` or `No`.
3. Depth two selects a different root from depth one on at least 6/24 states.
4. Mean within-state Spearman correlation between depth-two score and realized
   truth log-posterior gain is at least `0.20`.
5. That mean correlation exceeds the corresponding depth-one correlation by at
   least `0.10`.
6. Mean paired truth-NLL improvement, depth two minus depth one, is at least
   `0.02` nats, its paired 90% bootstrap lower bound is above zero, and depth two
   wins at least 14/24 states.
7. Mean paired truth-NLL improvement over the seeded-random root is at least
   `0.02` nats with a paired 90% bootstrap lower bound above zero.
8. Mean final entropy under depth two is not more than `0.02` nats worse than
   depth one.

Passing authorizes a fresh paired sequential confirmation with untouched
targets. Failure closes this exact fixed-support, epsilon-0.7, width-4/3 design;
it does not authorize threshold repair or a support/model swap on these targets.
