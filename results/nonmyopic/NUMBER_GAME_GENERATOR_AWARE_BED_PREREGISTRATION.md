# Number Game Generator-Aware BED Development Preregistration

Date frozen: 2026-07-28, before any Number Game model response.

## Question

Can depth-two BED select a better first query by planning over the LLM's
branch-conditioned hypothesis proposal process, rather than treating the current
particle support as fixed?

This is a fresh LLM-native environment motivated by Chari and Pattanaik,
*Wild Guesses and Mild Guesses in Active Concept Learning* (2026). It is a
reproduction and method extension, not an execution of official released code.

## Frozen Protocol

- Domain: integers `0..100`, binary membership labels.
- Proposal model: `google/gemini-2.5-flash`, reasoning disabled, temperature
  zero, strict JSON Schema.
- The LLM proposes exactly 24 executable predicates per call.
- Predicates use a restricted, audited Python expression grammar. Invalid,
  constant, history-inconsistent, and extension-duplicate predicates are
  discarded without semantic repair or retry.
- Particle weights are uniform after deduplication. This deliberately tests
  LLM support generation rather than LLM probability calibration.
- One initial proposal call is followed by both possible branch proposal calls
  for eight root candidates: exactly 17 accepted model calls.
- Candidate roots include the global myopic-EIG root, the global classical
  fixed-support depth-two root, additional distinct high-EIG partitions, two
  seeded positive-test roots, and one seeded random root.
- Generator-aware depth-two utility is immediate EIG plus expected best
  second-query EIG after branch regeneration. Each simulated branch support is
  the sampled current truth particle plus valid generated branch hypotheses.
  The truth particle is never silently filtered from its own rollout.
- Matched controls are one-step EIG, classical fixed-support depth two, and a
  deterministic random candidate. All deployed two-query endpoints use the
  same generated branch support after the realized first label.
- Development targets are the paper's 12 published easy, medium, and hard
  Number Game rules. Metrics are truth-extension coverage, posterior-predictive
  Brier score, best surviving-rule Hamming error, and survivor count after two
  labels.
- Seed: 26068. Raw responses remain private; compiled rules, hashes, selection,
  accounting, and aggregate/per-target metrics are public.

## Mechanics Gates

All must pass:

1. Exact 17 accepted calls with exact attempt accounting, no reasoning tokens,
   no forced exits, and cost at most `$0.50`.
2. At least 16 valid unique initial rules and at least eight valid unique rules
   in every branch.
3. Generated supports differ between the two labels at every candidate root.
4. The generator-aware root differs from both myopic EIG and classical
   fixed-support depth two.
5. Its generator-aware utility exceeds the myopic root by at least `0.01` nat.
6. The selected root is unchanged in at least 75% of initial-particle
   leave-one-out replays.

Failure closes this exact tree. It may motivate a prospectively changed prompt,
grammar, support rule, or model, but not a repair or rescore of the same raw
responses.

## Development Signal

The target endpoint is developmental, not confirmatory. It authorizes an
independent-tree confirmation only if generator-aware depth two:

- lowers mean Brier score by at least 5% versus myopic EIG;
- lowers mean best-rule Hamming error by at least 5% versus myopic EIG;
- does not reduce truth-extension coverage versus myopic EIG; and
- is no worse than classical fixed-support depth two on both mean error
  metrics.

Any later confirmation must freeze independent proposal-tree seeds and paired
criteria before those responses are requested.
